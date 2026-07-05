#!/usr/bin/env python3
"""FAST 3x3 decoder-type audit under offdiag_channelmean reconstruction loss.

Default mode is dry-run/preflight only. Real training requires
``--confirm-training``. Stage A trains the FAST VAE with a dummy canonical
logreg readout, and Stage B runs classifier-only logreg_l2 on saved latent
mu + Age + Sex. Ranking and recommendation use Stage B only.
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_EXE = "/home/diego/anaconda3/envs/vae_ad/bin/python"
TRAINING_SCRIPT = PROJECT_ROOT / "scripts/run_vae_clf_ad_inference.py"
READOUT_SCRIPT = PROJECT_ROOT / "scripts/revision_bspc_2026/run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep.py"

OUTPUT_ROOT = PROJECT_ROOT / "results/revision_bspc_2026/decoder_type_fast3x3_offdiag_channelmean"
BIG_DISK_ROOT = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026/decoder_type_fast3x3_offdiag_channelmean")

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

CHANNEL_SETS = [[1], [1, 0, 2]]
DECODER_TYPES = [
    ("convtranspose", "convtranspose"),
    ("upsample", "upsample_conv"),
]
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
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--big-disk-root", type=Path, default=BIG_DISK_ROOT)
    parser.add_argument("--candidate", default="all", help="Candidate run_key or 'all'.")
    parser.add_argument("--dry-run", action="store_true", help="Preflight only; do not train.")
    parser.add_argument("--confirm-training", action="store_true", help="Required for real Stage A/Stage B launch.")
    parser.add_argument("--resume", action="store_true", help="Skip candidates with complete Stage B readout.")
    parser.add_argument("--force-clean", action="store_true", help="Move stale candidate output to timestamped quarantine before running.")
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


def candidate_key(channels: Sequence[int], decoder_label: str) -> str:
    return f"{channel_key(channels)}_{decoder_label}"


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
    start = tokens.index(flag) + 1
    out: List[str] = []
    for token in tokens[start:]:
        if token.startswith("--"):
            break
        out.append(token)
    return out


def require_flag_values(tokens: Sequence[str], flag: str, expected: Sequence[str], context: str) -> None:
    got = values_after_flag(tokens, flag)
    if got != list(expected):
        raise RuntimeError(f"{context}: expected {flag} {' '.join(expected)}, got {got}")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    json.loads(path.read_text(encoding="utf-8"))


def markdown_table(df: pd.DataFrame, cols: Optional[Sequence[str]] = None, max_rows: int = 100) -> str:
    if df.empty:
        return "No rows.\n"
    if cols is None:
        cols = list(df.columns)
    sub = df.loc[:, [c for c in cols if c in df.columns]].head(max_rows)
    lines = ["| " + " | ".join(sub.columns) + " |", "| " + " | ".join(["---"] * len(sub.columns)) + " |"]
    for _, row in sub.iterrows():
        vals: List[str] = []
        for col in sub.columns:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append(f"{val:.6f}" if np.isfinite(val) else "NA")
            else:
                vals.append(str(val).replace("\n", " ").replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def write_pair(output_root: Path, name: str, df: pd.DataFrame, cols: Optional[Sequence[str]] = None, max_rows: int = 100) -> None:
    df.to_csv(output_root / f"{name}.csv", index=False)
    (output_root / f"{name}.md").write_text(markdown_table(df, cols=cols, max_rows=max_rows), encoding="utf-8")


def inspect_tensor(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as npz:
        key = "tensor" if "tensor" in npz.files else ("X" if "X" in npz.files else npz.files[0])
        arr = npz[key]
        python_bp = False
        if "python_bandpass_applied" in npz.files:
            raw = npz["python_bandpass_applied"]
            python_bp = bool(raw.item() if raw.shape == () else raw)
        return {
            "tensor_key": key,
            "shape": tuple(int(x) for x in arr.shape),
            "dtype": str(arr.dtype),
            "python_bandpass_applied": python_bp,
            "npz_files": list(npz.files),
        }


def output_dirs(output_root: Path, big_root: Path, run_key: str) -> Dict[str, Path]:
    run_dir = output_root / "runs" / run_key
    big_dir = big_root / run_key
    return {
        "run_dir": run_dir,
        "big_dir": big_dir,
        "readout_dir": run_dir / "classifier_only_readout",
    }


def candidate_config(output_root: Path, big_root: Path, channels: Sequence[int], decoder_label: str, decoder_type: str) -> Dict[str, Any]:
    key = candidate_key(channels, decoder_label)
    dirs = output_dirs(output_root, big_root, key)
    params = dict(FAST_PARAMS)
    params["channels_to_use"] = list(channels)
    params["decoder_type"] = decoder_type
    return {
        "run_name": f"decoder_type_fast3x3_offdiag_channelmean_v5_1b_{key}",
        "description": "FAST 3x3 decoder-type audit using offdiag_channelmean_sum. Stage A canonical classifier is dummy/ignored; Stage B logreg_l2 is the readout.",
        "created_utc": now_utc(),
        "dataset_branch": "v5_1b",
        "run_key": key,
        "channels": list(channels),
        "channel_names_master_in_tensor_order": [CHANNEL_NAMES[i] for i in range(7)],
        "selected_channel_names": channel_names(channels),
        "decoder_label": decoder_label,
        "decoder_type": decoder_type,
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
        "--readout-feature-sets",
        "z_plus_age_sex",
        "--outer-folds",
        str(int(params["outer_folds"])),
        "--inner-folds",
        str(int(params["inner_folds"])),
        "--reuse-latent-cache",
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
    decoder = values_after_flag(tokens, "--decoder_type")
    if decoder not in (["convtranspose"], ["upsample_conv"]):
        raise RuntimeError(f"Stage A decoder_type must be convtranspose or upsample_conv, got {decoder}")
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
    require_flag_values(tokens, "--readout-feature-sets", ["z_plus_age_sex"], "Stage B")
    require_flag_values(tokens, "--outer-folds", ["3"], "Stage B")
    require_flag_values(tokens, "--inner-folds", ["3"], "Stage B")


def validate_config(config: Dict[str, Any]) -> None:
    params = config["parameters"]
    required = {
        "outer_folds": 3,
        "inner_folds": 3,
        "latent_dim": 256,
        "epochs_vae": 960,
        "cyclical_beta_n_cycles": 12,
        "lr_scheduler_T0": 80,
        "beta_vae": 2.5,
        "recon_loss_mode": "offdiag_channelmean_sum",
        "dropout_rate_vae": 0.15,
        "vae_dropout_scope": "legacy_all",
        "vae_block_order": "legacy_act_norm",
        "vae_final_activation": "tanh",
        "intermediate_fc_dim_vae": "quarter",
        "vae_train_sampler_strategy": "none",
        "n_iter_logreg": 1,
    }
    for key, expected in required.items():
        if params.get(key) != expected:
            raise RuntimeError(f"{config['run_key']}.{key}: expected {expected!r}, got {params.get(key)!r}")
    if params["epochs_vae"] / params["cyclical_beta_n_cycles"] != 80:
        raise RuntimeError("Expected 80-epoch beta cycles")
    if params["classifier_types"] != ["logreg"]:
        raise RuntimeError("Stage A must use dummy canonical logreg only.")
    if params["decoder_type"] not in {"convtranspose", "upsample_conv"}:
        raise RuntimeError("decoder_type must be convtranspose or upsample_conv")
    if params["metadata_features"] != ["Age", "Sex"]:
        raise RuntimeError("metadata_features must be ['Age', 'Sex']")


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


def create_or_update_symlink(link_path: Path, target_path: Path, force_clean: bool = False) -> List[Path]:
    quarantines: List[Path] = []
    link_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink():
        if symlink_points_to(link_path, target_path):
            markers = stale_markers(link_path)
            if markers and not force_clean:
                raise RuntimeError(f"Refusing to run with stale candidate outputs in {link_path}. Use --force-clean.")
            if markers and force_clean:
                quarantine = quarantine_directory_contents(target_path)
                if quarantine is not None:
                    quarantines.append(quarantine)
            return quarantines
        if not force_clean:
            raise RuntimeError(f"Existing symlink points elsewhere: {link_path} -> {os.readlink(link_path)}. Use --force-clean.")
        quarantine = quarantine_existing_path(link_path)
        if quarantine is not None:
            quarantines.append(quarantine)
    elif link_path.exists():
        if not force_clean:
            raise RuntimeError(f"Refusing to replace existing non-symlink path: {link_path}. Use --force-clean.")
        quarantine = quarantine_existing_path(link_path)
        if quarantine is not None:
            quarantines.append(quarantine)
    if not link_path.exists() and not link_path.is_symlink():
        link_path.symlink_to(target_path, target_is_directory=True)
    if not link_path.is_symlink() or not symlink_points_to(link_path, target_path):
        raise RuntimeError(f"Failed to prepare run_dir symlink: {link_path} -> {target_path}")
    return quarantines


def prepare_run_symlink(row: pd.Series, force_clean: bool) -> List[Path]:
    run_dir = resolve(row["run_dir"])
    target = Path(row["big_disk_output_dir"])
    return create_or_update_symlink(run_dir, target, force_clean=force_clean)


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


def build_plan(output_root: Path, big_root: Path, python_executable: str) -> pd.DataFrame:
    configs_dir = output_root / "configs"
    rows: List[Dict[str, Any]] = []
    for channels in CHANNEL_SETS:
        for decoder_label, decoder_type in DECODER_TYPES:
            config = candidate_config(output_root, big_root, channels, decoder_label, decoder_type)
            validate_config(config)
            config_path = configs_dir / f"{config['run_key']}.json"
            write_json(config_path, config)
            stage_a = build_stage_a_command(config, python_executable)
            stage_b = build_stage_b_command(config, python_executable)
            rows.append(
                {
                    "candidate_index": len(rows) + 1,
                    "run_key": config["run_key"],
                    "channels": json.dumps(list(channels)),
                    "selected_channel_names": " | ".join(config["selected_channel_names"]),
                    "n_channels": len(channels),
                    "decoder_label": decoder_label,
                    "decoder_type": decoder_type,
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
    expected_keys = [
        candidate_key(channels, decoder_label)
        for channels in CHANNEL_SETS
        for decoder_label, _ in DECODER_TYPES
    ]
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


def collect_primary_from_readout(row: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    readout_dir = resolve(row["readout_dir"])
    foldwise_path = readout_dir / "classifier_sweep_foldwise_metrics.csv"
    pooled_path = readout_dir / "classifier_sweep_pooled_metrics.csv"
    subgroup_path = readout_dir / "classifier_sweep_subgroup_metrics_by_manufacturer.csv"
    if not all(p.exists() for p in [foldwise_path, pooled_path, subgroup_path]):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    foldwise = pd.read_csv(foldwise_path)
    pooled = pd.read_csv(pooled_path)
    subgroup = pd.read_csv(subgroup_path)
    for df in [foldwise, pooled, subgroup]:
        df["run_key"] = row["run_key"]
        df["channels"] = row["channels"]
        df["selected_channel_names"] = row["selected_channel_names"]
        df["decoder_label"] = row["decoder_label"]
        df["decoder_type"] = row["decoder_type"]
        df["n_channels"] = row["n_channels"]
    return pooled, foldwise, subgroup


def collect_rate_distortion_summary(plan: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in plan.iterrows():
        run_dir = resolve(row["run_dir"])
        beta_max = float(FAST_PARAMS["beta_vae"])
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
            d_val = float(best.get("D_val", np.nan))
            r_val = float(best.get("R_val_nats", np.nan))
            ratio = r_val / d_val if np.isfinite(d_val) and d_val != 0 else np.nan
            rows.append(
                {
                    "run_key": row["run_key"],
                    "channels": row["channels"],
                    "decoder_label": row["decoder_label"],
                    "decoder_type": row["decoder_type"],
                    "fold": fold,
                    "best_epoch": int(best["epoch"]),
                    "final_epoch": int(pd.to_numeric(df["epoch"], errors="coerce").max()),
                    "best_val_l_beta_max": float(best.get("L_val_betaMax", np.nan)),
                    "best_val_recon_D": d_val,
                    "best_val_kld_R_nats": r_val,
                    "best_val_kld_over_recon": ratio,
                    "best_val_beta_kld_over_recon": beta_max * ratio if np.isfinite(ratio) else np.nan,
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
                    "decoder_label": row["decoder_label"],
                    "decoder_type": row["decoder_type"],
                    "fold": fold,
                    "acc_raw": acc_raw,
                    "acc_latent": acc_latent,
                    "latent_minus_raw": acc_latent - acc_raw if np.isfinite(acc_raw) and np.isfinite(acc_latent) else np.nan,
                    "source_file": str(path),
                }
            )
    return pd.DataFrame(rows)


def aggregate_metrics(output_root: Path, plan: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    pooled_rows: List[pd.DataFrame] = []
    foldwise_rows: List[pd.DataFrame] = []
    subgroup_rows: List[pd.DataFrame] = []
    for _, row in plan.iterrows():
        pooled, foldwise, subgroup = collect_primary_from_readout(row)
        if not pooled.empty:
            pooled_rows.append(pooled)
        if not foldwise.empty:
            foldwise_rows.append(foldwise)
        if not subgroup.empty:
            subgroup_rows.append(subgroup)
    pooled_all = pd.concat(pooled_rows, ignore_index=True, sort=False) if pooled_rows else pd.DataFrame()
    foldwise_all = pd.concat(foldwise_rows, ignore_index=True, sort=False) if foldwise_rows else pd.DataFrame()
    subgroup_all = pd.concat(subgroup_rows, ignore_index=True, sort=False) if subgroup_rows else pd.DataFrame()
    if pooled_all.empty:
        placeholder = pd.DataFrame(columns=["run_key", "channels", "decoder_label", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "status"])
        for name in ["primary_results", "foldwise_metrics", "rate_distortion_by_candidate", "scanner_leakage_by_candidate"]:
            write_pair(output_root, name, placeholder)
        write_recommendation(output_root, placeholder, pd.DataFrame())
        return {"primary": placeholder, "foldwise": placeholder, "rate_distortion": placeholder, "scanner_leakage": placeholder}

    primary = pooled_all[
        (pooled_all["model_name"].eq(PRIMARY_MODEL))
        & (pooled_all["threshold_strategy"].eq(PRIMARY_THRESHOLD))
        & (pooled_all.get("readout_feature_set", "z_plus_age_sex").eq("z_plus_age_sex") if "readout_feature_set" in pooled_all.columns else True)
    ].copy()
    primary = primary.sort_values(["channels", "decoder_label"]).reset_index(drop=True)
    foldwise_primary = foldwise_all[
        (foldwise_all["model_name"].eq(PRIMARY_MODEL))
        & (foldwise_all["threshold_strategy"].eq(PRIMARY_THRESHOLD))
    ].copy()
    rd = collect_rate_distortion_summary(plan)
    leakage = collect_scanner_leakage_summary(plan)
    write_pair(
        output_root,
        "primary_results",
        primary.sort_values(["auc", "pr_auc"], ascending=False),
        ["run_key", "channels", "decoder_label", "decoder_type", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "tn", "fp", "fn", "tp"],
    )
    write_pair(
        output_root,
        "foldwise_metrics",
        foldwise_primary.sort_values(["run_key", "fold"]),
        ["run_key", "channels", "decoder_label", "fold", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "tn", "fp", "fn", "tp"],
        max_rows=100,
    )
    write_pair(output_root, "rate_distortion_by_candidate", rd, max_rows=100)
    write_pair(output_root, "scanner_leakage_by_candidate", leakage, max_rows=100)
    write_recommendation(output_root, primary, leakage)
    if not subgroup_all.empty:
        write_pair(output_root, "manufacturer_subgroup_metrics", subgroup_all, max_rows=200)
    return {"primary": primary, "foldwise": foldwise_primary, "rate_distortion": rd, "scanner_leakage": leakage}


def write_recommendation(output_root: Path, primary: pd.DataFrame, leakage: pd.DataFrame) -> None:
    lines = [
        "# Final Recommendation",
        "",
        "FAST 3x3 is a decoder-type screening analysis, not a final performance estimate.",
        "",
    ]
    if primary.empty:
        lines += [
            "No completed Stage B readouts were available at aggregation time.",
            "",
            "Run the launcher with `--confirm-training` to execute Stage A/Stage B, then rerun `--aggregate-only`.",
        ]
    else:
        lines += ["## Matched Decoder Comparisons", ""]
        for channels, sub in primary.groupby("channels"):
            conv = sub[sub["decoder_label"].eq("convtranspose")]
            ups = sub[sub["decoder_label"].eq("upsample")]
            if conv.empty or ups.empty:
                lines.append(f"- `{channels}`: incomplete matched decoder comparison.")
                continue
            c = conv.iloc[0]
            u = ups.iloc[0]
            delta_auc = float(u["auc"] - c["auc"])
            delta_pr = float(u["pr_auc"] - c["pr_auc"])
            delta_ba = float(u["balanced_accuracy"] - c["balanced_accuracy"])
            delta_f1 = float(u["f1"] - c["f1"])
            promote = (
                delta_auc > 0
                and delta_pr > 0
                and delta_ba >= -0.005
                and delta_f1 >= -0.005
                and float(u["sensitivity"]) >= float(c["sensitivity"]) - 0.02
            )
            lines += [
                f"- `{channels}`: upsample minus convtranspose AUC `{delta_auc:+.6f}`, PR-AUC `{delta_pr:+.6f}`, BA `{delta_ba:+.6f}`, F1 `{delta_f1:+.6f}`. FULL recommendation: `{'consider_full_confirmation' if promote else 'do_not_promote_from_fast'}`.",
            ]
        best = primary.sort_values(["auc", "pr_auc"], ascending=False).iloc[0]
        lines += [
            "",
            "## Best FAST Candidate",
            "",
            f"Best Stage B AUC candidate: `{best['run_key']}`, AUC `{best['auc']:.6f}`, PR-AUC `{best['pr_auc']:.6f}`, BA `{best['balanced_accuracy']:.6f}`, F1 `{best['f1']:.6f}`.",
            "",
            "Recommend FULL only if an `upsample_conv` candidate improves AUC and PR-AUC versus its matched `convtranspose` baseline without worsening BA/F1, sensitivity, scanner/manufacturer leakage, or rate-distortion.",
        ]
    (output_root / "final_recommendation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(output_root: Path, plan: pd.DataFrame, tensor_info: Dict[str, Any]) -> None:
    lines = [
        "# Decoder-Type FAST 3x3 Audit",
        "",
        "## Scope",
        "",
        "- Dataset: v5.1b final dataset.",
        "- Decoder comparison: `convtranspose` baseline versus `upsample_conv` candidate.",
        "- Channel subsets: `[1]` and `[1,0,2]`.",
        "- VAE objective: `offdiag_channelmean_sum`.",
        "- Conditional VAE: `OFF`.",
        "- Python bandpass: `OFF`.",
        "- Stage A classifier: dummy canonical `logreg`, `n_iter_logreg=1`, ignored for ranking.",
        "- Stage B readout: classifier-only `logreg_l2` on latent `mu + Age + Sex`.",
        "- Non-0.5 thresholds: true inner-CV OOF predictions only.",
        "- FAST folds: `outer_folds=3`, `inner_folds=3`.",
        "- FAST horizon: `epochs_vae=960`, `cyclical_beta_n_cycles=12`, cycle length `80`, `T0=80`.",
        "",
        "## Tensor",
        "",
        f"- Shape: `{tensor_info['shape']}`.",
        f"- dtype: `{tensor_info['dtype']}`.",
        f"- python_bandpass_applied: `{tensor_info['python_bandpass_applied']}`.",
        "",
        "## Planned Candidates",
        "",
        markdown_table(plan, ["candidate_index", "run_key", "channels", "decoder_label", "decoder_type", "recon_loss_mode", "outer_folds", "inner_folds", "latent_dim", "epochs_vae", "cyclical_beta_n_cycles"], max_rows=20),
        "",
        "## Outputs",
        "",
        "- `run_manifest.csv/.md`",
        "- `primary_results.csv/.md`",
        "- `foldwise_metrics.csv/.md`",
        "- `rate_distortion_by_candidate.csv/.md`",
        "- `scanner_leakage_by_candidate.csv/.md`",
        "- `final_recommendation.md`",
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
            "outer_folds": 3,
            "inner_folds": 3,
            "recon_loss_mode": "offdiag_channelmean_sum",
            "decoder_types": ["convtranspose", "upsample_conv"],
            "python_bandpass_off": True,
            "conditional_vae_off": True,
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
    if tensor_info["python_bandpass_applied"]:
        raise RuntimeError("Expected python_bandpass_applied=False")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(METADATA_PATH)
    plan = build_plan(output_root, args.big_disk_root, args.python_executable)
    validate_plan(plan)
    write_pair(
        output_root,
        "run_manifest",
        plan,
        ["candidate_index", "run_key", "channels", "selected_channel_names", "decoder_label", "decoder_type", "recon_loss_mode", "outer_folds", "inner_folds", "latent_dim", "epochs_vae", "cyclical_beta_n_cycles", "cycle_len", "lr_scheduler_T0", "status"],
        max_rows=20,
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
        print(plan[["candidate_index", "run_key", "channels", "decoder_label", "decoder_type", "outer_folds", "inner_folds", "latent_dim", "epochs_vae", "cyclical_beta_n_cycles"]].to_string(index=False))
        print("\nDry-run command preview:")
        for _, row in selected.iterrows():
            print(f"\n[{row['run_key']}] Stage A\n{row['stage_a_command']}")
            print(f"[{row['run_key']}] Stage B\n{row['stage_b_command']}")
        print("\nDry-run complete. No training launched.")
    else:
        for _, row in selected.iterrows():
            result = run_candidate(row, resume=bool(args.resume), force_clean=bool(args.force_clean))
            execution_rows.append(result)
            if result["status"] not in {"complete", "skipped_complete"}:
                write_command_log(output_root, args, plan, selected, execution_rows)
                raise SystemExit(f"Candidate {row['run_key']} failed: {result}")
        aggregate_metrics(output_root, plan)
    write_readme(output_root, plan, tensor_info)
    write_command_log(output_root, args, plan, selected, execution_rows)


if __name__ == "__main__":
    main()
