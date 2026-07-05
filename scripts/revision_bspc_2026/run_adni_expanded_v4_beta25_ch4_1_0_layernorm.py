#!/usr/bin/env python3
"""Dry-run/run wrapper for ADNI V4 [4,1,0] beta=2.5 LayerNorm experiment."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/runs/adni_expanded_v4_beta25_ch4_1_0_layernorm.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python-executable", default=None)
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def build_command(config: Dict[str, Any], python_executable: str) -> List[str]:
    paths = config["paths"]
    params = config["parameters"]
    command = [
        python_executable,
        str(resolve_path(paths["training_script"])),
        "--global_tensor_path",
        str(resolve_path(paths["global_tensor_path"])),
        "--metadata_path",
        str(resolve_path(paths["metadata_path"])),
        "--output_dir",
        str(resolve_path(paths["output_dir"])),
    ]
    ordered = [
        "channels_to_use",
        "classifier_types",
        "classifier_stratify_cols",
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
        "dropout_rate_vae",
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
        "tune_sampler_params",
        "mlp_classifier_hidden_layers",
        "n_iter_logreg",
        "n_iter_svm",
    ]
    unknown = sorted(set(params) - set(ordered))
    if unknown:
        raise RuntimeError(f"Unknown config parameters: {unknown}")
    for name in ordered:
        append_arg(command, name, params.get(name))
    return command


def preflight_inputs(config: Dict[str, Any]) -> None:
    for key in ["training_script", "global_tensor_path", "metadata_path"]:
        path = resolve_path(config["paths"][key])
        if not path.exists():
            raise FileNotFoundError(f"{key} not found: {path}")


def ensure_output_symlink_empty(config: Dict[str, Any]) -> Path:
    output_dir = resolve_path(config["paths"]["output_dir"])
    big_disk = Path(config["paths"]["big_disk_output_dir"])
    if not output_dir.exists():
        raise RuntimeError(
            "Output symlink is missing. Prepare it first:\n"
            f"  mkdir -p {shlex.quote(str(big_disk))}\n"
            f"  ln -s {shlex.quote(str(big_disk))} {shlex.quote(str(output_dir))}"
        )
    if not output_dir.is_symlink():
        raise RuntimeError(f"Refusing real run: output_dir is not a symlink: {output_dir}")
    if output_dir.resolve() != big_disk.resolve():
        raise RuntimeError(f"Refusing real run: output_dir resolves to {output_dir.resolve()}, expected {big_disk.resolve()}")
    if any(output_dir.iterdir()):
        raise RuntimeError(f"Refusing real run: output_dir target is not empty: {output_dir}")
    return output_dir


def write_manifest(config_path: Path, config: Dict[str, Any], command: List[str]) -> Path:
    output_dir = ensure_output_symlink_empty(config)
    manifest_path = output_dir / "run_manifest.json"
    params = config["parameters"]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "run_name": config.get("run_name"),
        "channels_to_use": params["channels_to_use"],
        "selected_channel_names": config.get("selected_channel_names", []),
        "beta_vae": params["beta_vae"],
        "use_layernorm_vae_fc": params["use_layernorm_vae_fc"],
        "trial_budgets": {"logreg": params.get("n_iter_logreg"), "svm": params.get("n_iter_svm")},
        "command": command,
        "command_shell": shlex.join(command),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    python_executable = args.python_executable or config.get("python_executable") or sys.executable
    preflight_inputs(config)
    command = build_command(config, python_executable)
    params = config["parameters"]
    output_dir = resolve_path(config["paths"]["output_dir"])
    big_disk = Path(config["paths"]["big_disk_output_dir"])
    print(f"Selected channels: {params['channels_to_use']}")
    print(f"Selected channel names: {', '.join(config.get('selected_channel_names', []))}")
    print(f"beta_vae: {params['beta_vae']}")
    print(f"use_layernorm_vae_fc: {params['use_layernorm_vae_fc']}")
    print(f"trial budgets: logreg={params.get('n_iter_logreg')}, svm={params.get('n_iter_svm')}")
    print(f"output_dir: {output_dir}")
    print(f"big_disk_output_dir: {big_disk}")
    print(f"output_dir_is_symlink: {output_dir.is_symlink()}")
    print(f"output_realpath: {output_dir.resolve() if output_dir.exists() else big_disk}")
    print("\nCommand:")
    print(shlex.join(command))
    if args.dry_run:
        print("\nDry-run requested. Training was not launched.")
        return 0
    manifest = write_manifest(args.config, config, command)
    print(f"Run manifest written: {manifest}")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
