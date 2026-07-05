#!/usr/bin/env python3
"""Dry-run/run wrapper for the ADNI expanded V4 [4,1,0], beta=4.6 experiment."""

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
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "runs" / "adni_expanded_v4_beta46_ch4_1_0.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and optionally execute the controlled V4 [4,1,0], beta=4.6 retraining command.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python-executable", default=None)
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
        "n_iter_rf",
        "n_iter_gb",
        "n_iter_xgb",
        "n_iter_mlp",
        "classifier_n_iter_json",
    ]
    unknown = sorted(set(params) - set(ordered))
    if unknown:
        raise RuntimeError(f"Unknown config parameters: {unknown}")
    for name in ordered:
        append_arg(command, name, params.get(name))
    return command


def soft_preflight(config: Dict[str, Any], dry_run: bool) -> None:
    paths = config["paths"]
    for key in ["training_script", "global_tensor_path", "metadata_path"]:
        path = resolve_path(paths[key])
        if path.exists():
            print(f"Preflight OK: {key}: {path}")
        elif dry_run:
            print(f"Preflight warning: {key} does not exist yet: {path}")
        else:
            raise FileNotFoundError(f"{key} not found: {path}")


def describe_output_path(config: Dict[str, Any]) -> None:
    output_dir = resolve_path(config["paths"]["output_dir"])
    big_disk = Path(config["paths"]["big_disk_output_dir"])
    print(f"Configured output_dir: {output_dir}")
    print(f"Configured big-disk target: {big_disk}")
    if output_dir.is_symlink():
        print(f"Output symlink target: {output_dir.resolve()}")
        print(f"Output realpath: {output_dir.resolve()}")
    elif output_dir.exists():
        print("Output path exists but is not a symlink.")
        print(f"Output realpath: {output_dir.resolve()}")
    else:
        print("Output path does not exist yet.")
        print(f"Expected output realpath after symlink preparation: {big_disk}")


def ensure_output_prepared_for_real_run(config: Dict[str, Any]) -> Path:
    output_dir = resolve_path(config["paths"]["output_dir"])
    big_disk = Path(config["paths"]["big_disk_output_dir"])
    if not output_dir.exists():
        raise RuntimeError(
            "Refusing to start training because output_dir is missing. Prepare the big-disk symlink first:\n"
            f"  mkdir -p {shlex.quote(str(big_disk))}\n"
            f"  ln -s {shlex.quote(str(big_disk))} {shlex.quote(str(output_dir))}"
        )
    if not output_dir.is_symlink():
        raise RuntimeError(f"Refusing to start training because output_dir is not a symlink: {output_dir}")
    if output_dir.resolve() != big_disk.resolve():
        raise RuntimeError(
            f"Refusing to start training because output symlink target is {output_dir.resolve()}, "
            f"expected {big_disk.resolve()}."
        )
    if any(output_dir.iterdir()):
        raise RuntimeError(f"Refusing to start training because output_dir target is not empty: {output_dir}")
    return output_dir


def write_manifest(config_path: Path, config: Dict[str, Any], command: List[str]) -> Path:
    output_dir = ensure_output_prepared_for_real_run(config)
    path = output_dir / "run_manifest.json"
    params = config["parameters"]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": False,
        "config_path": str(config_path),
        "run_name": config.get("run_name"),
        "selected_channels": params.get("channels_to_use"),
        "selected_channel_names": config.get("selected_channel_names"),
        "beta_vae": params.get("beta_vae"),
        "effective_requested_trial_budgets": {
            "logreg": params.get("n_iter_logreg"),
            "svm": params.get("n_iter_svm"),
        },
        "command": command,
        "command_shell": shlex.join(command),
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    python_executable = args.python_executable or config.get("python_executable") or sys.executable

    soft_preflight(config, args.dry_run)
    command = build_command(config, python_executable)

    params = config["parameters"]
    print(f"Selected channels: {params['channels_to_use']}")
    print(f"Selected channel names: {', '.join(config.get('selected_channel_names', []))}")
    print(f"Beta VAE: {params.get('beta_vae')}")
    print(f"Trial budgets: logreg={params.get('n_iter_logreg')}, svm={params.get('n_iter_svm')}")
    describe_output_path(config)
    print("\nCommand:")
    print(shlex.join(command))

    if args.dry_run:
        print("\nDry-run requested. Training was not launched.")
        return 0

    manifest_path = write_manifest(args.config, config, command)
    print(f"Run manifest written: {manifest_path}")
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
