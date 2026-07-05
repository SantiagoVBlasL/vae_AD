#!/usr/bin/env python3
"""Dry-run/run wrapper for ADNI_expanded_v3_all_available retraining."""

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
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "runs" / "adni_expanded_v3_beta25_static3.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and optionally execute the v3 expanded ADNI retraining command.",
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
        "classifier_calibrate",
        "classifier_use_class_weight",
        "gridsearch_scoring",
        "outer_folds",
        "inner_folds",
        "repeated_outer_folds_n_repeats",
        "epochs_vae",
        "vae_val_split_ratio",
        "early_stopping_patience_vae",
        "cyclical_beta_n_cycles",
        "beta_vae",
        "dropout_rate_vae",
        "latent_dim",
        "batch_size",
        "lr_scheduler_type",
        "lr_scheduler_T0",
        "lr_scheduler_eta_min",
        "weight_decay_vae",
        "n_jobs_gridsearch",
        "metadata_features",
        "save_fold_artefacts",
        "save_vae_training_history",
        "qc_analyze_distributions",
        "qc_check_scanner_leakage",
        "qc_rate_distortion",
        "qc_latent_information",
        "seed",
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


def write_manifest(config_path: Path, config: Dict[str, Any], command: List[str], dry_run: bool) -> Path:
    output_dir = resolve_path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / ("run_manifest.dryrun.json" if dry_run else "run_manifest.json")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(dry_run),
        "config_path": str(config_path),
        "run_name": config.get("run_name"),
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
    manifest_path = write_manifest(args.config, config, command, args.dry_run)
    print(f"Run manifest written: {manifest_path}")
    print("\nCommand:")
    print(shlex.join(command))
    if args.dry_run:
        print("\nDry-run requested. Training was not launched.")
        return 0
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
