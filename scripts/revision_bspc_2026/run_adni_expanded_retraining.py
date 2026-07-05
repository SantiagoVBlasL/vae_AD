#!/usr/bin/env python3
"""
Launch the ADNI_expanded_v1 beta=2.5 static-3-channel retraining run.

This wrapper only builds and executes the command for the canonical training
script. It does not modify scripts/run_vae_clf_ad_inference.py.

Dry-run behaviour:
  --dry-run  → validates inputs, prints command, writes run_manifest.dryrun.json.
  (no flag)  → full run; writes run_manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "runs" / "adni_expanded_v1_beta25_static3.json"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ADNI expanded retraining command from JSON config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command and write run_manifest.dryrun.json without launching training.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty output_dir.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume into an existing output_dir (implies --overwrite for directory check).",
    )
    parser.add_argument(
        "--python-executable",
        type=str,
        default=None,
        help="Python executable for the training script. Defaults to the current interpreter.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        default=False,
        help="Skip input validation before building the command.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_git_hash() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def resolve_path(value: str, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base / path


def append_arg(command: List[str], name: str, value: Any) -> None:
    flag = f"--{name}"
    if isinstance(value, bool):
        if value:
            command.append(flag)
        return
    if value is None:
        return
    command.append(flag)
    if isinstance(value, list):
        command.extend(str(v) for v in value)
    else:
        command.append(str(value))


# ---------------------------------------------------------------------------
# Preflight validation
# ---------------------------------------------------------------------------

def preflight_check(
    config: Dict[str, Any], overwrite: bool, resume: bool, dry_run: bool = False
) -> None:
    """Validate inputs and output-dir policy without loading the full tensor."""
    paths = config["paths"]
    global_tensor_path = resolve_path(paths["global_tensor_path"])
    metadata_path = resolve_path(paths["metadata_path"])
    output_dir = resolve_path(paths["output_dir"])

    if not global_tensor_path.exists():
        raise FileNotFoundError(f"global_tensor_path not found: {global_tensor_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata_path not found: {metadata_path}")

    # Load only the small subject_ids array to check N without reading the full tensor.
    with np.load(str(global_tensor_path), allow_pickle=True) as npz:
        if "global_tensor_data" not in npz.files:
            raise RuntimeError(
                f"NPZ missing required key 'global_tensor_data': {global_tensor_path}"
            )
        if "subject_ids" not in npz.files:
            raise RuntimeError(
                f"NPZ missing required key 'subject_ids': {global_tensor_path}"
            )
        n_tensor = int(np.asarray(npz["subject_ids"]).reshape(-1).shape[0])

    meta = pd.read_csv(metadata_path)
    n_meta = len(meta)
    if n_tensor != n_meta:
        raise RuntimeError(
            f"Tensor subject count ({n_tensor}) != metadata row count ({n_meta}). "
            "Re-run the builder or check your config paths."
        )

    required_cols = {"SubjectID", "ResearchGroup_Mapped", "Age", "Sex"}
    missing_cols = sorted(required_cols - set(meta.columns))
    if missing_cols:
        raise RuntimeError(f"Metadata missing required columns: {missing_cols}")

    if not dry_run and output_dir.exists() and any(output_dir.iterdir()):
        if resume:
            print(f"INFO: output_dir exists and is non-empty; resuming: {output_dir}")
        elif overwrite:
            print(f"INFO: output_dir exists; --overwrite passed: {output_dir}")
        else:
            raise RuntimeError(
                f"output_dir exists and is non-empty: {output_dir}\n"
                "Use --overwrite to proceed, --resume to resume, "
                "or choose a different output_dir."
            )

    print(
        f"Preflight OK: tensor N={n_tensor}, metadata N={n_meta}, "
        f"output_dir={'exists' if output_dir.exists() else 'does not exist'}"
    )


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------

def build_command(config: Dict[str, Any], python_executable: str) -> List[str]:
    paths = config["paths"]
    parameters = config["parameters"]

    training_script = resolve_path(paths["training_script"])
    global_tensor_path = resolve_path(paths["global_tensor_path"])
    metadata_path = resolve_path(paths["metadata_path"])
    output_dir = resolve_path(paths["output_dir"])

    for label, path in {
        "training_script": training_script,
        "global_tensor_path": global_tensor_path,
        "metadata_path": metadata_path,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    command = [
        python_executable,
        str(training_script),
        "--global_tensor_path",
        str(global_tensor_path),
        "--metadata_path",
        str(metadata_path),
        "--output_dir",
        str(output_dir),
    ]

    ordered_parameter_names = [
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

    unknown = sorted(set(parameters) - set(ordered_parameter_names))
    if unknown:
        raise RuntimeError(f"Unknown parameters in config: {unknown}")

    for name in ordered_parameter_names:
        if name in parameters:
            append_arg(command, name, parameters[name])

    return command


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def write_manifest(
    config_path: Path,
    config: Dict[str, Any],
    command: List[str],
    dry_run: bool,
) -> Path:
    output_dir = resolve_path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Unambiguous naming: dryrun suffix for dry runs.
    manifest_name = "run_manifest.dryrun.json" if dry_run else "run_manifest.json"
    manifest_path = output_dir / manifest_name

    tensor_path = resolve_path(config["paths"]["global_tensor_path"])
    metadata_path = resolve_path(config["paths"]["metadata_path"])

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(dry_run),
        "git_hash": get_git_hash(),
        "config_path": str(config_path.resolve()),
        "run_name": config.get("run_name"),
        "command": command,
        "command_shell": shlex.join(command),
        "inputs": {
            "global_tensor_path": str(tensor_path),
            "global_tensor_sha256": sha256_file(tensor_path) if tensor_path.exists() else None,
            "metadata_path": str(metadata_path),
            "metadata_sha256": sha256_file(metadata_path) if metadata_path.exists() else None,
        },
        "parameters": config["parameters"],
        "paths": {k: str(resolve_path(v)) for k, v in config["paths"].items()},
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    python_executable = args.python_executable or config.get("python_executable") or sys.executable

    if not args.skip_preflight:
        preflight_check(
            config,
            overwrite=args.overwrite,
            resume=args.resume,
            dry_run=args.dry_run,
        )

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
