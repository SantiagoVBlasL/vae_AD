#!/usr/bin/env python3
"""Dry-run/run wrapper for ADNI V4 [4,1,0] beta=2.5 half-FC experiment."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/runs/adni_expanded_v4_beta25_ch4_1_0_half_fc.json"
HELPER_PATH = PROJECT_ROOT / "scripts/revision_bspc_2026/run_adni_expanded_v4_beta25_ch4_1_0_linearout.py"

CONTROLLED_VALUES = {
    "channels_to_use": [4, 1, 0],
    "classifier_types": ["logreg", "svm"],
    "classifier_stratify_cols": ["Sex"],
    "metadata_features": ["Age", "Sex"],
    "num_conv_layers_encoder": 4,
    "decoder_type": "convtranspose",
    "beta_vae": 2.5,
    "dropout_rate_vae": 0.15,
    "latent_dim": 256,
    "vae_final_activation": "tanh",
    "intermediate_fc_dim_vae": "half",
    "use_layernorm_vae_fc": False,
    "n_iter_logreg": 300,
    "n_iter_svm": 300,
}


def load_helper_module() -> Any:
    spec = importlib.util.spec_from_file_location("linearout_wrapper_helpers", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import wrapper helpers from {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HELPERS = load_helper_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and optionally execute the ADNI V4 [4,1,0] half-FC full-run command.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python-executable", default=None)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_controlled_config(config: Dict[str, Any]) -> None:
    params = config["parameters"]
    for key, expected in CONTROLLED_VALUES.items():
        actual = params.get(key)
        if actual != expected:
            raise RuntimeError(f"Controlled experiment guard failed for {key}: got {actual!r}, expected {expected!r}")
    metadata_features = list(params.get("metadata_features", []))
    banned_features = {"Manufacturer", "Site", "Site3"}
    if banned_features.intersection(metadata_features):
        raise RuntimeError(f"Manufacturer/Site cannot be predictive features: {metadata_features}")
    if list(params.get("classifier_stratify_cols", [])) != ["Sex"]:
        raise RuntimeError("Primary split must stay the original Sex-stratified setup.")
    if params.get("vae_final_activation") != "tanh":
        raise RuntimeError("Half-FC experiment must keep vae_final_activation=tanh.")


def manifest_payload(config_path: Path, config: Dict[str, Any], command: List[str], dry_run: bool) -> Dict[str, Any]:
    params = config["parameters"]
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "config_path": str(config_path),
        "run_name": config.get("run_name"),
        "controlled_change": "intermediate_fc_dim_vae:quarter_to_half",
        "dataset": "ADNI_v4_only",
        "no_oasis": True,
        "selected_channels": params.get("channels_to_use"),
        "selected_channel_names": config.get("selected_channel_names"),
        "beta_vae": params.get("beta_vae"),
        "dropout_rate_vae": params.get("dropout_rate_vae"),
        "latent_dim": params.get("latent_dim"),
        "vae_final_activation": params.get("vae_final_activation"),
        "intermediate_fc_dim_vae": params.get("intermediate_fc_dim_vae"),
        "use_layernorm_vae_fc": params.get("use_layernorm_vae_fc"),
        "classifier_stratify_cols": params.get("classifier_stratify_cols"),
        "metadata_features": params.get("metadata_features"),
        "manufacturer_or_site_used_as_predictive_feature": bool(
            {"Manufacturer", "Site", "Site3"}.intersection(set(params.get("metadata_features", [])))
        ),
        "trial_budgets": {
            "logreg": params.get("n_iter_logreg"),
            "svm": params.get("n_iter_svm"),
        },
        "output_dir": str(resolve_path(config["paths"]["output_dir"])),
        "big_disk_output_dir": config["paths"]["big_disk_output_dir"],
        "command": command,
        "command_shell": shlex.join(command),
    }


def write_manifest(config_path: Path, config: Dict[str, Any], command: List[str]) -> Path:
    output_dir = HELPERS.ensure_output_prepared_for_real_run(config)
    path = output_dir / "run_manifest.json"
    path.write_text(
        json.dumps(manifest_payload(config_path, config, command, dry_run=False), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    python_executable = args.python_executable or config.get("python_executable") or sys.executable

    HELPERS.soft_preflight(config, args.dry_run)
    validate_controlled_config(config)
    command = HELPERS.build_command(config, python_executable)

    params = config["parameters"]
    print(f"Run name: {config.get('run_name')}")
    print("Controlled change: intermediate_fc_dim_vae quarter -> half")
    print(f"Selected channels: {params['channels_to_use']}")
    print(f"Selected channel names: {', '.join(config.get('selected_channel_names', []))}")
    print(
        "Architecture constants: "
        f"beta={params.get('beta_vae')}, latent_dim={params.get('latent_dim')}, "
        f"dropout={params.get('dropout_rate_vae')}, final_activation={params.get('vae_final_activation')}, "
        f"LayerNorm={params.get('use_layernorm_vae_fc')}"
    )
    print(f"Intermediate FC setting: {params.get('intermediate_fc_dim_vae')}")
    print(f"Classifier stratification extras: {params.get('classifier_stratify_cols')}")
    print(f"Predictive metadata features: {params.get('metadata_features')}")
    print("Manufacturer/Site predictive feature: False")
    print(f"Trial budgets: logreg={params.get('n_iter_logreg')}, svm={params.get('n_iter_svm')}")
    HELPERS.describe_output_path(config)
    print("\nCommand:")
    print(shlex.join(command))

    if args.dry_run:
        dry_manifest = manifest_payload(config_path, config, command, dry_run=True)
        print("\nDry-run manifest preview:")
        print(json.dumps(dry_manifest, indent=2, sort_keys=True))
        print("\nDry-run requested. Training was not launched.")
        return 0

    manifest_path = write_manifest(config_path, config, command)
    print(f"Run manifest written: {manifest_path}")
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
