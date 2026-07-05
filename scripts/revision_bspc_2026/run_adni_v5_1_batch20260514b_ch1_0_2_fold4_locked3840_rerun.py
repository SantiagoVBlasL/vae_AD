#!/usr/bin/env python3
"""Prepare/run a Fold-4-only locked-horizon rerun diagnostic for [1,0,2].

Default mode is dry-run/preflight. Real training requires --confirm-training.
There must be no scientific config diff versus the locked FULL baseline.

The actual training command additionally passes --outer_fold_indices_to_run 4
so no other outer folds are launched.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCKED_CONFIG = PROJECT_ROOT / "configs/runs/adni_v5_1_batch20260514b_ch1_0_2_mfrsplit_3840_final_candidate.json"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/runs/adni_v5_1_batch20260514b_ch1_0_2_fold4_locked3840_rerun.json"
TARGET_FOLD = 4
EXPECTED_SCIENTIFIC_DIFFS: dict[str, tuple[Any, Any]] = {}

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
    "vae_dropout_scope",
    "vae_block_order",
    "vae_train_sampler_strategy",
    "recon_loss_mode",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--locked-config", type=Path, default=LOCKED_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print commands without training.")
    parser.add_argument("--confirm-training", action="store_true", help="Required for real Fold-4 training launch.")
    parser.add_argument("--python-executable", default=None)
    parser.add_argument("--skip-stage-b-command", action="store_true", help="Only print/run Stage A.")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def normalized_params(config: dict[str, Any]) -> dict[str, Any]:
    params = dict(config["parameters"])
    params.setdefault("vae_dropout_scope", "legacy_all")
    params.setdefault("vae_block_order", "legacy_act_norm")
    params.setdefault("vae_train_sampler_strategy", "none")
    params.setdefault("recon_loss_mode", "mse_sum_batchmean_current")
    return params


def validate_scientific_diff(locked: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    locked_params = normalized_params(locked)
    cand_params = normalized_params(candidate)
    all_keys = sorted(set(locked_params) | set(cand_params))
    diffs: list[dict[str, Any]] = []
    unexpected: list[dict[str, Any]] = []
    for key in all_keys:
        if locked_params.get(key) != cand_params.get(key):
            item = {"parameter": key, "locked": locked_params.get(key), "candidate": cand_params.get(key)}
            diffs.append(item)
            if key not in EXPECTED_SCIENTIFIC_DIFFS or EXPECTED_SCIENTIFIC_DIFFS[key] != (locked_params.get(key), cand_params.get(key)):
                unexpected.append(item)
    if unexpected:
        raise RuntimeError("Unexpected scientific parameter diffs: " + json.dumps(unexpected, indent=2))
    for key, expected in EXPECTED_SCIENTIFIC_DIFFS.items():
        if (locked_params.get(key), cand_params.get(key)) != expected:
            raise RuntimeError(f"Missing expected diff for {key}: got {(locked_params.get(key), cand_params.get(key))}, expected {expected}")
    return diffs


def validate_config(config: dict[str, Any], locked: dict[str, Any]) -> list[dict[str, Any]]:
    diffs = validate_scientific_diff(locked, config)
    params = normalized_params(config)
    if params["channels_to_use"] != [1, 0, 2]:
        raise RuntimeError(f"Expected channels [1,0,2], got {params['channels_to_use']}")
    if params["outer_folds"] != 5 or params["inner_folds"] != 5:
        raise RuntimeError("This diagnostic must preserve outer_folds=5 and inner_folds=5.")
    if params["epochs_vae"] / params["cyclical_beta_n_cycles"] != 80:
        raise RuntimeError("Expected 80-epoch beta cycle length.")
    if params["lr_scheduler_T0"] != 80:
        raise RuntimeError("Expected lr_scheduler_T0=80.")
    if params["classifier_types"] != ["logreg", "svm"]:
        raise RuntimeError("Stage A must preserve locked logreg/svm diagnostics.")
    if params["n_iter_logreg"] <= 0 or params["n_iter_svm"] <= 0:
        raise RuntimeError("Stage A n_iter_logreg/n_iter_svm must remain positive.")
    return diffs


def validate_inputs(config: dict[str, Any]) -> None:
    paths = config["paths"]
    tensor_path = resolve(paths["global_tensor_path"])
    meta_path = resolve(paths["metadata_path"])
    script_path = resolve(paths["training_script"])
    for path in [tensor_path, meta_path, script_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    with np.load(tensor_path, allow_pickle=False) as zf:
        if bool(zf["python_bandpass_applied"]):
            raise RuntimeError("Expected python_bandpass_applied=False.")
        channel_names = [str(x) for x in zf["channel_names"].astype(str)]
    selected = [channel_names[i] for i in config["parameters"]["channels_to_use"]]
    if selected != config["selected_channel_names"]:
        raise RuntimeError(f"Selected channel-name mismatch: {selected} vs {config['selected_channel_names']}")
    meta = pd.read_csv(meta_path, usecols=lambda c: c in {"SubjectID", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex"})
    cnad = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])]
    if cnad.empty:
        raise RuntimeError("No CN/AD rows in metadata.")


def build_stage_a_command(config: dict[str, Any], python_executable: str) -> list[str]:
    paths = config["paths"]
    params = normalized_params(config)
    unknown = sorted(set(params) - set(PARAM_ORDER))
    if unknown:
        raise RuntimeError(f"Unknown config parameters: {unknown}")
    command = [
        python_executable,
        str(resolve(paths["training_script"])),
        "--global_tensor_path",
        str(resolve(paths["global_tensor_path"])),
        "--metadata_path",
        str(resolve(paths["metadata_path"])),
        "--output_dir",
        str(resolve(paths["output_dir"])),
    ]
    for name in PARAM_ORDER:
        append_arg(command, name, params.get(name))
    command.extend(["--outer_fold_indices_to_run", str(TARGET_FOLD)])
    validate_command(command)
    return command


def build_stage_b_command(config: dict[str, Any], python_executable: str) -> list[str]:
    run_dir = resolve(config["paths"]["output_dir"])
    output_dir = run_dir / "classifier_only_readout"
    return [
        python_executable,
        str(PROJECT_ROOT / "scripts/revision_bspc_2026/run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep.py"),
        "--run-dir",
        str(run_dir),
        "--output-dir",
        str(output_dir),
        "--models",
        "logreg_l2",
        "--outer-folds",
        "5",
        "--inner-folds",
        "5",
        "--folds-to-run",
        str(TARGET_FOLD),
        "--reuse-latent-cache",
    ]


def validate_command(command: list[str]) -> None:
    if "--outer_fold_indices_to_run" not in command:
        raise RuntimeError("Stage A command is missing --outer_fold_indices_to_run.")
    idx = command.index("--outer_fold_indices_to_run")
    if command[idx + 1] != str(TARGET_FOLD):
        raise RuntimeError(f"Stage A must run only fold {TARGET_FOLD}.")
    for i, token in enumerate(command[:-1]):
        if token.startswith("--n_iter") and command[i + 1] == "0":
            raise RuntimeError(f"Invalid zero Optuna trials in command: {token} 0")


def output_has_completed_run(path: Path) -> bool:
    if not path.exists():
        return False
    return any(path.glob("fold_*/vae_model_fold_*.pt")) or any(path.glob("all_folds_metrics_MULTI_*.csv"))


def ensure_real_output_ready(config: dict[str, Any]) -> None:
    output_dir = resolve(config["paths"]["output_dir"])
    if output_has_completed_run(output_dir):
        raise RuntimeError(f"Refusing to overwrite existing run output: {output_dir}")
    big_disk = Path(config["paths"]["big_disk_output_dir"])
    if not output_dir.exists():
        big_disk.mkdir(parents=True, exist_ok=True)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        output_dir.symlink_to(big_disk, target_is_directory=True)
    if not output_dir.is_symlink():
        raise RuntimeError(f"Expected output_dir symlink to big disk target: {output_dir}")
    if output_dir.resolve() != big_disk.resolve():
        raise RuntimeError(f"Unexpected symlink target {output_dir.resolve()} != {big_disk.resolve()}")


def write_dry_run_log(config: dict[str, Any], diffs: list[dict[str, Any]], stage_a: list[str], stage_b: list[str] | None) -> None:
    # Keep dry-run artifacts out of the real run directory so the future heavy
    # output path can remain a clean symlink to /media/diego/Datos.
    outdir = resolve(config["paths"]["output_dir"]).with_name(resolve(config["paths"]["output_dir"]).name + "_preflight")
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": True,
        "training_launched": False,
        "target_outer_fold": TARGET_FOLD,
        "scientific_diffs_vs_locked": diffs,
        "scientific_diff_policy": "no scientific parameter diffs vs locked; output paths/name differ only",
        "stage_a_command": stage_a,
        "stage_a_command_shell": shlex.join(stage_a),
        "stage_b_command": stage_b,
        "stage_b_command_shell": shlex.join(stage_b) if stage_b else None,
        "safety": {
            "tensor_modified": False,
            "metadata_modified": False,
            "ledger_modified": False,
            "full_5fold_training_launched": False,
        },
    }
    (outdir / "dry_run_command_log.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_command(command: list[str]) -> None:
    print(shlex.join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> int:
    args = parse_args()
    config = read_json(resolve(args.config))
    locked = read_json(resolve(args.locked_config))
    diffs = validate_config(config, locked)
    validate_inputs(config)
    python_executable = args.python_executable or config.get("python_executable") or sys.executable
    stage_a = build_stage_a_command(config, python_executable)
    stage_b = None if args.skip_stage_b_command else build_stage_b_command(config, python_executable)

    print(f"Config: {resolve(args.config)}")
    print(f"Run name: {config['run_name']}")
    print(f"Target fold: {TARGET_FOLD} only")
    print("Controlled scientific diffs vs locked:")
    print(pd.DataFrame(diffs).to_string(index=False))
    print("\nStage A command:")
    print(shlex.join(stage_a))
    if stage_b:
        print("\nStage B command:")
        print(shlex.join(stage_b))
    print("\nValidation: cycle_len=80, T0=80, Python bandpass OFF, 5x5 split preserved, fold restriction explicit.")

    if args.dry_run or not args.confirm_training:
        write_dry_run_log(config, diffs, stage_a, stage_b)
        if not args.dry_run:
            raise SystemExit("Refusing real training without --confirm-training. Use --dry-run for preflight.")
        print("Dry-run only: no training launched.")
        return 0

    ensure_real_output_ready(config)
    run_command(stage_a)
    if stage_b:
        run_command(stage_b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
