#!/usr/bin/env python3
"""Prepare/run FULL 5x5 manufacturer-balanced VAE sampler confirmation.

This run is a controlled perturbation of the locked current FULL [1,0,2]
manufacturer-aware 3840-epoch model.  The only scientific parameter change is:

    vae_train_sampler_strategy = manufacturer_balanced

Default mode is dry-run/no training.  Real execution requires
``--confirm-training``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "runs"
    / "adni_v5_1_batch20260514b_ch1_0_2_manufacturer_balanced_sampler_full_5x5.json"
)
LOCKED_SOURCE_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "runs"
    / "adni_v5_1_batch20260514b_ch1_0_2_mfrsplit_3840_final_candidate.json"
)
REQUESTED_STALE_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "runs"
    / "adni_v5_1_batch20260514b_ch1_0_2_final_candidate_baseline.json"
)
STAGE_B_SCRIPT = PROJECT_ROOT / "scripts/revision_bspc_2026/run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep.py"
COMPARISON_SCRIPT = PROJECT_ROOT / "scripts/revision_bspc_2026/compare_v5_1_batch20260514b_manufacturer_balanced_sampler_full_5x5.py"

EXPECTED_CHANNELS = [1, 0, 2]
EXPECTED_SELECTED_NAMES = [
    "Pearson_Full_FisherZ_Signed",
    "Pearson_OMST_GCE_Signed_Weighted",
    "MI_KNN_Symmetric",
]
EXPECTED_STRATIFY = ["Manufacturer"]
EXPECTED_METADATA = ["Age", "Sex"]

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
    "vae_train_sampler_strategy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print commands; do not launch training.")
    parser.add_argument("--confirm-training", action="store_true", help="Required for real Stage A/Stage B launch.")
    parser.add_argument("--skip-classifier-readout", action="store_true")
    parser.add_argument("--skip-comparison", action="store_true")
    parser.add_argument("--python-executable", default=None)
    parser.add_argument(
        "--output-base",
        type=Path,
        default=None,
        help=(
            "Optional heavy-output base. If omitted, VAE_AD_RESULTS_BASE is honored; "
            "otherwise config paths.big_disk_output_dir is used."
        ),
    )
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: expected {expected!r}, got {actual!r}")


def append_arg(cmd: List[str], name: str, value: Any) -> None:
    if isinstance(value, bool):
        if value:
            cmd.append(f"--{name}")
        return
    if value is None:
        return
    cmd.append(f"--{name}")
    if isinstance(value, list):
        cmd.extend(str(v) for v in value)
    else:
        cmd.append(str(value))


def values_after_flag(tokens: Sequence[str], flag: str) -> List[str]:
    if flag not in tokens:
        return []
    out: List[str] = []
    for token in tokens[tokens.index(flag) + 1 :]:
        if token.startswith("--"):
            break
        out.append(token)
    return out


def validate_config(config: Dict[str, Any]) -> None:
    params = config["parameters"]
    unknown = sorted(set(params) - set(PARAM_ORDER))
    if unknown:
        raise RuntimeError(f"Parameters not wired into Stage A command: {unknown}")
    require_equal(params["channels_to_use"], EXPECTED_CHANNELS, "channels_to_use")
    require_equal(config["selected_channel_names"], EXPECTED_SELECTED_NAMES, "selected_channel_names")
    require_equal(params["classifier_stratify_cols"], EXPECTED_STRATIFY, "classifier_stratify_cols")
    require_equal(params["vae_stratify_cols"], EXPECTED_STRATIFY, "vae_stratify_cols")
    require_equal(params["metadata_features"], EXPECTED_METADATA, "metadata_features")
    require_equal(params["vae_train_sampler_strategy"], "manufacturer_balanced", "vae_train_sampler_strategy")
    if "Sex" in params["classifier_stratify_cols"] or "Sex" in params["vae_stratify_cols"]:
        raise RuntimeError("Sex must remain metadata/covariate only, not a stratifier.")
    for key, expected in [
        ("outer_folds", 5),
        ("inner_folds", 5),
        ("repeated_outer_folds_n_repeats", 1),
        ("latent_dim", 256),
        ("epochs_vae", 3840),
        ("cyclical_beta_n_cycles", 48),
        ("early_stopping_patience_vae", 320),
        ("lr_scheduler_T0", 80),
        ("beta_vae", 2.5),
        ("dropout_rate_vae", 0.15),
        ("batch_size", 64),
        ("vae_final_activation", "tanh"),
        ("intermediate_fc_dim_vae", "quarter"),
        ("decoder_type", "convtranspose"),
        ("num_conv_layers_encoder", 4),
        ("norm_mode", "zscore_offdiag"),
        ("use_layernorm_vae_fc", False),
        ("classifier_calibrate", True),
        ("classifier_use_class_weight", True),
    ]:
        require_equal(params[key], expected, key)
    if params["epochs_vae"] / params["cyclical_beta_n_cycles"] != 80:
        raise RuntimeError("Expected 3840/48 = 80 cycle length.")
    for flag in ["qc_analyze_distributions", "qc_check_scanner_leakage", "qc_rate_distortion", "qc_latent_information"]:
        require_equal(params[flag], True, flag)


def verify_only_sampler_diff(config: Dict[str, Any]) -> pd.DataFrame:
    source = load_json(LOCKED_SOURCE_CONFIG)
    target_params = dict(config["parameters"])
    source_params = dict(source["parameters"])
    rows: List[Dict[str, Any]] = []
    keys = sorted(set(source_params) | set(target_params))
    allowed_param_diffs = {"vae_train_sampler_strategy"}
    unexpected: List[str] = []
    for key in keys:
        source_value = source_params.get(key, "<MISSING>")
        target_value = target_params.get(key, "<MISSING>")
        changed = source_value != target_value
        allowed = (not changed) or key in allowed_param_diffs
        if changed and not allowed:
            unexpected.append(key)
        rows.append(
            {
                "section": "parameters",
                "key": key,
                "source_locked_full": json.dumps(source_value, sort_keys=True),
                "target_sampler_full": json.dumps(target_value, sort_keys=True),
                "changed": changed,
                "allowed_change": allowed,
            }
        )
    if unexpected:
        raise RuntimeError(f"Unexpected parameter changes vs locked FULL config: {unexpected}")
    return pd.DataFrame(rows)


def inspect_requested_stale_config() -> Dict[str, Any]:
    if not REQUESTED_STALE_CONFIG.exists():
        return {"requested_config_exists": False}
    stale = load_json(REQUESTED_STALE_CONFIG)
    params = stale.get("parameters", {})
    return {
        "requested_config_exists": True,
        "requested_config_epochs_vae": params.get("epochs_vae"),
        "requested_config_classifier_stratify_cols": params.get("classifier_stratify_cols"),
        "requested_config_classifier_types": params.get("classifier_types"),
        "requested_config_note": "Not used as scientific source because it is not the locked current FULL model.",
    }


def inspect_tensor(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as zf:
        if bool(zf["python_bandpass_applied"]):
            raise RuntimeError("Refusing tensor with python_bandpass_applied=True")
        shape = tuple(int(x) for x in zf["global_tensor_data"].shape)
        channel_names = [str(x) for x in zf["channel_names"].astype(str)]
    selected = [channel_names[i] for i in EXPECTED_CHANNELS]
    require_equal(selected, EXPECTED_SELECTED_NAMES, "selected tensor channel names")
    return {"shape": shape, "python_bandpass_applied": False, "selected_channel_names": selected}


def inspect_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    meta = pd.read_csv(path)
    required = ["SubjectID", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex"]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise RuntimeError(f"Metadata missing required columns: {missing}")
    if meta["SubjectID"].duplicated().any():
        raise RuntimeError("Metadata contains duplicate SubjectID rows.")
    return meta


def heavy_target(config: Dict[str, Any], output_base: Path | None) -> Path:
    run_name = config["run_name"]
    if output_base is not None:
        return output_base / run_name
    env_base = os.environ.get("VAE_AD_RESULTS_BASE")
    if env_base:
        return Path(env_base) / run_name
    return Path(config["paths"]["big_disk_output_dir"])


def build_stage_a(config: Dict[str, Any], python_exe: str) -> List[str]:
    paths = config["paths"]
    params = config["parameters"]
    cmd = [
        python_exe,
        str(resolve(paths["training_script"])),
        "--global_tensor_path",
        str(resolve(paths["global_tensor_path"])),
        "--metadata_path",
        str(resolve(paths["metadata_path"])),
        "--output_dir",
        str(resolve(paths["output_dir"])),
    ]
    for name in PARAM_ORDER:
        append_arg(cmd, name, params.get(name))
    validate_stage_a(cmd)
    return cmd


def build_stage_b(config: Dict[str, Any], python_exe: str) -> List[str]:
    params = config["parameters"]
    out_dir = resolve(config["paths"]["output_dir"])
    cmd = [
        python_exe,
        str(STAGE_B_SCRIPT),
        "--run-dir",
        str(out_dir),
        "--output-dir",
        str(out_dir / "classifier_only_readout"),
        "--models",
        "logreg_l2",
        "--outer-folds",
        str(params["outer_folds"]),
        "--inner-folds",
        str(params["inner_folds"]),
        "--reuse-latent-cache",
        "--overwrite",
    ]
    validate_stage_b(cmd)
    return cmd


def build_comparison(config: Dict[str, Any], python_exe: str) -> List[str]:
    out_dir = resolve(config["paths"]["output_dir"])
    return [
        python_exe,
        str(COMPARISON_SCRIPT),
        "--candidate-run-dir",
        str(out_dir),
        "--candidate-readout-dir",
        str(out_dir / "classifier_only_readout"),
        "--output-dir",
        str(out_dir.parent / f"{config['run_name']}_comparison"),
        "--overwrite",
    ]


def validate_stage_a(cmd: Sequence[str]) -> None:
    for idx, token in enumerate(cmd[:-1]):
        if token.startswith("--n_iter_") and str(cmd[idx + 1]) == "0":
            raise RuntimeError(f"Invalid Stage A command contains {token} 0")
    require_equal(values_after_flag(cmd, "--channels_to_use"), ["1", "0", "2"], "Stage A channels")
    require_equal(values_after_flag(cmd, "--outer_folds"), ["5"], "Stage A outer_folds")
    require_equal(values_after_flag(cmd, "--inner_folds"), ["5"], "Stage A inner_folds")
    require_equal(values_after_flag(cmd, "--latent_dim"), ["256"], "Stage A latent_dim")
    require_equal(values_after_flag(cmd, "--epochs_vae"), ["3840"], "Stage A epochs_vae")
    require_equal(values_after_flag(cmd, "--cyclical_beta_n_cycles"), ["48"], "Stage A cycles")
    require_equal(values_after_flag(cmd, "--lr_scheduler_T0"), ["80"], "Stage A T0")
    require_equal(values_after_flag(cmd, "--vae_final_activation"), ["tanh"], "Stage A final activation")
    require_equal(values_after_flag(cmd, "--vae_train_sampler_strategy"), ["manufacturer_balanced"], "Stage A sampler")
    require_equal(values_after_flag(cmd, "--metadata_features"), ["Age", "Sex"], "Stage A metadata")
    require_equal(values_after_flag(cmd, "--classifier_stratify_cols"), ["Manufacturer"], "Stage A classifier stratify")
    require_equal(values_after_flag(cmd, "--vae_stratify_cols"), ["Manufacturer"], "Stage A VAE stratify")


def validate_stage_b(cmd: Sequence[str]) -> None:
    require_equal(values_after_flag(cmd, "--models"), ["logreg_l2"], "Stage B models")
    require_equal(values_after_flag(cmd, "--outer-folds"), ["5"], "Stage B outer-folds")
    require_equal(values_after_flag(cmd, "--inner-folds"), ["5"], "Stage B inner-folds")
    if "--reuse-latent-cache" not in cmd:
        raise RuntimeError("Stage B must reuse/cache fold latents.")


def prepare_output_symlink(config: Dict[str, Any], target: Path) -> None:
    output_dir = resolve(config["paths"]["output_dir"])
    target.mkdir(parents=True, exist_ok=True)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() or output_dir.is_symlink():
        if not output_dir.is_symlink() or output_dir.resolve() != target.resolve():
            raise RuntimeError(f"Output path exists but is not expected symlink: {output_dir} -> {target}")
    else:
        output_dir.symlink_to(target, target_is_directory=True)
    unexpected = [p.name for p in output_dir.iterdir() if p.name != "run_manifest.json"]
    if unexpected:
        raise RuntimeError(f"Refusing to train into non-empty output directory: {unexpected[:8]}")


def write_manifest(config_path: Path, config: Dict[str, Any], stage_a: Sequence[str], stage_b: Sequence[str], comparison: Sequence[str], target: Path) -> None:
    out_dir = resolve(config["paths"]["output_dir"])
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "run_name": config["run_name"],
        "heavy_output_target": str(target),
        "source_config_used": str(LOCKED_SOURCE_CONFIG),
        "only_scientific_parameter_change": "vae_train_sampler_strategy=manufacturer_balanced",
        "stage_a_command": list(stage_a),
        "stage_a_command_shell": shlex.join(stage_a),
        "stage_b_command": list(stage_b),
        "stage_b_command_shell": shlex.join(stage_b),
        "comparison_command": list(comparison),
        "comparison_command_shell": shlex.join(comparison),
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.dry_run and args.confirm_training:
        raise SystemExit("Use either --dry-run or --confirm-training, not both.")
    if not args.dry_run and not args.confirm_training:
        raise SystemExit("Refusing to train without --confirm-training. Use --dry-run for preflight.")

    config = load_json(args.config)
    validate_config(config)
    diff = verify_only_sampler_diff(config)
    stale_info = inspect_requested_stale_config()
    tensor_info = inspect_tensor(resolve(config["paths"]["global_tensor_path"]))
    meta = inspect_metadata(resolve(config["paths"]["metadata_path"]))
    python_exe = args.python_executable or config.get("python_executable") or sys.executable
    target = heavy_target(config, args.output_base)
    stage_a = build_stage_a(config, python_exe)
    stage_b = build_stage_b(config, python_exe)
    comparison = build_comparison(config, python_exe)

    print(f"Config: {args.config}")
    print(f"Run name: {config['run_name']}")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'REAL RUN'}")
    print(f"Source locked FULL config: {LOCKED_SOURCE_CONFIG}")
    print(f"Requested stale config note: {stale_info}")
    print("Scientific parameter diff vs locked FULL:")
    print(diff[diff["changed"]].to_string(index=False))
    print(f"Heavy output target: {target}")
    print(f"Local output path: {resolve(config['paths']['output_dir'])}")
    print(f"Tensor: shape={tensor_info['shape']}, python_bandpass_applied={tensor_info['python_bandpass_applied']}")
    print(f"Metadata rows={len(meta)}, CN={(meta.ResearchGroup_Mapped == 'CN').sum()}, AD={(meta.ResearchGroup_Mapped == 'AD').sum()}, MCI={(meta.ResearchGroup_Mapped == 'MCI').sum()}")
    print("\nStage A command:")
    print(shlex.join(stage_a))
    print("\nStage B command:")
    print(shlex.join(stage_b))
    print("\nComparison command:")
    print(shlex.join(comparison))

    if args.dry_run:
        print("\nDry-run complete. No training launched; no output symlink created.")
        return 0

    prepare_output_symlink(config, target)
    write_manifest(args.config, config, stage_a, stage_b, comparison, target)
    rc = subprocess.call(stage_a, cwd=PROJECT_ROOT)
    if rc != 0:
        return int(rc)
    if not args.skip_classifier_readout:
        rc = subprocess.call(stage_b, cwd=PROJECT_ROOT)
        if rc != 0:
            return int(rc)
    if not args.skip_comparison:
        rc = subprocess.call(comparison, cwd=PROJECT_ROOT)
        if rc != 0:
            return int(rc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
