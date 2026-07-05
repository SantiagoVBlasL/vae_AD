#!/usr/bin/env python3
"""
Dry-run / run wrapper for ADNI v5.1 batch20260514b final-candidate no-Python-bandpass baseline.

Channels [1,0,2] = Pearson_Full_FisherZ_Signed +
Pearson_OMST_GCE_Signed_Weighted + MI_KNN_Symmetric.

Architecture: beta=2.5, latent_dim=256, tanh, dropout=0.15,
quarter intermediate FC, no LayerNorm.

Dry-run:
    python scripts/revision_bspc_2026/run_adni_v5_1_batch20260514b_ch1_0_2_final_candidate_baseline.py --dry-run

Real run (requires explicit confirmation and output_dir symlink to /media/diego/Datos target):
    python scripts/revision_bspc_2026/run_adni_v5_1_batch20260514b_ch1_0_2_final_candidate_baseline.py --confirm-training
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "runs"
    / "adni_v5_1_batch20260514b_ch1_0_2_final_candidate_baseline.json"
)

EXPECTED_CHANNELS = [1, 0, 2]
EXPECTED_CLASSIFIERS = ["logreg", "svm"]
EXPECTED_METADATA_FEATURES = ["Age", "Sex"]
EXPECTED_STRATIFY_COLS = ["Sex"]
EXPECTED_CHANNEL_NAMES = [
    "Pearson_Full_FisherZ_Signed",
    "Pearson_OMST_GCE_Signed_Weighted",
    "MI_KNN_Symmetric",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and optionally execute the ADNI v5.1 batch20260514b final-candidate "
            "no-Python-bandpass baseline training command for channels [1,0,2]."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command and run preflight checks without launching training.",
    )
    parser.add_argument(
        "--confirm-training",
        action="store_true",
        help="Required for any real training launch. Dry-run does not need this flag.",
    )
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


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"Unexpected {label}: {actual!r} (expected {expected!r})")


def validate_config(config: Dict[str, Any]) -> None:
    params = config["parameters"]
    _require_equal(params.get("channels_to_use"), EXPECTED_CHANNELS, "channels_to_use")
    _require_equal(params.get("classifier_types"), EXPECTED_CLASSIFIERS, "classifier_types")
    _require_equal(params.get("metadata_features"), EXPECTED_METADATA_FEATURES, "metadata_features")
    _require_equal(params.get("classifier_stratify_cols"), EXPECTED_STRATIFY_COLS, "classifier_stratify_cols")
    _require_equal(params.get("classifier_calibrate"), True, "classifier_calibrate")
    _require_equal(params.get("classifier_use_class_weight"), True, "classifier_use_class_weight")
    _require_equal(params.get("beta_vae"), 2.5, "beta_vae")
    _require_equal(params.get("latent_dim"), 256, "latent_dim")
    _require_equal(params.get("dropout_rate_vae"), 0.15, "dropout_rate_vae")
    _require_equal(params.get("vae_final_activation"), "tanh", "vae_final_activation")
    _require_equal(params.get("intermediate_fc_dim_vae"), "quarter", "intermediate_fc_dim_vae")
    _require_equal(params.get("outer_folds"), 5, "outer_folds")
    _require_equal(params.get("inner_folds"), 5, "inner_folds")
    _require_equal(params.get("n_iter_logreg"), 300, "n_iter_logreg")
    _require_equal(params.get("n_iter_svm"), 300, "n_iter_svm")
    _require_equal(params.get("use_layernorm_vae_fc"), False, "use_layernorm_vae_fc")

    for qc_flag in [
        "qc_analyze_distributions",
        "qc_check_scanner_leakage",
        "qc_rate_distortion",
        "qc_latent_information",
    ]:
        _require_equal(params.get(qc_flag), True, qc_flag)

    _require_equal(config.get("selected_channel_names"), EXPECTED_CHANNEL_NAMES, "selected_channel_names")


def validate_training_metadata(path: Path, dry_run: bool) -> Dict[str, Any]:
    if not path.exists():
        if dry_run:
            print(f"[metadata] WARN: training-ready metadata does not exist yet: {path}")
            return {
                "path": str(path),
                "available": False,
                "rows": None,
                "cn": None,
                "ad": None,
                "mci": None,
                "cn_ad_total": None,
                "group_counts": {},
                "cn_ad_sex_counts": {},
            }
        raise FileNotFoundError(f"Training-ready metadata not found: {path}")

    meta = pd.read_csv(path)
    required = ["SubjectID", "ResearchGroup_Mapped", "Age", "Sex"]
    missing = [col for col in required if col not in meta.columns]
    if missing:
        raise RuntimeError(f"Metadata is missing required columns {missing}: {path}")

    duplicated = meta["SubjectID"][meta["SubjectID"].duplicated()].astype(str).tolist()
    if duplicated:
        raise RuntimeError(f"Duplicate SubjectID values in metadata: {duplicated[:10]}")

    cn_ad = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy()
    missing_demo = cn_ad[cn_ad["Age"].isna() | cn_ad["Sex"].isna()]
    if not missing_demo.empty:
        cols = ["SubjectID", "ResearchGroup_Mapped", "Age", "Sex"]
        raise RuntimeError(
            "CN/AD training pool has missing Age or Sex:\n"
            + missing_demo[cols].to_string(index=False)
        )

    group_counts = meta["ResearchGroup_Mapped"].fillna("UNKNOWN").value_counts().to_dict()
    sex_counts = cn_ad["Sex"].fillna("UNKNOWN").value_counts().to_dict()
    summary = {
        "path": str(path),
        "available": True,
        "rows": int(len(meta)),
        "cn": int((meta["ResearchGroup_Mapped"] == "CN").sum()),
        "ad": int((meta["ResearchGroup_Mapped"] == "AD").sum()),
        "mci": int((meta["ResearchGroup_Mapped"] == "MCI").sum()),
        "cn_ad_total": int(len(cn_ad)),
        "group_counts": {str(k): int(v) for k, v in group_counts.items()},
        "cn_ad_sex_counts": {str(k): int(v) for k, v in sex_counts.items()},
    }
    print(f"[metadata] Path: {path}")
    print(
        "[metadata] Rows={rows}, CN={cn}, AD={ad}, MCI={mci}, CN+AD={cn_ad_total}".format(
            **summary
        )
    )
    print(f"[metadata] CN/AD Sex counts: {summary['cn_ad_sex_counts']}")
    print("[metadata] CN/AD pool: all subjects have complete Age and Sex. OK.")
    return summary


def inspect_tensor_npz(path: Path, dry_run: bool) -> Dict[str, Any]:
    if not path.exists():
        if dry_run:
            print(f"[tensor] WARN: tensor does not exist yet: {path}")
            return {}
        raise FileNotFoundError(f"Global tensor not found: {path}")

    with np.load(path, allow_pickle=False) as zf:
        files = list(zf.files)
        if "python_bandpass_applied" not in files:
            raise RuntimeError(f"Tensor missing python_bandpass_applied flag: {path}")
        python_bandpass_applied = bool(zf["python_bandpass_applied"])
        if python_bandpass_applied:
            raise RuntimeError(f"Expected no Python bandpass, got python_bandpass_applied=True: {path}")
        channel_names = [str(x) for x in zf["channel_names"].astype(str)] if "channel_names" in files else []
        subject_count = int(len(zf["subject_ids"])) if "subject_ids" in files else None

    selected = [channel_names[i] for i in EXPECTED_CHANNELS] if channel_names else []
    if selected and selected != EXPECTED_CHANNEL_NAMES:
        raise RuntimeError(f"Tensor channel names for [1,0,2] are {selected}, expected {EXPECTED_CHANNEL_NAMES}")
    info = {
        "path": str(path),
        "subject_count": subject_count,
        "channel_names": channel_names,
        "selected_channel_names": selected,
        "python_bandpass_applied": python_bandpass_applied,
    }
    print(f"[tensor] Path: {path}")
    print(f"[tensor] Subjects={subject_count}, channels={len(channel_names)}, python_bandpass_applied=False")
    print(f"[tensor] Selected [1,0,2]: {selected}")
    return info


_PARAM_ORDER = [
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


def build_command(config: Dict[str, Any], python_executable: str) -> List[str]:
    paths = config["paths"]
    params = config["parameters"]
    unknown = sorted(set(params) - set(_PARAM_ORDER))
    if unknown:
        raise RuntimeError(f"Unknown config parameters: {unknown}")

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
    for name in _PARAM_ORDER:
        append_arg(command, name, params.get(name))
    return command


def soft_preflight(config: Dict[str, Any], dry_run: bool) -> Dict[str, Any]:
    paths = config["paths"]
    training_script = resolve_path(paths["training_script"])
    if training_script.exists():
        print(f"Preflight OK  : training_script: {training_script}")
    elif dry_run:
        print(f"Preflight WARN: training_script does not exist yet: {training_script}")
    else:
        raise FileNotFoundError(f"Required path not found: training_script: {training_script}")

    tensor_info = inspect_tensor_npz(resolve_path(paths["global_tensor_path"]), dry_run=dry_run)
    metadata_summary = validate_training_metadata(resolve_path(paths["metadata_path"]), dry_run=dry_run)
    return {"tensor": tensor_info, "metadata": metadata_summary}


def describe_output_path(config: Dict[str, Any]) -> None:
    output_dir = resolve_path(config["paths"]["output_dir"])
    big_disk = Path(config["paths"]["big_disk_output_dir"])
    print(f"\nOutput dir    : {output_dir}")
    if output_dir.is_symlink():
        print(f"Symlink target: {output_dir.resolve()}")
    elif output_dir.exists():
        print("Output path exists but is NOT a symlink.")
    else:
        print("Output path does not exist yet.")
    print(f"Big-disk target: {big_disk}")


def ensure_output_prepared_for_real_run(config: Dict[str, Any]) -> Path:
    output_dir = resolve_path(config["paths"]["output_dir"])
    big_disk = Path(config["paths"]["big_disk_output_dir"])

    if not output_dir.exists():
        raise RuntimeError(
            "Refusing to start training: output_dir is missing.\n"
            "Prepare the big-disk symlink first:\n"
            f"  mkdir -p {shlex.quote(str(big_disk))}\n"
            f"  ln -s {shlex.quote(str(big_disk))} {shlex.quote(str(output_dir))}"
        )
    if not output_dir.is_symlink():
        raise RuntimeError(f"Refusing to start training: output_dir is not a symlink: {output_dir}")
    resolved = output_dir.resolve()
    if resolved != big_disk.resolve():
        raise RuntimeError(
            f"Refusing to start training: symlink target is {resolved}, expected {big_disk.resolve()}."
        )
    non_manifest = [p for p in output_dir.iterdir() if p.name != "run_manifest.json"]
    if non_manifest:
        raise RuntimeError(
            f"Refusing to start training: output_dir is not empty ({len(non_manifest)} unexpected file(s)): "
            + ", ".join(p.name for p in non_manifest[:5])
        )
    return output_dir


def write_manifest(
    config_path: Path,
    config: Dict[str, Any],
    command: List[str],
    preflight: Dict[str, Any],
) -> Path:
    output_dir = resolve_path(config["paths"]["output_dir"])
    path = output_dir / "run_manifest.json"
    params = config["parameters"]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": False,
        "config_path": str(config_path),
        "run_name": config.get("run_name"),
        "selected_channels": params.get("channels_to_use"),
        "selected_channel_names": config.get("selected_channel_names"),
        "python_bandpass_applied": False,
        "metadata_path": config["paths"]["metadata_path"],
        "metadata_summary": preflight.get("metadata", {}),
        "tensor_summary": preflight.get("tensor", {}),
        "architecture": {
            "beta_vae": params.get("beta_vae"),
            "latent_dim": params.get("latent_dim"),
            "dropout_rate_vae": params.get("dropout_rate_vae"),
            "vae_final_activation": params.get("vae_final_activation"),
            "intermediate_fc_dim_vae": params.get("intermediate_fc_dim_vae"),
            "use_layernorm_vae_fc": params.get("use_layernorm_vae_fc"),
        },
        "classifiers": params.get("classifier_types"),
        "trial_budgets": {
            "logreg": params.get("n_iter_logreg"),
            "svm": params.get("n_iter_svm"),
        },
        "qc_enabled": {
            "qc_analyze_distributions": params.get("qc_analyze_distributions"),
            "qc_check_scanner_leakage": params.get("qc_check_scanner_leakage"),
            "qc_rate_distortion": params.get("qc_rate_distortion"),
            "qc_latent_information": params.get("qc_latent_information"),
        },
        "command": command,
        "command_shell": shlex.join(command),
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    validate_config(config)
    python_executable = args.python_executable or config.get("python_executable") or sys.executable
    if not args.dry_run and not args.confirm_training:
        raise SystemExit(
            "Refusing to launch training without --confirm-training. "
            "Run with --dry-run for preflight only."
        )

    print(f"Config        : {args.config}")
    print(f"Run name      : {config.get('run_name')}")
    print(f"Python        : {python_executable}")
    print(f"Mode          : {'DRY-RUN' if args.dry_run else 'REAL RUN (CONFIRMED)'}")
    print(f"Channels      : {EXPECTED_CHANNELS} ({', '.join(EXPECTED_CHANNEL_NAMES)})")
    print("Python bandpass: OFF")
    print("QC            : ON")

    print()
    preflight = soft_preflight(config, args.dry_run)
    command = build_command(config, python_executable)
    describe_output_path(config)

    params = config["parameters"]
    metadata = preflight["metadata"]
    if metadata.get("available"):
        print(
            f"\nSupervised    : CN={metadata['cn']}, AD={metadata['ad']}, "
            f"CN+AD={metadata['cn_ad_total']}"
        )
    else:
        print("\nSupervised    : not checked (metadata path unavailable in dry-run)")
    print(f"Metadata feats: {params.get('metadata_features')}")
    print(f"Stratify cols : {params.get('classifier_stratify_cols')}")
    print(f"Trial budgets : logreg={params.get('n_iter_logreg')}, svm={params.get('n_iter_svm')}")

    print("\nCommand:")
    print(shlex.join(command))

    if args.dry_run:
        print("\nDry-run complete. Training was NOT launched.")
        return 0

    ensure_output_prepared_for_real_run(config)
    manifest_path = write_manifest(args.config, config, command, preflight)
    print(f"\nRun manifest written: {manifest_path}")
    print("\nLaunching training...")
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
