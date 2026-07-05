#!/usr/bin/env python3
"""
Dry-run / run wrapper for ADNI v5 DPARSF-10000 no-Python-bandpass baseline.

Architecture: channels [4,1,0], beta=2.5, latent_dim=256, tanh, no LayerNorm.
Supervised pool: CN=147, AD=96 (subjects with complete Age+Sex).

Before launching training this wrapper:
  1. Generates a training-ready metadata CSV by removing subjects that cannot
     be used in supervised training with demographic features (Age, Sex).
  2. Validates all input paths and the output-dir symlink.
  3. Writes a run_manifest.json in the output dir (real run only).

Dry-run:
    python run_adni_v5_dparsf10000_no_pybandpass_ch4_1_0_baseline.py --dry-run

Real run (requires symlink to /media/diego/Datos target to be prepared first):
    python run_adni_v5_dparsf10000_no_pybandpass_ch4_1_0_baseline.py
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

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs" / "runs"
    / "adni_v5_dparsf10000_no_pybandpass_ch4_1_0_baseline.json"
)

# Subjects excluded from training-ready metadata.
# The training script (run_vae_clf_ad_inference.py) does not check
# exclude_from_supervised or filter NaN Age/Sex before building cn_ad_df.
# We remove these subjects from the metadata CSV passed to the training script.
SUBJECTS_TO_EXCLUDE: Dict[str, str] = {
    "035_S_6927": (
        "AD diagnosis confirmed but Age and Sex not found in any ADNI source; "
        "would enter supervised classifier with NaN demographics and imputed values."
    ),
    "128_S_2002": (
        "No diagnosis in any ADNI source (NaN ResearchGroup_Mapped); "
        "signal anomaly (96.9% near-zero values, scale_label=unknown); "
        "OMST hard fallback to MST; exclude_from_supervised=True."
    ),
    "114_S_6039": (
        "AD subject excluded from tensor extraction (not in global tensor). "
        "Including in metadata CSV would produce a dead row after load_data left join."
    ),
}

# Expected supervised counts after exclusion — sanity check.
EXPECTED_SUPERVISED_CN = 147
EXPECTED_SUPERVISED_AD = 96


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and optionally execute the ADNI v5 DPARSF-10000 no-Python-bandpass "
            "baseline training command."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help="Path to JSON config file.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Print the command and run all pre-flight checks "
            "(including metadata generation) without launching training."
        ),
    )
    parser.add_argument(
        "--python-executable", default=None,
        help="Override the Python executable (default: from config or sys.executable).",
    )
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


# ---------------------------------------------------------------------------
# Metadata preparation
# ---------------------------------------------------------------------------

def prepare_training_metadata(config: Dict[str, Any], dry_run: bool) -> Path:
    """
    Load source metadata, remove subjects in SUBJECTS_TO_EXCLUDE, verify
    expected counts, and write the filtered CSV to config.paths.metadata_path.

    Returns the path to the written CSV.
    """
    source_path = Path(config["paths"]["source_metadata_path"])
    target_path = Path(config["paths"]["metadata_path"])

    print(f"\n[metadata] Source: {source_path}")
    print(f"[metadata] Target: {target_path}")

    if not source_path.exists():
        raise FileNotFoundError(f"Source metadata not found: {source_path}")

    meta = pd.read_csv(source_path)
    n_source = len(meta)

    for sid, reason in SUBJECTS_TO_EXCLUDE.items():
        if sid in meta["SubjectID"].values:
            print(f"[metadata] Removing {sid}: {reason}")
        else:
            print(f"[metadata] {sid} not found in source (already absent).")

    meta_filtered = meta[~meta["SubjectID"].isin(SUBJECTS_TO_EXCLUDE)].copy()
    n_filtered = len(meta_filtered)
    n_removed = n_source - n_filtered

    print(f"[metadata] Rows: {n_source} → {n_filtered} (removed {n_removed})")

    # Sanity check supervised counts
    cn_count = int((meta_filtered["ResearchGroup_Mapped"] == "CN").sum())
    ad_count = int((meta_filtered["ResearchGroup_Mapped"] == "AD").sum())
    mci_count = int((meta_filtered["ResearchGroup_Mapped"] == "MCI").sum())
    unknown_count = int(meta_filtered["ResearchGroup_Mapped"].isna().sum())

    print(
        f"[metadata] Group counts: CN={cn_count}, AD={ad_count}, "
        f"MCI={mci_count}, Unknown/NaN={unknown_count}"
    )

    if cn_count != EXPECTED_SUPERVISED_CN or ad_count != EXPECTED_SUPERVISED_AD:
        raise RuntimeError(
            f"Unexpected supervised counts after filtering: "
            f"CN={cn_count} (expected {EXPECTED_SUPERVISED_CN}), "
            f"AD={ad_count} (expected {EXPECTED_SUPERVISED_AD}). "
            "Check source metadata or SUBJECTS_TO_EXCLUDE list."
        )

    # Check no remaining NaN Age/Sex in CN/AD pool
    cn_ad = meta_filtered[meta_filtered["ResearchGroup_Mapped"].isin(["CN", "AD"])]
    nan_age = cn_ad["Age"].isna().sum() if "Age" in cn_ad.columns else 0
    nan_sex = cn_ad["Sex"].isna().sum() if "Sex" in cn_ad.columns else 0
    if nan_age > 0 or nan_sex > 0:
        bad = cn_ad[cn_ad["Age"].isna() | cn_ad["Sex"].isna()][
            ["SubjectID", "ResearchGroup_Mapped", "Age", "Sex"]
        ]
        raise RuntimeError(
            f"Unexpected NaN demographics in CN/AD pool after filtering "
            f"(Age NaN={nan_age}, Sex NaN={nan_sex}):\n{bad.to_string()}"
        )
    print("[metadata] CN/AD pool: all subjects have complete Age and Sex. OK.")

    if dry_run:
        if target_path.exists():
            print(f"[metadata] (dry-run) Target already exists, would overwrite: {target_path}")
        else:
            print(f"[metadata] (dry-run) Would write {n_filtered} rows to: {target_path}")
    else:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        meta_filtered.to_csv(target_path, index=False)
        print(f"[metadata] Written {n_filtered} rows to: {target_path}")

    return target_path


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------

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


def build_command(
    config: Dict[str, Any],
    python_executable: str,
    training_metadata_path: Path,
) -> List[str]:
    paths = config["paths"]
    params = config["parameters"]

    unknown = sorted(set(params) - set(_PARAM_ORDER))
    if unknown:
        raise RuntimeError(f"Unknown config parameters (not in _PARAM_ORDER): {unknown}")

    command = [
        python_executable,
        str(resolve_path(paths["training_script"])),
        "--global_tensor_path",
        str(resolve_path(paths["global_tensor_path"])),
        "--metadata_path",
        str(training_metadata_path),
        "--output_dir",
        str(resolve_path(paths["output_dir"])),
    ]
    for name in _PARAM_ORDER:
        append_arg(command, name, params.get(name))
    return command


# ---------------------------------------------------------------------------
# Pre-flight and output preparation
# ---------------------------------------------------------------------------

def soft_preflight(config: Dict[str, Any], dry_run: bool) -> None:
    paths = config["paths"]
    for key in ["training_script", "global_tensor_path", "source_metadata_path"]:
        path = resolve_path(paths[key])
        if path.exists():
            print(f"Preflight OK  : {key}: {path}")
        elif dry_run:
            print(f"Preflight WARN: {key} does not exist yet: {path}")
        else:
            raise FileNotFoundError(f"Required path not found for {key}: {path}")


def describe_output_path(config: Dict[str, Any]) -> None:
    output_dir = resolve_path(config["paths"]["output_dir"])
    big_disk = Path(config["paths"]["big_disk_output_dir"])
    print(f"\nOutput dir   : {output_dir}")
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
            f"  ln -s {shlex.quote(str(big_disk))} "
            f"{shlex.quote(str(output_dir))}"
        )
    if not output_dir.is_symlink():
        raise RuntimeError(
            f"Refusing to start training: output_dir exists but is NOT a symlink: {output_dir}"
        )
    resolved = output_dir.resolve()
    if resolved != big_disk.resolve():
        raise RuntimeError(
            f"Refusing to start training: symlink target is {resolved}, "
            f"expected {big_disk.resolve()}."
        )
    contents = list(output_dir.iterdir())
    if contents:
        # run_manifest.json is written by this wrapper — allow it as the only file
        non_manifest = [p for p in contents if p.name != "run_manifest.json"]
        if non_manifest:
            raise RuntimeError(
                f"Refusing to start training: output_dir is not empty "
                f"({len(non_manifest)} unexpected file(s)): "
                + ", ".join(str(p.name) for p in non_manifest[:5])
            )
    return output_dir


def write_manifest(
    config_path: Path,
    config: Dict[str, Any],
    command: List[str],
    training_metadata_path: Path,
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
        "supervised_pool": {
            "CN": EXPECTED_SUPERVISED_CN,
            "AD": EXPECTED_SUPERVISED_AD,
            "total": EXPECTED_SUPERVISED_CN + EXPECTED_SUPERVISED_AD,
            "metadata_features": params.get("metadata_features"),
        },
        "excluded_from_training_metadata": {
            sid: reason for sid, reason in SUBJECTS_TO_EXCLUDE.items()
        },
        "training_metadata_path": str(training_metadata_path),
        "effective_trial_budgets": {
            "logreg": params.get("n_iter_logreg"),
            "svm": params.get("n_iter_svm"),
        },
        "command": command,
        "command_shell": shlex.join(command),
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    python_executable = (
        args.python_executable
        or config.get("python_executable")
        or sys.executable
    )

    print(f"Config        : {args.config}")
    print(f"Run name      : {config.get('run_name')}")
    print(f"Python        : {python_executable}")
    print(f"Mode          : {'DRY-RUN' if args.dry_run else 'REAL RUN'}")

    # 1. Pre-flight path checks
    print()
    soft_preflight(config, args.dry_run)

    # 2. Prepare training metadata (generates filtered CSV)
    training_metadata_path = prepare_training_metadata(config, dry_run=args.dry_run)

    # 3. Build command (uses generated metadata path)
    effective_metadata_path = (
        training_metadata_path
        if not args.dry_run or training_metadata_path.exists()
        else Path(config["paths"]["metadata_path"])
    )
    command = build_command(config, python_executable, effective_metadata_path)

    # 4. Describe output path
    describe_output_path(config)

    # 5. Print supervised pool info
    params = config["parameters"]
    print(f"\nChannels      : {params['channels_to_use']} "
          f"({', '.join(config.get('selected_channel_names', []))})")
    print(f"Supervised    : CN={EXPECTED_SUPERVISED_CN}, AD={EXPECTED_SUPERVISED_AD} "
          f"(with Age+Sex complete)")
    print(f"Metadata feats: {params.get('metadata_features')}")
    print(f"Trial budgets : logreg={params.get('n_iter_logreg')}, "
          f"svm={params.get('n_iter_svm')}")
    print(f"Excluded subjects (from supervised metadata):")
    for sid, reason in SUBJECTS_TO_EXCLUDE.items():
        print(f"  - {sid}: {reason}")

    # 6. Print command
    print("\nCommand:")
    print(shlex.join(command))

    if args.dry_run:
        print("\nDry-run complete. Training was NOT launched.")
        return 0

    # 7. Safety check: symlink + empty
    ensure_output_prepared_for_real_run(config)

    # 8. Write run manifest
    manifest_path = write_manifest(args.config, config, command, effective_metadata_path)
    print(f"\nRun manifest written: {manifest_path}")

    # 9. Launch training
    print("\nLaunching training...")
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
