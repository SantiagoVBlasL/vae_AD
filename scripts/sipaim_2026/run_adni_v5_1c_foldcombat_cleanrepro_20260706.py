#!/usr/bin/env python3
"""Guarded launcher for the exact matched FULL foldwise-ComBat clean rerun."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVISION_SCRIPTS = PROJECT_ROOT / "scripts" / "revision_bspc_2026"
if str(REVISION_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(REVISION_SCRIPTS))

from foldwise_combat_input_harmonization import dependency_status  # noqa: E402


RUN_NAME = (
    "recover035_latent384_beta3p75_"
    "foldcombat_mfr_age_sex_cleanrepro_20260706"
)
REFERENCE_CONFIG = (
    PROJECT_ROOT
    / "configs/runs/"
    "adni_v5_1c_recover035_latent384_beta3p75_T80_h10000_p560_full5x5.json"
)
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/runs/"
    "adni_v5_1c_recover035_latent384_beta3p75_"
    "foldcombat_mfr_age_sex_cleanrepro_20260706.json"
)
HARMONIZATION_KEYS = {
    "input_harmonization_mode": "foldwise_combat",
    "input_harmonization_batch_col": "Manufacturer",
    "input_harmonization_covariates": ["Age", "Sex"],
    "input_harmonization_excluded_covariates": ["ResearchGroup_Mapped"],
    "input_harmonization_fit_scope": "outer_train_dev_only",
    "input_harmonization_vectorization": "upper_offdiag_by_channel",
}
PROVENANCE_PATH_KEYS = {
    "output_dir",
    "big_disk_output_dir",
    "split_preview_csv",
    "split_preview_summary_csv",
}
EXACT_REQUIRED_PARAMETERS = {
    "channels_to_use": [1, 0, 2],
    "latent_dim": 384,
    "beta_vae": 3.75,
    "batch_size": 64,
    "lr_scheduler_T0": 80,
    "epochs_vae": 10000,
    "early_stopping_patience_vae": 560,
    "outer_folds": 5,
    "inner_folds": 5,
    "seed": 42,
    "metadata_features": ["Age", "Sex"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reference-config", type=Path, default=REFERENCE_CONFIG)
    parser.add_argument("--python-executable", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-training", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_config(reference: dict[str, Any], candidate: dict[str, Any]) -> None:
    if candidate.get("run_name") != RUN_NAME:
        raise RuntimeError(
            f"run_name must be {RUN_NAME!r}, got {candidate.get('run_name')!r}"
        )
    reference_parameters = dict(reference["parameters"])
    candidate_parameters = dict(candidate["parameters"])
    for key, expected in HARMONIZATION_KEYS.items():
        if candidate_parameters.get(key) != expected:
            raise RuntimeError(
                f"{key}: expected {expected!r}, got "
                f"{candidate_parameters.get(key)!r}"
            )
        candidate_parameters.pop(key)
    parameter_diff = {
        key: (reference_parameters.get(key), candidate_parameters.get(key))
        for key in sorted(set(reference_parameters) | set(candidate_parameters))
        if reference_parameters.get(key) != candidate_parameters.get(key)
    }
    if parameter_diff:
        raise RuntimeError(
            f"Unexpected non-harmonization parameter differences: {parameter_diff}"
        )
    for key, expected in EXACT_REQUIRED_PARAMETERS.items():
        if candidate["parameters"].get(key) != expected:
            raise RuntimeError(
                f"Required matched parameter {key}: expected {expected!r}, "
                f"got {candidate['parameters'].get(key)!r}"
            )
    for section in (
        "channel_names_master_in_tensor_order",
        "selected_channel_names",
        "split_strategy",
    ):
        if candidate.get(section) != reference.get(section):
            raise RuntimeError(f"Unexpected difference in {section}")
    for key, value in candidate["paths"].items():
        if key in PROVENANCE_PATH_KEYS:
            if RUN_NAME not in str(value):
                raise RuntimeError(
                    f"Unique provenance path {key} does not contain {RUN_NAME}: {value}"
                )
        elif value != reference["paths"].get(key):
            raise RuntimeError(
                f"Unexpected input path difference for {key}: "
                f"{reference['paths'].get(key)!r} -> {value!r}"
            )
    excluded = set(candidate["parameters"]["input_harmonization_excluded_covariates"])
    if "ResearchGroup_Mapped" not in excluded:
        raise RuntimeError("Diagnosis must be excluded from ComBat")
    if "ResearchGroup_Mapped" in set(
        candidate["parameters"]["input_harmonization_covariates"]
    ):
        raise RuntimeError("Diagnosis cannot be a preserved ComBat covariate")


def append_argument(command: list[str], name: str, value: Any) -> None:
    if isinstance(value, bool):
        if value:
            command.append(f"--{name}")
        return
    if value is None:
        return
    command.append(f"--{name}")
    if isinstance(value, list):
        command.extend(str(item) for item in value)
    else:
        command.append(str(value))


def values_after_flag(command: Sequence[str], flag: str) -> list[str]:
    if flag not in command:
        return []
    values: list[str] = []
    for token in list(command)[list(command).index(flag) + 1 :]:
        if token.startswith("--"):
            break
        values.append(token)
    return values


def build_training_command(
    config: dict[str, Any], python_executable: str, dry_run: bool
) -> list[str]:
    paths = config["paths"]
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
    for key, value in config["parameters"].items():
        append_argument(command, key, value)
    command.extend(
        [
            "--vae_required_metadata_cols",
            "ResearchGroup_Mapped",
            "Manufacturer",
            "Age",
            "Sex",
            "--vae_abort_if_val_split_fails",
        ]
    )
    if dry_run:
        command.append("--dry-run")
    required_tokens = {
        "--channels_to_use": ["1", "0", "2"],
        "--input_harmonization_mode": ["foldwise_combat"],
        "--input_harmonization_batch_col": ["Manufacturer"],
        "--input_harmonization_covariates": ["Age", "Sex"],
        "--input_harmonization_excluded_covariates": ["ResearchGroup_Mapped"],
        "--input_harmonization_fit_scope": ["outer_train_dev_only"],
        "--metadata_features": ["Age", "Sex"],
        "--vae_required_metadata_cols": [
            "ResearchGroup_Mapped",
            "Manufacturer",
            "Age",
            "Sex",
        ],
    }
    for flag, expected in required_tokens.items():
        actual = values_after_flag(command, flag)
        if actual != expected:
            raise RuntimeError(
                f"Command guard {flag}: expected {expected!r}, got {actual!r}"
            )
    if "--vae_abort_if_val_split_fails" not in command:
        raise RuntimeError("Missing --vae_abort_if_val_split_fails")
    return command


def validate_prepared_output(output_dir: Path) -> None:
    marker = output_dir / ".cleanrepro_prepared"
    manifest = output_dir / "provenance" / "live_source_sha256.txt"
    if not output_dir.is_dir() or not marker.is_file() or not manifest.is_file():
        raise RuntimeError(
            f"Output directory was not prepared by the guarded shell launcher: "
            f"{output_dir}"
        )
    forbidden = [
        output_dir / "run_config.json",
        output_dir / "fold_1",
        output_dir / "fold_2",
        output_dir / "fold_3",
        output_dir / "fold_4",
        output_dir / "fold_5",
    ]
    existing = [str(path) for path in forbidden if path.exists()]
    if existing:
        raise RuntimeError(
            "Refusing to reuse a partially populated output directory:\n"
            + "\n".join(existing)
        )
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"Snapshotted live source is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"Live source changed after snapshot: {relative}; "
                f"expected {expected}, got {actual}"
            )


def main() -> int:
    args = parse_args()
    reference = load_json(args.reference_config)
    candidate = load_json(args.config)
    validate_config(reference, candidate)
    python_executable = (
        args.python_executable
        or candidate.get("python_executable")
        or sys.executable
    )
    dependency = dependency_status()
    if not dependency.get("neurocombat_sklearn_CombatModel_available"):
        raise RuntimeError(f"ComBat dependency unavailable: {dependency}")
    command = build_training_command(
        candidate, python_executable, dry_run=bool(args.dry_run)
    )
    print(
        json.dumps(
            {
                "run_name": RUN_NAME,
                "dependency_status": dependency,
                "command": shlex.join(command),
                "dry_run": bool(args.dry_run),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.dry_run:
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if completed.returncode:
            raise SystemExit(completed.returncode)
        print("DRY_RUN_OK: training was not launched and no run output was written.")
        return 0
    if not args.confirm_training:
        raise SystemExit(
            "Refusing training without --confirm-training. Use --dry-run to validate."
        )
    output_dir = resolve(candidate["paths"]["output_dir"])
    validate_prepared_output(output_dir)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
