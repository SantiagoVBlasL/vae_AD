#!/usr/bin/env python3
"""Guarded launcher for Branch B: foldwise Manufacturer ComBat before VAE."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "scripts/revision_bspc_2026") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts/revision_bspc_2026"))

from foldwise_combat_input_harmonization import dependency_status  # noqa: E402

REFERENCE_CONFIG = PROJECT_ROOT / "configs/runs/adni_v5_1c_recover035_latent384_beta3p75_T80_h10000_p560_full5x5.json"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/runs/adni_v5_1c_recover035_latent384_beta3p75_foldcombat_mfr_age_sex_T80_h10000_p560_full5x5.json"
RUN_NAME = "recover035_latent384_beta3p75_foldcombat_mfr_age_sex_T80_h10000_p560_full5x5"
HARMONIZATION_KEYS = {
    "input_harmonization_mode": "foldwise_combat",
    "input_harmonization_batch_col": "Manufacturer",
    "input_harmonization_covariates": ["Age", "Sex"],
    "input_harmonization_excluded_covariates": ["ResearchGroup_Mapped"],
    "input_harmonization_fit_scope": "outer_train_dev_only",
    "input_harmonization_vectorization": "upper_offdiag_by_channel",
}
PATH_KEYS_ALLOWED = {"output_dir", "big_disk_output_dir", "split_preview_csv", "split_preview_summary_csv"}
DEFAULT_PREFLIGHT_OUT = PROJECT_ROOT / "results/revision_bspc_2026/foldcombat_real_training_wiring_preflight_20260607"
PREFLIGHT_SCRIPT = PROJECT_ROOT / "scripts/revision_bspc_2026/prepare_foldcombat_real_training_wiring_preflight_20260607.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reference-config", type=Path, default=REFERENCE_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="Validate only; no training.")
    parser.add_argument("--confirm-training", action="store_true", help="Required for real training.")
    parser.add_argument("--python-executable", default=None)
    parser.add_argument("--preflight-output-dir", type=Path, default=DEFAULT_PREFLIGHT_OUT)
    parser.add_argument(
        "--skip-guard-preflight",
        action="store_true",
        help="Internal/testing only: skip foldwise harmonization guard preflight.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(ref: dict[str, Any], cand: dict[str, Any]) -> None:
    if cand["run_name"] != RUN_NAME:
        raise RuntimeError(f"Unexpected run_name: {cand['run_name']}")
    r = dict(ref["parameters"])
    c = dict(cand["parameters"])
    for key, expected in HARMONIZATION_KEYS.items():
        if c.get(key) != expected:
            raise RuntimeError(f"{key}: expected {expected!r}, got {c.get(key)!r}")
        c.pop(key, None)
    diff = {k: (r.get(k), c.get(k)) for k in sorted(set(r) | set(c)) if r.get(k) != c.get(k)}
    if diff:
        raise RuntimeError(f"Unexpected non-harmonization scientific diff: {diff}")
    for section in ["selected_channel_names", "channel_names_master_in_tensor_order", "split_strategy"]:
        if ref.get(section) != cand.get(section):
            raise RuntimeError(f"Unexpected diff in {section}")
    for key, value in cand["paths"].items():
        if key in PATH_KEYS_ALLOWED:
            if RUN_NAME not in str(value):
                raise RuntimeError(f"Provenance path {key} does not contain run name: {value}")
        elif ref["paths"].get(key) != value:
            raise RuntimeError(f"Unexpected path diff for {key}: {ref['paths'].get(key)} -> {value}")


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


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


def values_after_flag(tokens: Sequence[str], flag: str) -> list[str]:
    if flag not in tokens:
        return []
    out: list[str] = []
    for token in list(tokens)[list(tokens).index(flag) + 1 :]:
        if token.startswith("--"):
            break
        out.append(token)
    return out


def build_training_command(config: dict[str, Any], python_exe: str, *, dry_run: bool) -> list[str]:
    paths = config["paths"]
    params = config["parameters"]
    command = [
        python_exe,
        str(resolve(paths["training_script"])),
        "--global_tensor_path",
        str(resolve(paths["global_tensor_path"])),
        "--metadata_path",
        str(resolve(paths["metadata_path"])),
        "--output_dir",
        str(resolve(paths["output_dir"])),
    ]
    for key, value in params.items():
        append_arg(command, key, value)
    command.extend(["--vae_required_metadata_cols", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex"])
    command.append("--vae_abort_if_val_split_fails")
    if dry_run:
        command.append("--dry-run")
    validate_training_command(command)
    return command


def validate_training_command(command: Sequence[str]) -> None:
    expected_flags = {
        "--channels_to_use": ["1", "0", "2"],
        "--input_harmonization_mode": ["foldwise_combat"],
        "--input_harmonization_batch_col": ["Manufacturer"],
        "--input_harmonization_covariates": ["Age", "Sex"],
        "--input_harmonization_excluded_covariates": ["ResearchGroup_Mapped"],
        "--input_harmonization_fit_scope": ["outer_train_dev_only"],
        "--input_harmonization_vectorization": ["upper_offdiag_by_channel"],
        "--metadata_features": ["Age", "Sex"],
        "--classifier_stratify_cols": ["Manufacturer"],
        "--vae_stratify_cols": ["Manufacturer"],
        "--vae_required_metadata_cols": ["ResearchGroup_Mapped", "Manufacturer", "Age", "Sex"],
    }
    for flag, expected in expected_flags.items():
        actual = values_after_flag(command, flag)
        if actual != expected:
            raise RuntimeError(f"{flag}: expected {expected!r}, got {actual!r}")
    if "--vae_abort_if_val_split_fails" not in command:
        raise RuntimeError("Training command must include --vae_abort_if_val_split_fails")


def run_guard_preflight(args: argparse.Namespace, python_exe: str) -> None:
    command = [
        python_exe,
        str(PREFLIGHT_SCRIPT),
        "--config",
        str(args.config),
        "--reference-config",
        str(args.reference_config),
        "--output-dir",
        str(args.preflight_output_dir),
        "--skip-launcher-dry-run",
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> int:
    args = parse_args()
    ref = load_json(args.reference_config)
    cand = load_json(args.config)
    validate_config(ref, cand)
    python_exe = args.python_executable or cand.get("python_executable") or sys.executable
    deps = dependency_status()
    print(json.dumps({"run_name": cand["run_name"], "dependency_status": deps}, indent=2, sort_keys=True))
    if not args.dry_run and not args.confirm_training:
        raise SystemExit("Refusing to launch training without --confirm-training. Use --dry-run for validation.")
    if not args.skip_guard_preflight:
        run_guard_preflight(args, python_exe)
    if args.dry_run:
        dry_command = build_training_command(cand, python_exe, dry_run=True)
        completed = subprocess.run(dry_command, cwd=PROJECT_ROOT, check=False)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
        print("Branch B trainer dry-run command:")
        print(shlex.join(dry_command))
        print("Dry-run/preflight OK. Training was NOT launched.")
        return 0

    train_command = build_training_command(cand, python_exe, dry_run=False)
    print("Branch B confirmed training command:")
    print(shlex.join(train_command))
    completed = subprocess.run(train_command, cwd=PROJECT_ROOT, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
