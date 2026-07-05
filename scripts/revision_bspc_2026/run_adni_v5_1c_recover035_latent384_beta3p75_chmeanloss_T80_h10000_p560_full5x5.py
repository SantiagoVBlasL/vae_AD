#!/usr/bin/env python3
"""Launcher/preflight for Branch A: channel-normalized reconstruction loss.

Real training is guarded by --confirm-training. The default path is dry-run.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_CONFIG = PROJECT_ROOT / "configs/runs/adni_v5_1c_recover035_latent384_beta3p75_T80_h10000_p560_full5x5.json"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/runs/adni_v5_1c_recover035_latent384_beta3p75_chmeanloss_T80_h10000_p560_full5x5.json"
RUN_NAME = "recover035_latent384_beta3p75_chmeanloss_T80_h10000_p560_full5x5"
EXPECTED_DIFF = {"recon_loss_mode": ("mse_sum_batchmean_current", "mse_offdiag_channel_mean_sum")}
PATH_KEYS_ALLOWED = {"output_dir", "big_disk_output_dir", "split_preview_csv", "split_preview_summary_csv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reference-config", type=Path, default=REFERENCE_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="Validate and call the core launcher with --dry-run.")
    parser.add_argument("--confirm-training", action="store_true", help="Required to launch Stage A training.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def effective_param_diff(ref: dict[str, Any], cand: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    r = dict(ref["parameters"])
    c = dict(cand["parameters"])
    return {k: (r.get(k), c.get(k)) for k in sorted(set(r) | set(c)) if r.get(k) != c.get(k)}


def validate_config(ref: dict[str, Any], cand: dict[str, Any]) -> None:
    if cand["run_name"] != RUN_NAME:
        raise RuntimeError(f"Unexpected run_name: {cand['run_name']}")
    diff = effective_param_diff(ref, cand)
    if diff != EXPECTED_DIFF:
        raise RuntimeError(f"Scientific diff mismatch. Expected {EXPECTED_DIFF}, got {diff}")
    for section in ["selected_channel_names", "channel_names_master_in_tensor_order", "split_strategy"]:
        if ref.get(section) != cand.get(section):
            raise RuntimeError(f"Unexpected diff in {section}")
    for key, value in cand["paths"].items():
        if key in PATH_KEYS_ALLOWED:
            if RUN_NAME not in str(value):
                raise RuntimeError(f"Provenance path {key} does not contain run name: {value}")
        elif ref["paths"].get(key) != value:
            raise RuntimeError(f"Unexpected path diff for {key}: {ref['paths'].get(key)} -> {value}")


def append_arg(cmd: list[str], name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            cmd.append(f"--{name}")
        return
    cmd.append(f"--{name}")
    if isinstance(value, list):
        cmd.extend(str(v) for v in value)
    else:
        cmd.append(str(value))


def stage_a_command(config: dict[str, Any], *, dry_run: bool) -> list[str]:
    python_exe = config.get("python_executable") or sys.executable
    cmd = [
        python_exe,
        str(resolve(config["paths"]["training_script"])),
        "--global_tensor_path",
        str(resolve(config["paths"]["global_tensor_path"])),
        "--metadata_path",
        str(resolve(config["paths"]["metadata_path"])),
        "--output_dir",
        str(resolve(config["paths"]["output_dir"])),
    ]
    for key, value in config["parameters"].items():
        append_arg(cmd, key, value)
    cmd.extend(["--vae_required_metadata_cols", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex"])
    cmd.append("--vae_abort_if_val_split_fails")
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def main() -> int:
    args = parse_args()
    if not args.dry_run and not args.confirm_training:
        raise SystemExit("Refusing to launch training without --confirm-training. Use --dry-run for preflight.")
    ref = load_json(args.reference_config)
    cand = load_json(args.config)
    validate_config(ref, cand)
    cmd = stage_a_command(cand, dry_run=args.dry_run)
    print(shlex.join(cmd))
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
