#!/usr/bin/env python3
"""Prepare/run decoder-specific dropout ablation for the current best ADNI model.

Candidate:
  recover035_latent384_beta3p75_encdrop0p15_decdrop0p10_T80_h10000_p560_full5x5

Reference:
  recover035_latent384_beta3p75_T80_h10000_p560_full5x5

Controlled scientific change:
  - dropout_rate_vae remains 0.15 as the global fallback
  - encoder_dropout_rate_vae = 0.15
  - decoder_dropout_rate_vae = 0.10

No real training is launched unless --confirm-training is supplied.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/runs/adni_v5_1c_recover035_latent384_beta3p75_encdrop0p15_decdrop0p10_T80_h10000_p560_full5x5.json"
)
REFERENCE_CONFIG = (
    PROJECT_ROOT
    / "configs/runs/adni_v5_1c_recover035_latent384_beta3p75_T80_h10000_p560_full5x5.json"
)
STAGE_B_SCRIPT = (
    PROJECT_ROOT / "scripts/revision_bspc_2026/run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep.py"
)

EXPECTED_RUN_NAME = "recover035_latent384_beta3p75_encdrop0p15_decdrop0p10_T80_h10000_p560_full5x5"
EXPECTED_CHANNELS = [1, 0, 2]
EXPECTED_LATENT_DIM = 384
EXPECTED_BETA = 3.75
EXPECTED_DROPOUT = 0.15
EXPECTED_ENCODER_DROPOUT = 0.15
EXPECTED_DECODER_DROPOUT = 0.10
EXPECTED_EPOCHS = 10000
EXPECTED_CYCLES = 125
EXPECTED_T0 = 80
EXPECTED_PATIENCE = 560
EXPECTED_VAE_COUNTS = {"CN": 300, "MCI": 250, "AD": 97}
EXPECTED_CLF_COUNTS = {"CN": 300, "AD": 97}
RECOVER_SUBJECT = "035_S_6927"
EXCLUDED_SUBJECT = "128_S_2002"

STALE_TOPLEVEL_NAMES = {"classifier_only_readout", "latent_cache", "run_manifest.json"}
STALE_PREFIXES = ("fold_", "all_folds_metrics", "summary_metrics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reference-config", type=Path, default=REFERENCE_CONFIG)
    parser.add_argument("--python-executable", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print commands only.")
    parser.add_argument("--confirm-training", action="store_true", help="Required to launch Stage A/Stage B.")
    parser.add_argument("--force-clean", action="store_true", help="Move existing output contents to quarantine before real training.")
    parser.add_argument("--skip-stage-b", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def append_arg(command: list[str], name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            command.append(f"--{name}")
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
    for token in tokens[list(tokens).index(flag) + 1:]:
        if token.startswith("--"):
            break
        out.append(token)
    return out


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: expected {expected!r}, got {actual!r}")


def validate_config(config: dict[str, Any], reference: dict[str, Any]) -> None:
    if config["run_name"] != EXPECTED_RUN_NAME:
        raise RuntimeError(f"Unexpected run_name: {config['run_name']!r}")
    params = dict(config["parameters"])
    ref_params = dict(reference["parameters"])
    ref_params.setdefault("encoder_dropout_rate_vae", None)
    ref_params.setdefault("decoder_dropout_rate_vae", None)
    params.setdefault("encoder_dropout_rate_vae", None)
    params.setdefault("decoder_dropout_rate_vae", None)

    allowed = {
        "encoder_dropout_rate_vae": (None, EXPECTED_ENCODER_DROPOUT),
        "decoder_dropout_rate_vae": (None, EXPECTED_DECODER_DROPOUT),
    }
    diffs = {k: (ref_params.get(k), params.get(k)) for k in sorted(set(ref_params) | set(params)) if ref_params.get(k) != params.get(k)}
    if diffs != allowed:
        raise RuntimeError(f"Unexpected scientific parameter diffs vs p=0.15 reference: {diffs}")

    if config["paths"]["global_tensor_path"] != reference["paths"]["global_tensor_path"]:
        raise RuntimeError("global_tensor_path changed relative to p=0.15 reference.")
    if config["paths"]["metadata_path"] != reference["paths"]["metadata_path"]:
        raise RuntimeError("metadata_path changed relative to p=0.15 reference.")

    checks = {
        "channels_to_use": EXPECTED_CHANNELS,
        "latent_dim": EXPECTED_LATENT_DIM,
        "beta_vae": EXPECTED_BETA,
        "dropout_rate_vae": EXPECTED_DROPOUT,
        "encoder_dropout_rate_vae": EXPECTED_ENCODER_DROPOUT,
        "decoder_dropout_rate_vae": EXPECTED_DECODER_DROPOUT,
        "vae_dropout_scope": "legacy_all",
        "epochs_vae": EXPECTED_EPOCHS,
        "cyclical_beta_n_cycles": EXPECTED_CYCLES,
        "lr_scheduler_T0": EXPECTED_T0,
        "early_stopping_patience_vae": EXPECTED_PATIENCE,
        "vae_final_activation": "tanh",
        "intermediate_fc_dim_vae": "quarter",
        "recon_loss_mode": "mse_sum_batchmean_current",
    }
    for key, expected in checks.items():
        require_equal(params.get(key), expected, f"parameters.{key}")


def metadata_counts(metadata_path: Path) -> tuple[dict[str, int], dict[str, int]]:
    meta = pd.read_csv(metadata_path)
    meta["SubjectID"] = meta["SubjectID"].astype(str)
    if RECOVER_SUBJECT not in set(meta["SubjectID"]):
        raise RuntimeError(f"{RECOVER_SUBJECT} missing from metadata.")
    if EXCLUDED_SUBJECT in set(meta["SubjectID"]):
        raise RuntimeError(f"{EXCLUDED_SUBJECT} unexpectedly present in metadata.")
    dx = meta["ResearchGroup_Mapped"].astype(str)
    vae_counts = dx.value_counts().to_dict()
    clf = meta[dx.isin(["CN", "AD"])]
    clf_counts = clf["ResearchGroup_Mapped"].astype(str).value_counts().to_dict()
    return {k: int(vae_counts.get(k, 0)) for k in ["CN", "MCI", "AD"]}, {k: int(clf_counts.get(k, 0)) for k in ["CN", "AD"]}


def stale_output_markers(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        return []
    markers = []
    for child in output_dir.iterdir():
        if child.name in STALE_TOPLEVEL_NAMES or child.name.startswith(STALE_PREFIXES):
            markers.append(child)
    for nested in output_dir.rglob("latent_cache"):
        if nested not in markers:
            markers.append(nested)
    return sorted(markers, key=lambda p: str(p))


def quarantine_existing_output_contents(output_dir: Path) -> Path:
    real_output = output_dir.resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quarantine = real_output.parent / f"{real_output.name}_quarantine_{timestamp}"
    suffix = 1
    while quarantine.exists():
        quarantine = real_output.parent / f"{real_output.name}_quarantine_{timestamp}_{suffix}"
        suffix += 1
    quarantine.mkdir(parents=True, exist_ok=False)
    for child in list(output_dir.iterdir()):
        child.rename(quarantine / child.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    return quarantine


def ensure_output_prepared(config: dict[str, Any], force_clean: bool) -> Path | None:
    output_dir = resolve(config["paths"]["output_dir"])
    big_disk = Path(config["paths"]["big_disk_output_dir"])
    if not output_dir.exists():
        raise RuntimeError(
            "Refusing real training: output_dir is missing. Create the target/symlink first:\n"
            f"  mkdir -p {shlex.quote(str(big_disk))}\n"
            f"  ln -s {shlex.quote(str(big_disk))} {shlex.quote(str(output_dir))}"
        )
    if not output_dir.is_symlink():
        raise RuntimeError(f"Refusing real training: output_dir is not a symlink: {output_dir}")
    if output_dir.resolve() != big_disk.resolve():
        raise RuntimeError(f"Refusing real training: symlink target is {output_dir.resolve()}, expected {big_disk.resolve()}")
    stale = stale_output_markers(output_dir)
    if stale and not force_clean:
        preview = "\n".join(f"  - {p}" for p in stale[:20])
        raise RuntimeError(f"Refusing real training: output_dir contains stale artifacts. Use --force-clean.\n{preview}")
    if stale and force_clean:
        return quarantine_existing_output_contents(output_dir)
    return None


def build_stage_a_command(config: dict[str, Any], python_exe: str) -> list[str]:
    paths = config["paths"]
    params = config["parameters"]
    command = [
        python_exe,
        str(resolve(paths["training_script"])),
        "--global_tensor_path", str(resolve(paths["global_tensor_path"])),
        "--metadata_path", str(resolve(paths["metadata_path"])),
        "--output_dir", str(resolve(paths["output_dir"])),
    ]
    for name, value in params.items():
        append_arg(command, name, value)
    command.extend(["--vae_required_metadata_cols", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex"])
    command.append("--vae_abort_if_val_split_fails")
    return command


def build_stage_b_command(config: dict[str, Any], python_exe: str) -> list[str]:
    outdir = resolve(config["paths"]["output_dir"])
    params = config["parameters"]
    return [
        python_exe,
        str(STAGE_B_SCRIPT),
        "--run-dir", str(outdir),
        "--output-dir", str(outdir / "classifier_only_readout"),
        "--outer-folds", str(params["outer_folds"]),
        "--inner-folds", str(params["inner_folds"]),
        "--models", "logreg_l2",
        "--reuse-latent-cache",
    ]


def validate_stage_a_command(command: Sequence[str]) -> None:
    require_equal(values_after_flag(command, "--encoder_dropout_rate_vae"), ["0.15"], "Stage A encoder_dropout_rate_vae")
    require_equal(values_after_flag(command, "--decoder_dropout_rate_vae"), ["0.1"], "Stage A decoder_dropout_rate_vae")
    require_equal(values_after_flag(command, "--dropout_rate_vae"), ["0.15"], "Stage A dropout_rate_vae")
    require_equal(values_after_flag(command, "--vae_dropout_scope"), ["legacy_all"], "Stage A vae_dropout_scope")
    require_equal(values_after_flag(command, "--channels_to_use"), ["1", "0", "2"], "Stage A channels_to_use")
    require_equal(values_after_flag(command, "--latent_dim"), ["384"], "Stage A latent_dim")
    require_equal(values_after_flag(command, "--beta_vae"), ["3.75"], "Stage A beta_vae")


def run_command(command: list[str]) -> None:
    print(shlex.join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> int:
    args = parse_args()
    config = load_json(resolve(args.config))
    reference = load_json(resolve(args.reference_config))
    validate_config(config, reference)
    python_exe = args.python_executable or config.get("python_executable") or sys.executable

    vae_counts, clf_counts = metadata_counts(resolve(config["paths"]["metadata_path"]))
    require_equal(vae_counts, EXPECTED_VAE_COUNTS, "VAE pool counts")
    require_equal(clf_counts, EXPECTED_CLF_COUNTS, "classifier pool counts")

    stage_a = build_stage_a_command(config, python_exe)
    stage_b = build_stage_b_command(config, python_exe)
    validate_stage_a_command(stage_a)
    output_dir = resolve(config["paths"]["output_dir"])
    stale = stale_output_markers(output_dir)

    print(f"Config: {resolve(args.config)}")
    print(f"Reference config: {resolve(args.reference_config)}")
    print(f"Run name: {config['run_name']}")
    print("Controlled diff vs p=0.15 reference: encoder_dropout_rate_vae None->0.15; decoder_dropout_rate_vae None->0.10")
    print(f"Global fallback dropout_rate_vae remains {EXPECTED_DROPOUT}")
    print(f"VAE pool counts: {vae_counts}")
    print(f"Classifier pool counts: {clf_counts}")
    print(f"Output dir: {output_dir}")
    print(f"Stale markers: {len(stale)}")
    if stale:
        for marker in stale[:20]:
            print(f"  - {marker}")
    print("\nStage A command:")
    print(shlex.join(stage_a))
    print("\nStage B command:")
    print(shlex.join(stage_b))

    if args.dry_run:
        print("\nDry-run OK. No training launched.")
        return 0
    if not args.confirm_training:
        raise SystemExit("Refusing to launch training without --confirm-training. Use --dry-run for validation.")

    quarantine = ensure_output_prepared(config, force_clean=args.force_clean)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": config["run_name"],
        "controlled_change": "encoder_dropout_rate_vae=0.15; decoder_dropout_rate_vae=0.10; dropout_rate_vae fallback remains 0.15",
        "vae_pool_counts": vae_counts,
        "classifier_pool_counts": clf_counts,
        "stage_a_command": stage_a,
        "stage_b_command": stage_b,
        "quarantine": str(quarantine) if quarantine else "",
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run_command(stage_a)
    if not args.skip_stage_b:
        run_command(stage_b)
    print("Training/readout completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
