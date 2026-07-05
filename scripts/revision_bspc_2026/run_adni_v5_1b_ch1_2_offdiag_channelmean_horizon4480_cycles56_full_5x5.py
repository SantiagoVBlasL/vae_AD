#!/usr/bin/env python3
"""Launcher for v5.1b [1,2] offdiag_channelmean FULL 5x5 confirmation.

Default behavior is dry-run/preflight only. Real training requires
--confirm-training. Compared with the v5.1b horizon4480/cycles56 final model,
the only allowed scientific differences are:

  channels_to_use: [1,0,2] -> [1,2]
  recon_loss_mode: mse_sum_batchmean_current -> offdiag_channelmean_sum

Output paths and descriptive metadata may differ. Tensor/metadata paths must
stay on the v5.1b no_pybandpass branch. This run is exploratory/sensitivity
only because the FAST pair [1,2] did not beat the FAST [1] subset.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_CONFIG = PROJECT_ROOT / "configs/runs/adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5.json"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/runs/adni_v5_1b_ch1_2_offdiag_channelmean_horizon4480_cycles56_full_5x5.json"
STAGE_B_SCRIPT = PROJECT_ROOT / "scripts/revision_bspc_2026/run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep.py"
COMPARISON_SCRIPT = PROJECT_ROOT / "scripts/revision_bspc_2026/compare_v5_1b_ch1_2_offdiag_channelmean_vs_final_and_ch1.py"
INTEGRITY_SCRIPT = PROJECT_ROOT / "scripts/revision_bspc_2026/audit_v5_1b_ch1_2_offdiag_channelmean_integrity.py"

EXPECTED_CHANNELS = [1, 2]
EXPECTED_SELECTED_NAMES = ["Pearson_Full_FisherZ_Signed", "MI_KNN_Symmetric"]
EXPECTED_PRIMARY_MODEL = "logreg_l2"
EXPECTED_PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
STALE_TOPLEVEL_NAMES = {"classifier_only_readout", "latent_cache", "run_manifest.json", "command_log.json"}
STALE_PREFIXES = ("fold_", "all_folds_metrics", "summary_metrics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-config", type=Path, default=SOURCE_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-training", action="store_true")
    parser.add_argument("--force-clean", action="store_true")
    parser.add_argument("--skip-classifier-readout", action="store_true")
    parser.add_argument("--skip-comparison", action="store_true")
    parser.add_argument("--python-executable", default=None)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: expected {expected!r}, got {actual!r}")


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


def values_after_flag(tokens: Sequence[str], flag: str) -> List[str]:
    if flag not in tokens:
        return []
    out: List[str] = []
    for token in tokens[tokens.index(flag) + 1 :]:
        if token.startswith("--"):
            break
        out.append(token)
    return out


def validate_controlled_diff(source: Dict[str, Any], target: Dict[str, Any]) -> None:
    source_params = dict(source["parameters"])
    target_params = dict(target["parameters"])
    allowed_param_diffs = {
        "channels_to_use": ([1, 0, 2], [1, 2]),
        "recon_loss_mode": ("mse_sum_batchmean_current", "offdiag_channelmean_sum"),
    }
    param_diffs = {
        key: (source_params.get(key), target_params.get(key))
        for key in sorted(set(source_params) | set(target_params))
        if source_params.get(key) != target_params.get(key)
    }
    if param_diffs != allowed_param_diffs:
        raise RuntimeError(f"Unexpected parameter diffs vs v5.1b horizon4480: {param_diffs}")
    require_equal(target["selected_channel_names"], EXPECTED_SELECTED_NAMES, "selected_channel_names")
    require_equal(source["channel_names_master_in_tensor_order"], target["channel_names_master_in_tensor_order"], "channel_names")
    source_paths = dict(source["paths"])
    target_paths = dict(target["paths"])
    allowed_path_diffs = {"output_dir", "big_disk_output_dir", "split_preview_csv", "split_preview_summary_csv"}
    diffs = {key for key in source_paths if source_paths.get(key) != target_paths.get(key)}
    if diffs != allowed_path_diffs:
        raise RuntimeError(f"Unexpected path diffs vs v5.1b horizon4480: {sorted(diffs)}")
    if source_paths["global_tensor_path"] != target_paths["global_tensor_path"]:
        raise RuntimeError("Tensor path changed unexpectedly.")
    if source_paths["metadata_path"] != target_paths["metadata_path"]:
        raise RuntimeError("Metadata path changed unexpectedly.")
    if "v5_1c" in target_paths["global_tensor_path"] or "v5_1c" in target_paths["metadata_path"]:
        raise RuntimeError("This run must use the v5.1b dataset, not v5.1c.")


def validate_config(config: Dict[str, Any], source: Dict[str, Any]) -> None:
    validate_controlled_diff(source, config)
    params = config["parameters"]
    required_values = {
        "channels_to_use": EXPECTED_CHANNELS,
        "outer_folds": 5,
        "inner_folds": 5,
        "latent_dim": 256,
        "epochs_vae": 4480,
        "cyclical_beta_n_cycles": 56,
        "lr_scheduler_T0": 80,
        "beta_vae": 2.5,
        "dropout_rate_vae": 0.15,
        "vae_dropout_scope": "legacy_all",
        "vae_block_order": "legacy_act_norm",
        "batch_size": 64,
        "vae_final_activation": "tanh",
        "recon_loss_mode": "offdiag_channelmean_sum",
        "intermediate_fc_dim_vae": "quarter",
        "decoder_type": "convtranspose",
        "num_conv_layers_encoder": 4,
        "norm_mode": "zscore_offdiag",
        "vae_train_sampler_strategy": "none",
        "classifier_calibrate": True,
        "classifier_use_class_weight": True,
    }
    for key, expected in required_values.items():
        require_equal(params.get(key), expected, key)
    require_equal(config["selected_channel_names"], EXPECTED_SELECTED_NAMES, "selected_channel_names")
    require_equal(params["metadata_features"], ["Age", "Sex"], "metadata_features")
    require_equal(params["classifier_stratify_cols"], ["Manufacturer"], "classifier_stratify_cols")
    require_equal(params["vae_stratify_cols"], ["Manufacturer"], "vae_stratify_cols")
    require_equal(params["classifier_types"], ["logreg", "svm"], "classifier_types")
    require_equal(params["n_iter_logreg"], 300, "n_iter_logreg")
    require_equal(params["n_iter_svm"], 300, "n_iter_svm")
    if "Sex" in params["classifier_stratify_cols"] or "Sex" in params["vae_stratify_cols"]:
        raise RuntimeError("Sex must remain metadata/covariate only.")
    if params["epochs_vae"] / params["cyclical_beta_n_cycles"] != 80:
        raise RuntimeError("Expected 4480/56 = 80.")


def inspect_tensor(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as zf:
        shape = tuple(int(x) for x in zf["global_tensor_data"].shape)
        bandpass = bool(zf["python_bandpass_applied"].item())
        channels = [str(x) for x in zf["channel_names"].astype(str)]
    if bandpass:
        raise RuntimeError("Refusing tensor with python_bandpass_applied=True")
    require_equal([channels[i] for i in EXPECTED_CHANNELS], EXPECTED_SELECTED_NAMES, "selected tensor channel names")
    return {"shape": shape, "python_bandpass_applied": bandpass}


def load_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["SubjectID", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Metadata missing columns: {missing}")
    if df["SubjectID"].astype(str).duplicated().any():
        raise RuntimeError("Metadata has duplicate SubjectID.")
    return df


def split_preview(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    params = config["parameters"]
    work = df.copy().reset_index(drop=True)
    work["strat_key"] = work["ResearchGroup_Mapped"].astype(str) + "|" + work["Manufacturer"].astype(str)
    if (work["strat_key"].value_counts() < int(params["outer_folds"])).any():
        raise RuntimeError("ResearchGroup_Mapped + Manufacturer 5-fold split is not feasible.")
    splitter = StratifiedKFold(n_splits=int(params["outer_folds"]), shuffle=True, random_state=int(params["seed"]))
    rows = []
    for fold, (_, test_idx) in enumerate(splitter.split(work, work["strat_key"]), start=1):
        sub = work.iloc[test_idx]
        rows.append(
            {
                "fold": fold,
                "N": int(len(sub)),
                "AD": int((sub["ResearchGroup_Mapped"] == "AD").sum()),
                "CN": int((sub["ResearchGroup_Mapped"] == "CN").sum()),
                "MCI": int((sub["ResearchGroup_Mapped"] == "MCI").sum()),
                "GE": int((sub["Manufacturer"] == "GE").sum()),
                "Philips": int((sub["Manufacturer"] == "Philips").sum()),
                "SIEMENS": int((sub["Manufacturer"] == "SIEMENS").sum()),
            }
        )
    out = pd.DataFrame(rows)
    out["valid_fold"] = (out[["AD", "CN", "MCI"]] > 0).all(axis=1) & (out[["GE", "Philips", "SIEMENS"]] > 0).all(axis=1)
    return out


def build_stage_a_command(config: Dict[str, Any], python_exe: str) -> List[str]:
    paths = config["paths"]
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
    for name, value in config["parameters"].items():
        append_arg(command, name, value)
    validate_stage_a_command(command)
    return command


def build_stage_b_command(config: Dict[str, Any], python_exe: str) -> List[str]:
    outdir = resolve(config["paths"]["output_dir"])
    params = config["parameters"]
    command = [
        python_exe,
        str(STAGE_B_SCRIPT),
        "--run-dir",
        str(outdir),
        "--output-dir",
        str(outdir / "classifier_only_readout"),
        "--outer-folds",
        str(params["outer_folds"]),
        "--inner-folds",
        str(params["inner_folds"]),
        "--models",
        EXPECTED_PRIMARY_MODEL,
        "--reuse-latent-cache",
    ]
    validate_stage_b_command(command)
    return command


def build_comparison_command(config: Dict[str, Any], python_exe: str) -> List[str]:
    outdir = resolve(config["paths"]["output_dir"])
    return [
        python_exe,
        str(COMPARISON_SCRIPT),
        "--candidate-run-dir",
        str(outdir),
        "--candidate-readout-dir",
        str(outdir / "classifier_only_readout"),
    ]


def build_integrity_command(config: Dict[str, Any], python_exe: str) -> List[str]:
    outdir = resolve(config["paths"]["output_dir"])
    return [
        python_exe,
        str(INTEGRITY_SCRIPT),
        "--candidate-run",
        str(outdir),
        "--candidate-readout",
        str(outdir / "classifier_only_readout"),
    ]


def validate_stage_a_command(command: Sequence[str]) -> None:
    for idx, token in enumerate(command[:-1]):
        if token.startswith("--n_iter_") and str(command[idx + 1]) == "0":
            raise RuntimeError(f"Invalid Stage A command contains {token} 0")
    require_equal(values_after_flag(command, "--channels_to_use"), ["1", "2"], "Stage A channels")
    require_equal(values_after_flag(command, "--recon_loss_mode"), ["offdiag_channelmean_sum"], "Stage A recon_loss_mode")
    require_equal(values_after_flag(command, "--outer_folds"), ["5"], "Stage A outer_folds")
    require_equal(values_after_flag(command, "--inner_folds"), ["5"], "Stage A inner_folds")
    require_equal(values_after_flag(command, "--epochs_vae"), ["4480"], "Stage A epochs_vae")
    require_equal(values_after_flag(command, "--cyclical_beta_n_cycles"), ["56"], "Stage A cycles")
    require_equal(values_after_flag(command, "--lr_scheduler_T0"), ["80"], "Stage A T0")
    require_equal(values_after_flag(command, "--vae_train_sampler_strategy"), ["none"], "Stage A sampler")


def validate_stage_b_command(command: Sequence[str]) -> None:
    require_equal(values_after_flag(command, "--outer-folds"), ["5"], "Stage B outer-folds")
    require_equal(values_after_flag(command, "--inner-folds"), ["5"], "Stage B inner-folds")
    require_equal(values_after_flag(command, "--models"), [EXPECTED_PRIMARY_MODEL], "Stage B model")


def symlink_points_to(link_path: Path, target_path: Path) -> bool:
    if not link_path.is_symlink():
        return False
    raw = Path(os.readlink(link_path))
    if not raw.is_absolute():
        raw = link_path.parent / raw
    return raw.resolve() == target_path.resolve()


def stale_output_markers(output_dir: Path) -> List[Path]:
    if not output_dir.exists():
        return []
    markers: List[Path] = []
    for child in output_dir.iterdir():
        if child.name in STALE_TOPLEVEL_NAMES or child.name.startswith(STALE_PREFIXES):
            markers.append(child)
    for nested in output_dir.rglob("latent_cache"):
        if nested not in markers:
            markers.append(nested)
    return sorted(markers, key=lambda p: str(p))


def quarantine_path(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quarantine = path.parent / f"{path.name}_quarantine_{stamp}"
    suffix = 1
    while quarantine.exists():
        quarantine = path.parent / f"{path.name}_quarantine_{stamp}_{suffix}"
        suffix += 1
    shutil.move(str(path), str(quarantine))
    return quarantine


def quarantine_existing_output_contents(output_dir: Path) -> Path:
    real_output = output_dir.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quarantine = real_output.parent / f"{real_output.name}_quarantine_{stamp}"
    suffix = 1
    while quarantine.exists():
        quarantine = real_output.parent / f"{real_output.name}_quarantine_{stamp}_{suffix}"
        suffix += 1
    quarantine.mkdir(parents=True, exist_ok=False)
    for child in list(output_dir.iterdir()):
        shutil.move(str(child), str(quarantine / child.name))
    return quarantine


def ensure_output_prepared(config: Dict[str, Any], force_clean: bool) -> Path | None:
    output_dir = resolve(config["paths"]["output_dir"])
    big_disk = Path(config["paths"]["big_disk_output_dir"])
    big_disk.mkdir(parents=True, exist_ok=True)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    quarantine = None
    if output_dir.exists() and not output_dir.is_symlink():
        if not force_clean:
            raise RuntimeError(f"Refusing real run: output_dir exists and is not a symlink: {output_dir}. Use --force-clean.")
        quarantine = quarantine_path(output_dir)
    if output_dir.is_symlink() and not symlink_points_to(output_dir, big_disk):
        if not force_clean:
            raise RuntimeError(f"Refusing real run: output_dir symlink target mismatch: {output_dir} -> {os.readlink(output_dir)}. Use --force-clean.")
        quarantine = quarantine_path(output_dir)
    if not output_dir.exists():
        output_dir.symlink_to(big_disk, target_is_directory=True)
    if not output_dir.is_symlink() or not symlink_points_to(output_dir, big_disk):
        raise RuntimeError(f"Failed to prepare output symlink {output_dir} -> {big_disk}")
    stale = stale_output_markers(output_dir)
    if stale and not force_clean:
        preview = "\n".join(f"  - {p}" for p in stale[:20])
        raise RuntimeError("Refusing real run: stale output markers exist. Use --force-clean to quarantine.\n" + preview)
    if stale and force_clean:
        quarantine = quarantine_existing_output_contents(output_dir)
    if any(output_dir.iterdir()):
        raise RuntimeError(f"Refusing real run: output_dir is not empty after clean policy: {output_dir}")
    return quarantine


def verify_fresh_checkpoints(config: Dict[str, Any], run_start_epoch: float) -> List[Dict[str, Any]]:
    outdir = resolve(config["paths"]["output_dir"])
    rows = []
    for fold in range(1, 6):
        path = outdir / f"fold_{fold}" / f"vae_model_fold_{fold}.pt"
        exists = path.exists()
        mtime = path.stat().st_mtime if exists else None
        rows.append(
            {
                "fold": fold,
                "path": str(path),
                "exists": exists,
                "mtime_epoch": mtime,
                "mtime_iso": datetime.fromtimestamp(mtime).isoformat() if mtime else "",
                "fresh_after_run_start": bool(exists and mtime and mtime > run_start_epoch),
            }
        )
    bad = [row for row in rows if not row["fresh_after_run_start"]]
    (outdir / "fresh_checkpoint_validation.json").write_text(
        json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(), "rows": rows, "all_fresh": not bad}, indent=2),
        encoding="utf-8",
    )
    if bad:
        raise RuntimeError("Fresh checkpoint validation failed: " + json.dumps(bad, indent=2))
    return rows


def write_manifest(config_path: Path, config: Dict[str, Any], stage_a: List[str], stage_b: List[str], comparison: List[str], integrity: List[str], quarantine: Path | None) -> None:
    outdir = resolve(config["paths"]["output_dir"])
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": config["run_name"],
        "config_path": str(config_path),
        "controlled_change_vs_v5_1b_horizon4480": "channels_to_use [1,0,2]->[1,2]; recon_loss_mode mse_sum_batchmean_current->offdiag_channelmean_sum",
        "stage_a_command": stage_a,
        "stage_a_command_shell": shlex.join(stage_a),
        "stage_b_command": stage_b,
        "stage_b_command_shell": shlex.join(stage_b),
        "comparison_command": comparison,
        "comparison_command_shell": shlex.join(comparison),
        "integrity_audit_command": integrity,
        "integrity_audit_command_shell": shlex.join(integrity),
        "quarantine_dir": str(quarantine) if quarantine else "",
        "ranking_source": "Stage B classifier-only logreg_l2",
        "primary_threshold_strategy": EXPECTED_PRIMARY_THRESHOLD,
        "threshold_selection": "true_inner_cv_oof",
    }
    (outdir / "run_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_command_log(config: Dict[str, Any], stage_a_rc: int | None, stage_b_rc: int | None, comparison_rc: int | None, quarantine: Path | None) -> None:
    outdir = resolve(config["paths"]["output_dir"])
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": config["run_name"],
        "training_launched": stage_a_rc is not None,
        "stage_a_returncode": stage_a_rc,
        "stage_b_returncode": stage_b_rc,
        "comparison_returncode": comparison_rc,
        "controlled_change_vs_v5_1b_horizon4480": "channels_to_use [1,0,2]->[1,2]; recon_loss_mode mse_sum_batchmean_current->offdiag_channelmean_sum",
        "quarantine_dir": str(quarantine) if quarantine else "",
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "existing_model_outputs_modified": False,
    }
    (outdir / "command_log.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    source = load_json(args.source_config)
    validate_config(config, source)
    python_exe = args.python_executable or config.get("python_executable") or sys.executable
    if not args.dry_run and not args.confirm_training:
        raise SystemExit("Refusing to launch training without --confirm-training. Use --dry-run for preflight only.")

    tensor_info = inspect_tensor(resolve(config["paths"]["global_tensor_path"]))
    metadata = load_metadata(resolve(config["paths"]["metadata_path"]))
    preview = split_preview(metadata, config)
    if not preview["valid_fold"].all():
        raise RuntimeError("5-fold split preview failed")
    preview.to_csv(resolve(config["paths"]["split_preview_summary_csv"]), index=False)

    stage_a = build_stage_a_command(config, python_exe)
    stage_b = build_stage_b_command(config, python_exe)
    comparison = build_comparison_command(config, python_exe)
    integrity = build_integrity_command(config, python_exe)
    outdir = resolve(config["paths"]["output_dir"])
    stale = stale_output_markers(outdir)

    print(f"Run name       : {config['run_name']}")
    print(f"Mode           : {'DRY-RUN' if args.dry_run else 'REAL RUN (CONFIRMED)'}")
    print("Config diff    : channels_to_use [1,0,2]->[1,2]; recon_loss_mode mse_sum_batchmean_current->offdiag_channelmean_sum")
    print(f"Tensor         : {config['paths']['global_tensor_path']}")
    print(f"Metadata       : {config['paths']['metadata_path']}")
    print(f"Tensor shape   : {tensor_info['shape']}; python_bandpass_applied={tensor_info['python_bandpass_applied']}")
    print(f"Metadata rows  : {len(metadata)}; AD={(metadata['ResearchGroup_Mapped']=='AD').sum()}, CN={(metadata['ResearchGroup_Mapped']=='CN').sum()}, MCI={(metadata['ResearchGroup_Mapped']=='MCI').sum()}")
    print("Channels       : [1,2] Pearson_Full_FisherZ_Signed + MI_KNN_Symmetric")
    print("VAE objective  : offdiag_channelmean_sum")
    print("VAE schedule   : 5x5, latent_dim=256, epochs=4480, cycles=56, cycle_len=80, T0=80")
    print("Readout        : Stage B classifier-only logreg_l2 + true inner-CV OOF thresholding")
    print("Clean-run      : refuses stale output unless --force-clean")
    print(preview.to_string(index=False))
    print("\nStage A command:")
    print(shlex.join(stage_a))
    print("\nStage B command:")
    print(shlex.join(stage_b))
    print("\nComparison command:")
    print(shlex.join(comparison))
    print("\nIntegrity audit command:")
    print(shlex.join(integrity))
    if stale:
        print(f"\nStale markers detected: {len(stale)}")
        for marker in stale[:20]:
            print(f"  - {marker}")
        if args.dry_run:
            print("Dry-run only: no cleanup performed.")
    else:
        print("\nNo stale markers detected.")
    if args.dry_run:
        print("Dry-run complete. Training was NOT launched.")
        return 0

    quarantine = ensure_output_prepared(config, args.force_clean)
    run_start = time.time()
    write_manifest(args.config, config, stage_a, stage_b, comparison, integrity, quarantine)
    stage_a_completed = subprocess.run(stage_a, cwd=PROJECT_ROOT, check=False)
    write_command_log(config, stage_a_completed.returncode, None, None, quarantine)
    if stage_a_completed.returncode != 0:
        return int(stage_a_completed.returncode)
    verify_fresh_checkpoints(config, run_start)
    if args.skip_classifier_readout:
        return 0
    stage_b_completed = subprocess.run(stage_b, cwd=PROJECT_ROOT, check=False)
    write_command_log(config, stage_a_completed.returncode, stage_b_completed.returncode, None, quarantine)
    if stage_b_completed.returncode != 0:
        return int(stage_b_completed.returncode)
    print("\nRun the integrity audit with:")
    print(shlex.join(integrity), flush=True)
    if args.skip_comparison:
        return 0
    comparison_completed = subprocess.run(comparison, cwd=PROJECT_ROOT, check=False)
    write_command_log(config, stage_a_completed.returncode, stage_b_completed.returncode, comparison_completed.returncode, quarantine)
    return int(comparison_completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
