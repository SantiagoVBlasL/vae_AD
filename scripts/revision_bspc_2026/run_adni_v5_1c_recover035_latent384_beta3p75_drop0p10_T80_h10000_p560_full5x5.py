#!/usr/bin/env python3
"""Launcher/preflight for recover035_latent384_beta3p75_drop0p10_T80_h10000_p560_full5x5.

Controlled change versus the current best all-eligible model:
  dropout_rate_vae: 0.15 -> 0.10

Everything else is held fixed. Default mode is dry-run/preflight. Real training
requires --confirm-training.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIG = PROJECT_ROOT / "configs/runs/adni_v5_1c_recover035_latent384_beta3p75_T80_h10000_p560_full5x5.json"
CANONICAL_TARGET_CONFIG = PROJECT_ROOT / "configs/runs/adni_v5_1c_recover035_latent384_beta3p75_drop0p10_T80_h10000_p560_full5x5.json"
PREFLIGHT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/recover035_latent384_beta3p75_drop0p10_T80_h10000_p560_full5x5_preflight"
STAGE_B_SCRIPT = PROJECT_ROOT / "scripts/revision_bspc_2026/run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep.py"

RECOVER_SUBJECT = "035_S_6927"
EXCLUDED_SUBJECT = "128_S_2002"
EXPECTED_VAE_COUNTS = {"CN": 300, "MCI": 250, "AD": 97}
EXPECTED_CLF_COUNTS = {"CN": 300, "AD": 97}
STALE_NAMES = {"classifier_only_readout", "latent_cache", "run_manifest.json"}
STALE_PREFIXES = ("fold_", "all_folds_metrics", "summary_metrics")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def diff_configs(source: Dict[str, Any], target: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    source_params = dict(source["parameters"])
    target_params = dict(target["parameters"])
    for key in sorted(set(source_params) | set(target_params)):
        old = source_params.get(key, "<MISSING>")
        new = target_params.get(key, "<MISSING>")
        if old != new:
            rows.append({"section": "parameters", "key": key, "source_value": old, "target_value": new})
    for section in ["paths", "split_strategy"]:
        src = source.get(section, {})
        tgt = target.get(section, {})
        for key in sorted(set(src) | set(tgt)):
            old = src.get(key, "<MISSING>")
            new = tgt.get(key, "<MISSING>")
            if old != new:
                rows.append({"section": section, "key": key, "source_value": old, "target_value": new})
    if source.get("run_name") != target.get("run_name"):
        rows.append({"section": "metadata", "key": "run_name", "source_value": source.get("run_name"), "target_value": target.get("run_name")})
    if source.get("description") != target.get("description"):
        rows.append({"section": "metadata", "key": "description", "source_value": source.get("description"), "target_value": target.get("description")})
    return pd.DataFrame(rows)


def validate_exact_scientific_diff(source: Dict[str, Any], target: Dict[str, Any]) -> None:
    source_params = dict(source["parameters"])
    target_params = dict(target["parameters"])
    if set(source_params) != set(target_params):
        raise RuntimeError(
            "Parameter keys differ: "
            f"missing={sorted(set(source_params) - set(target_params))}, "
            f"extra={sorted(set(target_params) - set(source_params))}"
        )
    diffs = {k: (source_params[k], target_params[k]) for k in source_params if source_params[k] != target_params[k]}
    if diffs != {"dropout_rate_vae": (0.15, 0.10)}:
        raise RuntimeError(f"Expected only dropout_rate_vae 0.15 -> 0.10, got {diffs}")
    for key in ["global_tensor_path", "metadata_path"]:
        if source["paths"][key] != target["paths"][key]:
            raise RuntimeError(f"{key} changed unexpectedly")
    for key in [
        "channels_to_use",
        "latent_dim",
        "beta_vae",
        "vae_dropout_scope",
        "epochs_vae",
        "cyclical_beta_n_cycles",
        "cyclical_beta_ratio_increase",
        "lr_scheduler_T0",
        "early_stopping_patience_vae",
        "batch_size",
        "decoder_type",
        "recon_loss_mode",
        "vae_final_activation",
        "intermediate_fc_dim_vae",
        "classifier_stratify_cols",
        "vae_stratify_cols",
        "metadata_features",
    ]:
        if source_params[key] != target_params[key]:
            raise RuntimeError(f"{key} changed unexpectedly")


def validate_pool(config: Dict[str, Any]) -> pd.DataFrame:
    meta = pd.read_csv(resolve(config["paths"]["metadata_path"]))
    if "tensor_idx" not in meta.columns and "tensor_index" in meta.columns:
        meta = meta.rename(columns={"tensor_index": "tensor_idx"})
    meta["SubjectID"] = meta["SubjectID"].astype(str)
    vae_counts = meta["ResearchGroup_Mapped"].value_counts().to_dict()
    clf = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy()
    clf_counts = clf["ResearchGroup_Mapped"].value_counts().to_dict()
    rows = []
    for dx, expected in EXPECTED_VAE_COUNTS.items():
        rows.append({"pool": "vae", "group": dx, "observed": int(vae_counts.get(dx, 0)), "expected": expected})
    for dx, expected in EXPECTED_CLF_COUNTS.items():
        rows.append({"pool": "classifier", "group": dx, "observed": int(clf_counts.get(dx, 0)), "expected": expected})
    rows.append({"pool": "required_subject", "group": RECOVER_SUBJECT, "observed": bool((meta["SubjectID"] == RECOVER_SUBJECT).any()), "expected": True})
    rows.append({"pool": "required_subject", "group": f"{EXCLUDED_SUBJECT}_vae_absent", "observed": bool((meta["SubjectID"] == EXCLUDED_SUBJECT).any()), "expected": False})
    rows.append({"pool": "required_subject", "group": f"{EXCLUDED_SUBJECT}_classifier_absent", "observed": bool((clf["SubjectID"] == EXCLUDED_SUBJECT).any()), "expected": False})
    out = pd.DataFrame(rows)
    failed = out[out["observed"].astype(str) != out["expected"].astype(str)]
    if not failed.empty:
        raise RuntimeError("Pool validation failed:\n" + failed.to_string(index=False))
    return out


def inspect_tensor(config: Dict[str, Any]) -> Dict[str, Any]:
    tensor_path = resolve(config["paths"]["global_tensor_path"])
    with np.load(tensor_path, allow_pickle=False) as npz:
        shape = tuple(int(v) for v in npz["global_tensor_data"].shape)
        channel_names = [str(x) for x in npz["channel_names"].astype(str)]
        subject_ids = [str(x) for x in npz["subject_ids"].astype(str)]
    selected = [channel_names[i] for i in config["parameters"]["channels_to_use"]]
    if RECOVER_SUBJECT not in subject_ids:
        raise RuntimeError(f"{RECOVER_SUBJECT} missing from tensor")
    if EXCLUDED_SUBJECT in subject_ids:
        # The source tensor may still contain historical rows in some branches. The decisive guard is metadata/pool exclusion.
        excluded_tensor_note = "present_in_tensor_but_excluded_from_metadata_pools"
    else:
        excluded_tensor_note = "absent_from_tensor"
    return {
        "tensor_path": str(tensor_path),
        "shape": shape,
        "selected_channel_names": selected,
        "recover_subject_tensor_index": subject_ids.index(RECOVER_SUBJECT),
        "excluded_subject_tensor_status": excluded_tensor_note,
    }


def stale_paths(output_dir: Path) -> List[Path]:
    if not output_dir.exists():
        return []
    stale: List[Path] = []
    for child in output_dir.iterdir():
        if child.name in STALE_NAMES or child.name.startswith(STALE_PREFIXES):
            stale.append(child)
    return sorted(stale, key=lambda p: str(p))


def quarantine_output(output_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quarantine = output_dir.parent / f"{output_dir.name}_quarantine_{timestamp}"
    suffix = 1
    while quarantine.exists():
        quarantine = output_dir.parent / f"{output_dir.name}_quarantine_{timestamp}_{suffix}"
        suffix += 1
    quarantine.mkdir(parents=True, exist_ok=False)
    for child in list(output_dir.iterdir()):
        shutil.move(str(child), str(quarantine / child.name))
    return quarantine


def build_stage_a_command(config: Dict[str, Any], python_exe: str) -> List[str]:
    command = [
        python_exe,
        str(resolve(config["paths"]["training_script"])),
        "--global_tensor_path",
        str(resolve(config["paths"]["global_tensor_path"])),
        "--metadata_path",
        str(resolve(config["paths"]["metadata_path"])),
        "--output_dir",
        str(resolve(config["paths"]["output_dir"])),
    ]
    for name, value in config["parameters"].items():
        append_arg(command, name, value)
    command.extend(["--vae_required_metadata_cols", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex"])
    command.append("--vae_abort_if_val_split_fails")
    return command


def build_stage_b_command(config: Dict[str, Any], python_exe: str) -> List[str]:
    out = resolve(config["paths"]["output_dir"])
    return [
        python_exe,
        str(STAGE_B_SCRIPT),
        "--run-dir",
        str(out),
        "--output-dir",
        str(out / "classifier_only_readout"),
        "--models",
        "logreg_l2",
        "--outer-folds",
        str(config["parameters"]["outer_folds"]),
        "--inner-folds",
        str(config["parameters"]["inner_folds"]),
        "--reuse-latent-cache",
    ]


def write_preflight(
    config: Dict[str, Any],
    diff: pd.DataFrame,
    pool: pd.DataFrame,
    tensor_info: Dict[str, Any],
    stale: List[Path],
    stage_a: List[str],
    stage_b: List[str],
    mode: str,
) -> None:
    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    diff.to_csv(PREFLIGHT_DIR / "config_diff.csv", index=False)
    pool.to_csv(PREFLIGHT_DIR / "pool_validation.csv", index=False)
    (PREFLIGHT_DIR / "tensor_validation.json").write_text(json.dumps(tensor_info, indent=2), encoding="utf-8")
    report = f"""# Dropout 0.10 Preflight

Mode: `{mode}`

## Controlled Diff

The only scientific parameter diff versus the current best model is:

`dropout_rate_vae: 0.15 -> 0.10`

Path/run-name changes are output bookkeeping only.

## Pool Validation

- VAE pool: CN=300, MCI=250, AD=97
- Classifier pool: CN=300, AD=97
- 035_S_6927 present.
- 128_S_2002 absent from VAE/classifier metadata pools.

## Tensor Validation

```json
{json.dumps(tensor_info, indent=2)}
```

## Stale Output Check

Stale markers found: `{len(stale)}`

{chr(10).join('- ' + str(p) for p in stale) if stale else 'No stale output markers found.'}

## Stage A Command

```bash
{' '.join(stage_a)}
```

## Stage B Command

```bash
{' '.join(stage_b)}
```

No training is launched unless `--confirm-training` is passed.
"""
    (PREFLIGHT_DIR / "dry_run_report.md").write_text(report, encoding="utf-8")
    command_log = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "config": rel(CANONICAL_TARGET_CONFIG),
        "source_config": rel(SOURCE_CONFIG),
        "output_dir": config["paths"]["output_dir"],
        "stage_a_command": stage_a,
        "stage_b_command": stage_b,
        "stale_output_markers": [str(p) for p in stale],
        "training_launched": False if mode != "real_training" else None,
    }
    (PREFLIGHT_DIR / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")


def run_command(command: Iterable[str]) -> None:
    subprocess.run(list(command), cwd=PROJECT_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", type=Path, default=CANONICAL_TARGET_CONFIG)
    p.add_argument("--source-config", type=Path, default=SOURCE_CONFIG)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--confirm-training", action="store_true")
    p.add_argument("--force-clean", action="store_true")
    p.add_argument("--python-executable", default=sys.executable)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source = load_json(resolve(args.source_config))
    target = load_json(resolve(args.config))
    validate_exact_scientific_diff(source, target)
    pool = validate_pool(target)
    tensor_info = inspect_tensor(target)
    diff = diff_configs(source, target)
    output_dir = resolve(target["paths"]["output_dir"])
    stale = stale_paths(output_dir)
    if stale and args.confirm_training and not args.force_clean:
        raise RuntimeError("Refusing real training with stale output markers. Use --force-clean to quarantine.")
    stage_a = build_stage_a_command(target, args.python_executable)
    stage_b = build_stage_b_command(target, args.python_executable)
    mode = "real_training" if args.confirm_training and not args.dry_run else "dry_run"
    write_preflight(target, diff, pool, tensor_info, stale, stage_a, stage_b, mode)

    print(f"Config diff rows: {len(diff)}")
    print(diff.to_string(index=False))
    print(pool.to_string(index=False))
    print(f"Stale output markers: {len(stale)}")
    if args.dry_run or not args.confirm_training:
        print(f"Dry-run OK. Preflight written to {PREFLIGHT_DIR}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    if stale and args.force_clean:
        quarantine = quarantine_output(output_dir)
        print(f"Moved existing output contents to quarantine: {quarantine}")
    run_command(stage_a)
    run_command(stage_b)


if __name__ == "__main__":
    main()
