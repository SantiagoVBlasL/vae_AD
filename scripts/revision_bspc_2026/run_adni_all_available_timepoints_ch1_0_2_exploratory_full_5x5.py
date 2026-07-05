#!/usr/bin/env python3
"""Launcher for exploratory ADNI all-timepoints [1,0,2] FULL 5x5.

Default mode is dry-run/preflight. Real training requires --confirm-training.
This branch changes only the input tensor/metadata branch relative to the
locked v5.1b horizon4480/cycles56 [1,0,2] model.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIG = PROJECT_ROOT / "configs/runs/adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5.json"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/runs/adni_all_available_timepoints_ch1_0_2_exploratory_full_5x5.json"
STAGE_B_SCRIPT = PROJECT_ROOT / "scripts/revision_bspc_2026/run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep.py"
COMPARISON_SCRIPT = PROJECT_ROOT / "scripts/revision_bspc_2026/compare_adni_all_available_timepoints_ch1_0_2_exploratory_full_5x5.py"
INTEGRITY_SCRIPT = PROJECT_ROOT / "scripts/revision_bspc_2026/audit_adni_all_available_timepoints_ch1_0_2_exploratory_integrity.py"

EXPECTED_CHANNELS = [1, 0, 2]
EXPECTED_SELECTED_NAMES = [
    "Pearson_Full_FisherZ_Signed",
    "Pearson_OMST_GCE_Signed_Weighted",
    "MI_KNN_Symmetric",
]
EXPECTED_PRIMARY_MODEL = "logreg_l2"
EXPECTED_PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-config", type=Path, default=SOURCE_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-training", action="store_true")
    parser.add_argument("--skip-classifier-readout", action="store_true")
    parser.add_argument("--skip-comparison", action="store_true")
    parser.add_argument("--skip-integrity-audit", action="store_true")
    parser.add_argument("--force-clean", action="store_true")
    parser.add_argument("--python-executable", default=None)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: expected {expected!r}, got {actual!r}")


def values_after_flag(tokens: Sequence[str], flag: str) -> list[str]:
    if flag not in tokens:
        return []
    out: list[str] = []
    for token in tokens[tokens.index(flag) + 1 :]:
        if token.startswith("--"):
            break
        out.append(token)
    return out


def normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    out = dict(params)
    out.setdefault("vae_dropout_scope", "legacy_all")
    out.setdefault("vae_block_order", "legacy_act_norm")
    out.setdefault("vae_train_sampler_strategy", "none")
    return out


def validate_config(config: dict[str, Any], source: dict[str, Any]) -> None:
    src_params = normalize_params(source["parameters"])
    cfg_params = normalize_params(config["parameters"])
    require_equal(cfg_params, src_params, "parameters")
    require_equal(config["selected_channel_names"], EXPECTED_SELECTED_NAMES, "selected_channel_names")
    require_equal(cfg_params["channels_to_use"], EXPECTED_CHANNELS, "channels_to_use")
    require_equal(cfg_params["outer_folds"], 5, "outer_folds")
    require_equal(cfg_params["inner_folds"], 5, "inner_folds")
    require_equal(cfg_params["epochs_vae"], 4480, "epochs_vae")
    require_equal(cfg_params["cyclical_beta_n_cycles"], 56, "cyclical_beta_n_cycles")
    require_equal(cfg_params["lr_scheduler_T0"], 80, "lr_scheduler_T0")
    require_equal(cfg_params["beta_vae"], 2.5, "beta_vae")
    require_equal(cfg_params["latent_dim"], 256, "latent_dim")
    require_equal(cfg_params["batch_size"], 64, "batch_size")
    require_equal(cfg_params["dropout_rate_vae"], 0.15, "dropout_rate_vae")
    require_equal(cfg_params["metadata_features"], ["Age", "Sex"], "metadata_features")
    require_equal(cfg_params["classifier_stratify_cols"], ["Manufacturer"], "classifier_stratify_cols")
    require_equal(cfg_params["vae_stratify_cols"], ["Manufacturer"], "vae_stratify_cols")
    if "Sex" in cfg_params["classifier_stratify_cols"] or "Sex" in cfg_params["vae_stratify_cols"]:
        raise RuntimeError("Sex must remain metadata/covariate only, not a stratifier.")
    allowed_path_changes = {
        "global_tensor_path",
        "metadata_path",
        "output_dir",
        "big_disk_output_dir",
        "split_preview_csv",
        "split_preview_summary_csv",
    }
    src_paths = source["paths"]
    cfg_paths = config["paths"]
    for key in src_paths:
        if key not in allowed_path_changes and cfg_paths.get(key) != src_paths.get(key):
            raise RuntimeError(f"Unexpected path change for {key}: {src_paths.get(key)} -> {cfg_paths.get(key)}")
    if "all_available_timepoints_ch1_0_2_exploratory" not in cfg_paths["global_tensor_path"]:
        raise RuntimeError("Config global_tensor_path does not point at all-timepoints exploratory branch.")
    if "all_available_timepoints_ch1_0_2_exploratory" not in cfg_paths["metadata_path"]:
        raise RuntimeError("Config metadata_path does not point at all-timepoints exploratory branch.")


def inspect_tensor(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as zf:
        tensor = zf["global_tensor_data"]
        channel_names = [str(x) for x in zf["channel_names"].astype(str)]
        python_bandpass = bool(zf["python_bandpass_applied"])
        original_n = zf["original_n_timepoints"] if "original_n_timepoints" in zf.files else None
    require_equal([channel_names[i] for i in EXPECTED_CHANNELS], EXPECTED_SELECTED_NAMES, "selected tensor channel names")
    if python_bandpass:
        raise RuntimeError("Python bandpass must be OFF.")
    if tensor.shape[1:] != (3, 131, 131):
        raise RuntimeError(f"Unexpected all-timepoints tensor shape: {tensor.shape}")
    return {
        "shape": tuple(int(x) for x in tensor.shape),
        "channel_names": channel_names,
        "python_bandpass_applied": python_bandpass,
        "has_original_n_timepoints": original_n is not None,
        "original_n_timepoints_min": float(np.nanmin(original_n)) if original_n is not None else np.nan,
        "original_n_timepoints_max": float(np.nanmax(original_n)) if original_n is not None else np.nan,
    }


def normalize_manufacturer(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    upper = text.upper()
    if "GE" in upper:
        return "GE"
    if "SIEMENS" in upper:
        return "SIEMENS"
    if "PHILIPS" in upper:
        return "Philips"
    return text or "UNKNOWN"


def load_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    meta = pd.read_csv(path)
    if "tensor_idx" not in meta.columns and "tensor_index" in meta.columns:
        meta = meta.rename(columns={"tensor_index": "tensor_idx"})
    required = ["SubjectID", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex", "tensor_idx"]
    missing = [col for col in required if col not in meta.columns]
    if missing:
        raise RuntimeError(f"Metadata missing required columns: {missing}")
    meta = meta.copy()
    meta["SubjectID"] = meta["SubjectID"].astype(str)
    meta["ResearchGroup_Mapped"] = meta["ResearchGroup_Mapped"].astype(str)
    meta["Manufacturer"] = meta["Manufacturer"].map(normalize_manufacturer)
    meta["Age"] = pd.to_numeric(meta["Age"], errors="coerce")
    meta["Sex"] = meta["Sex"].fillna("UNKNOWN").astype(str)
    meta["tensor_idx"] = meta["tensor_idx"].astype(int)
    if "original_n_TR" not in meta.columns and "original_n_timepoints" in meta.columns:
        meta["original_n_TR"] = meta["original_n_timepoints"]
    if "original_n_TR" not in meta.columns:
        raise RuntimeError("All-timepoints branch metadata must store original_n_TR or original_n_timepoints.")
    cn_ad = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])]
    if len(cn_ad) != 396:
        raise RuntimeError(f"Expected locked CN/AD classifier pool n=396, got {len(cn_ad)}.")
    require_equal(int(cn_ad["ResearchGroup_Mapped"].eq("CN").sum()), 300, "CN count")
    require_equal(int(cn_ad["ResearchGroup_Mapped"].eq("AD").sum()), 96, "AD count")
    return meta


def strat_key(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    tmp = df[list(cols)].copy()
    for col in cols:
        tmp[col] = tmp[col].fillna(f"{col}_UNKNOWN").astype(str)
    return tmp.apply(lambda row: "_".join(row.values.astype(str)), axis=1)


def count_fields(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"n": int(len(df))}
    for dx in ["AD", "CN", "MCI"]:
        out[dx] = int(df["ResearchGroup_Mapped"].eq(dx).sum())
    for mfr in ["GE", "SIEMENS", "Philips"]:
        label = "Siemens" if mfr == "SIEMENS" else mfr
        out[f"Manufacturer_{label}"] = int(df["Manufacturer"].eq(mfr).sum())
    for sex in ["F", "M"]:
        out[f"Sex_{sex}"] = int(df["Sex"].eq(sex).sum())
    out["mean_original_n_TR"] = float(pd.to_numeric(df.get("original_n_TR"), errors="coerce").mean())
    return out


def make_split_preview(meta: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = config["parameters"]
    seed = int(params["seed"])
    cn_ad = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy().reset_index(drop=True)
    strat_cols = ["ResearchGroup_Mapped", *params["classifier_stratify_cols"]]
    y_outer = strat_key(cn_ad, strat_cols)
    if (y_outer.value_counts() < int(params["outer_folds"])).any():
        raise RuntimeError("Manufacturer-aware outer 5-fold split is not feasible.")
    splitter = StratifiedKFold(n_splits=int(params["outer_folds"]), shuffle=True, random_state=seed)
    by_tensor = meta.set_index("tensor_idx", drop=False)
    all_tensor_idx = meta["tensor_idx"].to_numpy()
    rows: list[dict[str, Any]] = []
    subjects: list[dict[str, Any]] = []
    for fold, (train_dev_idx, test_idx) in enumerate(splitter.split(np.zeros(len(cn_ad)), y_outer), start=1):
        train_dev = cn_ad.iloc[train_dev_idx].copy()
        test = cn_ad.iloc[test_idx].copy()
        vae_pool_idx = np.setdiff1d(all_tensor_idx, test["tensor_idx"].to_numpy(), assume_unique=False)
        vae_pool = by_tensor.loc[vae_pool_idx].reset_index(drop=True)
        vae_key = strat_key(vae_pool, ["ResearchGroup_Mapped", *params["vae_stratify_cols"]])
        train_local, val_local = train_test_split(
            np.arange(len(vae_pool)),
            test_size=float(params["vae_val_split_ratio"]),
            stratify=vae_key,
            random_state=seed + fold + 9,
            shuffle=True,
        )
        for split_component, df in [
            ("classifier_train_dev", train_dev),
            ("classifier_test", test),
            ("vae_pool", vae_pool),
            ("vae_actual_train", vae_pool.iloc[train_local]),
            ("vae_internal_val", vae_pool.iloc[val_local]),
        ]:
            row = {"fold": fold, "split_component": split_component}
            row.update(count_fields(df))
            rows.append(row)
        for split_component, df in [("classifier_train_dev", train_dev), ("classifier_test", test)]:
            for _, r in df.iterrows():
                subjects.append(
                    {
                        "fold": fold,
                        "split_component": split_component,
                        "SubjectID": r["SubjectID"],
                        "tensor_idx": int(r["tensor_idx"]),
                        "ResearchGroup_Mapped": r["ResearchGroup_Mapped"],
                        "Manufacturer": r["Manufacturer"],
                        "Sex": r["Sex"],
                        "original_n_TR": r.get("original_n_TR", np.nan),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(subjects)


def build_stage_a_command(config: dict[str, Any], python_exe: str) -> list[str]:
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
    for name, value in params.items():
        append_arg(command, name, value)
    validate_stage_a_command(command)
    return command


def build_stage_b_command(config: dict[str, Any], python_exe: str) -> list[str]:
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


def build_comparison_command(config: dict[str, Any], python_exe: str) -> list[str]:
    outdir = resolve(config["paths"]["output_dir"])
    return [
        python_exe,
        str(COMPARISON_SCRIPT),
        "--candidate-run-dir",
        str(outdir),
        "--candidate-readout-dir",
        str(outdir / "classifier_only_readout"),
    ]


def build_integrity_command(config: dict[str, Any], python_exe: str) -> list[str]:
    outdir = resolve(config["paths"]["output_dir"])
    return [
        python_exe,
        str(INTEGRITY_SCRIPT),
        "--candidate-run-dir",
        str(outdir),
        "--candidate-readout-dir",
        str(outdir / "classifier_only_readout"),
    ]


def validate_stage_a_command(command: Sequence[str]) -> None:
    for idx, token in enumerate(command[:-1]):
        if token.startswith("--n_iter_") and str(command[idx + 1]) == "0":
            raise RuntimeError(f"Invalid Stage A command contains {token} 0")
    checks = {
        "--channels_to_use": ["1", "0", "2"],
        "--outer_folds": ["5"],
        "--inner_folds": ["5"],
        "--epochs_vae": ["4480"],
        "--cyclical_beta_n_cycles": ["56"],
        "--lr_scheduler_T0": ["80"],
        "--latent_dim": ["256"],
        "--batch_size": ["64"],
        "--beta_vae": ["2.5"],
        "--dropout_rate_vae": ["0.15"],
        "--metadata_features": ["Age", "Sex"],
        "--classifier_stratify_cols": ["Manufacturer"],
        "--vae_stratify_cols": ["Manufacturer"],
        "--classifier_types": ["logreg", "svm"],
        "--n_iter_logreg": ["300"],
        "--n_iter_svm": ["300"],
    }
    for flag, expected in checks.items():
        require_equal(values_after_flag(command, flag), expected, f"Stage A {flag}")


def validate_stage_b_command(command: Sequence[str]) -> None:
    require_equal(values_after_flag(command, "--outer-folds"), ["5"], "Stage B outer-folds")
    require_equal(values_after_flag(command, "--inner-folds"), ["5"], "Stage B inner-folds")
    require_equal(values_after_flag(command, "--models"), [EXPECTED_PRIMARY_MODEL], "Stage B models")


STALE_NAMES = {"classifier_only_readout", "latent_cache", "run_manifest.json"}
STALE_PREFIXES = ("fold_", "all_folds_metrics", "summary_metrics")


def stale_markers(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        return []
    markers: list[Path] = []
    for child in output_dir.iterdir():
        if child.name in STALE_NAMES or child.name.startswith(STALE_PREFIXES):
            markers.append(child)
    return sorted(markers, key=lambda p: str(p))


def ensure_output_symlink(config: dict[str, Any], force_clean: bool) -> Path | None:
    output_dir = resolve(config["paths"]["output_dir"])
    big_disk = Path(config["paths"]["big_disk_output_dir"])
    big_disk.mkdir(parents=True, exist_ok=True)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if not output_dir.exists():
        output_dir.symlink_to(big_disk, target_is_directory=True)
    if not output_dir.is_symlink():
        raise RuntimeError(f"Refusing real training: output_dir is not a symlink: {output_dir}")
    if output_dir.resolve() != big_disk.resolve():
        raise RuntimeError(f"Refusing real training: output_dir points to {output_dir.resolve()}, expected {big_disk.resolve()}")
    markers = stale_markers(output_dir)
    if markers and not force_clean:
        preview = "\n".join(f"  - {p}" for p in markers[:20])
        raise RuntimeError(f"Refusing real training: stale artifacts present. Use --force-clean to quarantine.\n{preview}")
    quarantine = None
    if markers and force_clean:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        quarantine = big_disk.parent / f"{big_disk.name}_quarantine_{timestamp}"
        quarantine.mkdir(parents=True, exist_ok=False)
        for child in list(output_dir.iterdir()):
            child.rename(quarantine / child.name)
    return quarantine


def verify_fresh_checkpoints(config: dict[str, Any], run_start_epoch: float) -> None:
    output_dir = resolve(config["paths"]["output_dir"])
    stale: list[str] = []
    for fold in range(1, int(config["parameters"]["outer_folds"]) + 1):
        ckpt = output_dir / f"fold_{fold}" / f"vae_model_fold_{fold}.pt"
        if not ckpt.exists() or ckpt.stat().st_mtime <= run_start_epoch:
            stale.append(str(ckpt))
    if stale:
        raise RuntimeError("Stage B blocked: missing/stale fold checkpoints:\n" + "\n".join(stale))


def write_manifest(config_path: Path, config: dict[str, Any], commands: dict[str, list[str]], quarantine: Path | None = None) -> None:
    output_dir = resolve(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "run_name": config["run_name"],
        "branch_label": "exploratory_upper_bound_confounding_stress_test",
        "controlled_change": "input tensor/metadata branch only: locked 140TR -> all-available-timepoints connectomes",
        "channels_to_use": EXPECTED_CHANNELS,
        "stage_a_command_shell": shlex.join(commands["stage_a"]),
        "stage_b_command_shell": shlex.join(commands["stage_b"]),
        "comparison_command_shell": shlex.join(commands["comparison"]),
        "integrity_audit_command_shell": shlex.join(commands["integrity"]),
        "quarantine_dir": str(quarantine) if quarantine else "",
        "promotion_guardrail": "Do not promote based on internal AUC alone; require no increased n_TR/site/manufacturer leakage and external OASIS robustness.",
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    source = load_json(args.source_config)
    validate_config(config, source)
    if not args.dry_run and not args.confirm_training:
        raise SystemExit("Refusing to launch training without --confirm-training. Use --dry-run for preflight.")

    python_exe = args.python_executable or config.get("python_executable") or sys.executable
    tensor_info = inspect_tensor(resolve(config["paths"]["global_tensor_path"]))
    meta = load_metadata(resolve(config["paths"]["metadata_path"]))
    preview_summary, preview_subjects = make_split_preview(meta, config)
    preview_summary_path = resolve(config["paths"]["split_preview_summary_csv"])
    preview_subjects_path = resolve(config["paths"]["split_preview_csv"])
    preview_summary_path.parent.mkdir(parents=True, exist_ok=True)
    preview_summary.to_csv(preview_summary_path, index=False)
    preview_subjects.to_csv(preview_subjects_path, index=False)

    commands = {
        "stage_a": build_stage_a_command(config, python_exe),
        "stage_b": build_stage_b_command(config, python_exe),
        "comparison": build_comparison_command(config, python_exe),
        "integrity": build_integrity_command(config, python_exe),
    }

    output_dir = resolve(config["paths"]["output_dir"])
    markers = stale_markers(output_dir)
    mode = "DRY-RUN" if args.dry_run else "REAL RUN (CONFIRMED)"
    print(f"Config           : {args.config}")
    print(f"Source config    : {args.source_config}")
    print(f"Run name         : {config['run_name']}")
    print(f"Mode             : {mode}")
    print("Controlled change: input tensor/metadata branch only (140TR -> all available timepoints)")
    print("Branch label     : exploratory_upper_bound_confounding_stress_test")
    print(f"Tensor shape     : {tensor_info['shape']}")
    print(f"Original n_TR    : min={tensor_info['original_n_timepoints_min']} max={tensor_info['original_n_timepoints_max']}")
    print(f"Channels         : {EXPECTED_CHANNELS} ({', '.join(EXPECTED_SELECTED_NAMES)})")
    print("Python bandpass  : OFF")
    print("Split strategy   : ResearchGroup_Mapped + Manufacturer")
    print("Sex role         : metadata/covariate only")
    print("No OASIS data    : confirmed; ADNI branch only")
    print(f"VAE schedule     : outer=5 inner=5 latent_dim=256 epochs=4480 cycles=56 T0=80 beta=2.5 batch=64 dropout=0.15")
    print(f"Stage B readout  : {EXPECTED_PRIMARY_MODEL}; threshold={EXPECTED_PRIMARY_THRESHOLD}; true inner-CV OOF")
    print("Promotion guard  : exploratory only; no promotion based on internal AUC alone")
    print("\nSplit preview:")
    cols = ["fold", "split_component", "n", "AD", "CN", "MCI", "Manufacturer_GE", "Manufacturer_Siemens", "Manufacturer_Philips", "mean_original_n_TR"]
    print(preview_summary[[c for c in cols if c in preview_summary.columns]].to_string(index=False))
    print("\nStage A command:")
    print(shlex.join(commands["stage_a"]))
    print("\nStage B command:")
    print(shlex.join(commands["stage_b"]))
    print("\nComparison command:")
    print(shlex.join(commands["comparison"]))
    print("\nIntegrity/confounding audit command:")
    print(shlex.join(commands["integrity"]))
    if markers:
        print(f"\nDetected stale markers: {len(markers)}")
        for marker in markers[:20]:
            print(f"  - {marker}")
    else:
        print("\nNo stale output markers detected.")
    if args.dry_run:
        print("\nDry-run complete. Training was NOT launched.")
        return 0

    quarantine = ensure_output_symlink(config, args.force_clean)
    write_manifest(args.config, config, commands, quarantine)
    run_start = time.time()
    stage_a_rc = subprocess.run(commands["stage_a"], cwd=PROJECT_ROOT, check=False).returncode
    if stage_a_rc != 0:
        return int(stage_a_rc)
    verify_fresh_checkpoints(config, run_start)
    if args.skip_classifier_readout:
        return 0
    stage_b_rc = subprocess.run(commands["stage_b"], cwd=PROJECT_ROOT, check=False).returncode
    if stage_b_rc != 0:
        return int(stage_b_rc)
    if not args.skip_comparison:
        subprocess.run(commands["comparison"], cwd=PROJECT_ROOT, check=False)
    if not args.skip_integrity_audit:
        subprocess.run(commands["integrity"], cwd=PROJECT_ROOT, check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
