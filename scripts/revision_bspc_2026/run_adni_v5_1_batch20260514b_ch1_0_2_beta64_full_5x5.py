#!/usr/bin/env python3
"""Prepare/run the controlled beta=6.4 FULL 5x5 confirmation.

Default behavior is preflight only. Real training requires
--confirm-training. The config is compared against the locked source config and
the only scientific parameter difference allowed is
beta_vae: 2.5 -> 6.4.

Tests stronger KL compression as a final single controlled FULL experiment.
Promotion rule: candidate must beat BOTH pooled OOF AUC > 0.778785 AND PR-AUC >= 0.551832
simultaneously. FAST result is NOT sufficient for promotion (FAST->FULL reversal precedent).
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIG = PROJECT_ROOT / "configs/runs/adni_v5_1_batch20260514b_ch1_0_2_mfrsplit_3840_final_candidate.json"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/runs/adni_v5_1_batch20260514b_ch1_0_2_beta64_full_5x5.json"

EXPECTED_CHANNELS = [1, 0, 2]
EXPECTED_SELECTED_NAMES = [
    "Pearson_Full_FisherZ_Signed",
    "Pearson_OMST_GCE_Signed_Weighted",
    "MI_KNN_Symmetric",
]
EXPECTED_METADATA_FEATURES = ["Age", "Sex"]
EXPECTED_STRATIFY_COLS = ["Manufacturer"]
EXPECTED_PRIMARY_MODEL = "logreg_l2"
EXPECTED_PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"

STAGE_B_SCRIPT = PROJECT_ROOT / "scripts/revision_bspc_2026/run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep.py"
COMPARISON_SCRIPT = PROJECT_ROOT / "scripts/revision_bspc_2026/compare_v5_1_batch20260514b_beta64_full_5x5.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-config", type=Path, default=SOURCE_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="Preflight only; do not launch training.")
    parser.add_argument("--confirm-training", action="store_true", help="Required for real Stage A/Stage B launch.")
    parser.add_argument("--skip-classifier-readout", action="store_true")
    parser.add_argument("--skip-comparison", action="store_true")
    parser.add_argument("--python-executable", default=None)
    parser.add_argument("--skip-preview-write", action="store_true")
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


def assert_only_beta64_scientific_diff(source: Dict[str, Any], target: Dict[str, Any]) -> None:
    src_params = dict(source["parameters"])
    tgt_params = dict(target["parameters"])
    if set(src_params) != set(tgt_params):
        raise RuntimeError(
            "Target config parameter keys differ from locked source config: "
            f"missing={sorted(set(src_params) - set(tgt_params))}, extra={sorted(set(tgt_params) - set(src_params))}"
        )
    diffs = {k: (src_params[k], tgt_params[k]) for k in src_params if src_params[k] != tgt_params[k]}
    require_equal(diffs, {"beta_vae": (2.5, 6.4)}, "scientific parameter diffs")

    if source["paths"]["global_tensor_path"] != target["paths"]["global_tensor_path"]:
        raise RuntimeError("global_tensor_path changed; this is not allowed.")
    if source["paths"]["metadata_path"] != target["paths"]["metadata_path"]:
        raise RuntimeError("metadata_path changed; this is not allowed.")
    if source["parameters"]["channels_to_use"] != target["parameters"]["channels_to_use"]:
        raise RuntimeError("channels_to_use changed; this is not allowed.")


def validate_config(config: Dict[str, Any], source_config: Dict[str, Any]) -> None:
    assert_only_beta64_scientific_diff(source_config, config)
    params = config["parameters"]
    require_equal(params["channels_to_use"], EXPECTED_CHANNELS, "channels_to_use")
    require_equal(config["selected_channel_names"], EXPECTED_SELECTED_NAMES, "selected_channel_names")
    require_equal(params["classifier_stratify_cols"], EXPECTED_STRATIFY_COLS, "classifier_stratify_cols")
    require_equal(params["vae_stratify_cols"], EXPECTED_STRATIFY_COLS, "vae_stratify_cols")
    require_equal(params["metadata_features"], EXPECTED_METADATA_FEATURES, "metadata_features")
    if "Sex" in params["classifier_stratify_cols"] or "Sex" in params["vae_stratify_cols"]:
        raise RuntimeError("Sex must remain metadata/covariate only.")
    for key, expected in [
        ("outer_folds", 5),
        ("inner_folds", 5),
        ("repeated_outer_folds_n_repeats", 1),
        ("latent_dim", 256),
        ("epochs_vae", 3840),
        ("cyclical_beta_n_cycles", 48),
        ("early_stopping_patience_vae", 320),
        ("lr_scheduler_type", "cosine_warm"),
        ("lr_scheduler_T0", 80),
        ("beta_vae", 6.4),
        ("dropout_rate_vae", 0.15),
        ("batch_size", 64),
        ("decoder_type", "convtranspose"),
        ("num_conv_layers_encoder", 4),
        ("norm_mode", "zscore_offdiag"),
        ("vae_final_activation", "tanh"),
        ("intermediate_fc_dim_vae", "quarter"),
        ("use_layernorm_vae_fc", False),
        ("classifier_calibrate", True),
        ("classifier_use_class_weight", True),
        ("n_iter_logreg", 300),
        ("n_iter_svm", 300),
    ]:
        require_equal(params[key], expected, key)
    if params["epochs_vae"] / params["cyclical_beta_n_cycles"] != 80:
        raise RuntimeError("Expected 3840/48 = 80 cycle length.")
    require_equal(params["lr_scheduler_T0"], 80, "lr_scheduler_T0")
    for flag in ["qc_analyze_distributions", "qc_check_scanner_leakage", "qc_rate_distortion", "qc_latent_information"]:
        require_equal(params[flag], True, flag)


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
    meta = pd.read_csv(path)
    if "tensor_idx" not in meta.columns and "tensor_index" in meta.columns:
        meta = meta.rename(columns={"tensor_index": "tensor_idx"})
    required = ["SubjectID", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex", "tensor_idx"]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise RuntimeError(f"Metadata missing required columns: {missing}")
    meta = meta.copy()
    meta["SubjectID"] = meta["SubjectID"].astype(str)
    meta["ResearchGroup_Mapped"] = meta["ResearchGroup_Mapped"].astype(str)
    meta["Manufacturer"] = meta["Manufacturer"].map(normalize_manufacturer)
    meta["Sex"] = meta["Sex"].fillna("UNKNOWN").astype(str)
    meta["Age"] = pd.to_numeric(meta["Age"], errors="coerce")
    meta["tensor_idx"] = meta["tensor_idx"].astype(int)
    if meta["SubjectID"].duplicated().any():
        raise RuntimeError("Metadata has duplicate SubjectID rows.")
    cn_ad = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])]
    if cn_ad["Age"].isna().any() or cn_ad["Sex"].isna().any():
        raise RuntimeError("CN/AD rows contain missing Age/Sex.")
    return meta


def inspect_tensor(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as zf:
        if bool(zf["python_bandpass_applied"]):
            raise RuntimeError("Refusing tensor with python_bandpass_applied=True")
        channel_names = [str(x) for x in zf["channel_names"].astype(str)]
        shape = tuple(int(x) for x in zf["global_tensor_data"].shape)
    selected = [channel_names[i] for i in EXPECTED_CHANNELS]
    require_equal(selected, EXPECTED_SELECTED_NAMES, "tensor selected channel names")
    return {"shape": shape, "selected_channel_names": selected, "python_bandpass_applied": False}


def strat_key(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    tmp = df[list(cols)].copy()
    for col in cols:
        tmp[col] = tmp[col].fillna(f"{col}_UNKNOWN").astype(str)
    return tmp.apply(lambda row: "_".join(row.values.astype(str)), axis=1)


def count_fields(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n": int(len(df))}
    for dx in ["AD", "CN", "MCI"]:
        out[dx] = int(df["ResearchGroup_Mapped"].eq(dx).sum())
    for mfr in ["GE", "SIEMENS", "Philips"]:
        label = "Siemens" if mfr == "SIEMENS" else mfr
        out[f"Manufacturer_{label}"] = int(df["Manufacturer"].eq(mfr).sum())
        for dx in ["AD", "CN", "MCI"]:
            out[f"{dx}_{label}"] = int((df["ResearchGroup_Mapped"].eq(dx) & df["Manufacturer"].eq(mfr)).sum())
    for sex in ["F", "M"]:
        out[f"Sex_{sex}"] = int(df["Sex"].eq(sex).sum())
    return out


def validate_component_counts(row: Dict[str, Any], component: str) -> List[str]:
    errors: List[str] = []
    diagnoses = ["AD", "CN"] if component in {"classifier_train_dev", "classifier_test"} else ["AD", "CN", "MCI"]
    for dx in diagnoses:
        if row[dx] <= 0:
            errors.append(f"{component} missing {dx}")
    for mfr in ["GE", "Siemens", "Philips"]:
        if row[f"Manufacturer_{mfr}"] <= 0:
            errors.append(f"{component} missing Manufacturer {mfr}")
    return errors


def make_split_preview(meta: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    params = config["parameters"]
    seed = int(params["seed"])
    n_splits = int(params["outer_folds"])
    cn_ad = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy().reset_index(drop=True)
    strat_cols = ["ResearchGroup_Mapped", *params["classifier_stratify_cols"]]
    y_outer = strat_key(cn_ad, strat_cols)
    if (y_outer.value_counts() < n_splits).any():
        raise RuntimeError("Manufacturer-aware outer 5-fold split is not feasible.")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    by_tensor = meta.set_index("tensor_idx", drop=False)
    all_tensor_idx = meta["tensor_idx"].to_numpy()
    rows: List[Dict[str, Any]] = []
    subjects: List[Dict[str, Any]] = []
    for fold, (train_dev_idx, test_idx) in enumerate(splitter.split(np.zeros(len(cn_ad)), y_outer), start=1):
        train_dev = cn_ad.iloc[train_dev_idx].copy()
        test = cn_ad.iloc[test_idx].copy()
        vae_pool_idx = np.setdiff1d(all_tensor_idx, test["tensor_idx"].to_numpy(), assume_unique=False)
        vae_pool = by_tensor.loc[vae_pool_idx].reset_index(drop=True)
        vae_cols = ["ResearchGroup_Mapped", *params["vae_stratify_cols"]]
        vae_key = strat_key(vae_pool, vae_cols)
        if (vae_key.value_counts() < 2).any():
            raise RuntimeError("Manufacturer-aware VAE internal validation split is not feasible.")
        train_local, val_local = train_test_split(
            np.arange(len(vae_pool)),
            test_size=float(params["vae_val_split_ratio"]),
            stratify=vae_key,
            random_state=seed + fold + 9,
            shuffle=True,
        )
        components = [
            ("classifier_train_dev", train_dev),
            ("classifier_test", test),
            ("vae_pool", vae_pool),
            ("vae_actual_train", vae_pool.iloc[train_local]),
            ("vae_internal_val", vae_pool.iloc[val_local]),
        ]
        for component, df in components:
            row: Dict[str, Any] = {
                "fold": fold,
                "split_component": component,
                "classifier_stratification_cols": "+".join(strat_cols),
                "vae_internal_val_stratification_cols": "+".join(vae_cols),
            }
            row.update(count_fields(df))
            errors = validate_component_counts(row, component)
            row["passes_required_representation_check"] = not errors
            row["representation_check_errors"] = " | ".join(errors)
            rows.append(row)
        for split_name, df in [("classifier_train_dev", train_dev), ("classifier_test", test)]:
            for _, r in df.iterrows():
                subjects.append(
                    {
                        "fold": fold,
                        "split_component": split_name,
                        "SubjectID": r["SubjectID"],
                        "tensor_idx": int(r["tensor_idx"]),
                        "ResearchGroup_Mapped": r["ResearchGroup_Mapped"],
                        "Manufacturer": r["Manufacturer"],
                        "Sex": r["Sex"],
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(subjects)


def build_stage_a_command(config: Dict[str, Any], python_exe: str) -> List[str]:
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


def build_stage_b_command(config: Dict[str, Any], python_exe: str) -> List[str]:
    paths = config["paths"]
    params = config["parameters"]
    outdir = resolve(paths["output_dir"])
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
        "--output-dir",
        str(OUT_COMPARISON_DIR),
        "--overwrite",
    ]


OUT_COMPARISON_DIR = PROJECT_ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_ch1_0_2_beta64_full_5x5_comparison"


def validate_stage_a_command(command: Sequence[str]) -> None:
    for idx, token in enumerate(command[:-1]):
        if token.startswith("--n_iter_") and str(command[idx + 1]) == "0":
            raise RuntimeError(f"Invalid Stage A command contains {token} 0")
    require_equal(values_after_flag(command, "--channels_to_use"), ["1", "0", "2"], "Stage A channels")
    require_equal(values_after_flag(command, "--classifier_types"), ["logreg", "svm"], "Stage A classifier_types")
    require_equal(values_after_flag(command, "--n_iter_logreg"), ["300"], "Stage A n_iter_logreg")
    require_equal(values_after_flag(command, "--n_iter_svm"), ["300"], "Stage A n_iter_svm")
    for flag, expected in [
        ("--outer_folds", ["5"]),
        ("--inner_folds", ["5"]),
        ("--latent_dim", ["256"]),
        ("--epochs_vae", ["3840"]),
        ("--cyclical_beta_n_cycles", ["48"]),
        ("--lr_scheduler_T0", ["80"]),
        ("--beta_vae", ["6.4"]),
        ("--dropout_rate_vae", ["0.15"]),
        ("--vae_final_activation", ["tanh"]),
        ("--intermediate_fc_dim_vae", ["quarter"]),
        ("--metadata_features", ["Age", "Sex"]),
        ("--classifier_stratify_cols", ["Manufacturer"]),
        ("--vae_stratify_cols", ["Manufacturer"]),
    ]:
        require_equal(values_after_flag(command, flag), expected, f"Stage A {flag}")
    # Explicitly confirm manufacturer-balanced sampler is NOT used
    if "--vae_train_sampler_strategy" in command:
        idx = command.index("--vae_train_sampler_strategy")
        sampler_val = command[idx + 1] if idx + 1 < len(command) else ""
        if sampler_val != "none":
            raise RuntimeError(f"vae_train_sampler_strategy must be 'none', got {sampler_val!r}")


def validate_stage_b_command(command: Sequence[str]) -> None:
    require_equal(values_after_flag(command, "--outer-folds"), ["5"], "Stage B outer-folds")
    require_equal(values_after_flag(command, "--inner-folds"), ["5"], "Stage B inner-folds")
    require_equal(values_after_flag(command, "--models"), [EXPECTED_PRIMARY_MODEL], "Stage B models")


def ensure_output_prepared(config: Dict[str, Any]) -> None:
    output_dir = resolve(config["paths"]["output_dir"])
    big_disk = Path(config["paths"]["big_disk_output_dir"])
    if not output_dir.exists():
        raise RuntimeError(
            "Refusing to start training: output_dir is missing. Create the external target/symlink first:\n"
            f"  mkdir -p {shlex.quote(str(big_disk))}\n"
            f"  ln -s {shlex.quote(str(big_disk))} {shlex.quote(str(output_dir))}"
        )
    if not output_dir.is_symlink():
        raise RuntimeError(f"Refusing to start training: output_dir is not a symlink: {output_dir}")
    if output_dir.resolve() != big_disk.resolve():
        raise RuntimeError(f"Refusing to start training: symlink target is {output_dir.resolve()}, expected {big_disk.resolve()}")
    unexpected = [p.name for p in output_dir.iterdir() if p.name not in {"run_manifest.json", "command_log.json"}]
    if unexpected:
        raise RuntimeError(f"Refusing to start training: output_dir is not empty: {unexpected[:8]}")


def write_manifest(config_path: Path, config: Dict[str, Any], stage_a: Sequence[str], stage_b: Sequence[str], comparison: Sequence[str]) -> Path:
    output_dir = resolve(config["paths"]["output_dir"])
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "run_name": config["run_name"],
        "controlled_change": "beta_vae 2.5 -> 6.4",
        "only_scientific_parameter_diff_verified": True,
        "channels_to_use": EXPECTED_CHANNELS,
        "selected_channel_names": EXPECTED_SELECTED_NAMES,
        "python_bandpass_applied": False,
        "promotion_rule": "Must beat BOTH AUC > 0.778785 AND PR-AUC >= 0.551832 simultaneously. FAST result is NOT sufficient.",
        "stage_a_command": list(stage_a),
        "stage_a_command_shell": shlex.join(stage_a),
        "stage_b_command": list(stage_b),
        "stage_b_command_shell": shlex.join(stage_b),
        "comparison_command": list(comparison),
        "comparison_command_shell": shlex.join(comparison),
        "ranking_source": "Stage B classifier-only logreg_l2",
        "primary_threshold_strategy": EXPECTED_PRIMARY_THRESHOLD,
        "threshold_selection": "true_inner_cv_oof",
    }
    path = output_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_command_log(config: Dict[str, Any], stage_a_rc: int | None, stage_b_rc: int | None, comparison_rc: int | None) -> None:
    output_dir = resolve(config["paths"]["output_dir"])
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": config["run_name"],
        "training_launched": stage_a_rc is not None,
        "stage_a_returncode": stage_a_rc,
        "stage_b_returncode": stage_b_rc,
        "comparison_returncode": comparison_rc,
        "controlled_change": "beta_vae 2.5 -> 6.4",
        "promotion_rule": "Must beat BOTH AUC > 0.778785 AND PR-AUC >= 0.551832 simultaneously.",
        "ranking_source": "Stage B classifier-only logreg_l2",
        "primary_threshold_strategy": EXPECTED_PRIMARY_THRESHOLD,
        "threshold_selection": "true_inner_cv_oof",
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
    }
    (output_dir / "command_log.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    source = load_json(args.source_config)
    validate_config(config, source)
    python_exe = args.python_executable or config.get("python_executable") or sys.executable
    if not args.dry_run and not args.confirm_training:
        raise SystemExit("Refusing to launch training without --confirm-training. Use --dry-run for preflight only.")

    tensor_info = inspect_tensor(resolve(config["paths"]["global_tensor_path"]))
    meta = load_metadata(resolve(config["paths"]["metadata_path"]))
    preview_summary, preview_subjects = make_split_preview(meta, config)
    if not preview_summary["passes_required_representation_check"].all():
        bad = preview_summary[~preview_summary["passes_required_representation_check"]]
        raise RuntimeError("Split preview failed representation checks:\n" + bad.to_string(index=False))
    if not args.skip_preview_write:
        preview_csv = resolve(config["paths"]["split_preview_csv"])
        preview_summary_csv = resolve(config["paths"]["split_preview_summary_csv"])
        preview_csv.parent.mkdir(parents=True, exist_ok=True)
        preview_subjects.to_csv(preview_csv, index=False)
        preview_summary.to_csv(preview_summary_csv, index=False)

    stage_a = build_stage_a_command(config, python_exe)
    stage_b = build_stage_b_command(config, python_exe)
    comparison = build_comparison_command(config, python_exe)
    params = config["parameters"]
    mode = "DRY-RUN" if args.dry_run else "REAL RUN (CONFIRMED)"
    print(f"Config         : {args.config}")
    print(f"Source config  : {args.source_config}")
    print(f"Run name       : {config['run_name']}")
    print(f"Mode           : {mode}")
    print("Controlled diff: beta_vae 2.5 -> 6.4")
    print(f"Channels       : {EXPECTED_CHANNELS} ({', '.join(EXPECTED_SELECTED_NAMES)})")
    print("Recon loss     : mse_sum_batchmean_current (locked default; not changed)")
    print(f"Final activation: {params['vae_final_activation']}")
    print(f"Intermediate FC: {params['intermediate_fc_dim_vae']}")
    print("Python bandpass: OFF")
    print("Split strategy : ResearchGroup_Mapped + Manufacturer")
    print("Sex role       : metadata/covariate only")
    print(f"VAE schedule   : outer={params['outer_folds']}, inner={params['inner_folds']}, latent_dim={params['latent_dim']}, epochs={params['epochs_vae']}, cycles={params['cyclical_beta_n_cycles']}, cycle_len={params['epochs_vae'] / params['cyclical_beta_n_cycles']}, T0={params['lr_scheduler_T0']}, batch_size={params['batch_size']}, dropout={params['dropout_rate_vae']}, beta_max={params['beta_vae']}")
    print(f"Stage A clf    : canonical {params['classifier_types']}, n_iter_logreg={params['n_iter_logreg']}, n_iter_svm={params['n_iter_svm']}")
    print(f"Stage B readout: {EXPECTED_PRIMARY_MODEL} + true inner-CV OOF {EXPECTED_PRIMARY_THRESHOLD}")
    print(f"Tensor shape   : {tensor_info['shape']}, python_bandpass_applied={tensor_info['python_bandpass_applied']}")
    print("Promotion rule : AUC > 0.778785 AND PR-AUC >= 0.551832 simultaneously (FULL only; FAST not sufficient)")
    print("[split] All folds/components have required diagnosis/manufacturer representation.")
    print(preview_summary[["fold", "split_component", "n", "AD", "CN", "MCI", "Manufacturer_GE", "Manufacturer_Siemens", "Manufacturer_Philips", "Sex_F", "Sex_M"]].to_string(index=False))
    print("\nStage A command:")
    print(shlex.join(stage_a))
    print("\nStage B classifier-only readout command:")
    print(shlex.join(stage_b))
    print("\nComparison command:")
    print(shlex.join(comparison))
    if args.dry_run:
        print("\nDry-run complete. Training was NOT launched.")
        return 0

    ensure_output_prepared(config)
    manifest = write_manifest(args.config, config, stage_a, stage_b, comparison)
    print(f"\nRun manifest written: {manifest}")
    completed = subprocess.run(stage_a, cwd=PROJECT_ROOT, check=False)
    write_command_log(config, completed.returncode, None, None)
    if completed.returncode != 0:
        return int(completed.returncode)
    if args.skip_classifier_readout:
        print("Stage A completed. Stage B skipped.")
        return 0
    readout_completed = subprocess.run(stage_b, cwd=PROJECT_ROOT, check=False)
    write_command_log(config, completed.returncode, readout_completed.returncode, None)
    if readout_completed.returncode != 0:
        return int(readout_completed.returncode)
    if args.skip_comparison:
        print("Stage B completed. Comparison skipped.")
        return 0
    comparison_completed = subprocess.run(comparison, cwd=PROJECT_ROOT, check=False)
    write_command_log(config, completed.returncode, readout_completed.returncode, comparison_completed.returncode)
    return int(comparison_completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
