#!/usr/bin/env python3
"""Prepare/run ADNI v5.1 batch20260514b [1,0,2] objective-v2 FULL 5x5.

Default mode is dry-run/no training. Real training requires
--confirm-training. Stage A trains the VAE with a dummy canonical logreg
readout only; Stage B is the classifier-only logreg_l2 readout used for
ranking.
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
DEFAULT_CONFIG = PROJECT_ROOT / "configs/runs/adni_v5_1_batch20260514b_ch1_0_2_objective_v2_full_5x5.json"

EXPECTED_CHANNELS = [1, 0, 2]
EXPECTED_SELECTED_NAMES = [
    "Pearson_Full_FisherZ_Signed",
    "Pearson_OMST_GCE_Signed_Weighted",
    "MI_KNN_Symmetric",
]
EXPECTED_CLASSIFIERS = ["logreg"]
EXPECTED_METADATA_FEATURES = ["Age", "Sex"]
EXPECTED_STRATIFY_COLS = ["Manufacturer"]
EXPECTED_PRIMARY_MODEL = "logreg_l2"
EXPECTED_PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"

PARAM_ORDER = [
    "channels_to_use",
    "classifier_types",
    "classifier_stratify_cols",
    "vae_stratify_cols",
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
    "recon_loss_mode",
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
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


def load_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: expected {expected!r}, got {actual!r}")


def values_after_flag(tokens: Sequence[str], flag: str) -> List[str]:
    if flag not in tokens:
        return []
    out: List[str] = []
    for token in tokens[tokens.index(flag) + 1 :]:
        if token.startswith("--"):
            break
        out.append(token)
    return out


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


def validate_config(config: Dict[str, Any]) -> None:
    params = config["parameters"]
    unknown = sorted(set(params) - set(PARAM_ORDER))
    if unknown:
        raise RuntimeError(f"Parameters not wired into Stage A command: {unknown}")

    require_equal(params["channels_to_use"], EXPECTED_CHANNELS, "channels_to_use")
    require_equal(config["selected_channel_names"], EXPECTED_SELECTED_NAMES, "selected_channel_names")
    require_equal(params["recon_loss_mode"], "offdiag_channelmean_sum", "recon_loss_mode")
    require_equal(params["vae_final_activation"], "tanh", "vae_final_activation")
    require_equal(params["classifier_types"], EXPECTED_CLASSIFIERS, "classifier_types")
    require_equal(params["n_iter_logreg"], 1, "n_iter_logreg")
    require_equal(params["n_iter_svm"], None, "n_iter_svm")
    require_equal(params["classifier_stratify_cols"], EXPECTED_STRATIFY_COLS, "classifier_stratify_cols")
    require_equal(params["vae_stratify_cols"], EXPECTED_STRATIFY_COLS, "vae_stratify_cols")
    require_equal(params["metadata_features"], EXPECTED_METADATA_FEATURES, "metadata_features")
    if "Sex" in params["classifier_stratify_cols"] or "Sex" in params["vae_stratify_cols"]:
        raise RuntimeError("Sex must remain metadata/covariate only, not a stratifier.")

    for key, expected in [
        ("outer_folds", 5),
        ("inner_folds", 5),
        ("repeated_outer_folds_n_repeats", 1),
        ("latent_dim", 256),
        ("epochs_vae", 3840),
        ("cyclical_beta_n_cycles", 48),
        ("early_stopping_patience_vae", 320),
        ("lr_scheduler_T0", 80),
        ("beta_vae", 2.5),
        ("dropout_rate_vae", 0.15),
        ("decoder_type", "convtranspose"),
        ("num_conv_layers_encoder", 4),
        ("norm_mode", "zscore_offdiag"),
        ("use_layernorm_vae_fc", False),
    ]:
        require_equal(params[key], expected, key)
    if params["epochs_vae"] / params["cyclical_beta_n_cycles"] != 80:
        raise RuntimeError("Expected 3840/48 = 80 cycle length.")
    require_equal(params["lr_scheduler_T0"], 80, "lr_scheduler_T0")
    for flag in ["qc_analyze_distributions", "qc_check_scanner_leakage", "qc_rate_distortion", "qc_latent_information"]:
        require_equal(params[flag], True, flag)

    require_equal(config.get("stage_a_classifier_outputs"), "dummy_logreg_ignored_for_ranking", "stage_a_classifier_outputs")
    readout = config["primary_readout"]
    require_equal(readout["model"], EXPECTED_PRIMARY_MODEL, "primary_readout.model")
    require_equal(readout["threshold_strategy"], EXPECTED_PRIMARY_THRESHOLD, "primary_readout.threshold_strategy")
    require_equal(readout["threshold_selection"], "true_inner_cv_oof", "primary_readout.threshold_selection")
    require_equal(readout["metadata_features"], EXPECTED_METADATA_FEATURES, "primary_readout.metadata_features")


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
    return {"shape": shape, "channel_names": channel_names, "selected_channel_names": selected, "python_bandpass_applied": False}


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
    return (
        pd.DataFrame(rows).sort_values(["fold", "split_component"]).reset_index(drop=True),
        pd.DataFrame(subjects).sort_values(["fold", "split_component", "SubjectID"]).reset_index(drop=True),
    )


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
    for name in PARAM_ORDER:
        append_arg(command, name, params.get(name))
    validate_stage_a_command(command)
    return command


def build_stage_b_command(config: Dict[str, Any], python_exe: str) -> List[str]:
    paths = config["paths"]
    params = config["parameters"]
    command = [
        python_exe,
        str(resolve(paths["classifier_only_script"])),
        "--run-dir",
        str(resolve(paths["output_dir"])),
        "--output-dir",
        str(resolve(paths["classifier_only_output_dir"])),
        "--outer-folds",
        str(params["outer_folds"]),
        "--inner-folds",
        str(params["inner_folds"]),
        "--models",
        EXPECTED_PRIMARY_MODEL,
        "--overwrite",
    ]
    validate_stage_b_command(command)
    return command


def build_comparison_command(config: Dict[str, Any], python_exe: str) -> List[str]:
    paths = config["paths"]
    return [
        python_exe,
        str(resolve(paths["comparison_script"])),
        "--objective-run-dir",
        str(resolve(paths["output_dir"])),
        "--objective-readout-dir",
        str(resolve(paths["classifier_only_output_dir"])),
        "--output-dir",
        str(resolve(paths["comparison_output_dir"])),
        "--overwrite",
    ]


def validate_stage_a_command(command: Sequence[str]) -> None:
    for idx, token in enumerate(command[:-1]):
        if token.startswith("--n_iter_") and str(command[idx + 1]) == "0":
            raise RuntimeError(f"Invalid Stage A command contains {token} 0")
    require_equal(values_after_flag(command, "--channels_to_use"), ["1", "0", "2"], "Stage A channels")
    require_equal(values_after_flag(command, "--classifier_types"), ["logreg"], "Stage A classifier_types")
    require_equal(values_after_flag(command, "--n_iter_logreg"), ["1"], "Stage A n_iter_logreg")
    if "--n_iter_svm" in command or "svm" in values_after_flag(command, "--classifier_types"):
        raise RuntimeError("Stage A must not include svm.")
    for flag, expected in [
        ("--outer_folds", ["5"]),
        ("--inner_folds", ["5"]),
        ("--latent_dim", ["256"]),
        ("--epochs_vae", ["3840"]),
        ("--cyclical_beta_n_cycles", ["48"]),
        ("--lr_scheduler_T0", ["80"]),
        ("--recon_loss_mode", ["offdiag_channelmean_sum"]),
        ("--vae_final_activation", ["tanh"]),
        ("--metadata_features", ["Age", "Sex"]),
        ("--classifier_stratify_cols", ["Manufacturer"]),
        ("--vae_stratify_cols", ["Manufacturer"]),
    ]:
        require_equal(values_after_flag(command, flag), expected, f"Stage A {flag}")


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
    unexpected = [p.name for p in output_dir.iterdir() if p.name != "run_manifest.json"]
    if unexpected:
        raise RuntimeError(f"Refusing to start training: output_dir is not empty: {unexpected[:8]}")


def write_manifest(
    config_path: Path,
    config: Dict[str, Any],
    stage_a: Sequence[str],
    stage_b: Sequence[str],
    comparison: Sequence[str],
    preview_summary: pd.DataFrame,
) -> Path:
    output_dir = resolve(config["paths"]["output_dir"])
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": False,
        "config_path": str(config_path),
        "run_name": config["run_name"],
        "selected_channels": EXPECTED_CHANNELS,
        "selected_channel_names": EXPECTED_SELECTED_NAMES,
        "python_bandpass_applied": False,
        "recon_loss_mode": config["parameters"]["recon_loss_mode"],
        "vae_final_activation": config["parameters"]["vae_final_activation"],
        "stage_a_classifier_outputs": config["stage_a_classifier_outputs"],
        "primary_readout": config["primary_readout"],
        "preview_all_required_representation_checks_passed": bool(preview_summary["passes_required_representation_check"].all()),
        "stage_a_command": list(stage_a),
        "stage_a_command_shell": shlex.join(stage_a),
        "stage_b_command": list(stage_b),
        "stage_b_command_shell": shlex.join(stage_b),
        "comparison_command": list(comparison),
        "comparison_command_shell": shlex.join(comparison),
    }
    path = output_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    validate_config(config)
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
    print(f"Run name       : {config['run_name']}")
    print(f"Mode           : {mode}")
    print(f"Channels       : {EXPECTED_CHANNELS} ({', '.join(EXPECTED_SELECTED_NAMES)})")
    print(f"Recon loss     : {params['recon_loss_mode']}")
    print(f"Final activation: {params['vae_final_activation']}")
    print("Python bandpass: OFF")
    print("Split strategy : ResearchGroup_Mapped + Manufacturer")
    print("Sex role       : metadata/covariate only")
    print(f"VAE schedule   : outer={params['outer_folds']}, inner={params['inner_folds']}, latent_dim={params['latent_dim']}, epochs={params['epochs_vae']}, cycles={params['cyclical_beta_n_cycles']}, cycle_len={params['epochs_vae'] / params['cyclical_beta_n_cycles']}, T0={params['lr_scheduler_T0']}")
    print("Stage A clf    : dummy canonical logreg, n_iter_logreg=1, ignored for ranking")
    print(f"Stage B readout: {EXPECTED_PRIMARY_MODEL} + true inner-CV OOF {EXPECTED_PRIMARY_THRESHOLD}")
    print(f"Tensor shape   : {tensor_info['shape']}, python_bandpass_applied={tensor_info['python_bandpass_applied']}")
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
    manifest = write_manifest(args.config, config, stage_a, stage_b, comparison, preview_summary)
    print(f"\nRun manifest written: {manifest}")
    completed = subprocess.run(stage_a, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        return int(completed.returncode)
    if args.skip_classifier_readout:
        print("Stage A completed. Stage B skipped.")
        return 0
    readout_completed = subprocess.run(stage_b, cwd=PROJECT_ROOT, check=False)
    if readout_completed.returncode != 0:
        return int(readout_completed.returncode)
    if args.skip_comparison:
        print("Stage B completed. Comparison skipped.")
        return 0
    comparison_completed = subprocess.run(comparison, cwd=PROJECT_ROOT, check=False)
    return int(comparison_completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
