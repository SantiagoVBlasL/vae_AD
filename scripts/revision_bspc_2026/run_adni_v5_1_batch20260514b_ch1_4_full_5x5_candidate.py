#!/usr/bin/env python3
"""Dry-run / run wrapper for ADNI v5.1 batch20260514b channel-[1,4] full 5x5 experiment.

This wrapper prepares the full 5x5 confirmation run for the best FAST pair
[1,4]. It refuses to train unless --confirm-training is explicitly
passed. In dry-run mode it validates the config, tensor/metadata, scheduler
arithmetic, Stage A/Stage B commands, and writes a lightweight split preview.
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
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "runs"
    / "adni_v5_1_batch20260514b_ch1_4_full_5x5_candidate.json"
)

EXPECTED_CHANNELS = [1, 4]
EXPECTED_CLASSIFIERS = ["logreg"]
EXPECTED_METADATA_FEATURES = ["Age", "Sex"]
EXPECTED_CLASSIFIER_STRATIFY_COLS = ["Manufacturer"]
EXPECTED_VAE_STRATIFY_COLS = ["Manufacturer"]
EXPECTED_CHANNEL_NAMES = [
    "Pearson_Full_FisherZ_Signed",
    "dFC_StdDev",
]
EXPECTED_MANUFACTURERS = ["GE", "Philips", "SIEMENS"]
EXPECTED_PRIMARY_READOUT_MODEL = "logreg_l2"
EXPECTED_PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and optionally execute the ADNI v5.1 batch20260514b channel-[1,4] "
            "full 5x5 confirmation command."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="Preflight only; do not launch training.")
    parser.add_argument("--confirm-training", action="store_true", help="Required for real training launch.")
    parser.add_argument(
        "--skip-classifier-readout",
        action="store_true",
        help="After a confirmed real Stage A run, do not launch the classifier-only Stage B readout.",
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="After a confirmed real Stage B readout, do not launch the read-only comparison script.",
    )
    parser.add_argument("--python-executable", default=None)
    parser.add_argument(
        "--skip-preview-write",
        action="store_true",
        help="Validate splits but do not write split preview CSV files.",
    )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"Unexpected {label}: {actual!r} (expected {expected!r})")


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
    require_equal(params.get("channels_to_use"), EXPECTED_CHANNELS, "channels_to_use")
    require_equal(params.get("classifier_types"), EXPECTED_CLASSIFIERS, "classifier_types")
    require_equal(params.get("metadata_features"), EXPECTED_METADATA_FEATURES, "metadata_features")
    require_equal(params.get("classifier_stratify_cols"), EXPECTED_CLASSIFIER_STRATIFY_COLS, "classifier_stratify_cols")
    require_equal(params.get("vae_stratify_cols"), EXPECTED_VAE_STRATIFY_COLS, "vae_stratify_cols")
    require_equal(config.get("stage_a_classifier_outputs"), "dummy_logreg_ignored_for_ranking", "stage_a_classifier_outputs")
    require_equal(params.get("classifier_calibrate"), False, "classifier_calibrate")
    require_equal(params.get("classifier_use_class_weight"), True, "classifier_use_class_weight")
    require_equal(params.get("beta_vae"), 2.5, "beta_vae")
    require_equal(params.get("epochs_vae"), 3840, "epochs_vae")
    require_equal(params.get("cyclical_beta_n_cycles"), 48, "cyclical_beta_n_cycles")
    require_equal(params.get("early_stopping_patience_vae"), 320, "early_stopping_patience_vae")
    require_equal(params.get("lr_scheduler_type"), "cosine_warm", "lr_scheduler_type")
    require_equal(params.get("lr_scheduler_T0"), 80, "lr_scheduler_T0")
    require_equal(params.get("latent_dim"), 256, "latent_dim")
    require_equal(params.get("dropout_rate_vae"), 0.15, "dropout_rate_vae")
    require_equal(params.get("vae_final_activation"), "tanh", "vae_final_activation")
    require_equal(params.get("intermediate_fc_dim_vae"), "quarter", "intermediate_fc_dim_vae")
    require_equal(params.get("outer_folds"), 5, "outer_folds")
    require_equal(params.get("inner_folds"), 5, "inner_folds")
    require_equal(params.get("n_iter_logreg"), 1, "n_iter_logreg")
    require_equal(params.get("n_iter_svm"), None, "n_iter_svm")
    require_equal(params.get("use_layernorm_vae_fc"), False, "use_layernorm_vae_fc")
    require_equal(config.get("selected_channel_names"), EXPECTED_CHANNEL_NAMES, "selected_channel_names")
    readout = config.get("primary_readout", {})
    require_equal(readout.get("model"), EXPECTED_PRIMARY_READOUT_MODEL, "primary_readout.model")
    require_equal(readout.get("threshold_strategy"), EXPECTED_PRIMARY_THRESHOLD, "primary_readout.threshold_strategy")
    require_equal(readout.get("threshold_selection"), "true_inner_cv_oof", "primary_readout.threshold_selection")
    require_equal(readout.get("metadata_features"), EXPECTED_METADATA_FEATURES, "primary_readout.metadata_features")

    for qc_flag in [
        "qc_analyze_distributions",
        "qc_check_scanner_leakage",
        "qc_rate_distortion",
        "qc_latent_information",
    ]:
        require_equal(params.get(qc_flag), True, qc_flag)

    cycle_len = params["epochs_vae"] / params["cyclical_beta_n_cycles"]
    if cycle_len != 80:
        raise RuntimeError(f"Expected epochs_vae/cyclical_beta_n_cycles = 80, got {cycle_len}")
    if params["lr_scheduler_T0"] != cycle_len:
        raise RuntimeError(f"Expected lr_scheduler_T0={cycle_len}, got {params['lr_scheduler_T0']}")


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


def validate_training_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Training-ready metadata not found: {path}")
    meta = pd.read_csv(path)
    if "tensor_idx" not in meta.columns and "tensor_index" in meta.columns:
        meta = meta.rename(columns={"tensor_index": "tensor_idx"})
    required = ["SubjectID", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex", "tensor_idx"]
    missing = [col for col in required if col not in meta.columns]
    if missing:
        raise RuntimeError(f"Metadata is missing required columns {missing}: {path}")
    duplicated = meta["SubjectID"][meta["SubjectID"].duplicated()].astype(str).tolist()
    if duplicated:
        raise RuntimeError(f"Duplicate SubjectID values in metadata: {duplicated[:10]}")
    meta = meta.copy()
    meta["SubjectID"] = meta["SubjectID"].astype(str)
    meta["ResearchGroup_Mapped"] = meta["ResearchGroup_Mapped"].astype(str)
    meta["Manufacturer"] = meta["Manufacturer"].map(normalize_manufacturer)
    meta["Sex"] = meta["Sex"].fillna("UNKNOWN").astype(str)
    meta["Age"] = pd.to_numeric(meta["Age"], errors="coerce")
    meta["tensor_idx"] = meta["tensor_idx"].astype(int)
    cn_ad = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])]
    missing_demo = cn_ad[cn_ad["Age"].isna() | cn_ad["Sex"].isna()]
    if not missing_demo.empty:
        cols = ["SubjectID", "ResearchGroup_Mapped", "Age", "Sex"]
        raise RuntimeError("CN/AD pool has missing Age or Sex:\n" + missing_demo[cols].to_string(index=False))
    print(f"[metadata] Path: {path}")
    print(f"[metadata] Rows={len(meta)}, CN={(meta.ResearchGroup_Mapped == 'CN').sum()}, AD={(meta.ResearchGroup_Mapped == 'AD').sum()}, MCI={(meta.ResearchGroup_Mapped == 'MCI').sum()}")
    return meta


def inspect_tensor_npz(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Global tensor not found: {path}")
    with np.load(path, allow_pickle=False) as zf:
        files = list(zf.files)
        if "python_bandpass_applied" not in files:
            raise RuntimeError(f"Tensor missing python_bandpass_applied flag: {path}")
        python_bandpass_applied = bool(zf["python_bandpass_applied"])
        if python_bandpass_applied:
            raise RuntimeError(f"Expected no Python bandpass, got python_bandpass_applied=True: {path}")
        channel_names = [str(x) for x in zf["channel_names"].astype(str)]
        subject_count = int(len(zf["subject_ids"]))
    selected = [channel_names[i] for i in EXPECTED_CHANNELS]
    require_equal(selected, EXPECTED_CHANNEL_NAMES, "tensor selected channel names for [1,4]")
    print(f"[tensor] Path: {path}")
    print(f"[tensor] Subjects={subject_count}, python_bandpass_applied=False")
    print(f"[tensor] Selected [1,4]: {selected}")
    return {"subject_count": subject_count, "selected_channel_names": selected, "python_bandpass_applied": False}


def strat_key(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    temp = df[list(cols)].copy()
    for col in cols:
        temp[col] = temp[col].fillna(f"{col}_Unknown").astype(str)
    return temp.apply(lambda row: "_".join(row.values.astype(str)), axis=1)


def count_fields(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n": int(len(df))}
    for dx in ["AD", "CN", "MCI"]:
        out[dx] = int(df["ResearchGroup_Mapped"].eq(dx).sum())
    for manufacturer in EXPECTED_MANUFACTURERS:
        key = "Siemens" if manufacturer == "SIEMENS" else manufacturer
        out[f"Manufacturer_{key}"] = int(df["Manufacturer"].eq(manufacturer).sum())
        for dx in ["AD", "CN", "MCI"]:
            out[f"{dx}_{key}"] = int((df["ResearchGroup_Mapped"].eq(dx) & df["Manufacturer"].eq(manufacturer)).sum())
    for sex in ["F", "M"]:
        out[f"Sex_{sex}"] = int(df["Sex"].eq(sex).sum())
    return out


def validate_component_counts(row: Dict[str, Any], split_component: str) -> List[str]:
    errors: List[str] = []
    if split_component in {"classifier_train_dev", "classifier_test"}:
        for dx in ["AD", "CN"]:
            if row[dx] <= 0:
                errors.append(f"{split_component} missing {dx}")
    else:
        for dx in ["AD", "CN", "MCI"]:
            if row[dx] <= 0:
                errors.append(f"{split_component} missing {dx}")
    for manufacturer in ["GE", "Siemens", "Philips"]:
        if row[f"Manufacturer_{manufacturer}"] <= 0:
            errors.append(f"{split_component} missing Manufacturer {manufacturer}")
    return errors


def make_split_preview(meta: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    params = config["parameters"]
    seed = int(params["seed"])
    n_splits = int(params["outer_folds"])
    cn_ad = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy().reset_index(drop=True)
    strat_cols = ["ResearchGroup_Mapped", *params["classifier_stratify_cols"]]
    y_outer = strat_key(cn_ad, strat_cols)
    outer_counts = y_outer.value_counts()
    outer_fallback = bool((outer_counts < n_splits).any())
    if outer_fallback:
        y_outer = cn_ad["ResearchGroup_Mapped"]
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_valid_tensor_idx = meta["tensor_idx"].astype(int).to_numpy()
    rows: List[Dict[str, Any]] = []
    subject_rows: List[Dict[str, Any]] = []

    for fold_idx, (train_dev_idx, test_idx) in enumerate(splitter.split(np.zeros(len(cn_ad)), y_outer), start=1):
        train_dev = cn_ad.iloc[train_dev_idx].copy()
        test = cn_ad.iloc[test_idx].copy()
        test_tensor_idx = test["tensor_idx"].astype(int).to_numpy()
        vae_pool_idx = np.setdiff1d(np.unique(all_valid_tensor_idx), np.unique(test_tensor_idx), assume_unique=False)
        vae_pool = meta.set_index("tensor_idx").loc[vae_pool_idx].reset_index()
        vae_strat_cols = ["ResearchGroup_Mapped", *params["vae_stratify_cols"]]
        vae_key = strat_key(vae_pool, vae_strat_cols)
        vae_fallback = bool((vae_key.value_counts() < 2).any())
        if vae_fallback:
            vae_key = vae_pool["ResearchGroup_Mapped"].fillna("RG_Unknown").astype(str)
        train_local, val_local = train_test_split(
            np.arange(len(vae_pool)),
            test_size=float(params["vae_val_split_ratio"]),
            stratify=vae_key,
            random_state=seed + (fold_idx - 1) + 10,
            shuffle=True,
        )
        components = [
            ("classifier_train_dev", train_dev, outer_fallback, int(outer_counts.min())),
            ("classifier_test", test, outer_fallback, int(outer_counts.min())),
            ("vae_pool", vae_pool, vae_fallback, int(vae_key.value_counts().min())),
            ("vae_actual_train", vae_pool.iloc[train_local], vae_fallback, int(vae_key.value_counts().min())),
            ("vae_internal_val", vae_pool.iloc[val_local], vae_fallback, int(vae_key.value_counts().min())),
        ]
        for component, df, fallback, min_cell in components:
            row: Dict[str, Any] = {
                "fold": fold_idx,
                "split_component": component,
                "classifier_stratification_cols": "+".join(strat_cols),
                "vae_internal_val_stratification_cols": "+".join(vae_strat_cols),
                "fallback_to_label_only": bool(fallback),
                "minimum_source_stratum_count": int(min_cell),
            }
            row.update(count_fields(df))
            errors = validate_component_counts(row, component)
            row["passes_required_representation_check"] = not errors
            row["representation_check_errors"] = " | ".join(errors)
            rows.append(row)
        for split_name, df in [("classifier_train_dev", train_dev), ("classifier_test", test)]:
            for _, r in df.iterrows():
                subject_rows.append(
                    {
                        "fold": fold_idx,
                        "split_component": split_name,
                        "SubjectID": r["SubjectID"],
                        "tensor_idx": int(r["tensor_idx"]),
                        "ResearchGroup_Mapped": r["ResearchGroup_Mapped"],
                        "Manufacturer": r["Manufacturer"],
                        "Sex": r["Sex"],
                    }
                )
    summary = pd.DataFrame(rows).sort_values(["fold", "split_component"]).reset_index(drop=True)
    subjects = pd.DataFrame(subject_rows).sort_values(["fold", "split_component", "SubjectID"]).reset_index(drop=True)
    return summary, subjects


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


def build_command(config: Dict[str, Any], python_executable: str) -> List[str]:
    paths = config["paths"]
    params = config["parameters"]
    unknown = sorted(set(params) - set(PARAM_ORDER))
    if unknown:
        raise RuntimeError(f"Unknown config parameters: {unknown}")
    command = [
        python_executable,
        str(resolve_path(paths["training_script"])),
        "--global_tensor_path",
        str(resolve_path(paths["global_tensor_path"])),
        "--metadata_path",
        str(resolve_path(paths["metadata_path"])),
        "--output_dir",
        str(resolve_path(paths["output_dir"])),
    ]
    for name in PARAM_ORDER:
        append_arg(command, name, params.get(name))
    return command


def build_readout_command(config: Dict[str, Any], python_executable: str) -> List[str]:
    paths = config["paths"]
    params = config["parameters"]
    return [
        python_executable,
        str(resolve_path(paths["classifier_only_script"])),
        "--run-dir",
        str(resolve_path(paths["output_dir"])),
        "--output-dir",
        str(resolve_path(paths["classifier_only_output_dir"])),
        "--outer-folds",
        str(params["outer_folds"]),
        "--inner-folds",
        str(params["inner_folds"]),
        "--models",
        EXPECTED_PRIMARY_READOUT_MODEL,
        "--overwrite",
    ]


def build_comparison_command(config: Dict[str, Any], python_executable: str) -> List[str]:
    paths = config["paths"]
    return [
        python_executable,
        str(resolve_path(paths["comparison_script"])),
        "--candidate-run-dir",
        str(resolve_path(paths["output_dir"])),
        "--candidate-readout-dir",
        str(resolve_path(paths["classifier_only_output_dir"])),
        "--output-dir",
        str(resolve_path(paths["comparison_output_dir"])),
        "--overwrite",
    ]


def validate_stage_commands(stage_a_command: Sequence[str], stage_b_command: Sequence[str]) -> None:
    for idx, token in enumerate(stage_a_command[:-1]):
        if token.startswith("--n_iter_") and str(stage_a_command[idx + 1]) == "0":
            raise RuntimeError(f"Invalid Stage A command: {token} 0")
    if "--classifier_types" not in stage_a_command:
        raise RuntimeError("Stage A command missing --classifier_types")
    clf_idx = stage_a_command.index("--classifier_types") + 1
    classifiers: List[str] = []
    for token in stage_a_command[clf_idx:]:
        if token.startswith("--"):
            break
        classifiers.append(token)
    require_equal(classifiers, EXPECTED_CLASSIFIERS, "Stage A classifier_types")
    if "svm" in classifiers:
        raise RuntimeError("Stage A must not include svm; canonical outputs are dummy/ignored")
    for required in ["--outer_folds", "5", "--inner_folds", "5", "--channels_to_use", "1", "4", "--n_iter_logreg", "1"]:
        if required not in stage_a_command:
            raise RuntimeError(f"Stage A command missing required token: {required}")
    for required in ["--outer-folds", "5", "--inner-folds", "5"]:
        if required not in stage_b_command:
            raise RuntimeError(f"Stage B command missing required token: {required}")
    if "--models" not in stage_b_command or EXPECTED_PRIMARY_READOUT_MODEL not in stage_b_command:
        raise RuntimeError(f"Stage B command must restrict readout to {EXPECTED_PRIMARY_READOUT_MODEL}")


def describe_output_path(config: Dict[str, Any]) -> None:
    output_dir = resolve_path(config["paths"]["output_dir"])
    big_disk = Path(config["paths"]["big_disk_output_dir"])
    print(f"\nOutput dir     : {output_dir}")
    if output_dir.is_symlink():
        print(f"Symlink target : {output_dir.resolve()}")
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
            f"Prepare target/symlink first:\n  mkdir -p {shlex.quote(str(big_disk))}\n"
            f"  ln -s {shlex.quote(str(big_disk))} {shlex.quote(str(output_dir))}"
        )
    if not output_dir.is_symlink():
        raise RuntimeError(f"Refusing to start training: output_dir is not a symlink: {output_dir}")
    if output_dir.resolve() != big_disk.resolve():
        raise RuntimeError(f"Refusing to start training: symlink target is {output_dir.resolve()}, expected {big_disk.resolve()}")
    unexpected = [p for p in output_dir.iterdir() if p.name != "run_manifest.json"]
    if unexpected:
        raise RuntimeError("Refusing to start training: output_dir is not empty: " + ", ".join(p.name for p in unexpected[:8]))
    return output_dir


def write_manifest(
    config_path: Path,
    config: Dict[str, Any],
    command: List[str],
    readout_command: List[str],
    comparison_command: List[str],
    preview_summary: pd.DataFrame,
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
        "python_bandpass_applied": False,
        "split_strategy": config.get("split_strategy"),
        "epochs_vae": params.get("epochs_vae"),
        "cyclical_beta_n_cycles": params.get("cyclical_beta_n_cycles"),
        "cycle_length_epochs": params.get("epochs_vae") / params.get("cyclical_beta_n_cycles"),
        "lr_scheduler_T0": params.get("lr_scheduler_T0"),
        "classifiers": params.get("classifier_types"),
        "stage_a_classifier_outputs": config.get("stage_a_classifier_outputs"),
        "primary_readout": config.get("primary_readout"),
        "metadata_features": params.get("metadata_features"),
        "preview_all_required_representation_checks_passed": bool(preview_summary["passes_required_representation_check"].all()),
        "command": command,
        "command_shell": shlex.join(command),
        "classifier_only_readout_command": readout_command,
        "classifier_only_readout_command_shell": shlex.join(readout_command),
        "comparison_command": comparison_command,
        "comparison_command_shell": shlex.join(comparison_command),
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    validate_config(config)
    python_executable = args.python_executable or config.get("python_executable") or sys.executable
    if not args.dry_run and not args.confirm_training:
        raise SystemExit("Refusing to launch training without --confirm-training. Use --dry-run for preflight only.")

    print(f"Config         : {args.config}")
    print(f"Run name       : {config.get('run_name')}")
    print(f"Mode           : {'DRY-RUN' if args.dry_run else 'REAL RUN (CONFIRMED)'}")
    print(f"Python         : {python_executable}")
    print(f"Channels       : {EXPECTED_CHANNELS} ({', '.join(EXPECTED_CHANNEL_NAMES)})")
    print("Python bandpass: OFF")
    print("QC             : ON")
    print("Stage A clf    : dummy canonical logreg, n_iter_logreg=1, ignored for ranking")
    print(f"Stage B readout: {EXPECTED_PRIMARY_READOUT_MODEL} + {EXPECTED_PRIMARY_THRESHOLD}")
    params = config["parameters"]
    print(f"Split strategy : ResearchGroup_Mapped + {params['classifier_stratify_cols']} (classifier outer/inner)")
    print(f"VAE val split  : ResearchGroup_Mapped + {params['vae_stratify_cols']}")
    print(f"Sex role       : metadata/covariate only")
    print(f"VAE schedule   : epochs={params['epochs_vae']}, cycles={params['cyclical_beta_n_cycles']}, cycle_len={params['epochs_vae'] / params['cyclical_beta_n_cycles']}, T0={params['lr_scheduler_T0']}")

    print()
    training_script = resolve_path(config["paths"]["training_script"])
    if not training_script.exists():
        raise FileNotFoundError(f"Training script not found: {training_script}")
    inspect_tensor_npz(resolve_path(config["paths"]["global_tensor_path"]))
    metadata = validate_training_metadata(resolve_path(config["paths"]["metadata_path"]))
    preview_summary, preview_subjects = make_split_preview(metadata, config)
    if not preview_summary["passes_required_representation_check"].all():
        bad = preview_summary[~preview_summary["passes_required_representation_check"]]
        raise RuntimeError("Split preview failed representation checks:\n" + bad.to_string(index=False))
    print("[split] All folds/components have required diagnosis/manufacturer representation.")
    print(preview_summary[["fold", "split_component", "n", "AD", "CN", "MCI", "Manufacturer_GE", "Manufacturer_Siemens", "Manufacturer_Philips", "Sex_F", "Sex_M"]].to_string(index=False))

    if not args.skip_preview_write:
        preview_csv = resolve_path(config["paths"]["split_preview_csv"])
        preview_summary_csv = resolve_path(config["paths"]["split_preview_summary_csv"])
        preview_csv.parent.mkdir(parents=True, exist_ok=True)
        preview_subjects.to_csv(preview_csv, index=False)
        preview_summary.to_csv(preview_summary_csv, index=False)
        print(f"[split] Subject assignment preview written: {preview_csv}")
        print(f"[split] Fold summary preview written     : {preview_summary_csv}")

    command = build_command(config, python_executable)
    readout_command = build_readout_command(config, python_executable)
    comparison_command = build_comparison_command(config, python_executable)
    validate_stage_commands(command, readout_command)
    describe_output_path(config)
    print(f"\nTrial budgets  : canonical dummy logreg={params.get('n_iter_logreg')}; svm={params.get('n_iter_svm')}")
    print("Stage A command:")
    print(shlex.join(command))
    print("\nStage B classifier-only readout command:")
    print(shlex.join(readout_command))
    print("\nComparison command:")
    print(shlex.join(comparison_command))

    if args.dry_run:
        print("\nDry-run complete. Training was NOT launched.")
        return 0

    ensure_output_prepared_for_real_run(config)
    manifest_path = write_manifest(args.config, config, command, readout_command, comparison_command, preview_summary)
    print(f"\nRun manifest written: {manifest_path}")
    print("Launching Stage A full VAE run...")
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
    if completed.returncode != 0:
        return int(completed.returncode)
    if args.skip_classifier_readout:
        print("Stage A completed. Stage B readout skipped by request.")
        return 0
    print("Launching Stage B classifier-only readout...")
    readout_completed = subprocess.run(readout_command, cwd=str(PROJECT_ROOT), check=False)
    if readout_completed.returncode != 0:
        return int(readout_completed.returncode)
    if args.skip_comparison:
        print("Stage B completed. Comparison skipped by request.")
        return 0
    print("Launching read-only comparison...")
    comparison_completed = subprocess.run(comparison_command, cwd=str(PROJECT_ROOT), check=False)
    return int(comparison_completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
