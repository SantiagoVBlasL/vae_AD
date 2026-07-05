#!/usr/bin/env python3
"""Greedy FAST forward channel selection for ADNI v5.1 batch20260514b.

The workflow is deliberately staged:

Stage A trains a fast VAE for each candidate channel set and saves fold
artefacts. The canonical logreg invoked by the VAE wrapper is not used for final
ranking.

Stage B runs the classifier-only readout on the saved fold-specific latent mu
features. The greedy ranking consumes only logreg_l2 with true inner-CV OOF
threshold selection.

Default mode is dry-run. Real training requires --confirm-training.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_EXE = "/home/diego/anaconda3/envs/vae_ad/bin/python"

TENSOR_PATH = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors/"
    "GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
)
METADATA_PATH = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass/"
    "training_ready_metadata_v5_1_batch20260514b_no_pybandpass.csv"
)

OUTPUT_BASENAME = "adni_v5_1_batch20260514b_greedy_fast_channel_selection_3x3"
LEGACY_OUTPUT_BASENAME = "adni_v5_1_batch20260514b_greedy_fast_channel_selection"

OUTPUT_ROOT = PROJECT_ROOT / f"results/revision_bspc_2026/{OUTPUT_BASENAME}"
BIG_DISK_ROOT = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026/"
    f"{OUTPUT_BASENAME}"
)
LEGACY_OUTPUT_ROOT = PROJECT_ROOT / f"results/revision_bspc_2026/{LEGACY_OUTPUT_BASENAME}"
LEGACY_BIG_DISK_ROOT = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026/"
    f"{LEGACY_OUTPUT_BASENAME}"
)

TRAINING_SCRIPT = PROJECT_ROOT / "scripts/run_vae_clf_ad_inference.py"
READOUT_SCRIPT = PROJECT_ROOT / "scripts/revision_bspc_2026/run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep.py"

ALL_CHANNELS = [0, 1, 2, 3, 4, 5, 6]
PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
SECONDARY_THRESHOLDS = ["fixed_0p5", "inner_oof_youden_j"]
MIN_MEAN_AUC_IMPROVEMENT = 0.01
CLEAR_SECONDARY_IMPROVEMENT = 0.01
EXPECTED_MANUFACTURERS = ["GE", "Philips", "SIEMENS"]
DIAGNOSES = ["AD", "CN", "MCI"]

FAST_PARAMS: Dict[str, Any] = {
    "classifier_types": ["logreg"],
    "classifier_stratify_cols": ["Manufacturer"],
    "vae_stratify_cols": ["Manufacturer"],
    "classifier_calibrate": False,
    "classifier_use_class_weight": True,
    "latent_features_type": "mu",
    "gridsearch_scoring": "roc_auc",
    "outer_folds": 3,
    "inner_folds": 3,
    "repeated_outer_folds_n_repeats": 1,
    "num_conv_layers_encoder": 4,
    "decoder_type": "convtranspose",
    "epochs_vae": 960,
    "vae_val_split_ratio": 0.2,
    "early_stopping_patience_vae": 240,
    "cyclical_beta_n_cycles": 12,
    "cyclical_beta_ratio_increase": 0.4,
    "beta_vae": 2.5,
    "dropout_rate_vae": 0.15,
    "latent_dim": 128,
    "batch_size": 64,
    "lr_vae": 0.0001,
    "lr_scheduler_type": "cosine_warm",
    "lr_scheduler_T0": 80,
    "lr_scheduler_eta_min": 5e-7,
    "lr_scheduler_patience_vae": 15,
    "weight_decay_vae": 5e-7,
    "vae_final_activation": "tanh",
    "intermediate_fc_dim_vae": "quarter",
    "use_layernorm_vae_fc": False,
    "n_jobs_gridsearch": 8,
    "metadata_features": ["Age", "Sex"],
    "norm_mode": "zscore_offdiag",
    "seed": 42,
    "num_workers": 4,
    "log_interval_epochs_vae": 10,
    "save_fold_artefacts": True,
    "save_vae_training_history": True,
    "qc_analyze_distributions": True,
    "qc_check_scanner_leakage": True,
    "qc_rate_distortion": True,
    "qc_latent_information": True,
    "qc_mi_n_neighbors": 3,
    "qc_mi_top_k": 10,
    "qc_rd_log_base": 2.0,
    "qc_tc_ridge": 1e-6,
    "qc_var_eps_active": 1e-4,
    "use_optuna_pruner": False,
    "use_smote": False,
    "tune_sampler_params": False,
    "mlp_classifier_hidden_layers": "64,16",
    # Stage A classifier is only a dummy required by the canonical VAE wrapper.
    # Final ranking must use Stage B classifier-only logreg_l2 readout.
    "n_iter_logreg": 1,
}

FAST_PREFLIGHT_EXPECTED: Dict[str, Any] = {
    "outer_folds": 3,
    "inner_folds": 3,
    "latent_dim": 128,
    "epochs_vae": 960,
    "cyclical_beta_n_cycles": 12,
    "lr_scheduler_T0": 80,
}
STALE_LEGACY_RUN_KEYS = ["ch0", "ch1"]

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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Greedy FAST channel selection for ADNI v5.1 batch20260514b.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true", help="Write state/plan only; do not train.")
    parser.add_argument("--run-step", choices=["singles"], default=None, help="Run a named greedy step.")
    parser.add_argument("--run-next-step", action="store_true", help="Run the next step inferred from completed metrics.")
    parser.add_argument("--resume", action="store_true", help="Skip candidates whose readout is already complete.")
    parser.add_argument("--max-channels", type=int, default=4)
    parser.add_argument("--confirm-training", action="store_true")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--python-executable", default=PYTHON_EXE)
    return parser.parse_args()


def channel_key(channels: Sequence[int]) -> str:
    return "ch" + "_".join(str(ch) for ch in channels)


def channels_json(channels: Sequence[int]) -> str:
    return json.dumps([int(ch) for ch in channels], separators=(",", ":"))


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def inspect_tensor(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as zf:
        files = set(zf.files)
        if "python_bandpass_applied" not in files:
            raise RuntimeError("Tensor missing python_bandpass_applied flag.")
        if bool(zf["python_bandpass_applied"]):
            raise RuntimeError("Refusing tensor with python_bandpass_applied=True.")
        if "channel_names" not in files:
            raise RuntimeError("Tensor missing channel_names.")
        channel_names = [str(x) for x in zf["channel_names"].astype(str)]
        subject_ids = np.asarray(zf["subject_ids"]).astype(str) if "subject_ids" in files else np.array([])
        tensor_key = "global_tensor_data" if "global_tensor_data" in files else "data"
        tensor_shape = tuple(int(x) for x in zf[tensor_key].shape) if tensor_key in files else None
    if max(ALL_CHANNELS) >= len(channel_names):
        raise RuntimeError(f"Requested channels {ALL_CHANNELS}, but tensor has {len(channel_names)} channels.")
    return {
        "channel_names": channel_names,
        "n_subjects": int(len(subject_ids)) if len(subject_ids) else None,
        "tensor_shape": tensor_shape,
        "python_bandpass_applied": False,
    }


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
    meta["Sex"] = meta["Sex"].fillna("UNKNOWN").astype(str)
    meta["Age"] = pd.to_numeric(meta["Age"], errors="coerce")
    meta["tensor_idx"] = meta["tensor_idx"].astype(int)
    duplicated = meta.loc[meta["SubjectID"].duplicated(), "SubjectID"].tolist()
    if duplicated:
        raise RuntimeError(f"Duplicate SubjectID values in training metadata: {duplicated[:8]}")
    cn_ad = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])]
    missing_demo = cn_ad[cn_ad["Age"].isna() | cn_ad["Sex"].isna()]
    if not missing_demo.empty:
        raise RuntimeError(
            "CN/AD pool has missing Age/Sex:\n"
            + missing_demo[["SubjectID", "ResearchGroup_Mapped", "Age", "Sex"]].head(10).to_string(index=False)
        )
    return meta


def strat_key(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    tmp = df[list(cols)].copy()
    for col in cols:
        tmp[col] = tmp[col].fillna(f"{col}_UNKNOWN").astype(str)
    return tmp.apply(lambda row: "_".join(row.values.astype(str)), axis=1)


def count_fields(df: pd.DataFrame) -> Dict[str, int]:
    out: Dict[str, int] = {"n": int(len(df))}
    for dx in DIAGNOSES:
        out[dx] = int(df["ResearchGroup_Mapped"].eq(dx).sum())
    for manufacturer in EXPECTED_MANUFACTURERS:
        label = "Siemens" if manufacturer == "SIEMENS" else manufacturer
        out[f"Manufacturer_{label}"] = int(df["Manufacturer"].eq(manufacturer).sum())
        for dx in DIAGNOSES:
            out[f"{dx}_{label}"] = int(
                (df["ResearchGroup_Mapped"].eq(dx) & df["Manufacturer"].eq(manufacturer)).sum()
            )
    for sex in ["F", "M"]:
        out[f"Sex_{sex}"] = int(df["Sex"].eq(sex).sum())
    return out


def representation_errors(row: Dict[str, Any], component: str) -> List[str]:
    errors: List[str] = []
    required_dx = ["AD", "CN"] if component.startswith("classifier") else DIAGNOSES
    for dx in required_dx:
        if int(row.get(dx, 0)) <= 0:
            errors.append(f"missing {dx}")
    for manufacturer in ["GE", "Siemens", "Philips"]:
        if int(row.get(f"Manufacturer_{manufacturer}", 0)) <= 0:
            errors.append(f"missing Manufacturer {manufacturer}")
    return errors


def write_split_preview(meta: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    seed = int(FAST_PARAMS["seed"])
    n_splits = int(FAST_PARAMS["outer_folds"])
    cn_ad = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy().reset_index(drop=True)
    strat_cols = ["ResearchGroup_Mapped", *FAST_PARAMS["classifier_stratify_cols"]]
    y_outer = strat_key(cn_ad, strat_cols)
    outer_counts = y_outer.value_counts()
    outer_fallback = bool((outer_counts < n_splits).any())
    if outer_fallback:
        y_outer = cn_ad["ResearchGroup_Mapped"]
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    by_tensor = meta.set_index("tensor_idx", drop=False)
    all_tensor_idx = meta["tensor_idx"].to_numpy()
    rows: List[Dict[str, Any]] = []
    subject_rows: List[Dict[str, Any]] = []
    for fold_idx, (train_dev_idx, test_idx) in enumerate(splitter.split(np.zeros(len(cn_ad)), y_outer), start=1):
        train_dev = cn_ad.iloc[train_dev_idx].copy()
        test = cn_ad.iloc[test_idx].copy()
        vae_pool_idx = np.setdiff1d(all_tensor_idx, test["tensor_idx"].astype(int).to_numpy(), assume_unique=False)
        vae_pool = by_tensor.loc[vae_pool_idx].reset_index(drop=True)
        vae_cols = ["ResearchGroup_Mapped", *FAST_PARAMS["vae_stratify_cols"]]
        vae_key = strat_key(vae_pool, vae_cols)
        vae_counts = vae_key.value_counts()
        vae_fallback = bool((vae_counts < 2).any())
        if vae_fallback:
            vae_key = vae_pool["ResearchGroup_Mapped"].astype(str)
            vae_counts = vae_key.value_counts()
        vae_train_idx, vae_val_idx = train_test_split(
            np.arange(len(vae_pool)),
            test_size=float(FAST_PARAMS["vae_val_split_ratio"]),
            stratify=vae_key,
            random_state=seed + 10 + (fold_idx - 1),
            shuffle=True,
        )
        components = [
            ("classifier_train_dev", train_dev, outer_fallback, int(outer_counts.min())),
            ("classifier_test", test, outer_fallback, int(outer_counts.min())),
            ("vae_pool", vae_pool, vae_fallback, int(vae_counts.min())),
            ("vae_actual_train", vae_pool.iloc[vae_train_idx], vae_fallback, int(vae_counts.min())),
            ("vae_internal_val", vae_pool.iloc[vae_val_idx], vae_fallback, int(vae_counts.min())),
        ]
        for component, df, fallback, min_cell in components:
            row: Dict[str, Any] = {
                "fold": fold_idx,
                "split_component": component,
                "classifier_stratification_cols": "+".join(strat_cols),
                "vae_internal_val_stratification_cols": "+".join(vae_cols),
                "fallback_to_label_only": bool(fallback),
                "minimum_source_stratum_count": int(min_cell),
            }
            row.update(count_fields(df))
            errs = representation_errors(row, component)
            row["passes_required_representation_check"] = not errs
            row["representation_check_errors"] = " | ".join(errs)
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
    summary = pd.DataFrame(rows).sort_values(["fold", "split_component"])
    subjects = pd.DataFrame(subject_rows).sort_values(["fold", "split_component", "SubjectID"])
    summary.to_csv(output_root / "split_preview_summary.csv", index=False)
    subjects.to_csv(output_root / "split_preview_subjects.csv", index=False)
    if not summary["passes_required_representation_check"].all():
        bad = summary[~summary["passes_required_representation_check"]]
        raise RuntimeError("Split representation failed:\n" + bad.to_string(index=False))
    return summary


def validate_schedule() -> None:
    cycle_len = FAST_PARAMS["epochs_vae"] / FAST_PARAMS["cyclical_beta_n_cycles"]
    if cycle_len != 80:
        raise RuntimeError(f"Expected cycle_len 80, got {cycle_len}")
    if FAST_PARAMS["lr_scheduler_T0"] != 80 or FAST_PARAMS["lr_scheduler_T0"] != cycle_len:
        raise RuntimeError("Expected lr_scheduler_T0 == cycle_len == 80")


def validate_fast_preflight(tensor_info: Optional[Dict[str, Any]] = None) -> None:
    """Hard guardrails for the FAST screening mode.

    These checks intentionally fail before planning or launching a candidate if
    the wrapper drifts from the agreed 3x3 FAST screen. The final full-run
    configuration remains separate and should be run as 5x5 after channel
    selection.
    """
    for name, expected in FAST_PREFLIGHT_EXPECTED.items():
        observed = FAST_PARAMS.get(name)
        if observed != expected:
            raise RuntimeError(f"FAST preflight failed: {name} must be {expected}, got {observed}")
    validate_schedule()

    classifier_stratify = list(FAST_PARAMS.get("classifier_stratify_cols", []))
    vae_stratify = list(FAST_PARAMS.get("vae_stratify_cols", []))
    metadata_features = list(FAST_PARAMS.get("metadata_features", []))
    if "Manufacturer" not in classifier_stratify:
        raise RuntimeError("FAST preflight failed: classifier_stratify_cols must include Manufacturer.")
    if "Manufacturer" not in vae_stratify:
        raise RuntimeError("FAST preflight failed: vae_stratify_cols must include Manufacturer.")
    if "Sex" in classifier_stratify:
        raise RuntimeError("FAST preflight failed: Sex must not be used for classifier stratification.")
    if "Sex" in vae_stratify:
        raise RuntimeError("FAST preflight failed: Sex must not be used for VAE stratification.")
    if metadata_features != ["Age", "Sex"]:
        raise RuntimeError(f"FAST preflight failed: metadata_features must be ['Age', 'Sex'], got {metadata_features}")
    if FAST_PARAMS.get("classifier_types") != ["logreg"]:
        raise RuntimeError("FAST preflight failed: Stage A must use only dummy canonical logreg.")
    if int(FAST_PARAMS.get("n_iter_logreg", -1)) != 1:
        raise RuntimeError("FAST preflight failed: Stage A dummy canonical logreg must use n_iter_logreg=1.")

    if tensor_info is not None and bool(tensor_info.get("python_bandpass_applied", True)):
        raise RuntimeError("FAST preflight failed: Python bandpass must be OFF.")


def output_dirs(output_root: Path, channels: Sequence[int]) -> Dict[str, Path]:
    key = channel_key(channels)
    run_dir = output_root / "runs" / key
    big_dir = BIG_DISK_ROOT / key
    readout_dir = run_dir / "classifier_only_readout"
    return {"run_dir": run_dir, "big_dir": big_dir, "readout_dir": readout_dir}


def candidate_config(
    output_root: Path,
    channels: Sequence[int],
    channel_names: Sequence[str],
    step: int,
    selected_before_step: Sequence[int],
) -> Dict[str, Any]:
    dirs = output_dirs(output_root, channels)
    params = dict(FAST_PARAMS)
    params["channels_to_use"] = list(channels)
    return {
        "run_key": channel_key(channels),
        "step": int(step),
        "channels": list(channels),
        "selected_before_step": list(selected_before_step),
        "selected_channel_names": [channel_names[ch] for ch in channels],
        "created_utc": now_utc(),
        "paths": {
            "training_script": str(TRAINING_SCRIPT.relative_to(PROJECT_ROOT)),
            "readout_script": str(READOUT_SCRIPT.relative_to(PROJECT_ROOT)),
            "global_tensor_path": str(TENSOR_PATH),
            "metadata_path": str(METADATA_PATH),
            "output_dir": str(dirs["run_dir"].relative_to(PROJECT_ROOT)),
            "big_disk_output_dir": str(dirs["big_dir"]),
            "readout_output_dir": str(dirs["readout_dir"].relative_to(PROJECT_ROOT)),
        },
        "split_strategy": {
            "classifier_outer": ["ResearchGroup_Mapped", "Manufacturer"],
            "classifier_inner": ["ResearchGroup_Mapped", "Manufacturer"],
            "vae_internal_val": ["ResearchGroup_Mapped", "Manufacturer"],
            "metadata_covariates_only": ["Age", "Sex"],
            "sex_primary_stratifier": False,
        },
        "stage_a": {
            "purpose": "fast_vae_fit_and_fold_artifact_export",
            "canonical_classifier": "logreg_dummy",
            "canonical_classifier_n_iter_logreg": 1,
            "canonical_classifier_output": "ignored_for_final_ranking",
            "final_ranking_allowed": False,
        },
        "stage_b": {
            "purpose": "classifier_only_readout_on_saved_latent_mu",
            "primary_model": PRIMARY_MODEL,
            "primary_threshold_strategy": PRIMARY_THRESHOLD,
            "secondary_threshold_strategies": SECONDARY_THRESHOLDS,
            "threshold_selection": "true_inner_cv_oof_required_for_non_0p5",
            "features": ["latent_mu", "Age", "Sex"],
            "outer_folds": int(FAST_PARAMS["outer_folds"]),
            "inner_folds": int(FAST_PARAMS["inner_folds"]),
        },
        "parameters": params,
    }


def build_train_command(config: Dict[str, Any], python_executable: str) -> List[str]:
    paths = config["paths"]
    params = config["parameters"]
    unknown = sorted(set(params) - set(PARAM_ORDER))
    if unknown:
        raise RuntimeError(f"Unknown training parameters: {unknown}")
    command = [
        python_executable,
        str((PROJECT_ROOT / paths["training_script"]).resolve()),
        "--global_tensor_path",
        paths["global_tensor_path"],
        "--metadata_path",
        paths["metadata_path"],
        "--output_dir",
        str((PROJECT_ROOT / paths["output_dir"]).resolve()),
    ]
    for name in PARAM_ORDER:
        append_arg(command, name, params.get(name))
    validate_stage_a_command(command)
    return command


def build_readout_command(config: Dict[str, Any], python_executable: str) -> List[str]:
    paths = config["paths"]
    params = config["parameters"]
    command = [
        python_executable,
        str((PROJECT_ROOT / paths["readout_script"]).resolve()),
        "--run-dir",
        str((PROJECT_ROOT / paths["output_dir"]).resolve()),
        "--output-dir",
        str((PROJECT_ROOT / paths["readout_output_dir"]).resolve()),
        "--outer-folds",
        str(int(params["outer_folds"])),
        "--inner-folds",
        str(int(params["inner_folds"])),
        "--overwrite",
    ]
    validate_readout_command(command)
    return command


def values_after_flag(tokens: Sequence[str], flag: str) -> List[str]:
    if flag not in tokens:
        return []
    start = tokens.index(flag) + 1
    values: List[str] = []
    for token in tokens[start:]:
        if token.startswith("--"):
            break
        values.append(token)
    return values


def require_flag_values(tokens: Sequence[str], flag: str, expected: Sequence[str], context: str) -> None:
    observed = values_after_flag(tokens, flag)
    if observed != list(expected):
        raise RuntimeError(f"{context}: expected {flag} {' '.join(expected)}, got {observed}")


def validate_stage_a_command(command: Sequence[str]) -> None:
    tokens = list(command)
    classifier_values = values_after_flag(tokens, "--classifier_types")
    if classifier_values != ["logreg"]:
        raise RuntimeError(f"Stage A must use only dummy logreg, got classifier_types={classifier_values}")
    if any(value.lower() == "svm" for value in classifier_values):
        raise RuntimeError("Stage A command must not include svm.")
    if "--n_iter_logreg" not in tokens:
        raise RuntimeError("Stage A command must include --n_iter_logreg 1.")
    n_iter_logreg_values = values_after_flag(tokens, "--n_iter_logreg")
    if n_iter_logreg_values != ["1"]:
        raise RuntimeError(f"Stage A dummy logreg must use n_iter_logreg=1, got {n_iter_logreg_values}")

    require_flag_values(tokens, "--outer_folds", ["3"], "Stage A FAST preflight")
    require_flag_values(tokens, "--inner_folds", ["3"], "Stage A FAST preflight")
    require_flag_values(tokens, "--latent_dim", ["128"], "Stage A FAST preflight")
    require_flag_values(tokens, "--epochs_vae", ["960"], "Stage A FAST preflight")
    require_flag_values(tokens, "--cyclical_beta_n_cycles", ["12"], "Stage A FAST preflight")
    require_flag_values(tokens, "--lr_scheduler_T0", ["80"], "Stage A FAST preflight")
    require_flag_values(tokens, "--metadata_features", ["Age", "Sex"], "Stage A FAST preflight")
    classifier_stratify = values_after_flag(tokens, "--classifier_stratify_cols")
    vae_stratify = values_after_flag(tokens, "--vae_stratify_cols")
    if "Manufacturer" not in classifier_stratify or "Sex" in classifier_stratify:
        raise RuntimeError(
            "Stage A FAST preflight: classifier_stratify_cols must include Manufacturer and must not include Sex."
        )
    if "Manufacturer" not in vae_stratify or "Sex" in vae_stratify:
        raise RuntimeError("Stage A FAST preflight: vae_stratify_cols must include Manufacturer and must not include Sex.")

    forbidden_n_iter_flags = {
        "--n_iter_svm",
        "--n_iter_rf",
        "--n_iter_gb",
        "--n_iter_xgb",
        "--n_iter_mlp",
    }
    present_forbidden = sorted(flag for flag in forbidden_n_iter_flags if flag in tokens)
    if present_forbidden:
        raise RuntimeError(f"Stage A command includes unused classifier n_iter flags: {present_forbidden}")

    for idx, token in enumerate(tokens):
        if token.startswith("--n_iter_"):
            if "=" in token:
                name, value = token.split("=", 1)
                if value == "0":
                    raise RuntimeError(f"Invalid zero Optuna trials in Stage A command: {name}=0")
            elif idx + 1 < len(tokens) and tokens[idx + 1] == "0":
                raise RuntimeError(f"Invalid zero Optuna trials in Stage A command: {token} 0")


def validate_readout_command(command: Sequence[str]) -> None:
    tokens = list(command)
    require_flag_values(tokens, "--outer-folds", ["3"], "Stage B FAST readout preflight")
    require_flag_values(tokens, "--inner-folds", ["3"], "Stage B FAST readout preflight")


def validate_planned_row_for_launch(row: pd.Series) -> None:
    train_cmd = shlex.split(str(row["train_command"]))
    validate_stage_a_command(train_cmd)
    readout_cmd = shlex.split(str(row["readout_command"]))
    validate_readout_command(readout_cmd)
    if str(row.get("stage_a_training_classifier", "")) != "canonical_logreg_dummy_ignored_for_ranking":
        raise RuntimeError("Stage A classifier outputs must be marked dummy/ignored before launch.")
    expected_readout = f"{PRIMARY_MODEL}_{PRIMARY_THRESHOLD}"
    if str(row.get("stage_b_primary_readout", "")) != expected_readout:
        raise RuntimeError(f"Stage B readout must be {expected_readout}.")
    if str(row.get("threshold_selection", "")) != "true_inner_cv_oof_for_non_0p5":
        raise RuntimeError("Stage B threshold selection must be true inner-CV OOF for non-0.5 thresholds.")


def initial_state(max_channels: int, channel_names: Sequence[str]) -> Dict[str, Any]:
    return {
        "created_utc": now_utc(),
        "updated_utc": now_utc(),
        "status": "initialized",
        "dataset": "adni_v5_1_batch20260514b_no_pybandpass",
        "python_bandpass_applied": False,
        "all_channels": ALL_CHANNELS,
        "channel_names": list(channel_names),
        "selected_channels": [],
        "selected_channel_names": [],
        "max_channels": int(max_channels),
        "min_mean_auc_improvement": MIN_MEAN_AUC_IMPROVEMENT,
        "clear_secondary_improvement": CLEAR_SECONDARY_IMPROVEMENT,
        "primary_ranking_metric": "mean_outer_roc_auc",
        "fast_screening_stage": True,
        "final_performance_estimate": False,
        "fast_outer_folds": int(FAST_PARAMS["outer_folds"]),
        "fast_inner_folds": int(FAST_PARAMS["inner_folds"]),
        "final_full_run_reference": {
            "outer_folds": 5,
            "inner_folds": 5,
            "note": "Use the selected channel set in the full 5x5 final run for performance reporting.",
        },
        "primary_readout": {
            "model": PRIMARY_MODEL,
            "threshold_strategy": PRIMARY_THRESHOLD,
            "threshold_selection": "true_inner_cv_oof",
        },
        "steps": [],
    }


def load_state(output_root: Path, max_channels: int, channel_names: Sequence[str]) -> Dict[str, Any]:
    path = output_root / "greedy_state.json"
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        state["max_channels"] = int(max_channels)
        return state
    return initial_state(max_channels=max_channels, channel_names=channel_names)


def write_state(output_root: Path, state: Dict[str, Any]) -> None:
    state["updated_utc"] = now_utc()
    (output_root / "greedy_state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def planned_candidates_from_state(state: Dict[str, Any], completed_metrics: pd.DataFrame, run_step: Optional[str]) -> Tuple[int, List[List[int]], List[int], str]:
    selected = [int(ch) for ch in state.get("selected_channels", [])]
    max_channels = int(state.get("max_channels", 4))
    if run_step == "singles" or not selected:
        return 1, [[ch] for ch in ALL_CHANNELS], selected, "singles"
    if len(selected) >= max_channels:
        return len(selected) + 1, [], selected, "max_channels_reached"
    remaining = [ch for ch in ALL_CHANNELS if ch not in selected]
    return len(selected) + 1, [selected + [ch] for ch in remaining], selected, "forward_candidates"


def readout_required_files(readout_dir: Path) -> List[Path]:
    return [
        readout_dir / "classifier_sweep_foldwise_metrics.csv",
        readout_dir / "classifier_sweep_pooled_metrics.csv",
        readout_dir / "classifier_sweep_thresholds_by_fold.csv",
        readout_dir / "classifier_sweep_subgroup_metrics_by_manufacturer.csv",
        readout_dir / "classifier_sweep_predictions.csv",
    ]


def readout_complete(readout_dir: Path) -> bool:
    return all(path.exists() for path in readout_required_files(readout_dir))


def verify_readout(readout_dir: Path, expected_outer_folds: Optional[int] = None, expected_inner_folds: Optional[int] = None) -> None:
    missing = [str(path) for path in readout_required_files(readout_dir) if not path.exists()]
    if missing:
        raise RuntimeError("Refusing aggregation: missing classifier-only readout metrics:\n" + "\n".join(missing))
    if expected_outer_folds is not None or expected_inner_folds is not None:
        command_log_path = readout_dir / "command_log.json"
        if not command_log_path.exists():
            raise RuntimeError(f"Readout is missing command_log.json with fold metadata: {readout_dir}")
        command_log = json.loads(command_log_path.read_text(encoding="utf-8"))
        if expected_outer_folds is not None and int(command_log.get("outer_folds", -1)) != int(expected_outer_folds):
            raise RuntimeError(
                f"Readout outer_folds={command_log.get('outer_folds')} does not match FAST outer_folds={expected_outer_folds}: {readout_dir}"
            )
        if expected_inner_folds is not None and int(command_log.get("inner_folds", -1)) != int(expected_inner_folds):
            raise RuntimeError(
                f"Readout inner_folds={command_log.get('inner_folds')} does not match FAST inner_folds={expected_inner_folds}: {readout_dir}"
            )
    thresholds = pd.read_csv(readout_dir / "classifier_sweep_thresholds_by_fold.csv")
    focus = thresholds[thresholds["model_name"].eq(PRIMARY_MODEL)].copy()
    required = set([PRIMARY_THRESHOLD, *SECONDARY_THRESHOLDS])
    found = set(focus["threshold_strategy"].astype(str))
    missing_strategies = sorted(required - found)
    if missing_strategies:
        raise RuntimeError(f"Readout missing required logreg_l2 threshold strategies: {missing_strategies}")
    nonfixed = focus[focus["threshold_strategy"].ne("fixed_0p5")]
    if nonfixed.empty or not nonfixed["threshold_selection_context"].eq("true_inner_cv_oof").all():
        raise RuntimeError("Non-0.5 logreg_l2 thresholds are not all true_inner_cv_oof.")
    if not nonfixed["inner_cv_context"].eq("ResearchGroup_Mapped+Manufacturer").all():
        raise RuntimeError("Inner CV context is not ResearchGroup_Mapped+Manufacturer for all logreg_l2 thresholds.")


def readout_complete_for_current_fast(readout_dir: Path) -> bool:
    try:
        verify_readout(
            readout_dir,
            expected_outer_folds=int(FAST_PARAMS["outer_folds"]),
            expected_inner_folds=int(FAST_PARAMS["inner_folds"]),
        )
        return True
    except Exception:
        return False


def metric_mean_se(values: pd.Series) -> Tuple[float, float]:
    v = pd.to_numeric(values, errors="coerce").dropna()
    if v.empty:
        return float("nan"), float("nan")
    mean = float(v.mean())
    se = float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0
    return mean, se


def summarize_readout(output_root: Path, channels: Sequence[int], channel_names: Sequence[str]) -> Dict[str, Any]:
    dirs = output_dirs(output_root, channels)
    readout_dir = dirs["readout_dir"]
    verify_readout(
        readout_dir,
        expected_outer_folds=int(FAST_PARAMS["outer_folds"]),
        expected_inner_folds=int(FAST_PARAMS["inner_folds"]),
    )
    foldwise = pd.read_csv(readout_dir / "classifier_sweep_foldwise_metrics.csv")
    pooled = pd.read_csv(readout_dir / "classifier_sweep_pooled_metrics.csv")
    subgroup = pd.read_csv(readout_dir / "classifier_sweep_subgroup_metrics_by_manufacturer.csv")
    thresholds = pd.read_csv(readout_dir / "classifier_sweep_thresholds_by_fold.csv")

    focus = foldwise[
        foldwise["model_name"].eq(PRIMARY_MODEL)
        & foldwise["threshold_strategy"].eq(PRIMARY_THRESHOLD)
    ].copy()
    if focus.empty:
        raise RuntimeError(f"No foldwise rows for {PRIMARY_MODEL}/{PRIMARY_THRESHOLD}: {readout_dir}")
    row: Dict[str, Any] = {
        "run_key": channel_key(channels),
        "channels": channels_json(channels),
        "n_channels": int(len(channels)),
        "selected_channel_names": " | ".join(channel_names[ch] for ch in channels),
        "run_dir": str(dirs["run_dir"].relative_to(PROJECT_ROOT)),
        "readout_dir": str(readout_dir.relative_to(PROJECT_ROOT)),
        "readout_model": PRIMARY_MODEL,
        "primary_threshold_strategy": PRIMARY_THRESHOLD,
        "threshold_selection_verified": True,
    }
    for metric in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]:
        mean, se = metric_mean_se(focus[metric])
        row[f"mean_outer_{metric}"] = mean
        row[f"se_outer_{metric}"] = se
    fold4 = focus[focus["fold"].eq(4)]
    if not fold4.empty:
        f4 = fold4.iloc[0]
        for metric in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]:
            row[f"fold4_{metric}"] = float(f4[metric])
    pooled_focus = pooled[
        pooled["model_name"].eq(PRIMARY_MODEL)
        & pooled["threshold_strategy"].eq(PRIMARY_THRESHOLD)
    ]
    if not pooled_focus.empty:
        p = pooled_focus.iloc[0]
        for metric in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "tn", "fp", "fn", "tp"]:
            if metric in pooled_focus.columns:
                row[f"pooled_{metric}"] = float(p[metric])
    for strategy in ["fixed_0p5", "inner_oof_youden_j"]:
        s = foldwise[
            foldwise["model_name"].eq(PRIMARY_MODEL)
            & foldwise["threshold_strategy"].eq(strategy)
        ]
        if not s.empty:
            for metric in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]:
                mean, _se = metric_mean_se(s[metric])
                row[f"{strategy}_mean_outer_{metric}"] = mean
    sub_focus = subgroup[
        subgroup["model_name"].eq(PRIMARY_MODEL)
        & subgroup["threshold_strategy"].eq(PRIMARY_THRESHOLD)
    ].copy()
    for manufacturer in ["GE", "Philips", "SIEMENS"]:
        s = sub_focus[sub_focus["Manufacturer"].eq(manufacturer)]
        label = "Siemens" if manufacturer == "SIEMENS" else manufacturer
        if not s.empty:
            for metric in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]:
                mean, _se = metric_mean_se(s[metric])
                row[f"manufacturer_{label}_{metric}"] = mean
    target_thresholds = thresholds[
        thresholds["model_name"].eq(PRIMARY_MODEL)
        & thresholds["threshold_strategy"].eq(PRIMARY_THRESHOLD)
    ]
    if not target_thresholds.empty:
        row["mean_selected_threshold"] = float(pd.to_numeric(target_thresholds["threshold"], errors="coerce").mean())
    return row


def aggregate_completed(output_root: Path, channel_names: Sequence[str]) -> pd.DataFrame:
    runs_root = output_root / "runs"
    if not runs_root.exists():
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    incomplete: List[str] = []
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir() or p.is_symlink()):
        key = run_dir.name
        if not key.startswith("ch"):
            continue
        try:
            channels = [int(x) for x in key.replace("ch", "").split("_") if x != ""]
        except ValueError:
            continue
        readout_dir = run_dir / "classifier_only_readout"
        if readout_complete_for_current_fast(readout_dir):
            rows.append(summarize_readout(output_root, channels, channel_names))
        elif readout_dir.exists() and any(readout_dir.iterdir()):
            # Stale readouts from a different FAST fold configuration must not
            # drive the current greedy selection; they are surfaced in
            # completed_runs.csv as stale instead of being aggregated.
            continue
    if incomplete:
        raise RuntimeError(
            "Refusing aggregation because these run dirs exist without complete classifier-only readout:\n"
            + "\n".join(incomplete)
        )
    if not rows:
        return pd.DataFrame()
    metrics = pd.DataFrame(rows).sort_values(["n_channels", "mean_outer_auc"], ascending=[True, False])
    return metrics


def greedy_trace(metrics: pd.DataFrame, max_channels: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if metrics.empty:
        return pd.DataFrame(), {"selected_channels": [], "status": "no_completed_metrics"}
    selected: List[int] = []
    previous: Optional[pd.Series] = None
    trace_rows: List[Dict[str, Any]] = []
    status_text = "in_progress"
    for step in range(1, max_channels + 1):
        if step == 1:
            candidates = metrics[metrics["n_channels"].eq(1)].copy()
        else:
            candidates = metrics[
                metrics["n_channels"].eq(step)
                & metrics["channels"].apply(lambda txt: set(selected).issubset(set(json.loads(txt))))
            ].copy()
        if candidates.empty:
            status_text = "waiting_for_step_metrics"
            break
        candidates = candidates.sort_values(
            ["mean_outer_auc", "mean_outer_pr_auc", "mean_outer_balanced_accuracy", "n_channels"],
            ascending=[False, False, False, True],
        )
        best = candidates.iloc[0]
        auc_delta = float("nan") if previous is None else float(best["mean_outer_auc"] - previous["mean_outer_auc"])
        pr_delta = float("nan") if previous is None else float(best["mean_outer_pr_auc"] - previous["mean_outer_pr_auc"])
        ba_delta = float("nan") if previous is None else float(best["mean_outer_balanced_accuracy"] - previous["mean_outer_balanced_accuracy"])
        stop = False
        stop_reason = ""
        if previous is not None and auc_delta < MIN_MEAN_AUC_IMPROVEMENT:
            if not (pr_delta >= CLEAR_SECONDARY_IMPROVEMENT and ba_delta >= CLEAR_SECONDARY_IMPROVEMENT):
                stop = True
                stop_reason = (
                    f"mean AUC delta {auc_delta:.4f} < {MIN_MEAN_AUC_IMPROVEMENT:.2f} "
                    "without clear PR-AUC and BA improvement"
                )
        trace_rows.append(
            {
                "step": step,
                "selected_run_key": best["run_key"],
                "selected_channels": best["channels"],
                "selected_channel_names": best["selected_channel_names"],
                "mean_outer_auc": best["mean_outer_auc"],
                "se_outer_auc": best["se_outer_auc"],
                "mean_outer_pr_auc": best["mean_outer_pr_auc"],
                "mean_outer_balanced_accuracy": best["mean_outer_balanced_accuracy"],
                "mean_outer_sensitivity": best["mean_outer_sensitivity"],
                "mean_outer_specificity": best["mean_outer_specificity"],
                "mean_outer_f1": best["mean_outer_f1"],
                "auc_delta_vs_previous": auc_delta,
                "pr_auc_delta_vs_previous": pr_delta,
                "balanced_accuracy_delta_vs_previous": ba_delta,
                "stop_after_step": stop,
                "stop_reason": stop_reason,
            }
        )
        selected = list(json.loads(best["channels"]))
        previous = best
        if stop:
            status_text = "stopped_no_meaningful_improvement"
            break
    state_update = {
        "selected_channels": selected,
        "status": status_text,
    }
    return pd.DataFrame(trace_rows), state_update


def candidate_status_path(run_dir: Path) -> Path:
    return run_dir / ".greedy_candidate_status.json"


def read_candidate_status(run_dir: Path) -> Dict[str, Any]:
    path = candidate_status_path(run_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "status_file_unreadable", "stage": "unknown", "message": str(path)}


def write_candidate_status(run_dir: Path, run_key: str, status: str, stage: str, message: str = "") -> None:
    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_key": run_key,
        "status": status,
        "stage": stage,
        "message": message,
        "updated_utc": now_utc(),
        "classifier_only_readout_complete": readout_complete_for_current_fast(run_dir / "classifier_only_readout"),
    }
    candidate_status_path(run_dir).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def recursive_find_first(payload: Any, key: str) -> Optional[Any]:
    if isinstance(payload, dict):
        if key in payload:
            return payload[key]
        for value in payload.values():
            found = recursive_find_first(value, key)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = recursive_find_first(value, key)
            if found is not None:
                return found
    return None


def as_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, list) and value:
        value = value[0]
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def fold_config_from_json(path: Path) -> Tuple[Optional[int], Optional[int], str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, None, f"{path}: unreadable JSON ({exc})"
    outer = as_optional_int(recursive_find_first(payload, "outer_folds"))
    inner = as_optional_int(recursive_find_first(payload, "inner_folds"))
    if outer is None or inner is None:
        command_text = recursive_find_first(payload, "train_command") or recursive_find_first(payload, "readout_command")
        if isinstance(command_text, str):
            tokens = shlex.split(command_text)
            outer = outer if outer is not None else as_optional_int(values_after_flag(tokens, "--outer_folds"))
            inner = inner if inner is not None else as_optional_int(values_after_flag(tokens, "--inner_folds"))
            outer = outer if outer is not None else as_optional_int(values_after_flag(tokens, "--outer-folds"))
            inner = inner if inner is not None else as_optional_int(values_after_flag(tokens, "--inner-folds"))
    return outer, inner, str(path)


def run_fold_config(run_dir: Path) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int], str]:
    stage_a_outer: Optional[int] = None
    stage_a_inner: Optional[int] = None
    stage_b_outer: Optional[int] = None
    stage_b_inner: Optional[int] = None
    evidence: List[str] = []

    for path in [run_dir / "run_config.json", run_dir / "config.json"]:
        if path.exists():
            outer, inner, source = fold_config_from_json(path)
            stage_a_outer = stage_a_outer if stage_a_outer is not None else outer
            stage_a_inner = stage_a_inner if stage_a_inner is not None else inner
            evidence.append(source)

    readout_log = run_dir / "classifier_only_readout" / "command_log.json"
    if readout_log.exists():
        outer, inner, source = fold_config_from_json(readout_log)
        stage_b_outer = outer
        stage_b_inner = inner
        evidence.append(source)

    root_status = run_dir / ".greedy_candidate_status.json"
    if root_status.exists():
        outer, inner, source = fold_config_from_json(root_status)
        stage_a_outer = stage_a_outer if stage_a_outer is not None else outer
        stage_a_inner = stage_a_inner if stage_a_inner is not None else inner
        evidence.append(source)

    if stage_a_outer is None or stage_a_inner is None:
        for path in list(run_dir.glob("**/run_config.json"))[:10]:
            outer, inner, source = fold_config_from_json(path)
            stage_a_outer = stage_a_outer if stage_a_outer is not None else outer
            stage_a_inner = stage_a_inner if stage_a_inner is not None else inner
            evidence.append(source)
            if stage_a_outer is not None and stage_a_inner is not None:
                break
    return stage_a_outer, stage_a_inner, stage_b_outer, stage_b_inner, " | ".join(evidence)


def unique_existing_paths(paths: Sequence[Path]) -> List[Path]:
    seen: set[str] = set()
    out: List[Path] = []
    for path in paths:
        if not (path.exists() or path.is_symlink()):
            continue
        key = str(path.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def mark_stale_outer5_outputs(output_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    legacy_roots = [
        ("legacy_local", LEGACY_OUTPUT_ROOT / "runs"),
        ("legacy_big_disk", LEGACY_BIG_DISK_ROOT),
        ("current_output_root", output_root / "runs"),
    ]
    for run_key in STALE_LEGACY_RUN_KEYS:
        candidates = unique_existing_paths([root / run_key for _, root in legacy_roots])
        for run_dir in candidates:
            stage_a_outer, stage_a_inner, stage_b_outer, stage_b_inner, evidence = run_fold_config(run_dir)
            stale_outer5 = stage_a_outer == 5 or stage_b_outer == 5
            status = "stale_outer5" if stale_outer5 else "not_stale_or_unknown"
            reason = ""
            if stale_outer5:
                reason = "Existing candidate output was generated with outer_folds=5 and is invalid for FAST 3x3 ranking."
                write_candidate_status(run_dir, run_key, "stale_outer5", "preflight", reason)
            rows.append(
                {
                    "run_key": run_key,
                    "path": str(run_dir),
                    "resolved_path": str(run_dir.resolve(strict=False)),
                    "stage_a_outer_folds": stage_a_outer,
                    "stage_a_inner_folds": stage_a_inner,
                    "stage_b_outer_folds": stage_b_outer,
                    "stage_b_inner_folds": stage_b_inner,
                    "status": status,
                    "reason": reason,
                    "evidence": evidence,
                    "marker_path": str(candidate_status_path(run_dir)) if stale_outer5 else "",
                }
            )
    stale = pd.DataFrame(rows)
    if stale.empty:
        stale = pd.DataFrame(
            columns=[
                "run_key",
                "path",
                "resolved_path",
                "stage_a_outer_folds",
                "stage_a_inner_folds",
                "stage_b_outer_folds",
                "stage_b_inner_folds",
                "status",
                "reason",
                "evidence",
                "marker_path",
            ]
        )
    stale.to_csv(output_root / "stale_outer5_outputs.csv", index=False)
    return stale


def completed_runs_table(output_root: Path, candidates: List[List[int]], channel_names: Sequence[str]) -> pd.DataFrame:
    rows = []
    for channels in candidates:
        dirs = output_dirs(output_root, channels)
        readout_dir = dirs["readout_dir"]
        status_info = read_candidate_status(dirs["run_dir"])
        raw_readout_complete = readout_complete(readout_dir)
        readout_is_complete = readout_complete_for_current_fast(readout_dir)
        if readout_is_complete:
            status = "complete"
        elif raw_readout_complete:
            stage_a_outer, _stage_a_inner, stage_b_outer, _stage_b_inner, _evidence = run_fold_config(dirs["run_dir"])
            status = "stale_outer5" if stage_a_outer == 5 or stage_b_outer == 5 else "stale_readout_wrong_fast_fold_config"
        elif status_info:
            status = str(status_info.get("status", "not_complete"))
        else:
            status = "not_complete"
        rows.append(
            {
                "run_key": channel_key(channels),
                "channels": channels_json(channels),
                "selected_channel_names": " | ".join(channel_names[ch] for ch in channels),
                "run_dir": str(dirs["run_dir"].relative_to(PROJECT_ROOT)),
                "readout_dir": str(readout_dir.relative_to(PROJECT_ROOT)),
                "vae_run_dir_exists": dirs["run_dir"].exists(),
                "classifier_only_readout_complete": readout_is_complete,
                "raw_classifier_only_readout_files_exist": raw_readout_complete,
                "status": status,
                "last_stage": status_info.get("stage", "") if status_info else "",
                "last_message": status_info.get("message", "") if status_info else "",
                "last_updated_utc": status_info.get("updated_utc", "") if status_info else "",
            }
        )
    return pd.DataFrame(rows)


def write_empty_csv(path: Path, columns: Sequence[str]) -> None:
    if not path.exists():
        pd.DataFrame(columns=list(columns)).to_csv(path, index=False)


def write_outputs(
    output_root: Path,
    state: Dict[str, Any],
    candidates: List[List[int]],
    step: int,
    step_label: str,
    channel_names: Sequence[str],
    planned: pd.DataFrame,
    metrics: pd.DataFrame,
    trace: pd.DataFrame,
) -> None:
    write_state(output_root, state)
    planned.to_csv(output_root / "planned_candidates_by_step.csv", index=False)
    completed = completed_runs_table(output_root, candidates, channel_names)
    completed.to_csv(output_root / "completed_runs.csv", index=False)
    if metrics.empty:
        write_empty_csv(
            output_root / "channel_set_metrics.csv",
            [
                "run_key",
                "channels",
                "n_channels",
                "mean_outer_auc",
                "se_outer_auc",
                "mean_outer_pr_auc",
                "mean_outer_balanced_accuracy",
                "mean_outer_sensitivity",
                "mean_outer_specificity",
                "mean_outer_f1",
            ],
        )
    else:
        metrics.to_csv(output_root / "channel_set_metrics.csv", index=False)
    if trace.empty:
        write_empty_csv(
            output_root / "greedy_selection_trace.csv",
            [
                "step",
                "selected_run_key",
                "selected_channels",
                "mean_outer_auc",
                "auc_delta_vs_previous",
                "stop_after_step",
                "stop_reason",
            ],
        )
    else:
        trace.to_csv(output_root / "greedy_selection_trace.csv", index=False)
    if metrics.empty:
        write_empty_csv(
            output_root / "final_greedy_ranking.csv",
            [
                "rank",
                "run_key",
                "channels",
                "n_channels",
                "mean_outer_auc",
                "mean_outer_pr_auc",
                "mean_outer_balanced_accuracy",
                "mean_outer_sensitivity",
                "mean_outer_specificity",
            ],
        )
    else:
        ranking = metrics.sort_values(
            ["mean_outer_auc", "mean_outer_pr_auc", "mean_outer_balanced_accuracy", "n_channels"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
        ranking.to_csv(output_root / "final_greedy_ranking.csv", index=False)
    write_readme(output_root, state, planned, metrics, trace, step, step_label)


def build_planned_candidates(
    output_root: Path,
    candidates: List[List[int]],
    step: int,
    selected_before_step: Sequence[int],
    channel_names: Sequence[str],
    python_executable: str,
) -> pd.DataFrame:
    config_dir = output_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for idx, channels in enumerate(candidates, start=1):
        cfg = candidate_config(output_root, channels, channel_names, step=step, selected_before_step=selected_before_step)
        cfg_path = config_dir / f"{cfg['run_key']}.json"
        cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        train_command = build_train_command(cfg, python_executable)
        readout_command = build_readout_command(cfg, python_executable)
        dirs = output_dirs(output_root, channels)
        rows.append(
            {
                "step": step,
                "candidate_index": idx,
                "run_key": cfg["run_key"],
                "channels": channels_json(channels),
                "n_channels": len(channels),
                "selected_before_step": channels_json(selected_before_step),
                "candidate_added_channel": channels[-1] if channels else "",
                "selected_channel_names": " | ".join(cfg["selected_channel_names"]),
                "vae_output_dir": str(dirs["run_dir"].relative_to(PROJECT_ROOT)),
                "big_disk_output_dir": str(dirs["big_dir"]),
                "readout_output_dir": str(dirs["readout_dir"].relative_to(PROJECT_ROOT)),
                "config_path": str(cfg_path.relative_to(PROJECT_ROOT)),
                "stage_a_training_classifier": "canonical_logreg_dummy_ignored_for_ranking",
                "stage_a_n_iter_logreg": 1,
                "stage_b_primary_readout": f"{PRIMARY_MODEL}_{PRIMARY_THRESHOLD}",
                "fast_outer_folds": int(FAST_PARAMS["outer_folds"]),
                "fast_inner_folds": int(FAST_PARAMS["inner_folds"]),
                "threshold_selection": "true_inner_cv_oof_for_non_0p5",
                "python_bandpass_applied": False,
                "status": "planned",
                "train_command": shlex.join(train_command),
                "readout_command": shlex.join(readout_command),
            }
        )
    return pd.DataFrame(rows)


def write_command_log(
    output_root: Path,
    mode: str,
    planned: pd.DataFrame,
    tensor_info: Dict[str, Any],
    metadata: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    payload = {
        "created_utc": now_utc(),
        "mode": mode,
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "tensor_path": str(TENSOR_PATH),
        "metadata_path": str(METADATA_PATH),
        "tensor_shape": tensor_info["tensor_shape"],
        "tensor_subjects": tensor_info["n_subjects"],
        "metadata_rows": int(len(metadata)),
        "python_bandpass_applied": False,
        "max_channels": int(args.max_channels),
        "dry_run": mode == "dry-run",
        "vae_retrained": mode != "dry-run",
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "primary_model": PRIMARY_MODEL,
        "primary_threshold_strategy": PRIMARY_THRESHOLD,
        "threshold_selection": "true_inner_cv_oof_required",
        "fast_screening_stage": True,
        "final_performance_estimate": False,
        "fast_outer_folds": int(FAST_PARAMS["outer_folds"]),
        "fast_inner_folds": int(FAST_PARAMS["inner_folds"]),
        "final_full_run_reference": {
            "outer_folds": 5,
            "inner_folds": 5,
            "note": "FAST is for channel screening only. Report final performance from a full 5x5 run.",
        },
        "planned_candidates": planned[["step", "run_key", "channels", "train_command", "readout_command"]].to_dict(orient="records"),
    }
    (output_root / "command_log.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_readme(
    output_root: Path,
    state: Dict[str, Any],
    planned: pd.DataFrame,
    metrics: pd.DataFrame,
    trace: pd.DataFrame,
    step: int,
    step_label: str,
) -> None:
    lines = [
        "# ADNI v5.1 batch20260514b Greedy FAST Channel Selection",
        "",
        "This wrapper replaces the preplanned FAST ablation with a constructive greedy workflow.",
        "",
        "**FAST is a channel-screening stage only. It is not the final performance estimate.**",
        "The selected channel set must be rerun with the final full-run configuration using 5 outer folds and 5 inner folds before reporting performance.",
        "",
        "## Current State",
        "",
        f"- Planned step: `{step}` / `{step_label}`",
        f"- Planned candidates in this step: `{len(planned)}`",
        f"- Selected channels before this step: `{state.get('selected_channels', [])}`",
        f"- Max channels: `{state.get('max_channels')}`",
        f"- Completed channel sets with verified readout: `{0 if metrics.empty else len(metrics)}`",
        "- Python bandpass: `OFF`",
        "- v5.1_gecn9: not used",
        "- Training launched by this invocation: no, unless `--run-step/--run-next-step --confirm-training` was used",
        "",
        "## Method",
        "",
        "- Step 1 evaluates all single channels `[0]` through `[6]`.",
        "- Each subsequent step fixes the current selected set and adds one remaining channel.",
        "- Ranking uses mean outer ROC-AUC from `logreg_l2` classifier-only readout on saved fold latent `mu` + Age + Sex.",
        f"- FAST screening folds: `{FAST_PARAMS['outer_folds']}x{FAST_PARAMS['inner_folds']}` outer/inner CV.",
        "- Final full-run folds remain `5x5` and are not changed by this wrapper.",
        "- Non-0.5 thresholds must be selected from true inner-CV out-of-fold predictions only.",
        "- Primary operating point: `inner_oof_target_sens_ge_0p70_max_spec`.",
        "- Secondary operating points: `fixed_0p5`, `inner_oof_youden_j`.",
        "- Stop rule: stop if mean AUC improves by less than 0.01 unless PR-AUC and balanced accuracy both improve clearly.",
        "",
        "## Split Constraints",
        "",
        "- Supervised classifier pool: AD vs CN only.",
        "- VAE train pool: CN + AD + MCI excluding the outer-test fold.",
        "- Classifier outer/inner split: `ResearchGroup_Mapped + Manufacturer`.",
        "- VAE internal validation split: `ResearchGroup_Mapped + Manufacturer`.",
        "- Sex is a metadata covariate only, not a stratification key.",
        "",
        "## Safety",
        "",
        "- Stage A uses a dummy canonical `logreg` with `n_iter_logreg=1` only to satisfy the VAE wrapper.",
        "- The canonical logreg emitted by Stage A is ignored for final ranking.",
        "- Aggregation refuses incomplete classifier-only readouts.",
        "- Existing tensors, metadata, and ledger files are not modified.",
        "",
        "## Outputs",
        "",
        "- `greedy_state.json`",
        "- `planned_candidates_by_step.csv`",
        "- `completed_runs.csv`",
        "- `channel_set_metrics.csv`",
        "- `greedy_selection_trace.csv`",
        "- `final_greedy_ranking.csv`",
        "- `stale_outer5_outputs.csv`",
        "- `command_log.json`",
    ]
    if not trace.empty:
        lines.extend(["", "## Latest Greedy Trace", ""])
        for row in trace.itertuples(index=False):
            lines.append(
                f"- Step {row.step}: `{row.selected_channels}` mean AUC={row.mean_outer_auc:.4f}, "
                f"PR-AUC={row.mean_outer_pr_auc:.4f}, BA={row.mean_outer_balanced_accuracy:.4f}"
            )
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_run_dir(run_dir: Path, big_dir: Path, resume: bool) -> None:
    if run_dir.exists() or run_dir.is_symlink():
        if resume:
            return
        if any(run_dir.iterdir()):
            raise RuntimeError(f"Refusing to overwrite non-empty run dir: {run_dir}")
        return
    if BIG_DISK_ROOT.parent.exists():
        big_dir.mkdir(parents=True, exist_ok=True)
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        run_dir.symlink_to(big_dir, target_is_directory=True)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)


def run_candidate(row: pd.Series, resume: bool) -> None:
    run_dir = PROJECT_ROOT / str(row["vae_output_dir"])
    readout_dir = PROJECT_ROOT / str(row["readout_output_dir"])
    big_dir = Path(str(row["big_disk_output_dir"]))
    validate_planned_row_for_launch(row)
    if resume and readout_complete_for_current_fast(readout_dir):
        print(f"[skip] {row['run_key']} readout already complete")
        write_candidate_status(run_dir, str(row["run_key"]), "complete", "stage_b", "readout already complete; skipped by --resume")
        return
    prepare_run_dir(run_dir, big_dir, resume=resume)
    if resume and run_dir.exists() and not readout_complete_for_current_fast(readout_dir):
        print(f"[retry] {row['run_key']} is not complete; retrying Stage A/Stage B")
        write_candidate_status(run_dir, str(row["run_key"]), "retrying_incomplete", "preflight", "retry requested by --resume")
    train_cmd = shlex.split(str(row["train_command"]))
    readout_cmd = shlex.split(str(row["readout_command"]))
    validate_stage_a_command(train_cmd)
    print(f"[stage A] {row['run_key']}")
    write_candidate_status(run_dir, str(row["run_key"]), "running", "stage_a", "fast VAE + dummy canonical logreg")
    completed = subprocess.run(train_cmd, cwd=str(PROJECT_ROOT), check=False)
    if completed.returncode != 0:
        write_candidate_status(
            run_dir,
            str(row["run_key"]),
            "failed_stage_a",
            "stage_a",
            f"training failed with code {completed.returncode}",
        )
        raise RuntimeError(f"Training failed for {row['run_key']} with code {completed.returncode}")
    print(f"[stage B] {row['run_key']}")
    write_candidate_status(run_dir, str(row["run_key"]), "running", "stage_b", "classifier-only logreg_l2 readout")
    completed = subprocess.run(readout_cmd, cwd=str(PROJECT_ROOT), check=False)
    if completed.returncode != 0:
        write_candidate_status(
            run_dir,
            str(row["run_key"]),
            "failed_stage_b",
            "stage_b",
            f"readout failed with code {completed.returncode}",
        )
        raise RuntimeError(f"Readout failed for {row['run_key']} with code {completed.returncode}")
    verify_readout(
        readout_dir,
        expected_outer_folds=int(FAST_PARAMS["outer_folds"]),
        expected_inner_folds=int(FAST_PARAMS["inner_folds"]),
    )
    write_candidate_status(run_dir, str(row["run_key"]), "complete", "stage_b", "classifier-only readout verified")


def print_dry_run_command_preview(planned: pd.DataFrame) -> None:
    if planned.empty:
        return
    print("\nExact Stage A commands:")
    for row in planned.itertuples(index=False):
        print(f"[{row.run_key}] {row.train_command}")
    print("\nExact Stage B commands:")
    for row in planned.itertuples(index=False):
        print(f"[{row.run_key}] {row.readout_command}")


def main() -> int:
    args = parse_args()
    real_run_requested = bool(args.run_step or args.run_next_step)
    mode = "run" if real_run_requested else "dry-run"
    if args.dry_run:
        mode = "dry-run"
        real_run_requested = False
    if args.run_step and args.run_next_step:
        raise SystemExit("Use either --run-step or --run-next-step, not both.")
    if real_run_requested and not args.confirm_training:
        raise SystemExit("Refusing real training without --confirm-training.")
    if args.max_channels < 1 or args.max_channels > len(ALL_CHANNELS):
        raise SystemExit(f"--max-channels must be 1..{len(ALL_CHANNELS)}")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "runs").mkdir(parents=True, exist_ok=True)
    validate_fast_preflight()
    if not TRAINING_SCRIPT.exists():
        raise FileNotFoundError(TRAINING_SCRIPT)
    if not READOUT_SCRIPT.exists():
        raise FileNotFoundError(READOUT_SCRIPT)

    tensor_info = inspect_tensor(TENSOR_PATH)
    validate_fast_preflight(tensor_info)
    metadata = load_metadata(METADATA_PATH)
    stale_outer5 = mark_stale_outer5_outputs(output_root)
    split_summary = write_split_preview(metadata, output_root)
    state = load_state(output_root, max_channels=args.max_channels, channel_names=tensor_info["channel_names"])
    state["max_channels"] = int(args.max_channels)

    metrics = aggregate_completed(output_root, tensor_info["channel_names"])
    if metrics.empty:
        trace = pd.DataFrame()
        state_update = {"selected_channels": [], "status": "waiting_for_singles"}
    else:
        trace, state_update = greedy_trace(metrics, max_channels=args.max_channels)
    state.update(state_update)
    state["selected_channel_names"] = [tensor_info["channel_names"][ch] for ch in state.get("selected_channels", [])]

    run_step = args.run_step if args.run_step else None
    if args.run_next_step:
        run_step = None
    step, candidates, selected_before, step_label = planned_candidates_from_state(state, metrics, run_step=run_step)
    planned = build_planned_candidates(
        output_root,
        candidates,
        step=step,
        selected_before_step=selected_before,
        channel_names=tensor_info["channel_names"],
        python_executable=args.python_executable,
    )
    state["last_planned_step"] = {
        "step": step,
        "label": step_label,
        "n_candidates": int(len(planned)),
        "candidate_channels": planned["channels"].tolist() if not planned.empty else [],
        "created_utc": now_utc(),
    }

    write_outputs(
        output_root=output_root,
        state=state,
        candidates=candidates,
        step=step,
        step_label=step_label,
        channel_names=tensor_info["channel_names"],
        planned=planned,
        metrics=metrics,
        trace=trace,
    )
    write_command_log(output_root, mode, planned, tensor_info, metadata, args)

    print(f"Mode: {mode}")
    print(f"Output root: {output_root}")
    print(f"Tensor subjects: {tensor_info['n_subjects']}")
    print(f"Metadata rows: {len(metadata)}")
    print("Python bandpass: OFF")
    print("Split: ResearchGroup_Mapped + Manufacturer")
    print("Sex role: metadata/covariate only")
    print(f"FAST screening folds: outer={FAST_PARAMS['outer_folds']} inner={FAST_PARAMS['inner_folds']}")
    print("Final full-run folds remain 5x5; FAST is not the final performance estimate.")
    print("VAE schedule: epochs=960 cycles=12 cycle_len=80 T0=80")
    print(f"Current selected channels: {state.get('selected_channels', [])}")
    print(f"Planned step: {step} ({step_label})")
    print(f"Planned candidates: {len(planned)}")
    print(f"Stale outer5 outputs marked/reported: {int(stale_outer5['status'].eq('stale_outer5').sum())}")
    if not planned.empty:
        print(planned[["step", "candidate_index", "run_key", "channels", "stage_b_primary_readout", "status"]].to_string(index=False))
        print_dry_run_command_preview(planned)
    if mode == "dry-run":
        print("Dry-run complete. No training/readout was launched.")
        return 0

    for _, row in planned.iterrows():
        run_candidate(row, resume=args.resume)
    metrics = aggregate_completed(output_root, tensor_info["channel_names"])
    trace, state_update = greedy_trace(metrics, max_channels=args.max_channels)
    state.update(state_update)
    state["selected_channel_names"] = [tensor_info["channel_names"][ch] for ch in state.get("selected_channels", [])]
    write_outputs(
        output_root=output_root,
        state=state,
        candidates=candidates,
        step=step,
        step_label=step_label,
        channel_names=tensor_info["channel_names"],
        planned=planned,
        metrics=metrics,
        trace=trace,
    )
    print("Run step complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
