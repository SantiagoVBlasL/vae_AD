#!/usr/bin/env python3
"""Prepare and optionally run a fast channel ablation on ADNI v5.1 batch20260514b.

This script is intentionally conservative: by default it performs only a
dry-run/preflight, writes per-channel-set configs and split previews, and prints
the commands that would be launched. Training requires both --run and
--confirm-training.
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

OUTPUT_ROOT = PROJECT_ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_fast_channel_ablation"
BIG_DISK_ROOT = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026/"
    "adni_v5_1_batch20260514b_fast_channel_ablation"
)

TRAINING_SCRIPT = PROJECT_ROOT / "scripts/run_vae_clf_ad_inference.py"

CHANNEL_SETS: List[Tuple[str, str, List[int]]] = [
    ("single_ch0", "single", [0]),
    ("single_ch1", "single", [1]),
    ("single_ch2", "single", [2]),
    ("single_ch3", "single", [3]),
    ("single_ch4", "single", [4]),
    ("single_ch5", "single", [5]),
    ("single_ch6", "single", [6]),
    ("pair_ch1_0", "pair", [1, 0]),
    ("pair_ch1_2", "pair", [1, 2]),
    ("pair_ch0_2", "pair", [0, 2]),
    ("pair_ch1_5", "pair", [1, 5]),
    ("pair_ch1_6", "pair", [1, 6]),
    ("pair_ch2_5", "pair", [2, 5]),
    ("triple_ch1_0_2", "triple", [1, 0, 2]),
    ("triple_ch1_0_5", "triple", [1, 0, 5]),
    ("triple_ch1_2_5", "triple", [1, 2, 5]),
    ("triple_ch1_0_6", "triple", [1, 0, 6]),
    ("quad_ch1_0_2_5", "quad", [1, 0, 2, 5]),
]

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
    "outer_folds": 5,
    "inner_folds": 5,
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
    "n_iter_logreg": 80,
    "n_iter_svm": 0,
}

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare/run fast ADNI v5.1 batch20260514b channel ablation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true", help="Preflight and write plan only.")
    parser.add_argument("--run", action="store_true", help="Launch planned runs.")
    parser.add_argument("--confirm-training", action="store_true", help="Required together with --run.")
    parser.add_argument("--only-run-key", action="append", default=None, help="Limit execution to selected run_key(s).")
    parser.add_argument("--python-executable", default=PYTHON_EXE)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


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
            raise RuntimeError("Tensor is missing python_bandpass_applied flag.")
        if bool(zf["python_bandpass_applied"]):
            raise RuntimeError("Refusing to use tensor with python_bandpass_applied=True.")
        if "channel_names" not in files:
            raise RuntimeError("Tensor is missing channel_names.")
        channel_names = [str(x) for x in zf["channel_names"].astype(str)]
        n_subjects = int(len(zf["subject_ids"])) if "subject_ids" in files else None
        tensor_key = "global_tensor_data" if "global_tensor_data" in files else "data"
        tensor_shape = tuple(int(x) for x in zf[tensor_key].shape) if tensor_key in files else None
    max_requested = max(max(channels) for _, _, channels in CHANNEL_SETS)
    if max_requested >= len(channel_names):
        raise RuntimeError(
            f"Requested channel index {max_requested}, but tensor has only {len(channel_names)} channels."
        )
    if tensor_shape is not None and tensor_shape[1] != len(channel_names):
        raise RuntimeError(f"Tensor channel axis {tensor_shape[1]} != channel_names {len(channel_names)}")
    return {
        "channel_names": channel_names,
        "n_subjects": n_subjects,
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
        raise RuntimeError(f"Missing metadata columns: {missing}")
    meta = meta.copy()
    meta["SubjectID"] = meta["SubjectID"].astype(str)
    meta["ResearchGroup_Mapped"] = meta["ResearchGroup_Mapped"].astype(str)
    meta["Manufacturer"] = meta["Manufacturer"].map(normalize_manufacturer)
    meta["Sex"] = meta["Sex"].fillna("UNKNOWN").astype(str)
    meta["Age"] = pd.to_numeric(meta["Age"], errors="coerce")
    meta["tensor_idx"] = meta["tensor_idx"].astype(int)
    duplicates = meta.loc[meta["SubjectID"].duplicated(), "SubjectID"].tolist()
    if duplicates:
        raise RuntimeError(f"Duplicate SubjectID values in metadata: {duplicates[:8]}")
    cn_ad = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])]
    missing_demo = cn_ad[cn_ad["Age"].isna() | cn_ad["Sex"].isna()]
    if not missing_demo.empty:
        raise RuntimeError(
            "CN/AD pool has missing Age or Sex:\n"
            + missing_demo[["SubjectID", "ResearchGroup_Mapped", "Age", "Sex"]].to_string(index=False)
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


def make_split_preview(meta: pd.DataFrame, output_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    all_tensor_idx = meta["tensor_idx"].astype(int).to_numpy()
    by_tensor = meta.set_index("tensor_idx", drop=False)
    rows: List[Dict[str, Any]] = []
    subject_rows: List[Dict[str, Any]] = []

    for fold_idx, (train_dev_idx, test_idx) in enumerate(splitter.split(np.zeros(len(cn_ad)), y_outer), start=1):
        train_dev = cn_ad.iloc[train_dev_idx].copy()
        test = cn_ad.iloc[test_idx].copy()
        test_tensor_idx = test["tensor_idx"].astype(int).to_numpy()
        vae_pool_idx = np.setdiff1d(np.unique(all_tensor_idx), np.unique(test_tensor_idx), assume_unique=False)
        vae_pool = by_tensor.loc[vae_pool_idx].reset_index(drop=True)
        vae_strat_cols = ["ResearchGroup_Mapped", *FAST_PARAMS["vae_stratify_cols"]]
        vae_key = strat_key(vae_pool, vae_strat_cols)
        vae_counts = vae_key.value_counts()
        vae_fallback = bool((vae_counts < 2).any())
        if vae_fallback:
            vae_key = vae_pool["ResearchGroup_Mapped"].fillna("UNKNOWN").astype(str)
            vae_counts = vae_key.value_counts()
        train_local, val_local = train_test_split(
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
            ("vae_actual_train", vae_pool.iloc[train_local], vae_fallback, int(vae_counts.min())),
            ("vae_internal_val", vae_pool.iloc[val_local], vae_fallback, int(vae_counts.min())),
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
            errors = representation_errors(row, component)
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
    summary.to_csv(output_root / "split_preview_summary.csv", index=False)
    subjects.to_csv(output_root / "split_preview_subjects.csv", index=False)
    return summary, subjects


def make_config(run_key: str, group: str, channels: List[int], channel_names: List[str], output_root: Path) -> Dict[str, Any]:
    selected_names = [channel_names[i] for i in channels]
    local_run_dir = output_root / "runs" / run_key
    big_run_dir = BIG_DISK_ROOT / run_key
    params = dict(FAST_PARAMS)
    params["channels_to_use"] = channels
    return {
        "run_name": f"adni_v5_1_batch20260514b_fast_channel_ablation_{run_key}",
        "description": (
            "Exploratory FAST channel screening on ADNI v5.1 batch20260514b, "
            "manufacturer-aware splits, no Python bandpass. Primary intended readout "
            "is post-hoc logreg_l2 with true inner-OOF target sensitivity >=0.70 max specificity."
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "channel_group": group,
        "channel_names_master_in_tensor_order": channel_names,
        "selected_channel_names": selected_names,
        "paths": {
            "training_script": str(TRAINING_SCRIPT.relative_to(PROJECT_ROOT)),
            "global_tensor_path": str(TENSOR_PATH),
            "metadata_path": str(METADATA_PATH),
            "output_dir": str(local_run_dir.relative_to(PROJECT_ROOT)),
            "big_disk_output_dir": str(big_run_dir),
        },
        "split_strategy": {
            "classifier_outer": ["ResearchGroup_Mapped", "Manufacturer"],
            "classifier_inner": ["ResearchGroup_Mapped", "Manufacturer"],
            "vae_internal_val": ["ResearchGroup_Mapped", "Manufacturer"],
            "metadata_covariates_only": ["Age", "Sex"],
            "sex_primary_stratifier": False,
        },
        "fast_readout_plan": {
            "primary_classifier": "logreg_l2",
            "classifier_stage": "posthoc_on_saved_fold_latent_mu",
            "threshold_rules": ["fixed_0p5", "inner_oof_target_sensitivity_0p70_max_specificity"],
            "threshold_selection": "true_inner_cv_oof_required",
            "outer_test_threshold_leakage": False,
            "canonical_training_classifier_note": (
                "The VAE wrapper launches only the lightweight canonical logreg readout so that "
                "fold artefacts are saved; paper-facing FAST ranking should use the post-hoc "
                "logreg_l2 inner-OOF readout on saved latents."
            ),
        },
        "parameters": params,
    }


def build_command(config: Dict[str, Any], python_executable: str) -> List[str]:
    paths = config["paths"]
    params = config["parameters"]
    unknown = sorted(set(params) - set(PARAM_ORDER))
    if unknown:
        raise RuntimeError(f"Unknown config parameters: {unknown}")
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
    return command


def write_plan_files(
    output_root: Path,
    tensor_info: Dict[str, Any],
    meta: pd.DataFrame,
    configs: List[Dict[str, Any]],
    python_executable: str,
) -> pd.DataFrame:
    configs_dir = output_root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    (output_root / "runs").mkdir(parents=True, exist_ok=True)

    plan_rows: List[Dict[str, Any]] = []
    for run_index, config in enumerate(configs, start=1):
        run_key = config["run_name"].replace("adni_v5_1_batch20260514b_fast_channel_ablation_", "")
        config_path = configs_dir / f"{run_key}.json"
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        command = build_command(config, python_executable)
        channels = config["parameters"]["channels_to_use"]
        plan_rows.append(
            {
                "run_index": run_index,
                "run_key": run_key,
                "channel_group": config["channel_group"],
                "channels": json.dumps(channels),
                "n_channels": len(channels),
                "selected_channel_names": " | ".join(config["selected_channel_names"]),
                "supported_by_tensor": True,
                "config_path": str(config_path.relative_to(PROJECT_ROOT)),
                "local_output_dir": config["paths"]["output_dir"],
                "big_disk_output_dir": config["paths"]["big_disk_output_dir"],
                "status": "planned_not_started",
                "python_bandpass_applied": False,
                "split_strategy": "ResearchGroup_Mapped+Manufacturer",
                "sex_role": "metadata_covariate_only",
                "primary_readout_classifier": "logreg_l2",
                "threshold_rules": "fixed_0p5 | inner_oof_target_sensitivity_0p70_max_specificity",
                "command": shlex.join(command),
            }
        )

    plan = pd.DataFrame(plan_rows)
    plan.to_csv(output_root / "planned_runs.csv", index=False)

    ranking = plan[["run_index", "run_key", "channels", "n_channels", "selected_channel_names"]].copy()
    for col in [
        "auc",
        "pr_auc",
        "sensitivity",
        "specificity",
        "balanced_accuracy",
        "f1",
        "cn_ge_specificity",
        "ad_ge_sensitivity",
    ]:
        ranking[col] = np.nan
    ranking["ranking_status"] = "planned_not_run"
    ranking.to_csv(output_root / "channel_ablation_summary_ranking.csv", index=False)

    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "dry_run_plan" if True else "run",
        "planned_runs": int(len(plan)),
        "python_executable": python_executable,
        "tensor_path": str(TENSOR_PATH),
        "metadata_path": str(METADATA_PATH),
        "tensor_shape": tensor_info["tensor_shape"],
        "tensor_subjects": tensor_info["n_subjects"],
        "metadata_rows": int(len(meta)),
        "python_bandpass_applied": False,
        "fast_params": FAST_PARAMS,
        "commands": plan[["run_key", "command"]].to_dict(orient="records"),
    }
    (output_root / "command_log.json").write_text(
        json.dumps(command_log, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return plan


def write_readme(output_root: Path, plan: pd.DataFrame, split_summary: pd.DataFrame, meta: pd.DataFrame) -> None:
    dx_counts = meta["ResearchGroup_Mapped"].value_counts().to_dict()
    cn = meta[meta["ResearchGroup_Mapped"].eq("CN")]
    ad = meta[meta["ResearchGroup_Mapped"].eq("AD")]
    lines = [
        "# ADNI v5.1 batch20260514b FAST Channel Ablation",
        "",
        "This is an exploratory screening plan. Training has not been launched by the dry-run.",
        "",
        "## Dataset",
        "",
        f"- Metadata rows: {len(meta)}",
        f"- AD/CN/MCI: AD={dx_counts.get('AD', 0)}, CN={dx_counts.get('CN', 0)}, MCI={dx_counts.get('MCI', 0)}",
        f"- CN by manufacturer: GE={int(cn['Manufacturer'].eq('GE').sum())}, Siemens={int(cn['Manufacturer'].eq('SIEMENS').sum())}, Philips={int(cn['Manufacturer'].eq('Philips').sum())}",
        f"- AD by manufacturer: GE={int(ad['Manufacturer'].eq('GE').sum())}, Siemens={int(ad['Manufacturer'].eq('SIEMENS').sum())}, Philips={int(ad['Manufacturer'].eq('Philips').sum())}",
        "- Python bandpass: OFF",
        "- Ledger/tensors modified: no",
        "",
        "## Planned Runs",
        "",
        f"- Planned channel-set runs: {len(plan)}",
        f"- Single-channel runs: {int(plan['channel_group'].eq('single').sum())}",
        f"- Pair runs: {int(plan['channel_group'].eq('pair').sum())}",
        f"- Triple runs: {int(plan['channel_group'].eq('triple').sum())}",
        f"- Quad runs: {int(plan['channel_group'].eq('quad').sum())}",
        "- The `[1,0,2,5]` run is supported because the tensor has channels 0 through 6.",
        "",
        "## FAST Config",
        "",
        "- latent_dim: 128",
        "- epochs_vae: 960",
        "- cyclical_beta_n_cycles: 12",
        "- cycle length: 80 epochs",
        "- lr_scheduler_T0: 80",
        "- early_stopping_patience_vae: 240",
        "- beta_vae: 2.5",
        "- dropout: 0.15",
        "- VAE final activation: tanh",
        "- LayerNorm: OFF",
        "- Canonical classifier during VAE wrapper: logreg only, class_weight balanced",
        "- Primary intended readout for ranking: logreg_l2 on saved fold latent mu, with fixed 0.5 and true inner-OOF target sensitivity >=0.70 max specificity thresholds.",
        "",
        "## Split Validation",
        "",
        "- Classifier outer split: ResearchGroup_Mapped + Manufacturer",
        "- VAE internal validation split: ResearchGroup_Mapped + Manufacturer",
        "- Sex: metadata/covariate only, not primary stratifier",
        f"- Required representation checks passed: {bool(split_summary['passes_required_representation_check'].all())}",
        "",
        "## Files",
        "",
        "- `planned_runs.csv`: one row per channel-set run and the exact launch command.",
        "- `configs/`: one JSON config per planned channel-set run.",
        "- `split_preview_summary.csv` and `split_preview_subjects.csv`: split feasibility preview.",
        "- `channel_ablation_summary_ranking.csv`: placeholder ranking table; metrics remain empty until runs complete.",
        "",
        "## Next Command After Explicit Confirmation",
        "",
        "```bash",
        f"{PYTHON_EXE} scripts/revision_bspc_2026/run_adni_v5_1_batch20260514b_fast_channel_ablation.py --run --confirm-training",
        "```",
        "",
        "Do not run this until the training is explicitly approved.",
    ]
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_fast_schedule() -> None:
    cycle_len = FAST_PARAMS["epochs_vae"] / FAST_PARAMS["cyclical_beta_n_cycles"]
    if cycle_len != 80:
        raise RuntimeError(f"Expected 960/12=80, got {cycle_len}")
    if FAST_PARAMS["lr_scheduler_T0"] != 80:
        raise RuntimeError(f"Expected lr_scheduler_T0=80, got {FAST_PARAMS['lr_scheduler_T0']}")
    if FAST_PARAMS["lr_scheduler_T0"] != cycle_len:
        raise RuntimeError("lr_scheduler_T0 must equal cycle length.")


def main() -> int:
    args = parse_args()
    if args.run and not args.confirm_training:
        raise SystemExit("Refusing to train without --confirm-training.")
    if args.dry_run and args.run:
        raise SystemExit("Choose either --dry-run or --run, not both.")
    mode = "run" if args.run else "dry-run"

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    validate_fast_schedule()
    if not TRAINING_SCRIPT.exists():
        raise FileNotFoundError(TRAINING_SCRIPT)

    tensor_info = inspect_tensor(TENSOR_PATH)
    meta = load_metadata(METADATA_PATH)
    configs = [
        make_config(run_key, group, channels, tensor_info["channel_names"], output_root)
        for run_key, group, channels in CHANNEL_SETS
    ]
    split_summary, _ = make_split_preview(meta, output_root)
    if not split_summary["passes_required_representation_check"].all():
        bad = split_summary[~split_summary["passes_required_representation_check"]]
        raise RuntimeError("Split preview failed:\n" + bad.to_string(index=False))
    plan = write_plan_files(output_root, tensor_info, meta, configs, args.python_executable)
    write_readme(output_root, plan, split_summary, meta)

    print(f"Mode: {mode}")
    print(f"Output root: {output_root}")
    print(f"Tensor subjects: {tensor_info['n_subjects']}")
    print(f"Training-ready rows: {len(meta)}")
    print("Python bandpass: OFF")
    print("Split: ResearchGroup_Mapped + Manufacturer")
    print("Sex role: metadata/covariate only")
    print("FAST schedule: epochs=960, cycles=12, cycle_len=80, T0=80")
    print(f"Planned runs: {len(plan)}")
    print(plan[["run_index", "run_key", "channels", "n_channels", "status"]].to_string(index=False))

    if mode == "dry-run":
        print("Dry-run complete. Training was NOT launched.")
        return 0

    run_keys = set(args.only_run_key or plan["run_key"].tolist())
    unknown = sorted(run_keys - set(plan["run_key"]))
    if unknown:
        raise RuntimeError(f"Unknown --only-run-key values: {unknown}")

    selected = plan[plan["run_key"].isin(run_keys)].sort_values("run_index")
    print(f"Launching {len(selected)} run(s).")
    for _, row in selected.iterrows():
        command = shlex.split(row["command"])
        print(f"[launch] {row['run_key']}")
        completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
        if completed.returncode != 0:
            print(f"[error] {row['run_key']} failed with code {completed.returncode}", file=sys.stderr)
            return int(completed.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
