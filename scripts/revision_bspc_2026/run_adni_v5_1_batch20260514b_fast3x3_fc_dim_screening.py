#!/usr/bin/env python3
"""FAST 3x3 intermediate-FC screening for ADNI v5.1 [1,0,2].

The experiment is isolated from paper-ready results. Stage A trains the FAST
VAE with a dummy canonical logreg readout only. Stage B runs classifier-only
logreg_l2 on saved latent mu + Age + Sex, and all ranking comes from Stage B.
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
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_EXE = "/home/diego/anaconda3/envs/vae_ad/bin/python"
OUTPUT_ROOT = PROJECT_ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_fast3x3_fc_dim_screening"
BIG_DISK_ROOT = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026/adni_v5_1_batch20260514b_fast3x3_fc_dim_screening")
TENSOR_PATH = Path("/media/diego/Datos/vae_AD_data/revision_bspc_2026/adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors/GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz")
METADATA_PATH = Path("/media/diego/Datos/vae_AD_data/revision_bspc_2026/adni_expanded_v5_1_batch20260514b_no_pybandpass/training_ready_metadata_v5_1_batch20260514b_no_pybandpass.csv")
TRAINING_SCRIPT = PROJECT_ROOT / "scripts/run_vae_clf_ad_inference.py"
READOUT_SCRIPT = PROJECT_ROOT / "scripts/revision_bspc_2026/run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep.py"

CHANNELS = [1, 0, 2]
CHANNEL_NAMES = [
    "Pearson_Full_FisherZ_Signed",
    "Pearson_OMST_GCE_Signed_Weighted",
    "MI_KNN_Symmetric",
]
FC_CANDIDATES = ["quarter", "1024", "512", "0"]
PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
CONTROL_FC = "quarter"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Preflight only; do not train.")
    parser.add_argument("--confirm-training", action="store_true", help="Required to launch Stage A/Stage B.")
    parser.add_argument("--resume", action="store_true", help="Skip candidates with complete Stage B outputs.")
    parser.add_argument("--python-executable", default=PYTHON_EXE)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--big-disk-root", type=Path, default=BIG_DISK_ROOT)
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def append_arg(cmd: List[str], name: str, value: Any) -> None:
    if isinstance(value, bool):
        if value:
            cmd.append(f"--{name}")
        return
    if value is None:
        return
    cmd.append(f"--{name}")
    if isinstance(value, list):
        cmd.extend(str(v) for v in value)
    else:
        cmd.append(str(value))


def values_after_flag(tokens: Sequence[str], flag: str) -> List[str]:
    if flag not in tokens:
        return []
    out: List[str] = []
    for token in tokens[tokens.index(flag) + 1 :]:
        if token.startswith("--"):
            break
        out.append(token)
    return out


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: expected {expected!r}, got {actual!r}")


def inspect_tensor() -> Dict[str, Any]:
    if not TENSOR_PATH.exists():
        raise FileNotFoundError(TENSOR_PATH)
    with np.load(TENSOR_PATH, allow_pickle=False) as zf:
        shape = tuple(int(x) for x in zf["global_tensor_data"].shape)
        channel_names = [str(x) for x in zf["channel_names"].astype(str)]
        python_bandpass_applied = bool(zf["python_bandpass_applied"])
    if python_bandpass_applied:
        raise RuntimeError("Refusing tensor with python_bandpass_applied=True")
    selected = [channel_names[i] for i in CHANNELS]
    require_equal(selected, CHANNEL_NAMES, "selected channel names")
    return {"shape": shape, "channel_names": channel_names, "selected_channel_names": selected, "python_bandpass_applied": False}


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


def load_metadata() -> pd.DataFrame:
    meta = pd.read_csv(METADATA_PATH)
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
    for sex in ["F", "M"]:
        out[f"Sex_{sex}"] = int(df["Sex"].eq(sex).sum())
    return out


def split_preview(meta: pd.DataFrame) -> pd.DataFrame:
    cn_ad = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy().reset_index(drop=True)
    y_outer = strat_key(cn_ad, ["ResearchGroup_Mapped", "Manufacturer"])
    if (y_outer.value_counts() < 3).any():
        raise RuntimeError("3-fold ResearchGroup_Mapped+Manufacturer split is not feasible.")
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    by_tensor = meta.set_index("tensor_idx", drop=False)
    all_tensor_idx = meta["tensor_idx"].to_numpy()
    rows: List[Dict[str, Any]] = []
    for fold, (train_dev_idx, test_idx) in enumerate(splitter.split(np.zeros(len(cn_ad)), y_outer), start=1):
        train_dev = cn_ad.iloc[train_dev_idx].copy()
        test = cn_ad.iloc[test_idx].copy()
        vae_pool_idx = np.setdiff1d(all_tensor_idx, test["tensor_idx"].to_numpy(), assume_unique=False)
        vae_pool = by_tensor.loc[vae_pool_idx].reset_index(drop=True)
        vae_key = strat_key(vae_pool, ["ResearchGroup_Mapped", "Manufacturer"])
        if (vae_key.value_counts() < 2).any():
            raise RuntimeError("VAE internal split is not feasible.")
        train_local, val_local = train_test_split(
            np.arange(len(vae_pool)),
            test_size=0.2,
            stratify=vae_key,
            random_state=42 + fold + 9,
            shuffle=True,
        )
        for component, df in [
            ("classifier_train_dev", train_dev),
            ("classifier_test", test),
            ("vae_pool", vae_pool),
            ("vae_actual_train", vae_pool.iloc[train_local]),
            ("vae_internal_val", vae_pool.iloc[val_local]),
        ]:
            row: Dict[str, Any] = {"fold": fold, "split_component": component}
            row.update(count_fields(df))
            row["passes_required_representation_check"] = (
                row["AD"] > 0
                and row["CN"] > 0
                and (component.startswith("classifier") or row["MCI"] > 0)
                and row["Manufacturer_GE"] > 0
                and row["Manufacturer_Siemens"] > 0
                and row["Manufacturer_Philips"] > 0
            )
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out["passes_required_representation_check"].all():
        raise RuntimeError("Split preview failed representation checks.")
    return out


def run_name(fc_dim: str) -> str:
    return f"fc_{str(fc_dim).replace('/', '_')}"


def candidate_paths(output_root: Path, big_disk_root: Path, fc_dim: str) -> Dict[str, Path]:
    label = run_name(fc_dim)
    local_run = output_root / "runs" / label
    big_run = big_disk_root / "runs" / label
    return {
        "local_run": local_run,
        "big_run": big_run,
        "readout": local_run / "classifier_only_readout",
    }


def stage_a_command(python_exe: str, output_root: Path, big_disk_root: Path, fc_dim: str) -> List[str]:
    paths = candidate_paths(output_root, big_disk_root, fc_dim)
    params: Dict[str, Any] = {
        "channels_to_use": CHANNELS,
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
        "recon_loss_mode": "mse_sum_batchmean_current",
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
        "intermediate_fc_dim_vae": fc_dim,
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
        "n_iter_logreg": 1,
    }
    order = [
        "channels_to_use", "classifier_types", "classifier_stratify_cols", "vae_stratify_cols",
        "classifier_calibrate", "classifier_use_class_weight", "latent_features_type", "gridsearch_scoring",
        "outer_folds", "inner_folds", "repeated_outer_folds_n_repeats", "num_conv_layers_encoder",
        "decoder_type", "epochs_vae", "vae_val_split_ratio", "early_stopping_patience_vae",
        "cyclical_beta_n_cycles", "cyclical_beta_ratio_increase", "beta_vae", "recon_loss_mode",
        "dropout_rate_vae", "latent_dim", "batch_size", "lr_vae", "lr_scheduler_type",
        "lr_scheduler_T0", "lr_scheduler_eta_min", "lr_scheduler_patience_vae", "weight_decay_vae",
        "vae_final_activation", "intermediate_fc_dim_vae", "use_layernorm_vae_fc", "n_jobs_gridsearch",
        "metadata_features", "norm_mode", "seed", "num_workers", "log_interval_epochs_vae",
        "save_fold_artefacts", "save_vae_training_history", "qc_analyze_distributions",
        "qc_check_scanner_leakage", "qc_rate_distortion", "qc_latent_information", "qc_mi_n_neighbors",
        "qc_mi_top_k", "qc_rd_log_base", "qc_tc_ridge", "qc_var_eps_active", "use_optuna_pruner",
        "use_smote", "tune_sampler_params", "mlp_classifier_hidden_layers", "n_iter_logreg",
    ]
    cmd = [
        python_exe,
        str(TRAINING_SCRIPT),
        "--global_tensor_path", str(TENSOR_PATH),
        "--metadata_path", str(METADATA_PATH),
        "--output_dir", str(paths["local_run"]),
    ]
    for name in order:
        append_arg(cmd, name, params[name])
    validate_stage_a(cmd, fc_dim)
    return cmd


def stage_b_command(python_exe: str, output_root: Path, big_disk_root: Path, fc_dim: str) -> List[str]:
    paths = candidate_paths(output_root, big_disk_root, fc_dim)
    cmd = [
        python_exe,
        str(READOUT_SCRIPT),
        "--run-dir", str(paths["local_run"]),
        "--output-dir", str(paths["readout"]),
        "--outer-folds", "3",
        "--inner-folds", "3",
        "--models", PRIMARY_MODEL,
        "--overwrite",
    ]
    validate_stage_b(cmd)
    return cmd


def validate_stage_a(cmd: Sequence[str], fc_dim: str) -> None:
    for i, token in enumerate(cmd[:-1]):
        if token.startswith("--n_iter_") and str(cmd[i + 1]) == "0":
            raise RuntimeError(f"Invalid Stage A command contains {token} 0")
    require_equal(values_after_flag(cmd, "--channels_to_use"), ["1", "0", "2"], "Stage A channels")
    require_equal(values_after_flag(cmd, "--classifier_types"), ["logreg"], "Stage A classifier_types")
    require_equal(values_after_flag(cmd, "--n_iter_logreg"), ["1"], "Stage A n_iter_logreg")
    if "--n_iter_svm" in cmd or "svm" in values_after_flag(cmd, "--classifier_types"):
        raise RuntimeError("Stage A must not include svm.")
    for flag, expected in [
        ("--outer_folds", ["3"]),
        ("--inner_folds", ["3"]),
        ("--latent_dim", ["128"]),
        ("--epochs_vae", ["960"]),
        ("--cyclical_beta_n_cycles", ["12"]),
        ("--lr_scheduler_T0", ["80"]),
        ("--recon_loss_mode", ["mse_sum_batchmean_current"]),
        ("--vae_final_activation", ["tanh"]),
        ("--intermediate_fc_dim_vae", [str(fc_dim)]),
        ("--classifier_stratify_cols", ["Manufacturer"]),
        ("--vae_stratify_cols", ["Manufacturer"]),
        ("--metadata_features", ["Age", "Sex"]),
    ]:
        require_equal(values_after_flag(cmd, flag), expected, f"Stage A {flag}")


def validate_stage_b(cmd: Sequence[str]) -> None:
    require_equal(values_after_flag(cmd, "--outer-folds"), ["3"], "Stage B outer-folds")
    require_equal(values_after_flag(cmd, "--inner-folds"), ["3"], "Stage B inner-folds")
    require_equal(values_after_flag(cmd, "--models"), [PRIMARY_MODEL], "Stage B models")


def readout_complete(path: Path) -> bool:
    return all((path / name).exists() for name in ["classifier_sweep_pooled_metrics.csv", "classifier_sweep_foldwise_metrics.csv", "classifier_sweep_predictions.csv", "command_log.json"])


def prepare_run_dir(paths: Dict[str, Path], resume: bool) -> None:
    local_run = paths["local_run"]
    big_run = paths["big_run"]
    big_run.mkdir(parents=True, exist_ok=True)
    local_run.parent.mkdir(parents=True, exist_ok=True)
    if local_run.exists() or local_run.is_symlink():
        if local_run.is_symlink() and local_run.resolve() == big_run.resolve():
            if resume:
                return
            if not any(local_run.iterdir()):
                return
        if any(local_run.iterdir()):
            if resume:
                return
            raise RuntimeError(f"Refusing to overwrite non-empty run dir: {local_run}")
    else:
        local_run.symlink_to(big_run, target_is_directory=True)


def planned_runs(output_root: Path, big_disk_root: Path, python_exe: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for fc in FC_CANDIDATES:
        paths = candidate_paths(output_root, big_disk_root, fc)
        stage_a = stage_a_command(python_exe, output_root, big_disk_root, fc)
        stage_b = stage_b_command(python_exe, output_root, big_disk_root, fc)
        rows.append(
            {
                "candidate": run_name(fc),
                "intermediate_fc_dim_vae": fc,
                "channels_to_use": json.dumps(CHANNELS),
                "recon_loss_mode": "mse_sum_batchmean_current",
                "vae_final_activation": "tanh",
                "outer_folds": 3,
                "inner_folds": 3,
                "latent_dim": 128,
                "epochs_vae": 960,
                "cyclical_beta_n_cycles": 12,
                "lr_scheduler_T0": 80,
                "python_bandpass_applied": False,
                "stage_a_classifier": "dummy_logreg_ignored",
                "stage_b_model": PRIMARY_MODEL,
                "stage_b_threshold": PRIMARY_THRESHOLD,
                "run_dir": str(paths["local_run"]),
                "big_disk_run_dir": str(paths["big_run"]),
                "readout_dir": str(paths["readout"]),
                "stage_a_command": shlex.join(stage_a),
                "stage_b_command": shlex.join(stage_b),
            }
        )
    return pd.DataFrame(rows)


def binary_metrics(y_true: Sequence[int], y_score: Sequence[float], y_pred: Sequence[int]) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    pred = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    spec = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    return {
        "n": int(len(y)),
        "n_cn": int((y == 0).sum()),
        "n_ad": int((y == 1).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float((tp + tn) / len(y)) if len(y) else float("nan"),
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": float(np.nanmean([sens, spec])),
        "f1": float((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else float("nan"),
        "auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan"),
        "pr_auc": float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else float("nan"),
    }


def aggregate_outputs(output_root: Path, big_disk_root: Path, planned: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    main_rows: List[Dict[str, Any]] = []
    foldwise_rows: List[pd.DataFrame] = []
    pred_rows: List[pd.DataFrame] = []
    for fc in FC_CANDIDATES:
        paths = candidate_paths(output_root, big_disk_root, fc)
        readout = paths["readout"]
        if not readout_complete(readout):
            continue
        pooled = pd.read_csv(readout / "classifier_sweep_pooled_metrics.csv")
        row = pooled[(pooled["model_name"].eq(PRIMARY_MODEL)) & (pooled["threshold_strategy"].eq(PRIMARY_THRESHOLD))]
        if len(row) != 1:
            raise RuntimeError(f"{fc}: expected one primary pooled row, found {len(row)}")
        r = row.iloc[0].to_dict()
        main = {"candidate": run_name(fc), "intermediate_fc_dim_vae": fc, "feature_source": "StageB_latent_mu_Age_Sex"}
        for col in ["n", "n_cn", "n_ad", "tn", "fp", "fn", "tp", "accuracy", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "predicted_ad_rate"]:
            main[col] = r.get(col, np.nan)
        main_rows.append(main)
        fw = pd.read_csv(readout / "classifier_sweep_foldwise_metrics.csv")
        fw = fw[(fw["model_name"].eq(PRIMARY_MODEL)) & (fw["threshold_strategy"].eq(PRIMARY_THRESHOLD))].copy()
        fw.insert(0, "candidate", run_name(fc))
        fw.insert(1, "intermediate_fc_dim_vae", fc)
        foldwise_rows.append(fw)
        pred = pd.read_csv(readout / "classifier_sweep_predictions.csv")
        pred = pred[(pred["model_name"].eq(PRIMARY_MODEL)) & (pred["threshold_strategy"].eq(PRIMARY_THRESHOLD))].copy()
        pred.insert(0, "candidate", run_name(fc))
        pred.insert(1, "intermediate_fc_dim_vae", fc)
        pred_rows.append(pred)
    if not main_rows:
        return {"main": pd.DataFrame(), "foldwise": pd.DataFrame(), "predictions": pd.DataFrame(), "manufacturer": pd.DataFrame(), "sex": pd.DataFrame(), "fold4": pd.DataFrame()}
    main_df = pd.DataFrame(main_rows).sort_values("auc", ascending=False).reset_index(drop=True)
    control_auc = float(main_df.loc[main_df["intermediate_fc_dim_vae"].eq(CONTROL_FC), "auc"].iloc[0])
    control_pr = float(main_df.loc[main_df["intermediate_fc_dim_vae"].eq(CONTROL_FC), "pr_auc"].iloc[0])
    control_ba = float(main_df.loc[main_df["intermediate_fc_dim_vae"].eq(CONTROL_FC), "balanced_accuracy"].iloc[0])
    control_f1 = float(main_df.loc[main_df["intermediate_fc_dim_vae"].eq(CONTROL_FC), "f1"].iloc[0])
    main_df["delta_auc_vs_control"] = main_df["auc"] - control_auc
    main_df["delta_pr_auc_vs_control"] = main_df["pr_auc"] - control_pr
    main_df["delta_ba_vs_control"] = main_df["balanced_accuracy"] - control_ba
    main_df["delta_f1_vs_control"] = main_df["f1"] - control_f1
    main_df["passes_full5x5_decision_rule"] = (
        (main_df["delta_auc_vs_control"] >= 0.015)
        & (main_df["delta_pr_auc_vs_control"] >= 0.0)
        & (main_df["delta_ba_vs_control"] >= -0.01)
        & (main_df["delta_f1_vs_control"] >= -0.01)
    )
    foldwise_df = pd.concat(foldwise_rows, ignore_index=True)
    pred_df = pd.concat(pred_rows, ignore_index=True)
    manufacturer = subgroup_metrics(pred_df, "Manufacturer")
    sex = subgroup_metrics(pred_df, "Sex")
    fold4 = fold4_table(pred_df)
    return {"main": main_df, "foldwise": foldwise_df, "predictions": pred_df, "manufacturer": manufacturer, "sex": sex, "fold4": fold4}


def subgroup_metrics(pred: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for keys, sub in pred.groupby(["candidate", "intermediate_fc_dim_vae", group_col], dropna=False):
        candidate, fc, group = keys
        row = {"candidate": candidate, "intermediate_fc_dim_vae": fc, group_col: group}
        row.update(binary_metrics(sub["y_true"], sub["y_score"], sub["y_pred"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values([group_col, "auc"], ascending=[True, False]).reset_index(drop=True)


def fold4_table(pred: pd.DataFrame) -> pd.DataFrame:
    fold4 = pred[pred["fold"].eq(4)].copy()
    if fold4.empty:
        fold4 = pred[pred["fold"].eq(3)].copy()
        fold4["fold4_note"] = "FAST_3x3_has_no_fold4; using fold3 weak-fold placeholder"
    else:
        fold4["fold4_note"] = "fold4"
    fold4["diagnosis"] = np.where(fold4["y_true"].eq(1), "AD", "CN")
    fold4["prediction"] = np.where(fold4["y_pred"].eq(1), "AD_like", "CN_like")
    fold4["error_type"] = np.select(
        [fold4["y_true"].eq(1) & fold4["y_pred"].eq(0), fold4["y_true"].eq(0) & fold4["y_pred"].eq(1)],
        ["false_negative_AD", "false_positive_CN"],
        default="correct",
    )
    fold4["margin_to_threshold"] = fold4["y_score"].astype(float) - fold4["threshold"].astype(float)
    keep = [
        "candidate", "intermediate_fc_dim_vae", "SubjectID", "diagnosis", "Manufacturer", "Age", "Sex",
        "source_batch", "source_label", "tensor_source", "fold", "fold4_note", "threshold", "y_score",
        "margin_to_threshold", "prediction", "error_type", "y_true", "y_pred",
    ]
    return fold4[[c for c in keep if c in fold4.columns]].sort_values(["candidate", "error_type", "Manufacturer", "SubjectID"]).reset_index(drop=True)


def write_md(path: Path, df: pd.DataFrame) -> None:
    if df.empty:
        path.write_text("_No rows._\n", encoding="utf-8")
        return
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append("" if pd.isna(val) else f"{float(val):.4f}")
            else:
                vals.append("" if pd.isna(val) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def md_lines(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return ["_No rows._"]
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append("" if pd.isna(val) else f"{float(val):.4f}")
            else:
                vals.append("" if pd.isna(val) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def write_readme(output_root: Path, dry_run: bool, planned: pd.DataFrame, agg: Dict[str, pd.DataFrame] | None = None) -> None:
    lines = [
        "# ADNI v5.1 batch20260514b FAST 3x3 FC-Dim Screening",
        "",
        "Scope: controlled FAST architecture screening for `intermediate_fc_dim_vae` on channels `[1,0,2]`.",
        "",
        "- This does not modify tensors, metadata, ledger, or paper-ready result directories.",
        "- Python bandpass: OFF.",
        "- Stage A classifier output: dummy canonical `logreg`, `n_iter_logreg=1`, ignored for ranking.",
        "- Stage B ranking: classifier-only `logreg_l2` on saved latent `mu + Age + Sex`.",
        f"- Primary threshold: `{PRIMARY_THRESHOLD}` selected by true inner-CV OOF predictions.",
        "- FAST profile: 3x3, latent_dim=128, epochs=960, cycles=12, T0=80.",
        "",
        "## Candidates",
        "",
        *md_lines(planned[["candidate", "intermediate_fc_dim_vae", "run_dir"]]),
        "",
        "## Decision Rule",
        "",
        "Recommend a candidate for FULL 5x5 only if it improves FAST control AUC by at least +0.015, does not decrease PR-AUC, and does not materially worsen BA/F1 or manufacturer subgroup robustness.",
        "",
        f"- dry_run: `{str(dry_run).lower()}`",
    ]
    if agg and not agg["main"].empty:
        lines.extend(["", "## Main Results", "", *md_lines(agg["main"])])
        passed = agg["main"][agg["main"]["passes_full5x5_decision_rule"]]
        if passed.empty:
            lines.append("\nNo candidate passed the FULL 5x5 recommendation rule.")
        else:
            lines.append("\nCandidates passing the FULL 5x5 recommendation rule: " + ", ".join(passed["candidate"].astype(str)))
        lines.append("\nFAST 3x3 has no true Fold 4; `fc_dim_fold4_deep_dive.*` uses fold 3 as a labeled weak-fold placeholder.")
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_aggregate_files(output_root: Path, agg: Dict[str, pd.DataFrame]) -> None:
    pairs = [
        ("fc_dim_main_comparison", "main"),
        ("fc_dim_foldwise_comparison", "foldwise"),
        ("fc_dim_subgroup_by_manufacturer", "manufacturer"),
        ("fc_dim_subgroup_by_sex", "sex"),
        ("fc_dim_fold4_deep_dive", "fold4"),
    ]
    for stem, key in pairs:
        df = agg[key]
        df.to_csv(output_root / f"{stem}.csv", index=False)
        write_md(output_root / f"{stem}.md", df)


def main() -> int:
    args = parse_args()
    output_root = resolve(args.output_root)
    big_disk_root = args.big_disk_root
    output_root.mkdir(parents=True, exist_ok=True)
    tensor_info = inspect_tensor()
    meta = load_metadata()
    preview = split_preview(meta)
    preview.to_csv(output_root / "split_preview_summary.csv", index=False)
    planned = planned_runs(output_root, big_disk_root, args.python_executable)
    planned.to_csv(output_root / "planned_runs.csv", index=False)
    manifest = {
        "created_utc": now_utc(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "tensor_info": tensor_info,
        "fc_candidates": FC_CANDIDATES,
        "intermediate_fc_zero_supported": True,
        "stage_a_classifier_outputs": "dummy_logreg_ignored_for_ranking",
        "stage_b_model": PRIMARY_MODEL,
        "stage_b_threshold": PRIMARY_THRESHOLD,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
    }
    (output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(output_root, dry_run=(args.dry_run or not args.confirm_training), planned=planned)

    print("=== FAST 3x3 FC-Dim Screening ===")
    print(f"Mode: {'DRY-RUN' if args.dry_run or not args.confirm_training else 'REAL RUN'}")
    print(planned[["candidate", "intermediate_fc_dim_vae", "outer_folds", "inner_folds", "latent_dim", "epochs_vae", "stage_a_classifier", "stage_b_model"]].to_string(index=False))
    for row in planned.itertuples(index=False):
        print(f"\n[{row.candidate}] Stage A command:")
        print(row.stage_a_command)
        print(f"[{row.candidate}] Stage B command:")
        print(row.stage_b_command)
    if args.dry_run or not args.confirm_training:
        command_log = {
            "created_utc": now_utc(),
            "dry_run": True,
            "training_launched": False,
            "tensor_modified": False,
            "metadata_modified": False,
            "ledger_modified": False,
            "planned_candidates": planned["candidate"].tolist(),
        }
        (output_root / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("\nDry-run complete. No training launched.")
        return 0

    completed: List[Dict[str, Any]] = []
    for fc in FC_CANDIDATES:
        paths = candidate_paths(output_root, big_disk_root, fc)
        if args.resume and readout_complete(paths["readout"]):
            print(f"[skip] {run_name(fc)} readout already complete")
            completed.append({"candidate": run_name(fc), "status": "skipped_complete"})
            continue
        prepare_run_dir(paths, resume=args.resume)
        stage_a = stage_a_command(args.python_executable, output_root, big_disk_root, fc)
        stage_b = stage_b_command(args.python_executable, output_root, big_disk_root, fc)
        print(f"[run] Stage A {run_name(fc)}")
        ret = subprocess.run(stage_a, cwd=PROJECT_ROOT, check=False)
        if ret.returncode != 0:
            raise SystemExit(ret.returncode)
        print(f"[run] Stage B {run_name(fc)}")
        ret = subprocess.run(stage_b, cwd=PROJECT_ROOT, check=False)
        if ret.returncode != 0:
            raise SystemExit(ret.returncode)
        completed.append({"candidate": run_name(fc), "status": "completed"})

    agg = aggregate_outputs(output_root, big_disk_root, planned)
    write_aggregate_files(output_root, agg)
    write_readme(output_root, dry_run=False, planned=planned, agg=agg)
    command_log = {
        "created_utc": now_utc(),
        "dry_run": False,
        "training_launched": True,
        "completed": completed,
        "ranking_source": "StageB classifier-only logreg_l2 only",
        "threshold_selection": "true_inner_cv_oof",
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
    }
    (output_root / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("\n=== Stage B primary ranking ===")
    print(agg["main"][["candidate", "intermediate_fc_dim_vae", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "passes_full5x5_decision_rule"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
