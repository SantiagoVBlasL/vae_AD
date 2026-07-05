#!/usr/bin/env python3
"""FAST 3x3 decoder-final-activation screening for ADNI v5.1 [1,0,2].

Stage A trains FAST VAEs with a dummy canonical logreg readout. Stage B runs
classifier-only logreg_l2 on saved latent mu + Age + Sex. Ranking uses only
Stage B with true inner-CV OOF threshold selection.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_EXE = "/home/diego/anaconda3/envs/vae_ad/bin/python"
OUTPUT_ROOT = PROJECT_ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_final_activation_fast3x3_ch1_0_2"
BIG_DISK_ROOT = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026/adni_v5_1_batch20260514b_final_activation_fast3x3_ch1_0_2")
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
PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
CONTROL_RUN = "control_tanh"

CANDIDATES = [
    {"run_name": "control_tanh", "vae_final_activation": "tanh"},
    {"run_name": "final_activation_none", "vae_final_activation": "none"},
    {"run_name": "final_activation_linear", "vae_final_activation": "linear"},
]

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
]


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


def md_table(df: pd.DataFrame, digits: int = 4) -> str:
    if df.empty:
        return "_No rows._\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals: List[str] = []
        for col in cols:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append("" if pd.isna(val) else f"{float(val):.{digits}f}")
            else:
                vals.append("" if pd.isna(val) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


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


def common_params(activation: str) -> Dict[str, Any]:
    return {
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
        "vae_final_activation": activation,
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
        "n_iter_logreg": 1,
    }


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


def candidate_paths(output_root: Path, big_disk_root: Path, run_name: str) -> Dict[str, Path]:
    local_run = output_root / "runs" / run_name
    big_run = big_disk_root / "runs" / run_name
    return {"local_run": local_run, "big_run": big_run, "readout": local_run / "classifier_only_readout"}


def stage_a_command(python_exe: str, output_root: Path, big_disk_root: Path, candidate: Dict[str, str]) -> List[str]:
    params = common_params(candidate["vae_final_activation"])
    paths = candidate_paths(output_root, big_disk_root, candidate["run_name"])
    cmd = [
        python_exe,
        str(TRAINING_SCRIPT),
        "--global_tensor_path",
        str(TENSOR_PATH),
        "--metadata_path",
        str(METADATA_PATH),
        "--output_dir",
        str(paths["local_run"]),
    ]
    for name in PARAM_ORDER:
        append_arg(cmd, name, params[name])
    validate_stage_a(cmd, candidate)
    return cmd


def stage_b_command(python_exe: str, output_root: Path, big_disk_root: Path, candidate: Dict[str, str]) -> List[str]:
    paths = candidate_paths(output_root, big_disk_root, candidate["run_name"])
    cmd = [
        python_exe,
        str(READOUT_SCRIPT),
        "--run-dir",
        str(paths["local_run"]),
        "--output-dir",
        str(paths["readout"]),
        "--outer-folds",
        "3",
        "--inner-folds",
        "3",
        "--models",
        PRIMARY_MODEL,
        "--overwrite",
    ]
    validate_stage_b(cmd)
    return cmd


def validate_stage_a(cmd: Sequence[str], candidate: Dict[str, str]) -> None:
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
        ("--beta_vae", ["2.5"]),
        ("--dropout_rate_vae", ["0.15"]),
        ("--batch_size", ["64"]),
        ("--recon_loss_mode", ["mse_sum_batchmean_current"]),
        ("--vae_final_activation", [candidate["vae_final_activation"]]),
        ("--intermediate_fc_dim_vae", ["quarter"]),
        ("--classifier_stratify_cols", ["Manufacturer"]),
        ("--vae_stratify_cols", ["Manufacturer"]),
        ("--metadata_features", ["Age", "Sex"]),
        ("--norm_mode", ["zscore_offdiag"]),
    ]:
        require_equal(values_after_flag(cmd, flag), expected, f"Stage A {flag}")


def validate_stage_b(cmd: Sequence[str]) -> None:
    require_equal(values_after_flag(cmd, "--outer-folds"), ["3"], "Stage B outer-folds")
    require_equal(values_after_flag(cmd, "--inner-folds"), ["3"], "Stage B inner-folds")
    require_equal(values_after_flag(cmd, "--models"), [PRIMARY_MODEL], "Stage B models")


def readout_complete(path: Path) -> bool:
    return all(
        (path / name).exists()
        for name in [
            "classifier_sweep_pooled_metrics.csv",
            "classifier_sweep_foldwise_metrics.csv",
            "classifier_sweep_predictions.csv",
            "classifier_sweep_thresholds_by_fold.csv",
            "command_log.json",
        ]
    )


def prepare_run_dir(paths: Dict[str, Path], resume: bool) -> None:
    local_run = paths["local_run"]
    big_run = paths["big_run"]
    big_run.mkdir(parents=True, exist_ok=True)
    local_run.parent.mkdir(parents=True, exist_ok=True)
    if local_run.exists() or local_run.is_symlink():
        if local_run.is_symlink() and local_run.resolve() == big_run.resolve():
            if resume or not any(local_run.iterdir()):
                return
        if any(local_run.iterdir()):
            if resume:
                return
            raise RuntimeError(f"Refusing to overwrite non-empty run dir: {local_run}")
    else:
        local_run.symlink_to(big_run, target_is_directory=True)


def planned_runs(output_root: Path, big_disk_root: Path, python_exe: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for cand in CANDIDATES:
        paths = candidate_paths(output_root, big_disk_root, cand["run_name"])
        stage_a = stage_a_command(python_exe, output_root, big_disk_root, cand)
        stage_b = stage_b_command(python_exe, output_root, big_disk_root, cand)
        rows.append(
            {
                "run_name": cand["run_name"],
                "channels_to_use": json.dumps(CHANNELS),
                "recon_loss_mode": "mse_sum_batchmean_current",
                "vae_final_activation": cand["vae_final_activation"],
                "intermediate_fc_dim_vae": "quarter",
                "outer_folds": 3,
                "inner_folds": 3,
                "latent_dim": 128,
                "epochs_vae": 960,
                "cyclical_beta_n_cycles": 12,
                "lr_scheduler_T0": 80,
                "batch_size": 64,
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


def binary_metrics(y_true: Sequence[int], y_score: Sequence[float], y_pred: Sequence[int]) -> Dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    pred = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "n": int(len(y)),
        "n_cn": int((y == 0).sum()),
        "n_ad": int((y == 1).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "auc": roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan,
        "pr_auc": average_precision_score(y, score) if len(np.unique(y)) == 2 else np.nan,
        "accuracy": (tn + tp) / len(y) if len(y) else np.nan,
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": np.nanmean([sens, spec]),
        "f1": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan,
    }


def aggregate_outputs(output_root: Path, big_disk_root: Path) -> Dict[str, pd.DataFrame]:
    main_rows: List[Dict[str, Any]] = []
    foldwise_parts: List[pd.DataFrame] = []
    pred_parts: List[pd.DataFrame] = []
    recon_rows: List[Dict[str, Any]] = []
    rd_rows: List[Dict[str, Any]] = []

    for cand in CANDIDATES:
        paths = candidate_paths(output_root, big_disk_root, cand["run_name"])
        readout = paths["readout"]
        if not readout_complete(readout):
            continue
        pooled = pd.read_csv(readout / "classifier_sweep_pooled_metrics.csv")
        row = pooled[(pooled["model_name"].eq(PRIMARY_MODEL)) & (pooled["threshold_strategy"].eq(PRIMARY_THRESHOLD))]
        if len(row) != 1:
            raise RuntimeError(f"{cand['run_name']}: expected one primary pooled row, found {len(row)}")
        r = row.iloc[0].to_dict()
        main = {
            "run_name": cand["run_name"],
            "vae_final_activation": cand["vae_final_activation"],
            "intermediate_fc_dim_vae": "quarter",
            "feature_source": "StageB_latent_mu_Age_Sex",
        }
        for col in ["n", "n_cn", "n_ad", "tn", "fp", "fn", "tp", "accuracy", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "predicted_ad_rate"]:
            main[col] = r.get(col, np.nan)
        main_rows.append(main)

        fw = pd.read_csv(readout / "classifier_sweep_foldwise_metrics.csv")
        fw = fw[(fw["model_name"].eq(PRIMARY_MODEL)) & (fw["threshold_strategy"].eq(PRIMARY_THRESHOLD))].copy()
        fw.insert(0, "run_name", cand["run_name"])
        fw.insert(1, "vae_final_activation", cand["vae_final_activation"])
        foldwise_parts.append(fw)

        pred = pd.read_csv(readout / "classifier_sweep_predictions.csv")
        pred = pred[(pred["model_name"].eq(PRIMARY_MODEL)) & (pred["threshold_strategy"].eq(PRIMARY_THRESHOLD))].copy()
        pred.insert(0, "run_name", cand["run_name"])
        pred.insert(1, "vae_final_activation", cand["vae_final_activation"])
        pred_parts.append(pred)

        for fold in range(1, 4):
            fdir = paths["local_run"] / f"fold_{fold}"
            for source in ["norm", "recon"]:
                dist_path = fdir / f"fold_{fold}_dist_{source}.csv"
                if dist_path.exists():
                    dist = pd.read_csv(dist_path)
                    for _, drow in dist.iterrows():
                        out = {
                            "run_name": cand["run_name"],
                            "vae_final_activation": cand["vae_final_activation"],
                            "fold": fold,
                            "source": source,
                            **drow.to_dict(),
                        }
                        out["touches_or_exceeds_abs_1"] = bool(float(out["min"]) <= -0.999 or float(out["max"]) >= 0.999)
                        out["outside_minus1_1"] = bool(float(out["min"]) < -1.0 or float(out["max"]) > 1.0)
                        recon_rows.append(out)
            rd_path = fdir / f"fold_{fold}_rate_distortion.csv"
            if rd_path.exists():
                rd = pd.read_csv(rd_path)
                for row_type, idx in [("final_epoch", rd.index[-1]), ("best_val_beta_max", rd["L_val_betaMax"].idxmin())]:
                    rrd = rd.loc[idx].to_dict()
                    out = {"run_name": cand["run_name"], "vae_final_activation": cand["vae_final_activation"], "fold": fold, "row_type": row_type, **rrd}
                    out["beta2p5_R_over_D_val"] = 2.5 * out["R_val_nats"] / out["D_val"] if out["D_val"] else np.nan
                    out["beta2p5_R_over_D_train"] = 2.5 * out["R_train_nats"] / out["D_train"] if out["D_train"] else np.nan
                    rd_rows.append(out)

    if not main_rows:
        empty = pd.DataFrame()
        return {"main": empty, "foldwise": empty, "predictions": empty, "manufacturer": empty, "sex": empty, "reconstruction": empty, "rate_distortion": empty, "fold4": empty}

    main_df = pd.DataFrame(main_rows).sort_values("auc", ascending=False).reset_index(drop=True)
    control = main_df[main_df["run_name"].eq(CONTROL_RUN)].iloc[0]
    for col in ["auc", "pr_auc", "balanced_accuracy", "f1", "sensitivity", "specificity"]:
        main_df[f"delta_{col}_vs_tanh_control"] = main_df[col] - float(control[col])

    foldwise_df = pd.concat(foldwise_parts, ignore_index=True) if foldwise_parts else pd.DataFrame()
    pred_df = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    manufacturer = subgroup_metrics(pred_df, "Manufacturer") if not pred_df.empty else pd.DataFrame()
    sex = subgroup_metrics(pred_df, "Sex") if not pred_df.empty else pd.DataFrame()
    fold4 = fold4_deep_dive(pred_df) if not pred_df.empty else pd.DataFrame()
    reconstruction = pd.DataFrame(recon_rows)
    rd = pd.DataFrame(rd_rows)

    return {
        "main": main_df,
        "foldwise": foldwise_df,
        "predictions": pred_df,
        "manufacturer": manufacturer,
        "sex": sex,
        "reconstruction": reconstruction,
        "rate_distortion": rd,
        "fold4": fold4,
    }


def subgroup_metrics(predictions: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for keys, sub in predictions.groupby(["run_name", "vae_final_activation", group_col], dropna=False):
        run_name, activation, group_value = keys
        row = {"run_name": run_name, "vae_final_activation": activation, group_col: group_value}
        row.update(binary_metrics(sub["y_true"], sub["y_score"], sub["y_pred"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values([group_col, "auc", "run_name"], ascending=[True, False, True]).reset_index(drop=True)


def fold4_deep_dive(predictions: pd.DataFrame) -> pd.DataFrame:
    fold4 = predictions[predictions["fold"].eq(4)].copy()
    if fold4.empty:
        # FAST uses 3 outer folds; keep the requested filename with weak-fold details.
        weak_rows = []
        for run_name, sub in predictions.groupby("run_name", dropna=False):
            fold_metrics = []
            for fold, fsub in sub.groupby("fold"):
                m = binary_metrics(fsub["y_true"], fsub["y_score"], fsub["y_pred"])
                fold_metrics.append((fold, m["auc"], m["balanced_accuracy"]))
            if fold_metrics:
                weak_fold = sorted(fold_metrics, key=lambda x: (x[1], x[2]))[0][0]
                weak = sub[sub["fold"].eq(weak_fold)].copy()
                weak["diagnosis"] = np.where(weak["y_true"].eq(1), "AD", "CN")
                weak["prediction"] = np.where(weak["y_pred"].eq(1), "AD_like", "CN_like")
                weak["error_type"] = np.select(
                    [
                        weak["y_true"].eq(1) & weak["y_pred"].eq(0),
                        weak["y_true"].eq(0) & weak["y_pred"].eq(1),
                    ],
                    ["false_negative_AD", "false_positive_CN"],
                    default="correct",
                )
                weak["weak_fold_note"] = f"FAST 3x3 has no fold 4; this is weakest fold {weak_fold} for {run_name}."
                weak_rows.append(weak)
        fold4 = pd.concat(weak_rows, ignore_index=True) if weak_rows else pd.DataFrame()
    if fold4.empty:
        return fold4
    fold4["margin_to_threshold"] = fold4["y_score"].astype(float) - fold4["threshold"].astype(float)
    keep = [
        "run_name",
        "vae_final_activation",
        "SubjectID",
        "ResearchGroup_Mapped",
        "diagnosis",
        "Manufacturer",
        "Age",
        "Sex",
        "source_batch",
        "source_label",
        "tensor_source",
        "fold",
        "threshold",
        "y_score",
        "margin_to_threshold",
        "prediction",
        "error_type",
        "weak_fold_note",
    ]
    return fold4[[c for c in keep if c in fold4.columns]].sort_values(["run_name", "error_type", "Manufacturer", "SubjectID"]).reset_index(drop=True)


def write_df(output_root: Path, basename: str, df: pd.DataFrame) -> None:
    df.to_csv(output_root / f"{basename}.csv", index=False)
    (output_root / f"{basename}.md").write_text(md_table(df), encoding="utf-8")


def activation_promotion_summary(main: pd.DataFrame, foldwise: pd.DataFrame, manufacturer: pd.DataFrame, sex: pd.DataFrame) -> List[str]:
    lines = [
        "## Promotion Decision",
        "",
        "Promotion rule for a non-tanh candidate: FAST AUC +0.025 over tanh, PR-AUC nondecreasing, AUC better in at least 2/3 folds, no material BA/F1 or subgroup fragility, and plausible QC behavior.",
        "",
    ]
    if main.empty or CONTROL_RUN not in set(main["run_name"]):
        lines.append("No completed Stage B results are available yet.")
        return lines
    control = main[main["run_name"].eq(CONTROL_RUN)].iloc[0]
    recommended = []
    for _, row in main[~main["run_name"].eq(CONTROL_RUN)].iterrows():
        run_name = row["run_name"]
        delta_auc = float(row["auc"] - control["auc"])
        delta_pr = float(row["pr_auc"] - control["pr_auc"])
        delta_ba = float(row["balanced_accuracy"] - control["balanced_accuracy"])
        delta_f1 = float(row["f1"] - control["f1"])
        fold_ok = False
        if not foldwise.empty:
            fw = foldwise[foldwise["run_name"].isin([CONTROL_RUN, run_name])]
            wide = fw.pivot_table(index="fold", columns="run_name", values="auc", aggfunc="first")
            if CONTROL_RUN in wide and run_name in wide:
                fold_ok = int((wide[run_name] > wide[CONTROL_RUN]).sum()) >= 2
        subgroup_fragile = False
        for table, group_col in [(manufacturer, "Manufacturer"), (sex, "Sex")]:
            if table.empty:
                continue
            sub = table[table["run_name"].isin([CONTROL_RUN, run_name])]
            if sub.empty:
                continue
            wide_ba = sub.pivot_table(index=group_col, columns="run_name", values="balanced_accuracy", aggfunc="first")
            if CONTROL_RUN in wide_ba and run_name in wide_ba and ((wide_ba[run_name] - wide_ba[CONTROL_RUN]) < -0.05).any():
                subgroup_fragile = True
        passes = (
            delta_auc >= 0.025
            and delta_pr >= 0.0
            and fold_ok
            and delta_ba >= -0.01
            and delta_f1 >= -0.01
            and not subgroup_fragile
        )
        if passes:
            recommended.append(run_name)
        lines.append(
            f"- `{run_name}`: delta AUC={delta_auc:+.4f}, delta PR-AUC={delta_pr:+.4f}, "
            f"delta BA={delta_ba:+.4f}, delta F1={delta_f1:+.4f}, improves >=2/3 folds={fold_ok}, "
            f"subgroup_fragility={subgroup_fragile}. Promote={passes}."
        )
    lines.append("")
    lines.append("Recommended FULL 5x5 follow-up: " + (", ".join(recommended) if recommended else "none."))
    return lines


def write_readme(output_root: Path, planned: pd.DataFrame, split_df: pd.DataFrame, aggregate: Dict[str, pd.DataFrame], dry_run: bool) -> None:
    lines = [
        "# ADNI v5.1 batch20260514b Final-Activation FAST 3x3 Screening",
        "",
        "Purpose: test whether removing decoder `tanh` improves the current best `[1,0,2]` VAE pipeline without modifying the paper-ready results.",
        "",
        "## Safety",
        "",
        "- Tensor modification: NO",
        "- Metadata modification: NO",
        "- Ledger modification: NO",
        "- Python bandpass: OFF",
        "- Stage A classifier: dummy canonical logreg only, `n_iter_logreg=1`, ignored for ranking",
        "- Stage B readout: classifier-only `logreg_l2` on latent `mu + Age + Sex`",
        "- Thresholding: true inner-CV OOF `inner_oof_target_sens_ge_0p70_max_spec`",
        "",
        "## FAST Profile",
        "",
        "- channels `[1,0,2]`",
        "- `intermediate_fc_dim_vae=quarter` (current/default, not fc0)",
        "- `recon_loss_mode=mse_sum_batchmean_current`",
        "- outer/inner folds `3x3`",
        "- latent_dim `128`, epochs `960`, cycles `12`, T0 `80`, batch_size `64`",
        "- split `ResearchGroup_Mapped + Manufacturer`; Sex is metadata/covariate only",
        "",
        "## Planned Runs",
        "",
        md_table(planned[["run_name", "vae_final_activation", "intermediate_fc_dim_vae", "outer_folds", "inner_folds", "latent_dim", "epochs_vae"]]),
        "",
    ]
    if dry_run:
        lines.extend(["## Status", "", "Dry-run/preflight only. No training launched.", ""])
    else:
        main = aggregate["main"]
        if main.empty:
            lines.extend(["## Status", "", "No completed Stage B metrics found yet.", ""])
        else:
            lines.extend(["## Main Results", "", md_table(main), ""])
            lines.extend(activation_promotion_summary(main, aggregate["foldwise"], aggregate["manufacturer"], aggregate["sex"]))
            lines.append("")
    lines.extend(["## Split Preview", "", md_table(split_df.head(20)), ""])
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_manifest_and_log(
    output_root: Path,
    planned: pd.DataFrame,
    split_df: pd.DataFrame,
    tensor_info: Dict[str, Any],
    dry_run: bool,
    stage_results: List[Dict[str, Any]] | None = None,
) -> None:
    manifest = {
        "created_utc": now_utc(),
        "run_family": "final_activation_fast3x3_ch1_0_2",
        "dry_run": dry_run,
        "channels_to_use": CHANNELS,
        "selected_channel_names": CHANNEL_NAMES,
        "recon_loss_mode": "mse_sum_batchmean_current",
        "intermediate_fc_dim_vae": "quarter",
        "candidate_activations": [c["vae_final_activation"] for c in CANDIDATES],
        "outer_folds": 3,
        "inner_folds": 3,
        "latent_dim": 128,
        "epochs_vae": 960,
        "cyclical_beta_n_cycles": 12,
        "lr_scheduler_T0": 80,
        "batch_size": 64,
        "split": "ResearchGroup_Mapped+Manufacturer",
        "sex_role": "metadata/covariate only",
        "python_bandpass_applied": False,
        "tensor_info": tensor_info,
        "primary_readout": {
            "model": PRIMARY_MODEL,
            "threshold_strategy": PRIMARY_THRESHOLD,
            "threshold_selection": "true_inner_cv_oof",
        },
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
    }
    command_log = {
        **manifest,
        "training_launched": not dry_run,
        "planned_stage_commands": planned[["run_name", "stage_a_command", "stage_b_command"]].to_dict(orient="records"),
        "stage_results": stage_results or [],
        "split_preview_rows": int(len(split_df)),
    }
    (output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_command(cmd: Sequence[str]) -> int:
    print("Launching:", shlex.join(cmd), flush=True)
    completed = subprocess.run(list(cmd), cwd=PROJECT_ROOT)
    return int(completed.returncode)


def main() -> int:
    args = parse_args()
    output_root = args.output_root if args.output_root.is_absolute() else PROJECT_ROOT / args.output_root
    big_disk_root = args.big_disk_root
    output_root.mkdir(parents=True, exist_ok=True)

    tensor_info = inspect_tensor()
    meta = load_metadata()
    split_df = split_preview(meta)
    planned = planned_runs(output_root, big_disk_root, args.python_executable)
    planned.to_csv(output_root / "planned_runs.csv", index=False)
    split_df.to_csv(output_root / "split_preview_summary.csv", index=False)

    print("=== Final activation FAST 3x3 preflight ===")
    print("Output root:", output_root)
    print("Channels:", CHANNELS, CHANNEL_NAMES)
    print("Recon loss: mse_sum_batchmean_current")
    print("Intermediate FC: quarter/current default (not fc0)")
    print("Activations:", ", ".join(c["vae_final_activation"] for c in CANDIDATES))
    print("Profile: outer=3 inner=3 latent_dim=128 epochs=960 cycles=12 T0=80 batch_size=64")
    print("Python bandpass: OFF")
    print("Split: ResearchGroup_Mapped + Manufacturer; Sex covariate only")
    print("Stage A: dummy logreg n_iter_logreg=1; Stage B: logreg_l2 true inner-CV OOF")
    for _, row in planned.iterrows():
        print(f"\n[{row['run_name']}] Stage A:")
        print(row["stage_a_command"])
        print(f"[{row['run_name']}] Stage B:")
        print(row["stage_b_command"])

    if args.dry_run:
        write_manifest_and_log(output_root, planned, split_df, tensor_info, dry_run=True)
        write_readme(output_root, planned, split_df, {"main": pd.DataFrame(), "foldwise": pd.DataFrame(), "manufacturer": pd.DataFrame(), "sex": pd.DataFrame()}, dry_run=True)
        print("DRY RUN OK: no training launched.")
        return 0

    if not args.confirm_training:
        raise RuntimeError("Refusing real training without --confirm-training. Use --dry-run for preflight.")

    stage_results: List[Dict[str, Any]] = []
    for cand in CANDIDATES:
        paths = candidate_paths(output_root, big_disk_root, cand["run_name"])
        if args.resume and readout_complete(paths["readout"]):
            print(f"Skipping completed candidate with --resume: {cand['run_name']}")
            stage_results.append({"run_name": cand["run_name"], "stage_a_returncode": "skipped", "stage_b_returncode": "skipped"})
            continue
        prepare_run_dir(paths, resume=args.resume)
        stage_a = stage_a_command(args.python_executable, output_root, big_disk_root, cand)
        code_a = run_command(stage_a)
        if code_a != 0:
            stage_results.append({"run_name": cand["run_name"], "stage_a_returncode": code_a, "stage_b_returncode": None})
            write_manifest_and_log(output_root, planned, split_df, tensor_info, dry_run=False, stage_results=stage_results)
            raise RuntimeError(f"Stage A failed for {cand['run_name']} with code {code_a}")
        stage_b = stage_b_command(args.python_executable, output_root, big_disk_root, cand)
        code_b = run_command(stage_b)
        stage_results.append({"run_name": cand["run_name"], "stage_a_returncode": code_a, "stage_b_returncode": code_b})
        if code_b != 0:
            write_manifest_and_log(output_root, planned, split_df, tensor_info, dry_run=False, stage_results=stage_results)
            raise RuntimeError(f"Stage B failed for {cand['run_name']} with code {code_b}")

    aggregate = aggregate_outputs(output_root, big_disk_root)
    write_df(output_root, "final_activation_main_comparison", aggregate["main"])
    write_df(output_root, "final_activation_foldwise_comparison", aggregate["foldwise"])
    write_df(output_root, "final_activation_subgroup_by_manufacturer", aggregate["manufacturer"])
    write_df(output_root, "final_activation_subgroup_by_sex", aggregate["sex"])
    write_df(output_root, "final_activation_reconstruction_qc", aggregate["reconstruction"])
    write_df(output_root, "final_activation_rate_distortion_qc", aggregate["rate_distortion"])
    write_df(output_root, "final_activation_fold4_deep_dive", aggregate["fold4"])
    write_readme(output_root, planned, split_df, aggregate, dry_run=False)
    write_manifest_and_log(output_root, planned, split_df, tensor_info, dry_run=False, stage_results=stage_results)
    print("=== Final activation FAST 3x3 completed ===")
    if not aggregate["main"].empty:
        print(aggregate["main"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
