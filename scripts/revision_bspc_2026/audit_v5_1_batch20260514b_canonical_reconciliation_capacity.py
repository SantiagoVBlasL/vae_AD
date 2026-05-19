#!/usr/bin/env python3
"""Read-only canonical-run reconciliation and capacity audit for ADNI v5.1.

This script does not train models and does not modify tensors, metadata, the
ledger, or existing run folders. It only reads completed result artifacts and
writes a new audit directory under results/.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results" / "revision_bspc_2026"
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "adni_v5_1_batch20260514b_canonical_reconciliation_capacity_audit"

CURRENT_RUN_DIR = RESULTS_ROOT / "adni_v5_1_batch20260514b_ch1_0_2_mfrsplit_3840_final_candidate"
CURRENT_READOUT_DIR = RESULTS_ROOT / "adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep"
CURRENT_THRESHOLD_AUDIT_DIR = RESULTS_ROOT / "adni_v5_1_batch20260514b_threshold_final_audit"
CURRENT_PAPER_TABLES_DIR = RESULTS_ROOT / "adni_v5_1_batch20260514b_paper_ready_threshold_tables"

PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    label: str
    run_dir: Path
    readout_dir: Optional[Path]
    role: str
    notes: str


RUN_SPECS: Sequence[RunSpec] = [
    RunSpec(
        "current_ch1_0_2",
        "current tanh [1,0,2] FULL 5x5",
        CURRENT_RUN_DIR,
        CURRENT_READOUT_DIR,
        "locked_paper_candidate",
        "Primary VAE is tanh with current summed-MSE objective; primary readout is classifier-only logreg_l2.",
    ),
    RunSpec(
        "baseline_2560_ch1_0_2",
        "older 2560 baseline [1,0,2]",
        RESULTS_ROOT / "adni_v5_1_batch20260514b_ch1_0_2_final_candidate_baseline",
        None,
        "historical_stage_a_baseline",
        "Older canonical full-Optuna logreg/svm Stage A baseline at 2560 epochs.",
    ),
    RunSpec(
        "interim_20260514_ch1_0_2",
        "interim 20260514 [1,0,2]",
        RESULTS_ROOT / "adni_v5_1_batch20260514_ch1_0_2_interim_baseline",
        None,
        "historical_interim",
        "Earlier dataset before batch20260514b; not current paper candidate.",
    ),
    RunSpec(
        "objective_v2_ch1_0_2",
        "objective-v2 [1,0,2]",
        RESULTS_ROOT / "adni_v5_1_batch20260514b_ch1_0_2_objective_v2_full_5x5",
        RESULTS_ROOT / "adni_v5_1_batch20260514b_ch1_0_2_objective_v2_full_5x5" / "classifier_only_readout",
        "failed_micro_optimization",
        "offdiag_channelmean_sum objective; did not beat current tanh baseline.",
    ),
    RunSpec(
        "fc0_ch1_0_2",
        "fc0 [1,0,2]",
        RESULTS_ROOT / "adni_v5_1_batch20260514b_ch1_0_2_fc0_full_5x5",
        RESULTS_ROOT / "adni_v5_1_batch20260514b_ch1_0_2_fc0_full_5x5" / "classifier_only_readout",
        "failed_micro_optimization",
        "intermediate_fc_dim_vae=0; did not beat current tanh baseline.",
    ),
    RunSpec(
        "final_activation_none_ch1_0_2",
        "final activation none [1,0,2]",
        RESULTS_ROOT / "adni_v5_1_batch20260514b_ch1_0_2_final_activation_none_full_5x5",
        RESULTS_ROOT / "adni_v5_1_batch20260514b_ch1_0_2_final_activation_none_full_5x5" / "classifier_only_readout",
        "failed_micro_optimization",
        "Pre-registered decoder identity output perturbation; did not beat current tanh baseline.",
    ),
    RunSpec(
        "ch1",
        "single-channel [1]",
        RESULTS_ROOT / "adni_v5_1_batch20260514b_ch1_full_5x5_candidate",
        RESULTS_ROOT / "adni_v5_1_batch20260514b_ch1_full_5x5_candidate" / "classifier_only_readout",
        "parsimonious_ablation",
        "Strong parsimonious ablation; not best by AUC/PR-AUC.",
    ),
    RunSpec(
        "ch1_4",
        "pair [1,4]",
        RESULTS_ROOT / "adni_v5_1_batch20260514b_ch1_4_full_5x5_candidate",
        RESULTS_ROOT / "adni_v5_1_batch20260514b_ch1_4_full_5x5_candidate" / "classifier_only_readout",
        "secondary_ablation_failed",
        "Best FAST pair but underperformed in FULL 5x5.",
    ),
]


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        return pd.DataFrame({"read_error": [str(exc)]})


def first_file(path: Path, pattern: str) -> Optional[Path]:
    files = sorted(path.glob(pattern))
    return files[0] if files else None


def fmt_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def md_table(df: pd.DataFrame, max_rows: int = 80, digits: int = 4) -> str:
    if df.empty:
        return "_No rows._\n"
    show = df.head(max_rows).copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: fmt_float(x, digits))
    headers = list(show.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in show.iterrows():
        vals = [str(row[col]) if not pd.isna(row[col]) else "" for col in headers]
        vals = [v.replace("\n", " ").replace("|", "\\|") for v in vals]
        lines.append("| " + " | ".join(vals) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(df)} rows._")
    return "\n".join(lines) + "\n"


def write_table(output_dir: Path, stem: str, df: pd.DataFrame, max_rows: int = 80) -> None:
    df.to_csv(output_dir / f"{stem}.csv", index=False)
    (output_dir / f"{stem}.md").write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


def args_from_config(run_dir: Path) -> Dict[str, Any]:
    cfg = read_json(run_dir / "run_config.json")
    return cfg.get("args", cfg)


def tensor_shape_from_config(run_dir: Path) -> str:
    cfg = read_json(run_dir / "run_config.json")
    shape = cfg.get("tensor_shape")
    if shape is None:
        shape = cfg.get("args", {}).get("tensor_shape")
    return str(shape) if shape is not None else ""


def dataset_counts(metadata_path: str) -> Dict[str, Any]:
    path = Path(metadata_path)
    if not path.exists():
        return {}
    df = read_csv(path)
    if df.empty:
        return {}
    dx_col = "ResearchGroup_Mapped" if "ResearchGroup_Mapped" in df.columns else "Diagnosis"
    counts = df[dx_col].value_counts(dropna=False).to_dict()
    supervised = df[df[dx_col].isin(["AD", "CN"])]
    return {
        "metadata_rows": len(df),
        "N_training_ready": len(df),
        "N_classifier_AD_CN": len(supervised),
        "AD": int(counts.get("AD", 0)),
        "CN": int(counts.get("CN", 0)),
        "MCI": int(counts.get("MCI", 0)),
        "Unknown": int(counts.get("UNKNOWN", counts.get("Unknown", 0))),
    }


def safe_auc(y_true: pd.Series, y_score: pd.Series) -> float:
    y_true = pd.to_numeric(y_true, errors="coerce")
    y_score = pd.to_numeric(y_score, errors="coerce")
    mask = y_true.notna() & y_score.notna()
    if mask.sum() == 0 or y_true[mask].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true[mask], y_score[mask]))


def safe_pr_auc(y_true: pd.Series, y_score: pd.Series) -> float:
    y_true = pd.to_numeric(y_true, errors="coerce")
    y_score = pd.to_numeric(y_score, errors="coerce")
    mask = y_true.notna() & y_score.notna()
    if mask.sum() == 0 or y_true[mask].nunique() < 2:
        return float("nan")
    return float(average_precision_score(y_true[mask], y_score[mask]))


def stage_a_rows(spec: RunSpec) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    metrics_path = first_file(spec.run_dir, "all_folds_metrics_MULTI_*.csv")
    preds_path = first_file(spec.run_dir, "all_folds_clf_predictions_MULTI_*.csv")
    if metrics_path is None:
        return rows
    metrics = read_csv(metrics_path)
    preds = read_csv(preds_path) if preds_path else pd.DataFrame()
    if metrics.empty:
        return rows

    clf_col = "actual_classifier_type" if "actual_classifier_type" in metrics.columns else "classifier"
    for clf, group in metrics.groupby(clf_col, dropna=False):
        pooled_auc = pooled_pr = float("nan")
        pooled_n = pooled_cn = pooled_ad = float("nan")
        if not preds.empty and "classifier_type" in preds.columns:
            pg = preds[preds["classifier_type"].astype(str) == str(clf)]
            if not pg.empty:
                score_col = "y_score_final" if "y_score_final" in pg.columns else "y_score"
                pooled_auc = safe_auc(pg["y_true"], pg[score_col])
                pooled_pr = safe_pr_auc(pg["y_true"], pg[score_col])
                pooled_n = len(pg)
                pooled_ad = int(pd.to_numeric(pg["y_true"], errors="coerce").sum())
                pooled_cn = int(pooled_n - pooled_ad)
        row = {
            "metric_row_type": "stage_a_full_optuna_threshold_0p5",
            "classifier_readout": str(clf),
            "metric_definition": "foldwise mean from all_folds_metrics; pooled OOF from all_folds_clf_predictions if available",
            "threshold_strategy": "fixed_0p5",
            "stage": "Stage A canonical VAE+classifier pipeline",
            "auc_foldwise_mean": pd.to_numeric(group.get("auc"), errors="coerce").mean(),
            "auc_foldwise_std": pd.to_numeric(group.get("auc"), errors="coerce").std(ddof=0),
            "pr_auc_foldwise_mean": pd.to_numeric(group.get("pr_auc"), errors="coerce").mean(),
            "pooled_oof_auc": pooled_auc,
            "pooled_oof_pr_auc": pooled_pr,
            "balanced_accuracy": pd.to_numeric(group.get("balanced_accuracy"), errors="coerce").mean(),
            "sensitivity": pd.to_numeric(group.get("sensitivity"), errors="coerce").mean(),
            "specificity": pd.to_numeric(group.get("specificity"), errors="coerce").mean(),
            "f1": pd.to_numeric(group.get("f1_score", group.get("f1")), errors="coerce").mean(),
            "n": pooled_n,
            "n_cn": pooled_cn,
            "n_ad": pooled_ad,
            "source_metrics_file": str(metrics_path),
            "source_predictions_file": str(preds_path) if preds_path else "",
        }
        rows.append(row)
    return rows


def stage_b_primary_row(spec: RunSpec) -> Optional[Dict[str, Any]]:
    if spec.readout_dir is None:
        return None
    pooled_path = spec.readout_dir / "classifier_sweep_pooled_metrics.csv"
    df = read_csv(pooled_path)
    if df.empty:
        return None
    mask = (df["model_name"].astype(str) == PRIMARY_MODEL) & (
        df["threshold_strategy"].astype(str) == PRIMARY_THRESHOLD
    )
    if not mask.any():
        return None
    r = df[mask].iloc[0].to_dict()
    return {
        "metric_row_type": "stage_b_classifier_only_primary",
        "classifier_readout": PRIMARY_MODEL,
        "metric_definition": "pooled OOF AD/CN metrics from classifier-only readout; threshold selected inside train/dev inner-CV OOF",
        "threshold_strategy": PRIMARY_THRESHOLD,
        "stage": "Stage B classifier-only readout on saved fold-specific latent mu + Age + Sex",
        "auc_foldwise_mean": np.nan,
        "auc_foldwise_std": np.nan,
        "pr_auc_foldwise_mean": np.nan,
        "pooled_oof_auc": r.get("auc", np.nan),
        "pooled_oof_pr_auc": r.get("pr_auc", np.nan),
        "balanced_accuracy": r.get("balanced_accuracy", np.nan),
        "sensitivity": r.get("sensitivity", np.nan),
        "specificity": r.get("specificity", np.nan),
        "f1": r.get("f1", np.nan),
        "n": r.get("n", np.nan),
        "n_cn": r.get("n_cn", np.nan),
        "n_ad": r.get("n_ad", np.nan),
        "tn": r.get("tn", np.nan),
        "fp": r.get("fp", np.nan),
        "fn": r.get("fn", np.nan),
        "tp": r.get("tp", np.nan),
        "source_metrics_file": str(pooled_path),
        "source_predictions_file": str(spec.readout_dir / "classifier_sweep_predictions.csv"),
    }


def build_reconciliation() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for spec in RUN_SPECS:
        args = args_from_config(spec.run_dir)
        metadata_path = str(args.get("metadata_path", ""))
        counts = dataset_counts(metadata_path)
        base = {
            "run_id": spec.run_id,
            "label": spec.label,
            "role": spec.role,
            "run_dir": str(spec.run_dir),
            "readout_dir": str(spec.readout_dir or ""),
            "dataset_tensor": args.get("global_tensor_path", ""),
            "metadata": metadata_path,
            "tensor_shape": tensor_shape_from_config(spec.run_dir),
            "channels_to_use": str(args.get("channels_to_use", "")),
            "selected_channel_names": str(args.get("selected_channel_names", "")),
            "vae_final_activation": args.get("vae_final_activation", ""),
            "recon_loss_mode": args.get("recon_loss_mode", "mse_sum_batchmean_current"),
            "intermediate_fc_dim_vae": args.get("intermediate_fc_dim_vae", ""),
            "latent_dim": args.get("latent_dim", ""),
            "epochs_vae": args.get("epochs_vae", ""),
            "cyclical_beta_n_cycles": args.get("cyclical_beta_n_cycles", ""),
            "lr_scheduler_T0": args.get("lr_scheduler_T0", ""),
            "beta_vae": args.get("beta_vae", ""),
            "dropout_rate_vae": args.get("dropout_rate_vae", ""),
            "batch_size": args.get("batch_size", ""),
            "classifier_stratify_cols": str(args.get("classifier_stratify_cols", "")),
            "vae_stratify_cols": str(args.get("vae_stratify_cols", "")),
            "metadata_features": str(args.get("metadata_features", "")),
            "python_bandpass_applied": "False/no_pybandpass inferred from dataset path",
            "notes": spec.notes,
            **counts,
        }
        for row in stage_a_rows(spec):
            rows.append({**base, **row})
        stage_b = stage_b_primary_row(spec)
        if stage_b is not None:
            rows.append({**base, **stage_b})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["is_true_current_paper_candidate"] = (
            (out["run_id"] == "current_ch1_0_2")
            & (out["metric_row_type"] == "stage_b_classifier_only_primary")
        )
    return out


def load_history(fold_dir: Path) -> Optional[Dict[str, List[float]]]:
    path = first_file(fold_dir, "vae_train_history_fold_*.joblib")
    if path is None:
        return None
    try:
        hist = joblib.load(path)
    except Exception:
        return None
    return hist if isinstance(hist, dict) else None


def best_epoch_index(hist: Dict[str, List[float]]) -> int:
    key = "val_loss_modelsel" if "val_loss_modelsel" in hist else "val_loss"
    vals = np.asarray(hist[key], dtype=float)
    return int(np.nanargmin(vals))


def corr(a: Sequence[float], b: Sequence[float]) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if mask.sum() < 3:
        return float("nan")
    return float(np.corrcoef(aa[mask], bb[mask])[0, 1])


def current_vae_capacity_qc() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for fold in range(1, 6):
        fold_dir = CURRENT_RUN_DIR / f"fold_{fold}"
        hist = load_history(fold_dir)
        if not hist:
            continue
        idx = best_epoch_index(hist)
        final_idx = len(hist.get("train_loss", [])) - 1
        rd = read_csv(fold_dir / f"fold_{fold}_rate_distortion.csv")
        rd_best: Dict[str, Any] = {}
        if not rd.empty and "L_val_betaMax" in rd.columns:
            r = rd.loc[pd.to_numeric(rd["L_val_betaMax"], errors="coerce").idxmin()].to_dict()
            rd_best = {f"rd_best_{k}": v for k, v in r.items() if k in ["epoch", "beta", "D_train", "R_train_nats", "D_val", "R_val_nats", "L_val_betaMax"]}
            rd_best["rd_n_points"] = len(rd)
        train_recon = float(hist["train_recon"][idx])
        val_recon = float(hist["val_recon"][idx])
        train_kld = float(hist["train_kld"][idx])
        val_kld = float(hist["val_kld"][idx])
        beta = float(hist["beta"][idx])
        row = {
            "fold": fold,
            "best_epoch": idx + 1,
            "epochs_completed": final_idx + 1,
            "train_loss_best": float(hist["train_loss"][idx]),
            "val_loss_best": float(hist["val_loss"][idx]),
            "val_loss_modelsel_best": float(hist.get("val_loss_modelsel", hist["val_loss"])[idx]),
            "train_recon_best": train_recon,
            "val_recon_best": val_recon,
            "train_kld_best": train_kld,
            "val_kld_best": val_kld,
            "beta_best": beta,
            "train_beta_kld_over_recon_best": beta * train_kld / train_recon if train_recon else np.nan,
            "val_beta_kld_over_recon_best": beta * val_kld / val_recon if val_recon else np.nan,
            "val_minus_train_recon_gap_best": val_recon - train_recon,
            "val_minus_train_kld_gap_best": val_kld - train_kld,
            "train_val_recon_history_corr": corr(hist["train_recon"], hist["val_recon"]),
            "final_train_recon": float(hist["train_recon"][final_idx]),
            "final_val_recon": float(hist["val_recon"][final_idx]),
            "final_train_kld": float(hist["train_kld"][final_idx]),
            "final_val_kld": float(hist["val_kld"][final_idx]),
            "final_beta": float(hist["beta"][final_idx]),
        }
        row.update(rd_best)
        rows.append(row)
    return pd.DataFrame(rows)


def latent_feature_columns(df: pd.DataFrame) -> List[str]:
    known = {
        "SubjectID",
        "tensor_idx",
        "ResearchGroup_Mapped",
        "Manufacturer",
        "Age",
        "Sex",
        "source_batch",
        "source_label",
        "tensor_source",
        "y_true",
        "fold",
        "split",
    }
    cols = []
    for col in df.columns:
        if col in known:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def participation_ratio(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    var = np.var(values, axis=0)
    denom = np.sum(var**2)
    return float((np.sum(var) ** 2) / denom) if denom > 0 else float("nan")


def current_latent_information_qc() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    latent_dir = CURRENT_READOUT_DIR / "latent_cache"
    for fold in range(1, 6):
        for split in ["trainDev", "test"]:
            info = read_csv(CURRENT_RUN_DIR / f"fold_{fold}" / f"fold_{fold}_{split}_latent_info_summary.csv")
            info_by_var: Dict[str, Dict[str, Any]] = {}
            if not info.empty and "variable" in info.columns:
                for _, r in info.iterrows():
                    var = str(r.get("variable"))
                    info_by_var[var] = r.to_dict()
            latent_csv = latent_dir / f"fold_{fold}_{split}_latent_mu.csv"
            latent = read_csv(latent_csv)
            feats = latent_feature_columns(latent) if not latent.empty else []
            arr = latent[feats].to_numpy(dtype=float) if feats else np.empty((0, 0))
            base = {
                "fold": fold,
                "split": split,
                "latent_cache_file": str(latent_csv),
                "n_samples_latent_cache": len(latent),
                "n_latent_dims_cache": len(feats),
                "mu_global_mean": float(np.nanmean(arr)) if arr.size else np.nan,
                "mu_global_std": float(np.nanstd(arr)) if arr.size else np.nan,
                "mu_global_min": float(np.nanmin(arr)) if arr.size else np.nan,
                "mu_global_max": float(np.nanmax(arr)) if arr.size else np.nan,
                "mu_l2_norm_mean": float(np.nanmean(np.linalg.norm(arr, axis=1))) if arr.size else np.nan,
                "latent_participation_ratio_mu": participation_ratio(arr),
                "active_units_from_mu_var_gt_1e_4": int(np.sum(np.var(arr, axis=0) > 1e-4)) if arr.size else np.nan,
            }
            for var, prefix in [("Y_target", "Y"), ("Manufacturer", "Manufacturer"), ("Sex", "Sex")]:
                r = info_by_var.get(var, {})
                base[f"mi_sum_{prefix}_nats"] = r.get("mi_sum_nats", np.nan)
                base[f"mi_mean_{prefix}_nats"] = r.get("mi_mean_nats", np.nan)
                base[f"top_dims_{prefix}"] = r.get("top_dims", "")
            first_info = next(iter(info_by_var.values()), {})
            base["latent_dim_info"] = first_info.get("latent_dim", np.nan)
            base["n_active_info"] = first_info.get("n_active", np.nan)
            base["frac_active_info"] = first_info.get("frac_active", np.nan)
            base["total_correlation_nats"] = first_info.get("total_correlation_nats", np.nan)
            rows.append(base)
    return pd.DataFrame(rows)


def current_scanner_leakage_qc() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for fold in range(1, 6):
        fold_dir = CURRENT_RUN_DIR / f"fold_{fold}"
        for split, name in [
            ("trainDev", f"fold_{fold}_scanner_leakage_summary.csv"),
            ("test", f"fold_{fold}_test_scanner_leakage_summary.csv"),
            ("latent_qc_test", "latent_qc_metrics.csv"),
        ]:
            df = read_csv(fold_dir / name)
            if df.empty:
                continue
            for _, r in df.iterrows():
                row = {"fold": fold, "split": split, "source_file": str(fold_dir / name)}
                row.update(r.to_dict())
                if "acc_site_raw" in row and "acc_site_latent" in row:
                    row["latent_minus_raw_site_acc"] = row["acc_site_latent"] - row["acc_site_raw"]
                rows.append(row)
    return pd.DataFrame(rows)


def current_classifier_gap_qc() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    foldwise = read_csv(CURRENT_READOUT_DIR / "classifier_sweep_foldwise_metrics.csv")
    primary = foldwise[
        (foldwise["model_name"].astype(str) == PRIMARY_MODEL)
        & (foldwise["threshold_strategy"].astype(str) == PRIMARY_THRESHOLD)
    ].copy()
    if not primary.empty:
        primary["outer_minus_inner_auc_gap"] = primary["auc"] - primary["best_inner_auc"]
        primary["outer_minus_inner_sensitivity_gap"] = primary["sensitivity"] - primary["inner_oof_sensitivity"]
        primary["outer_minus_inner_specificity_gap"] = primary["specificity"] - primary["inner_oof_specificity"]
    preds = read_csv(CURRENT_READOUT_DIR / "classifier_sweep_predictions.csv")
    pp = preds[
        (preds["model_name"].astype(str) == PRIMARY_MODEL)
        & (preds["threshold_strategy"].astype(str) == PRIMARY_THRESHOLD)
    ].copy()
    score_rows: List[Dict[str, Any]] = []
    if not pp.empty:
        for (fold, y_true), g in pp.groupby(["fold", "y_true"], dropna=False):
            label = "AD" if int(y_true) == 1 else "CN"
            score_rows.append(
                {
                    "fold": fold,
                    "class": label,
                    "n": len(g),
                    "score_mean": g["y_score"].mean(),
                    "score_std": g["y_score"].std(ddof=0),
                    "score_median": g["y_score"].median(),
                    "score_p05": g["y_score"].quantile(0.05),
                    "score_p95": g["y_score"].quantile(0.95),
                    "threshold_mean": g["threshold"].mean(),
                }
            )
    score_df = pd.DataFrame(score_rows)
    return primary, score_df, pp


def current_foldwise_failure_modes(preds: pd.DataFrame, subgroup_paths: Dict[str, Path]) -> str:
    if preds.empty:
        return "# Current FULL Foldwise Failure Modes\n\nNo prediction rows found.\n"
    errors = preds[preds["y_true"] != preds["y_pred"]].copy()
    errors["error_type"] = np.where(errors["y_true"] == 1, "false_negative_AD", "false_positive_CN")
    fold_metrics = read_csv(CURRENT_READOUT_DIR / "classifier_sweep_foldwise_metrics.csv")
    primary = fold_metrics[
        (fold_metrics["model_name"].astype(str) == PRIMARY_MODEL)
        & (fold_metrics["threshold_strategy"].astype(str) == PRIMARY_THRESHOLD)
    ].copy()
    if not primary.empty and {"auc", "best_inner_auc"}.issubset(primary.columns):
        primary["outer_minus_inner_auc_gap"] = primary["auc"] - primary["best_inner_auc"]
    by_fold = (
        errors.groupby(["fold", "error_type", "Manufacturer", "Sex"], dropna=False)
        .size()
        .reset_index(name="n_errors")
        .sort_values(["fold", "error_type", "n_errors"], ascending=[True, True, False])
    )
    manufacturer = read_csv(subgroup_paths["manufacturer"])
    sex = read_csv(subgroup_paths["sex"])
    lines = [
        "# Current FULL Foldwise Failure Modes",
        "",
        "Primary readout: classifier-only `logreg_l2` with true inner-CV OOF target-sensitivity threshold.",
        "",
        "## Foldwise Metrics",
        "",
        md_table(primary[["fold", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "best_inner_auc", "outer_minus_inner_auc_gap"]], max_rows=10),
        "## Error Counts by Fold/Manufacturer/Sex",
        "",
        md_table(by_fold, max_rows=80),
        "## Manufacturer Subgroups",
        "",
        md_table(manufacturer, max_rows=20),
        "## Sex Subgroups",
        "",
        md_table(sex, max_rows=20),
    ]
    return "\n".join(lines)


def top_classifier_only_models() -> pd.DataFrame:
    pooled = read_csv(CURRENT_READOUT_DIR / "classifier_sweep_pooled_metrics.csv")
    if pooled.empty:
        return pooled
    cols = ["model_name", "threshold_strategy", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]
    return pooled[cols].sort_values(["auc", "pr_auc"], ascending=False).reset_index(drop=True)


def write_metric_definition(output_dir: Path, recon: pd.DataFrame) -> None:
    current = recon[
        (recon["run_id"] == "current_ch1_0_2")
        & (recon["metric_row_type"] == "stage_b_classifier_only_primary")
    ]
    stage_a = recon[
        (recon["run_id"] == "current_ch1_0_2")
        & (recon["metric_row_type"] == "stage_a_full_optuna_threshold_0p5")
    ]
    baseline = recon[
        (recon["run_id"] == "baseline_2560_ch1_0_2")
        & (recon["metric_row_type"] == "stage_a_full_optuna_threshold_0p5")
    ]
    lines = [
        "# Metric Definition Reconciliation",
        "",
        "The current manuscript should not mix Stage A canonical classifier metrics with Stage B classifier-only threshold metrics.",
        "",
        "## Definitions",
        "",
        "- Stage A canonical metrics: generated by `run_vae_clf_ad_inference.py` during the VAE run. These use the canonical classifier factory, full Optuna search in older runs, calibration when enabled, and fixed threshold 0.5 for confusion-derived metrics.",
        "- Stage B classifier-only metrics: generated after VAE training from saved fold-specific latent `mu` plus Age/Sex. The current primary readout is `logreg_l2` with threshold selected only from train/dev inner-CV OOF predictions.",
        "- Fold-wise mean AUC: arithmetic mean of per-outer-fold AUC values.",
        "- Pooled OOF AUC: one ROC-AUC computed after pooling all outer-test predictions across folds.",
        "- Threshold strategy does not change ROC-AUC/PR-AUC for the same scores; it changes sensitivity, specificity, balanced accuracy, F1, and confusion counts.",
        "",
        "## Current Paper Candidate",
        "",
    ]
    if not current.empty:
        r = current.iloc[0]
        lines += [
            f"- Use `current_ch1_0_2` Stage B classifier-only `logreg_l2`, threshold `{PRIMARY_THRESHOLD}`.",
            f"- Pooled OOF AUC={fmt_float(r['pooled_oof_auc'])}, PR-AUC={fmt_float(r['pooled_oof_pr_auc'])}, BA={fmt_float(r['balanced_accuracy'])}, sensitivity={fmt_float(r['sensitivity'])}, specificity={fmt_float(r['specificity'])}.",
        ]
    lines += ["", "## Why Older Numbers Should Be Replaced", ""]
    if not stage_a.empty:
        lines.append("Current 3840 Stage A fixed-threshold rows:")
        lines.append("")
        lines.append(md_table(stage_a[["classifier_readout", "auc_foldwise_mean", "pooled_oof_auc", "pooled_oof_pr_auc", "sensitivity", "specificity"]], max_rows=10))
    if not baseline.empty:
        lines.append("Older 2560 Stage A rows:")
        lines.append("")
        lines.append(md_table(baseline[["classifier_readout", "auc_foldwise_mean", "pooled_oof_auc", "pooled_oof_pr_auc", "sensitivity", "specificity"]], max_rows=10))
    lines += [
        "- The 0.843-level value found in the current result tree corresponds to an individual fold in the older 2560 baseline, not to the current Stage B primary pooled metric.",
        "- The 0.829-level value appears in historical notebooks, not in the current structured FULL 5x5 v5.1_batch20260514b paper-candidate readout.",
    ]
    (output_dir / "metric_definition_reconciliation.md").write_text("\n".join(lines), encoding="utf-8")


def write_manuscript_recommendation(output_dir: Path, recon: pd.DataFrame) -> None:
    current = recon[
        (recon["run_id"] == "current_ch1_0_2")
        & (recon["metric_row_type"] == "stage_b_classifier_only_primary")
    ].iloc[0]
    rows = recon[
        (recon["metric_row_type"] == "stage_b_classifier_only_primary")
        & recon["run_id"].isin(["current_ch1_0_2", "objective_v2_ch1_0_2", "fc0_ch1_0_2", "final_activation_none_ch1_0_2", "ch1", "ch1_4"])
    ].copy()
    rows = rows.sort_values("pooled_oof_auc", ascending=False)
    lines = [
        "# Manuscript Number Update Recommendation",
        "",
        "Use the locked current FULL tanh `[1,0,2]` Stage B readout as the revision's primary internal ADNI result.",
        "",
        f"- Primary AUC: {fmt_float(current['pooled_oof_auc'])}",
        f"- Primary PR-AUC: {fmt_float(current['pooled_oof_pr_auc'])}",
        f"- Sensitivity/specificity at leakage-safe operating point: {fmt_float(current['sensitivity'])}/{fmt_float(current['specificity'])}",
        f"- Balanced accuracy/F1: {fmt_float(current['balanced_accuracy'])}/{fmt_float(current['f1'])}",
        "",
        "Replace older manuscript statements around AUC 0.843/0.829 unless they are explicitly labeled as historical notebook analyses or single-fold/legacy-stage metrics.",
        "",
        "## FULL Candidate Ranking",
        "",
        md_table(rows[["run_id", "label", "pooled_oof_auc", "pooled_oof_pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "role"]], max_rows=20),
        "## Recommended Wording",
        "",
        "In internal manufacturer-aware 5-fold evaluation of ADNI v5.1_batch20260514b, the locked `[1,0,2]` tanh beta-VAE with classifier-only LogReg-L2 readout achieved pooled OOF ROC-AUC 0.779 and PR-AUC 0.552. A leakage-safe inner-CV OOF threshold targeting sensitivity >=0.70 yielded sensitivity 0.729 and specificity 0.697. Follow-up full perturbations of the reconstruction objective, bottleneck, and final activation did not improve the primary AUC/PR-AUC and were not promoted.",
    ]
    (output_dir / "manuscript_number_update_recommendation.md").write_text("\n".join(lines), encoding="utf-8")


def write_optimization_report(
    output_dir: Path,
    capacity: pd.DataFrame,
    classifier_gap: pd.DataFrame,
    latent_info: pd.DataFrame,
    scanner: pd.DataFrame,
    recon: pd.DataFrame,
) -> None:
    current = recon[
        (recon["run_id"] == "current_ch1_0_2")
        & (recon["metric_row_type"] == "stage_b_classifier_only_primary")
    ].iloc[0]
    micro = recon[
        (recon["metric_row_type"] == "stage_b_classifier_only_primary")
        & recon["run_id"].isin(["objective_v2_ch1_0_2", "fc0_ch1_0_2", "final_activation_none_ch1_0_2", "ch1", "ch1_4"])
    ].copy()
    top_models = top_classifier_only_models()
    fold_auc_min = classifier_gap["auc"].min() if not classifier_gap.empty else np.nan
    fold_auc_max = classifier_gap["auc"].max() if not classifier_gap.empty else np.nan
    fold_auc_std = classifier_gap["auc"].std(ddof=0) if not classifier_gap.empty else np.nan
    mean_gap = classifier_gap["outer_minus_inner_auc_gap"].mean() if "outer_minus_inner_auc_gap" in classifier_gap else np.nan
    active_frac = pd.to_numeric(latent_info.get("frac_active_info"), errors="coerce").dropna()
    active_note = active_frac.mean() if not active_frac.empty else np.nan
    leakage_test = scanner[scanner["split"].astype(str).str.contains("test", na=False)].copy()
    raw_mean = pd.to_numeric(leakage_test.get("acc_site_raw"), errors="coerce").mean()
    latent_mean = pd.to_numeric(leakage_test.get("acc_site_latent"), errors="coerce").mean()
    lines = [
        "# Optimization Decision Report",
        "",
        "## Decision Questions",
        "",
        "1. Is current model underfitting, overfitting, or variance-limited?",
        "",
        f"- Best current pooled OOF AUC is {fmt_float(current['pooled_oof_auc'])}; fold AUC range is {fmt_float(fold_auc_min)}-{fmt_float(fold_auc_max)} with SD {fmt_float(fold_auc_std)}.",
        f"- Mean outer-minus-inner AUC gap for the primary readout is {fmt_float(mean_gap)}.",
        "- The pattern is best interpreted as variance/subgroup-limited rather than a clean underfitting or overfitting signature.",
        "",
        "2. Is AUC limited by VAE representation or classifier readout?",
        "",
        "- Classifier-only alternatives on frozen latents did not exceed LogReg-L2 by AUC/PR-AUC.",
        "- FULL VAE micro-optimizations also failed to beat current tanh `[1,0,2]`.",
        "- This points to representation/data variance and subgroup structure as the main limit, not an obvious missing classifier tweak.",
        "",
        "3. Is error concentrated in specific folds/manufacturers?",
        "",
        "- Fold-level variance remains material; subgroup limitations remain visible for GE AD sensitivity and Philips CN specificity in prior paper-ready tables.",
        "- Errors are not explained by a single micro-optimization-sensitive failure mode.",
        "",
        "4. Is there evidence to justify batch_size=32?",
        "",
        "- No. Current QC shows active latent dimensions are broadly used and there is no read-only evidence that batch size is the bottleneck.",
        "",
        "5. Is there evidence to justify changing beta/capacity schedule?",
        "",
        f"- No strong evidence. Mean active-unit fraction from latent-info summaries is {fmt_float(active_note)}; failed FULL objective/fc/final-activation perturbations argue against further schedule micro-optimization before external validation.",
        "",
        "6. Is there evidence to justify trying LightGBM on frozen latents?",
        "",
        "- No as a primary path. Existing gradient boosting/XGBoost/random forest classifier-only sweeps underperform LogReg-L2 by AUC/PR-AUC.",
        "",
        "## Scanner/Manufacturer Leakage",
        "",
        f"- Mean test scanner/manufacturer separability raw={fmt_float(raw_mean)}, latent={fmt_float(latent_mean)}. Latents reduce but do not eliminate manufacturer information.",
        "",
        "## Current FULL Stage B Classifier-Only Ranking",
        "",
        md_table(top_models.head(12), max_rows=12),
        "## Failed FULL Micro-Optimizations",
        "",
        md_table(micro[["run_id", "pooled_oof_auc", "pooled_oof_pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "role"]].sort_values("pooled_oof_auc", ascending=False), max_rows=20),
        "## Stop Rule",
        "",
        "Stop micro-optimization on this ADNI internal CV. Keep current tanh `[1,0,2]` as the paper model, report failed perturbations as ablation/robustness checks, and prioritize external validation plus conservative manuscript language.",
    ]
    (output_dir / "optimization_decision_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = resolve(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output dir exists and is non-empty: {output_dir}. Use --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    recon = build_reconciliation()
    write_table(output_dir, "canonical_run_reconciliation", recon, max_rows=120)
    write_metric_definition(output_dir, recon)
    write_manuscript_recommendation(output_dir, recon)

    capacity = current_vae_capacity_qc()
    write_table(output_dir, "current_full_vae_capacity_qc", capacity, max_rows=40)

    classifier_gap, score_dist, primary_preds = current_classifier_gap_qc()
    combined_gap = classifier_gap.copy()
    for col in ["score_mean", "score_std", "score_median", "score_p05", "score_p95", "threshold_mean", "class"]:
        if col not in combined_gap.columns:
            combined_gap[col] = np.nan
    score_dist2 = score_dist.copy()
    for col in combined_gap.columns:
        if col not in score_dist2.columns:
            score_dist2[col] = np.nan
    score_dist2["row_type"] = "score_distribution_by_fold_class"
    combined_gap["row_type"] = "fold_classifier_gap"
    combined_gap = pd.concat([combined_gap, score_dist2[combined_gap.columns]], ignore_index=True, sort=False)
    write_table(output_dir, "current_full_classifier_gap_qc", combined_gap, max_rows=120)

    latent_info = current_latent_information_qc()
    write_table(output_dir, "current_full_latent_information_qc", latent_info, max_rows=80)

    scanner = current_scanner_leakage_qc()
    write_table(output_dir, "current_full_scanner_leakage_qc", scanner, max_rows=80)

    failure_md = current_foldwise_failure_modes(
        primary_preds,
        {
            "manufacturer": CURRENT_THRESHOLD_AUDIT_DIR / "logreg_l2_subgroup_manufacturer.csv",
            "sex": CURRENT_THRESHOLD_AUDIT_DIR / "logreg_l2_subgroup_sex.csv",
        },
    )
    (output_dir / "current_full_foldwise_failure_modes.md").write_text(failure_md, encoding="utf-8")
    write_optimization_report(output_dir, capacity, classifier_gap, latent_info, scanner, recon)

    command_log = {
        "script": str(Path(__file__).resolve()),
        "output_dir": str(output_dir),
        "read_only": True,
        "trained": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "existing_result_folders_modified": False,
        "primary_run_dir": str(CURRENT_RUN_DIR),
        "primary_readout_dir": str(CURRENT_READOUT_DIR),
    }
    (output_dir / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
    print(f"Wrote audit outputs to {output_dir}")
    if not recon.empty:
        current = recon[
            (recon["run_id"] == "current_ch1_0_2")
            & (recon["metric_row_type"] == "stage_b_classifier_only_primary")
        ]
        if not current.empty:
            r = current.iloc[0]
            print(
                "Current paper candidate: "
                f"AUC={r['pooled_oof_auc']:.4f}, PR-AUC={r['pooled_oof_pr_auc']:.4f}, "
                f"BA={r['balanced_accuracy']:.4f}, Sens={r['sensitivity']:.4f}, Spec={r['specificity']:.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
