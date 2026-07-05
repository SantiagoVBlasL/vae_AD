#!/usr/bin/env python3
"""Compare ADNI V4 [4,1,0] tanh, manufacturer-stratified sensitivity, and linear-output runs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results/revision_bspc_2026/auc_sprint_adni_only/ch4_1_0_model_decision_table"
)
DEFAULT_FROZEN_MU_LDA_SWEEP_DIR = (
    PROJECT_ROOT / "results/revision_bspc_2026/auc_sprint_adni_only/latent_representation_sweep_mu_logvar_ch4_1_0"
)

RUN_SPECS = [
    {
        "run_key": "original_tanh_main",
        "label": "Original tanh main",
        "role": "main_model",
        "decision": "main_candidate",
        "run_dir": PROJECT_ROOT / "results/revision_bspc_2026/adni_expanded_v4_beta25_ch4_1_0",
    },
    {
        "run_key": "tanh_mfrstrat_sensitivity",
        "label": "Tanh manufacturer-stratified sensitivity",
        "role": "scanner_balanced_sensitivity",
        "decision": "sensitivity_analysis",
        "run_dir": PROJECT_ROOT
        / "results/revision_bspc_2026/auc_sprint_adni_only/adni_expanded_v4_beta25_ch4_1_0_mfrstrat",
    },
    {
        "run_key": "linearout_main",
        "label": "Linear-output main",
        "role": "candidate_main_same_split",
        "decision": "negative_control",
        "run_dir": PROJECT_ROOT
        / "results/revision_bspc_2026/auc_sprint_adni_only/adni_expanded_v4_beta25_ch4_1_0_linearout",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frozen-mu-lda-sweep-dir", type=Path, default=DEFAULT_FROZEN_MU_LDA_SWEEP_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    path = resolve(path)
    path.mkdir(parents=True, exist_ok=True)
    generated = [
        "model_comparison_metrics.csv",
        "model_comparison_fold_metrics.csv",
        "latent_scanner_qc_summary.csv",
        "reconstruction_summary.csv",
        "reconstruction_channel_summaries.csv",
        "comparison_manifest.json",
        "README.md",
    ]
    existing = [path / name for name in generated if (path / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{path} already contains comparison outputs; pass --overwrite")
    if overwrite:
        for file_path in existing:
            file_path.unlink()
    return path


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def numeric(value: Any) -> float:
    return pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]


def run_realpath(run_dir: Path) -> Path:
    return run_dir.resolve() if run_dir.exists() else run_dir


def first_match(run_dir: Path, pattern: str) -> Optional[Path]:
    root = run_realpath(run_dir)
    matches = sorted(root.glob(pattern)) if root.exists() else []
    return matches[0] if matches else None


def ece_score(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return np.nan
    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        mask = (y_score >= low) & (y_score <= high) if i == n_bins - 1 else (y_score >= low) & (y_score < high)
        if mask.any():
            ece += (mask.sum() / n) * abs(float(y_score[mask].mean()) - float(y_true[mask].mean()))
    return float(ece)


def metrics_from_scores(y: np.ndarray, score: np.ndarray) -> Dict[str, Any]:
    score = np.clip(np.asarray(score, dtype=float), 1e-6, 1.0 - 1e-6)
    pred = (score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    has_both = len(set(y.tolist())) == 2
    return {
        "n": int(len(y)),
        "n_CN": int((y == 0).sum()),
        "n_AD": int((y == 1).sum()),
        "roc_auc": roc_auc_score(y, score) if has_both else np.nan,
        "pr_auc": average_precision_score(y, score) if has_both else np.nan,
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "sensitivity_AD": tp / (tp + fn) if tp + fn else np.nan,
        "specificity_CN": tn / (tn + fp) if tn + fp else np.nan,
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "brier": brier_score_loss(y, score),
        "ece_10_bins": ece_score(y, score),
        "threshold_0p5_TN": int(tn),
        "threshold_0p5_FP": int(fp),
        "threshold_0p5_FN": int(fn),
        "threshold_0p5_TP": int(tp),
    }


def load_pipeline_fold_metrics(run_dir: Path) -> pd.DataFrame:
    path = first_match(run_dir, "all_folds_metrics_MULTI_*.csv")
    if path is None:
        return pd.DataFrame()
    df = safe_read_csv(path)
    if df.empty:
        return df
    clf_col = "actual_classifier_type" if "actual_classifier_type" in df.columns else "classifier"
    df = df.rename(columns={clf_col: "classifier", "f1_score": "f1"}).copy()
    keep = [
        "fold",
        "classifier",
        "auc",
        "pr_auc",
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
    ]
    for col in keep:
        if col not in df.columns:
            df[col] = np.nan
    return df[keep]


def load_predictions(run_dir: Path) -> pd.DataFrame:
    path = first_match(run_dir, "all_folds_clf_predictions_MULTI_*.csv")
    if path is None:
        return pd.DataFrame()
    df = safe_read_csv(path)
    if df.empty:
        return df
    clf_col = "classifier_type" if "classifier_type" in df.columns else "classifier"
    df = df.rename(columns={clf_col: "classifier"}).copy()
    score_col = "y_score_final" if "y_score_final" in df.columns else "y_score_cal" if "y_score_cal" in df.columns else "y_score_raw"
    required = ["fold", "classifier", "y_true", score_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Prediction table {path} is missing columns: {missing}")
    out = df[["fold", "classifier", "SubjectID", "y_true", score_col]].copy()
    out = out.rename(columns={score_col: "y_score"})
    out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce").astype("Int64")
    out["y_score"] = pd.to_numeric(out["y_score"], errors="coerce")
    return out.dropna(subset=["y_true", "y_score"]).copy()


def fold_probability_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if preds.empty:
        return pd.DataFrame()
    for (fold, classifier), group in preds.groupby(["fold", "classifier"], sort=True):
        y = group["y_true"].astype(int).to_numpy()
        score = group["y_score"].astype(float).to_numpy()
        row = {"fold": int(fold), "classifier": classifier}
        row.update(metrics_from_scores(y, score))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_model_metrics(
    run_key: str,
    label: str,
    role: str,
    decision: str,
    run_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pipeline = load_pipeline_fold_metrics(run_dir)
    preds = load_predictions(run_dir)
    prob_folds = fold_probability_metrics(preds)
    if pipeline.empty and prob_folds.empty:
        row = {
            "run_key": run_key,
            "label": label,
            "role": role,
            "decision": decision,
            "run_dir": str(run_dir),
            "run_realpath": str(run_realpath(run_dir)),
            "status": "missing_or_incomplete",
            "note": "No all_folds metrics/prediction CSVs found.",
        }
        return pd.DataFrame([row]), pd.DataFrame()

    if not pipeline.empty:
        fold_df = pipeline.merge(
            prob_folds[["fold", "classifier", "brier", "ece_10_bins"]] if not prob_folds.empty else pd.DataFrame(),
            on=["fold", "classifier"],
            how="left",
        )
    else:
        fold_df = prob_folds.rename(
            columns={
                "roc_auc": "auc",
                "sensitivity_AD": "sensitivity",
                "specificity_CN": "specificity",
            }
        )

    fold_df.insert(0, "run_key", run_key)
    fold_df.insert(1, "label", label)
    fold_df.insert(2, "role", role)
    fold_df.insert(3, "decision", decision)

    rows: List[Dict[str, Any]] = []
    for classifier, group in fold_df.groupby("classifier", sort=True):
        pred_group = preds[preds["classifier"] == classifier]
        pooled = (
            metrics_from_scores(
                pred_group["y_true"].astype(int).to_numpy(),
                pred_group["y_score"].astype(float).to_numpy(),
            )
            if not pred_group.empty
            else {}
        )
        row: Dict[str, Any] = {
            "run_key": run_key,
            "label": label,
            "role": role,
            "decision": decision,
            "run_dir": str(run_dir),
            "run_realpath": str(run_realpath(run_dir)),
            "status": "complete",
            "classifier": classifier,
            "n_folds": int(group["fold"].nunique()),
            "auc_mean_fold": float(pd.to_numeric(group["auc"], errors="coerce").mean()),
            "auc_sd_fold": float(pd.to_numeric(group["auc"], errors="coerce").std(ddof=1)),
            "pr_auc_mean_fold": float(pd.to_numeric(group["pr_auc"], errors="coerce").mean()),
            "pr_auc_sd_fold": float(pd.to_numeric(group["pr_auc"], errors="coerce").std(ddof=1)),
            "balanced_accuracy_mean_fold": float(pd.to_numeric(group["balanced_accuracy"], errors="coerce").mean()),
            "sensitivity_AD_mean_fold": float(pd.to_numeric(group["sensitivity"], errors="coerce").mean()),
            "specificity_CN_mean_fold": float(pd.to_numeric(group["specificity"], errors="coerce").mean()),
            "f1_mean_fold": float(pd.to_numeric(group["f1"], errors="coerce").mean()),
            "brier_mean_fold": float(pd.to_numeric(group.get("brier"), errors="coerce").mean()),
            "ece_10_bins_mean_fold": float(pd.to_numeric(group.get("ece_10_bins"), errors="coerce").mean()),
        }
        for key, value in pooled.items():
            row[f"{key}_pooled"] = value
        rows.append(row)
    return pd.DataFrame(rows), fold_df


def parse_latent_info(run_key: str, run_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    root = run_realpath(run_dir)
    for fold in range(1, 6):
        path = root / f"fold_{fold}/fold_{fold}_test_latent_info_summary.csv"
        df = safe_read_csv(path)
        if df.empty:
            continue
        y = df[df["variable"].astype(str) == "Y_target"]
        mfr = df[df["variable"].astype(str) == "Manufacturer"]
        mi_y = numeric(y["mi_sum_nats"].iloc[0]) if not y.empty else np.nan
        mi_mfr = numeric(mfr["mi_sum_nats"].iloc[0]) if not mfr.empty else np.nan
        base = y if not y.empty else df
        rows.append(
            {
                "run_key": run_key,
                "fold": fold,
                "MI_Y_test": mi_y,
                "MI_Manufacturer_test": mi_mfr,
                "MI_Manufacturer_over_Y_test": mi_mfr / mi_y if pd.notna(mi_y) and mi_y else np.nan,
                "active_units_test": numeric(base["n_active"].iloc[0]) if "n_active" in base.columns else np.nan,
                "TC_test": numeric(base["total_correlation_nats"].iloc[0]) if "total_correlation_nats" in base.columns else np.nan,
            }
        )
    return pd.DataFrame(rows)


def parse_scanner_leakage(run_key: str, run_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    root = run_realpath(run_dir)
    for fold in range(1, 6):
        path = root / f"fold_{fold}/fold_{fold}_test_scanner_leakage_summary.csv"
        df = safe_read_csv(path)
        if df.empty:
            continue
        row = df.iloc[0]
        rows.append(
            {
                "run_key": run_key,
                "fold": fold,
                "acc_site_latent_test": numeric(row.get("acc_site_latent")),
                "acc_site_raw_test": numeric(row.get("acc_site_raw")),
                "site_chance_level_test": numeric(row.get("chance_level")),
            }
        )
    return pd.DataFrame(rows)


def parse_reconstruction(run_key: str, run_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    channel_rows: List[Dict[str, Any]] = []
    root = run_realpath(run_dir)
    for fold in range(1, 6):
        rd = safe_read_csv(root / f"fold_{fold}/fold_{fold}_rate_distortion.csv")
        row: Dict[str, Any] = {"run_key": run_key, "fold": fold}
        if not rd.empty:
            val_loss = pd.to_numeric(rd.get("L_val_betaMax"), errors="coerce")
            idx = int(val_loss.idxmin()) if val_loss.notna().any() else int(pd.to_numeric(rd.get("D_val"), errors="coerce").idxmin())
            best = rd.loc[idx]
            row.update(
                {
                    "best_epoch": numeric(best.get("epoch")),
                    "early_stop_epoch": numeric(pd.to_numeric(rd.get("epoch"), errors="coerce").max()),
                    "recon_train_at_best": numeric(best.get("D_train")),
                    "recon_val_at_best": numeric(best.get("D_val")),
                    "reconstruction_gap_val_minus_train": numeric(best.get("D_val")) - numeric(best.get("D_train")),
                    "kl_train_at_best": numeric(best.get("R_train_nats")),
                    "kl_val_at_best": numeric(best.get("R_val_nats")),
                }
            )
        rows.append(row)

        recon = safe_read_csv(root / f"fold_{fold}/fold_{fold}_dist_recon.csv")
        norm = safe_read_csv(root / f"fold_{fold}/fold_{fold}_dist_norm.csv")
        if recon.empty or "channel" not in recon.columns:
            continue
        norm_lookup = norm.set_index("channel").to_dict("index") if not norm.empty and "channel" in norm.columns else {}
        for _, r in recon.iterrows():
            channel = str(r["channel"])
            n = norm_lookup.get(channel, {})
            channel_rows.append(
                {
                    "run_key": run_key,
                    "fold": fold,
                    "channel": channel,
                    "recon_mean": numeric(r.get("mean")),
                    "recon_std": numeric(r.get("std")),
                    "recon_min": numeric(r.get("min")),
                    "recon_max": numeric(r.get("max")),
                    "norm_mean": numeric(n.get("mean")),
                    "norm_std": numeric(n.get("std")),
                    "norm_min": numeric(n.get("min")),
                    "norm_max": numeric(n.get("max")),
                    "recon_minus_norm_mean": numeric(r.get("mean")) - numeric(n.get("mean")) if n else np.nan,
                    "recon_hits_tanh_boundary": bool(numeric(r.get("min")) <= -0.999 or numeric(r.get("max")) >= 0.999),
                    "norm_outside_tanh_range": bool(numeric(n.get("min")) < -1.0 or numeric(n.get("max")) > 1.0) if n else np.nan,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(channel_rows)


def artifact_time_span_sec(run_dir: Path) -> float:
    root = run_realpath(run_dir)
    if not root.exists():
        return np.nan
    files = [p for p in root.rglob("*") if p.is_file()]
    if not files:
        return np.nan
    mtimes = [p.stat().st_mtime for p in files]
    return float(max(mtimes) - min(mtimes))


def summarize_qc(latent: pd.DataFrame, scanner: pd.DataFrame) -> pd.DataFrame:
    qc = latent.merge(scanner, on=["run_key", "fold"], how="outer")
    if qc.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for run_key, group in qc.groupby("run_key", sort=False):
        row = {"run_key": run_key, "n_qc_folds": int(group["fold"].nunique())}
        for col in [
            "MI_Y_test",
            "MI_Manufacturer_test",
            "MI_Manufacturer_over_Y_test",
            "active_units_test",
            "TC_test",
            "acc_site_latent_test",
            "acc_site_raw_test",
            "site_chance_level_test",
        ]:
            row[f"{col}_mean"] = float(pd.to_numeric(group.get(col), errors="coerce").mean())
            row[f"{col}_sd"] = float(pd.to_numeric(group.get(col), errors="coerce").std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_reconstruction(recon: pd.DataFrame) -> pd.DataFrame:
    if recon.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for run_key, group in recon.groupby("run_key", sort=False):
        row = {"run_key": run_key, "n_recon_folds": int(group["fold"].nunique())}
        for col in [
            "best_epoch",
            "early_stop_epoch",
            "recon_train_at_best",
            "recon_val_at_best",
            "reconstruction_gap_val_minus_train",
            "kl_train_at_best",
            "kl_val_at_best",
        ]:
            row[f"{col}_mean"] = float(pd.to_numeric(group.get(col), errors="coerce").mean())
            row[f"{col}_sd"] = float(pd.to_numeric(group.get(col), errors="coerce").std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def load_frozen_mu_lda_candidate(sweep_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics_path = sweep_dir / "latent_representation_sweep_metrics.csv"
    fold_path = sweep_dir / "latent_representation_sweep_fold_metrics.csv"
    if not metrics_path.exists() or not fold_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    metrics = safe_read_csv(metrics_path)
    folds = safe_read_csv(fold_path)
    if metrics.empty or folds.empty:
        return pd.DataFrame(), pd.DataFrame()

    row = metrics[
        (metrics.get("representation", pd.Series(dtype=str)).astype(str) == "mu")
        & (metrics.get("classifier", pd.Series(dtype=str)).astype(str) == "lda_shrinkage")
        & (metrics.get("status", pd.Series(dtype=str)).astype(str) == "ok")
    ]
    if row.empty:
        return pd.DataFrame(), pd.DataFrame()
    r = row.iloc[0]
    out = pd.DataFrame(
        [
            {
                "run_key": "frozen_mu_lda_shrinkage",
                "label": "Frozen tanh mu + LDA shrinkage",
                "role": "downstream_only_candidate",
                "decision": "main_candidate",
                "run_dir": str(sweep_dir),
                "run_realpath": str(run_realpath(sweep_dir)),
                "status": "complete",
                "classifier": "lda_shrinkage",
                "n_folds": numeric(r.get("n_folds")),
                "auc_mean_fold": numeric(r.get("roc_auc_mean_fold")),
                "auc_sd_fold": numeric(r.get("roc_auc_sd_fold")),
                "pr_auc_mean_fold": numeric(r.get("pr_auc_mean_fold")),
                "pr_auc_sd_fold": numeric(r.get("pr_auc_sd_fold")),
                "balanced_accuracy_mean_fold": numeric(r.get("balanced_accuracy_mean_fold")),
                "sensitivity_AD_mean_fold": numeric(r.get("sensitivity_mean_fold")),
                "specificity_CN_mean_fold": numeric(r.get("specificity_mean_fold")),
                "f1_mean_fold": numeric(r.get("f1_mean_fold")),
                "brier_mean_fold": numeric(r.get("brier_mean_fold")),
                "ece_10_bins_mean_fold": numeric(r.get("ece_mean_fold")),
                "n_pooled": numeric(r.get("n_total")),
                "n_CN_pooled": numeric(r.get("n_CN")),
                "n_AD_pooled": numeric(r.get("n_AD")),
                "roc_auc_pooled": numeric(r.get("roc_auc_pooled")),
                "pr_auc_pooled": numeric(r.get("pr_auc_pooled")),
                "balanced_accuracy_pooled": numeric(r.get("balanced_accuracy_pooled")),
                "sensitivity_AD_pooled": numeric(r.get("sensitivity_pooled")),
                "specificity_CN_pooled": numeric(r.get("specificity_pooled")),
                "f1_pooled": numeric(r.get("f1_pooled")),
                "brier_pooled": numeric(r.get("brier_pooled")),
                "ece_10_bins_pooled": numeric(r.get("ece_pooled")),
                "note": "Downstream-only frozen-latent classifier result; no VAE retraining.",
            }
        ]
    )

    fold_sub = folds[
        (folds.get("representation", pd.Series(dtype=str)).astype(str) == "mu")
        & (folds.get("classifier", pd.Series(dtype=str)).astype(str) == "lda_shrinkage")
    ].copy()
    if not fold_sub.empty:
        fold_sub.insert(0, "run_key", "frozen_mu_lda_shrinkage")
        fold_sub.insert(1, "label", "Frozen tanh mu + LDA shrinkage")
        fold_sub.insert(2, "role", "downstream_only_candidate")
        fold_sub.insert(3, "decision", "main_candidate")
        fold_sub = fold_sub.rename(
            columns={
                "roc_auc": "auc",
                "sensitivity": "sensitivity_AD",
                "specificity": "specificity_CN",
            }
        )
    return out, fold_sub


def markdown_table(df: pd.DataFrame, columns: List[str], n: int = 20) -> str:
    if df.empty:
        return "_No rows available._"
    display = df.copy()
    for col in columns:
        if col not in display.columns:
            display[col] = np.nan
    display = display[columns].head(n).copy()
    for col in display.columns:
        if pd.api.types.is_numeric_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            display[col] = display[col].fillna("").astype(str)
    header = "| " + " | ".join(display.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in display.to_numpy()]
    return "\n".join([header, sep, *rows])


def write_readme(
    path: Path,
    metrics: pd.DataFrame,
    qc_summary: pd.DataFrame,
    recon_summary: pd.DataFrame,
    unavailable: List[Dict[str, Any]],
) -> None:
    ok = metrics[metrics["status"] == "complete"].copy()
    best_cols = [
        "run_key",
        "decision",
        "role",
        "classifier",
        "auc_mean_fold",
        "auc_sd_fold",
        "roc_auc_pooled",
        "pr_auc_mean_fold",
        "balanced_accuracy_mean_fold",
        "sensitivity_AD_mean_fold",
        "specificity_CN_mean_fold",
        "f1_mean_fold",
        "brier_mean_fold",
        "ece_10_bins_mean_fold",
        "artifact_time_span_sec",
    ]
    if not ok.empty:
        ok = ok.sort_values(["decision", "auc_mean_fold"], ascending=[True, False])
    lines = [
        "# V4 [4,1,0] Model Decision Table",
        "",
        "## Scope",
        "Compares the original tanh main run, the completed linear-output run, the manufacturer-stratified scanner-balanced sensitivity run, and the frozen-latent mu + LDA shrinkage downstream-only result when available.",
        "",
        "- ADNI only.",
        "- Manufacturer-stratified run is reported as sensitivity analysis, not the main model.",
        "- Manufacturer is not used as a predictive feature in these run configs.",
        "- Pooled metrics are recomputed from held-out fold predictions using `y_score_final` when available.",
        "- Frozen-latent rows do not retrain or alter the CVAE; their VAE QC/reconstruction context is inherited from the tanh baseline.",
        "",
        "## Interpretation",
        "- Linearout did not validate the tanh-saturation hypothesis as an AUC-improvement path.",
        "- Manufacturer-stratified splitting is useful as scanner-balanced sensitivity analysis, but it is not the main AUC-improvement path.",
        "- Logvar/posterior uncertainty did not help in the prior no-retraining sweep.",
        "- Keep the tanh [4,1,0] baseline as the current main CVAE model.",
        "- Next minimal step: add LDA shrinkage as an official downstream classifier candidate and/or perform checkpoint selection using only train/dev downstream validation.",
        "",
        "## Classifier Metrics",
        "",
        markdown_table(ok, best_cols),
        "",
        "## Latent/Scanner QC",
        "",
        markdown_table(
            qc_summary,
            [
                "run_key",
                "n_qc_folds",
                "MI_Manufacturer_over_Y_test_mean",
                "acc_site_latent_test_mean",
                "active_units_test_mean",
                "TC_test_mean",
            ],
        ),
        "",
        "## Reconstruction Summary",
        "",
        markdown_table(
            recon_summary,
            [
                "run_key",
                "n_recon_folds",
                "best_epoch_mean",
                "early_stop_epoch_mean",
                "recon_train_at_best_mean",
                "recon_val_at_best_mean",
                "reconstruction_gap_val_minus_train_mean",
            ],
        ),
    ]
    if unavailable:
        lines.extend(["", "## Unavailable Runs", ""])
        for item in unavailable:
            lines.append(f"- `{item['run_key']}`: {item['note']} Path: `{item['run_dir']}`")
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "Only compare `linearout_main` against `original_tanh_main` as the controlled final-activation experiment. Treat `tanh_mfrstrat_sensitivity` as scanner-balance sensitivity context, and treat `frozen_mu_lda_shrinkage` as a downstream-only classifier candidate.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    outdir = prepare_output_dir(args.output_dir, args.overwrite)

    metric_frames: List[pd.DataFrame] = []
    fold_frames: List[pd.DataFrame] = []
    latent_frames: List[pd.DataFrame] = []
    scanner_frames: List[pd.DataFrame] = []
    recon_frames: List[pd.DataFrame] = []
    channel_frames: List[pd.DataFrame] = []
    unavailable: List[Dict[str, Any]] = []

    for spec in RUN_SPECS:
        run_key = spec["run_key"]
        run_dir = resolve(spec["run_dir"])
        metrics, folds = summarize_model_metrics(run_key, spec["label"], spec["role"], spec["decision"], run_dir)
        metrics["artifact_time_span_sec"] = artifact_time_span_sec(run_dir)
        metric_frames.append(metrics)
        if not folds.empty:
            folds["artifact_time_span_sec"] = artifact_time_span_sec(run_dir)
            fold_frames.append(folds)
        if (metrics["status"] != "complete").all():
            unavailable.append({"run_key": run_key, "run_dir": str(run_dir), "note": str(metrics.iloc[0].get("note"))})
            continue
        latent_frames.append(parse_latent_info(run_key, run_dir))
        scanner_frames.append(parse_scanner_leakage(run_key, run_dir))
        recon, channels = parse_reconstruction(run_key, run_dir)
        recon_frames.append(recon)
        channel_frames.append(channels)

    frozen_metrics, frozen_folds = load_frozen_mu_lda_candidate(resolve(args.frozen_mu_lda_sweep_dir))
    if not frozen_metrics.empty:
        frozen_metrics["artifact_time_span_sec"] = artifact_time_span_sec(resolve(args.frozen_mu_lda_sweep_dir))
        metric_frames.append(frozen_metrics)
    if not frozen_folds.empty:
        frozen_folds["artifact_time_span_sec"] = artifact_time_span_sec(resolve(args.frozen_mu_lda_sweep_dir))
        fold_frames.append(frozen_folds)

    metrics_df = pd.concat(metric_frames, ignore_index=True, sort=False)
    fold_df = pd.concat(fold_frames, ignore_index=True, sort=False) if fold_frames else pd.DataFrame()
    latent_df = pd.concat(latent_frames, ignore_index=True, sort=False) if latent_frames else pd.DataFrame()
    scanner_df = pd.concat(scanner_frames, ignore_index=True, sort=False) if scanner_frames else pd.DataFrame()
    recon_df = pd.concat(recon_frames, ignore_index=True, sort=False) if recon_frames else pd.DataFrame()
    channel_df = pd.concat(channel_frames, ignore_index=True, sort=False) if channel_frames else pd.DataFrame()

    qc_summary = summarize_qc(latent_df, scanner_df)
    recon_summary = summarize_reconstruction(recon_df)

    if not frozen_metrics.empty:
        for frame in [qc_summary, recon_summary]:
            if not frame.empty and (frame["run_key"] == "original_tanh_main").any():
                inherited = frame[frame["run_key"] == "original_tanh_main"].copy()
                inherited["run_key"] = "frozen_mu_lda_shrinkage"
                frame.drop(frame[frame["run_key"] == "frozen_mu_lda_shrinkage"].index, inplace=True, errors="ignore")
                frame.loc[len(frame)] = inherited.iloc[0]
        if not channel_df.empty and (channel_df["run_key"] == "original_tanh_main").any():
            inherited_channels = channel_df[channel_df["run_key"] == "original_tanh_main"].copy()
            inherited_channels["run_key"] = "frozen_mu_lda_shrinkage"
            channel_df = pd.concat([channel_df, inherited_channels], ignore_index=True, sort=False)

    metrics_df = metrics_df.merge(qc_summary, on="run_key", how="left").merge(recon_summary, on="run_key", how="left")

    metrics_df.to_csv(outdir / "model_comparison_metrics.csv", index=False)
    fold_df.to_csv(outdir / "model_comparison_fold_metrics.csv", index=False)
    qc_summary.to_csv(outdir / "latent_scanner_qc_summary.csv", index=False)
    recon_summary.to_csv(outdir / "reconstruction_summary.csv", index=False)
    channel_df.to_csv(outdir / "reconstruction_channel_summaries.csv", index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(outdir),
        "run_specs": [
            {**{k: v for k, v in spec.items() if k != "run_dir"}, "run_dir": str(resolve(spec["run_dir"]))}
            for spec in RUN_SPECS
        ],
        "frozen_mu_lda_sweep_dir": str(resolve(args.frozen_mu_lda_sweep_dir)),
        "unavailable": unavailable,
        "manufacturer_stratified_role": "scanner_balanced_sensitivity_not_main_model",
    }
    (outdir / "comparison_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(outdir / "README.md", metrics_df, qc_summary, recon_summary, unavailable)

    print(f"Comparison written to: {outdir}")
    cols = ["run_key", "decision", "role", "classifier", "status", "auc_mean_fold", "auc_sd_fold", "roc_auc_pooled"]
    print(metrics_df[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
