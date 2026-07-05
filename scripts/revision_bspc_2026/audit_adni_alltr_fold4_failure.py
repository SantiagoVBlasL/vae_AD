#!/usr/bin/env python3
"""Fold-4 failure audit for exploratory ADNI all-timepoints run.

This is a read-only audit: it consumes existing Stage B readouts, metadata,
latent caches, and QC summaries, then writes a separate audit package.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"

LOCKED_RUN = RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5"
LOCKED_READOUT = LOCKED_RUN / "classifier_only_readout"
ALLTR_RUN = RESULTS / "adni_all_available_timepoints_ch1_0_2_exploratory_full_5x5"
ALLTR_READOUT = ALLTR_RUN / "classifier_only_readout"
ALLTR_METADATA = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_all_available_timepoints_ch1_0_2_exploratory/"
    "training_ready_metadata_adni_all_available_timepoints_ch1_0_2_exploratory.csv"
)
DEFAULT_OUTPUT = RESULTS / "adni_all_available_timepoints_fold4_failure_audit"

PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
FIXED_THRESHOLD = "fixed_0p5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--locked-run-dir", type=Path, default=LOCKED_RUN)
    parser.add_argument("--locked-readout-dir", type=Path, default=LOCKED_READOUT)
    parser.add_argument("--alltr-run-dir", type=Path, default=ALLTR_RUN)
    parser.add_argument("--alltr-readout-dir", type=Path, default=ALLTR_READOUT)
    parser.add_argument("--metadata", type=Path, default=ALLTR_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_table(df: pd.DataFrame, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    (out_dir / f"{stem}.md").write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")


def q25(x: pd.Series) -> float:
    return float(pd.to_numeric(x, errors="coerce").quantile(0.25))


def q75(x: pd.Series) -> float:
    return float(pd.to_numeric(x, errors="coerce").quantile(0.75))


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def binary_metrics(df: pd.DataFrame) -> dict[str, Any]:
    y = df["y_true"].astype(int).to_numpy()
    score = df["y_score"].astype(float).to_numpy()
    pred = df["y_pred"].astype(int).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)
    return {
        "n": int(len(df)),
        "n_cn": int((y == 0).sum()),
        "n_ad": int((y == 1).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": float(np.nanmean([sens, spec])),
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
    }


def add_error_type(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    conditions = [
        (out["y_true"].eq(0) & out["y_pred"].eq(0), "TN"),
        (out["y_true"].eq(0) & out["y_pred"].eq(1), "FP"),
        (out["y_true"].eq(1) & out["y_pred"].eq(0), "FN"),
        (out["y_true"].eq(1) & out["y_pred"].eq(1), "TP"),
    ]
    out["error_type"] = ""
    for mask, label in conditions:
        out.loc[mask, "error_type"] = label
    return out


def load_predictions(readout_dir: Path, run_id: str, label: str, threshold: str = PRIMARY_THRESHOLD) -> pd.DataFrame:
    pred = pd.read_csv(readout_dir / "classifier_sweep_predictions.csv")
    mask = pred["model_name"].astype(str).eq(PRIMARY_MODEL) & pred["threshold_strategy"].astype(str).eq(threshold)
    if "readout_feature_set" in pred.columns:
        mask &= pred["readout_feature_set"].astype(str).eq("z_plus_age_sex")
    out = pred.loc[mask].copy()
    out.insert(0, "run_id", run_id)
    out.insert(1, "run_label", label)
    return add_error_type(out)


def load_foldwise(readout_dir: Path, run_id: str, label: str) -> pd.DataFrame:
    fold = pd.read_csv(readout_dir / "classifier_sweep_foldwise_metrics.csv")
    mask = fold["model_name"].astype(str).eq(PRIMARY_MODEL) & fold["threshold_strategy"].astype(str).isin(
        [PRIMARY_THRESHOLD, FIXED_THRESHOLD]
    )
    if "readout_feature_set" in fold.columns:
        mask &= fold["readout_feature_set"].astype(str).eq("z_plus_age_sex")
    out = fold.loc[mask].copy()
    out.insert(0, "run_id", run_id)
    out.insert(1, "run_label", label)
    out["fold_group"] = np.where(out["fold"].astype(int).eq(4), "fold4", "other_folds")
    return out


def load_metadata(path: Path) -> pd.DataFrame:
    meta = pd.read_csv(path)
    if "original_n_TR" not in meta.columns and "original_n_timepoints" in meta.columns:
        meta["original_n_TR"] = meta["original_n_timepoints"]
    if "SiteCode" not in meta.columns:
        meta["SiteCode"] = meta["SubjectID"].astype(str).str.slice(0, 3)
    return meta


def augment_predictions(pred: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "SubjectID",
        "original_n_TR",
        "original_n_timepoints",
        "locked_n_timepoints_used",
        "branch_n_timepoints_used",
        "SiteCode",
        "Site3",
        "Age",
        "AgeBin",
        "Sex",
        "Manufacturer",
        "ResearchGroup_Mapped",
    ]
    keep = [c for c in keep if c in meta.columns]
    merged = pred.merge(meta[keep].drop_duplicates("SubjectID"), on="SubjectID", how="left", suffixes=("", "_meta"))
    for col in ["Manufacturer", "Age", "Sex", "ResearchGroup_Mapped"]:
        meta_col = f"{col}_meta"
        if meta_col in merged.columns:
            merged[col] = merged[col].where(merged[col].notna(), merged[meta_col])
            merged = merged.drop(columns=[meta_col])
    if "SiteCode" not in merged.columns:
        merged["SiteCode"] = merged["SubjectID"].astype(str).str.slice(0, 3)
    return merged


def fold_metrics_summary(foldwise: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "run_id",
        "run_label",
        "fold",
        "fold_group",
        "threshold_strategy",
        "threshold",
        "n",
        "n_cn",
        "n_ad",
        "tn",
        "fp",
        "fn",
        "tp",
        "auc",
        "pr_auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "best_inner_auc",
        "best_params",
    ]
    return foldwise[[c for c in cols if c in foldwise.columns]].sort_values(["threshold_strategy", "run_id", "fold"])


def score_distribution(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, grp in pred.groupby(["run_id", "run_label", "fold", "fold_group", "ResearchGroup_Mapped"], dropna=False):
        run_id, run_label, fold, fold_group, diagnosis = keys
        score = pd.to_numeric(grp["y_score"], errors="coerce")
        threshold = pd.to_numeric(grp["threshold"], errors="coerce")
        rows.append(
            {
                "run_id": run_id,
                "run_label": run_label,
                "fold": fold,
                "fold_group": fold_group,
                "diagnosis": diagnosis,
                "n": int(len(grp)),
                "score_mean": float(score.mean()),
                "score_sd": float(score.std(ddof=1)),
                "score_median": float(score.median()),
                "score_q25": q25(score),
                "score_q75": q75(score),
                "score_min": float(score.min()),
                "score_max": float(score.max()),
                "threshold": float(threshold.iloc[0]) if len(threshold) else np.nan,
                "fraction_above_threshold": float((score >= threshold.iloc[0]).mean()) if len(threshold) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["run_id", "fold", "diagnosis"])


def error_transition(locked: pd.DataFrame, alltr: pd.DataFrame) -> pd.DataFrame:
    lcols = [
        "SubjectID",
        "fold",
        "y_true",
        "y_score",
        "y_pred",
        "threshold",
        "error_type",
        "ResearchGroup_Mapped",
        "Manufacturer",
        "Age",
        "Sex",
        "SiteCode",
        "original_n_TR",
    ]
    acols = lcols.copy()
    left = locked[[c for c in lcols if c in locked.columns]].copy().add_suffix("_locked")
    right = alltr[[c for c in acols if c in alltr.columns]].copy().add_suffix("_alltr")
    out = left.merge(right, left_on="SubjectID_locked", right_on="SubjectID_alltr", how="inner")
    out["SubjectID"] = out["SubjectID_locked"]
    out["y_true"] = out["y_true_locked"]
    out["fold"] = out["fold_alltr"]
    out["ResearchGroup_Mapped"] = out.get("ResearchGroup_Mapped_alltr", out.get("ResearchGroup_Mapped_locked"))
    out["Manufacturer"] = out.get("Manufacturer_alltr", out.get("Manufacturer_locked"))
    out["SiteCode"] = out.get("SiteCode_alltr", out.get("SiteCode_locked"))
    out["Age"] = out.get("Age_alltr", out.get("Age_locked"))
    out["Sex"] = out.get("Sex_alltr", out.get("Sex_locked"))
    out["original_n_TR"] = out.get("original_n_TR_alltr", out.get("original_n_TR_locked"))
    locked_correct = out["error_type_locked"].isin(["TN", "TP"])
    alltr_correct = out["error_type_alltr"].isin(["TN", "TP"])
    out["transition"] = np.select(
        [
            locked_correct & alltr_correct,
            (~locked_correct) & alltr_correct,
            locked_correct & (~alltr_correct),
            (~locked_correct) & (~alltr_correct) & out["error_type_locked"].eq("FP") & out["error_type_alltr"].eq("FP"),
            (~locked_correct) & (~alltr_correct) & out["error_type_locked"].eq("FN") & out["error_type_alltr"].eq("FN"),
        ],
        ["stable_correct", "fixed_by_alltr", "new_error_alltr", "stable_fp", "stable_fn"],
        default="both_error_changed",
    )
    out["fold_group"] = np.where(out["fold"].astype(int).eq(4), "fold4", "other_folds")
    keep = [
        "SubjectID",
        "fold",
        "fold_group",
        "ResearchGroup_Mapped",
        "y_true",
        "Manufacturer",
        "SiteCode",
        "Age",
        "Sex",
        "original_n_TR",
        "error_type_locked",
        "y_score_locked",
        "threshold_locked",
        "error_type_alltr",
        "y_score_alltr",
        "threshold_alltr",
        "transition",
    ]
    return out[[c for c in keep if c in out.columns]].sort_values(["fold", "transition", "SubjectID"])


def error_transition_summary(transitions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, grp in transitions.groupby(["fold_group", "transition", "ResearchGroup_Mapped"], dropna=False):
        fold_group, transition, diagnosis = keys
        rows.append(
            {
                "fold_group": fold_group,
                "transition": transition,
                "diagnosis": diagnosis,
                "n": int(len(grp)),
                "n_fold4": int(grp["fold"].eq(4).sum()),
                "mean_n_TR": float(pd.to_numeric(grp["original_n_TR"], errors="coerce").mean()),
                "median_n_TR": float(pd.to_numeric(grp["original_n_TR"], errors="coerce").median()),
                "mean_age": float(pd.to_numeric(grp["Age"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["fold_group", "transition", "diagnosis"])


def numeric_distribution(df: pd.DataFrame, value_col: str, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, grp in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        vals = pd.to_numeric(grp[value_col], errors="coerce")
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update(
            {
                "metric": value_col,
                "n": int(vals.notna().sum()),
                "mean": float(vals.mean()),
                "sd": float(vals.std(ddof=1)),
                "median": float(vals.median()),
                "q25": q25(vals),
                "q75": q75(vals),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "gt_140": int((vals > 140).sum()) if value_col == "original_n_TR" else np.nan,
                "gt_160": int((vals > 160).sum()) if value_col == "original_n_TR" else np.nan,
                "gt_180": int((vals > 180).sum()) if value_col == "original_n_TR" else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def metadata_summaries(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = pred.drop_duplicates(["SubjectID", "fold"]).copy()
    base["fold_group"] = np.where(base["fold"].astype(int).eq(4), "fold4", "other_folds")
    numeric = pd.concat(
        [
            numeric_distribution(base, "original_n_TR", ["fold", "fold_group", "ResearchGroup_Mapped"]),
            numeric_distribution(base, "Age", ["fold", "fold_group", "ResearchGroup_Mapped"]),
        ],
        ignore_index=True,
    )
    mfr = (
        base.groupby(["fold", "fold_group", "ResearchGroup_Mapped", "Manufacturer"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["fold", "ResearchGroup_Mapped", "Manufacturer"])
    )
    site = (
        base.groupby(["fold", "fold_group", "ResearchGroup_Mapped", "SiteCode"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["fold", "ResearchGroup_Mapped", "SiteCode"])
    )
    sex = (
        base.groupby(["fold", "fold_group", "ResearchGroup_Mapped", "Sex"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["fold", "ResearchGroup_Mapped", "Sex"])
    )
    return numeric, mfr, site, sex


def load_latents(readout_dir: Path, meta: pd.DataFrame, run_id: str, label: str) -> pd.DataFrame:
    parts = []
    for path in sorted((readout_dir / "latent_cache").glob("fold_*_test_latent_mu.csv")):
        fold = int(path.name.split("_")[1])
        df = pd.read_csv(path)
        df["fold"] = fold
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    lat = pd.concat(parts, ignore_index=True)
    keep = [c for c in ["SubjectID", "original_n_TR", "Manufacturer", "SiteCode", "ResearchGroup_Mapped", "Age", "Sex"] if c in meta.columns]
    lat = lat.merge(meta[keep].drop_duplicates("SubjectID"), on="SubjectID", how="left", suffixes=("", "_meta"))
    lat.insert(0, "run_id", run_id)
    lat.insert(1, "run_label", label)
    return lat


def corr_abs(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    mask = np.isfinite(y)
    if mask.sum() < 5 or np.nanstd(y[mask]) == 0:
        return np.array([])
    x = x[mask]
    y = y[mask]
    x_std = np.nanstd(x, axis=0)
    valid = x_std > 0
    if not valid.any():
        return np.array([])
    x = x[:, valid]
    xz = (x - np.nanmean(x, axis=0)) / np.nanstd(x, axis=0)
    yz = (y - np.nanmean(y)) / np.nanstd(y)
    return np.abs(np.nanmean(xz * yz[:, None], axis=0))


def centroid_distance(x: np.ndarray, labels: pd.Series) -> float:
    labs = labels.astype(str)
    if labs.nunique() != 2:
        return float("nan")
    a, b = [x[labs.eq(v).to_numpy()] for v in sorted(labs.unique())]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    pooled_sd = np.nanstd(x, axis=0)
    pooled_sd[pooled_sd == 0] = 1.0
    delta = (np.nanmean(a, axis=0) - np.nanmean(b, axis=0)) / pooled_sd
    return float(np.linalg.norm(delta) / np.sqrt(x.shape[1]))


def category_centroid_dispersion(x: np.ndarray, labels: pd.Series, min_count: int = 3) -> tuple[float, int]:
    labs = labels.astype(str)
    counts = labs.value_counts()
    valid = [v for v, n in counts.items() if n >= min_count]
    if len(valid) < 2:
        return float("nan"), len(valid)
    centroids = []
    pooled_sd = np.nanstd(x, axis=0)
    pooled_sd[pooled_sd == 0] = 1.0
    for value in valid:
        centroids.append(np.nanmean(x[labs.eq(value).to_numpy()], axis=0) / pooled_sd)
    dists = [float(np.linalg.norm(a - b) / np.sqrt(x.shape[1])) for a, b in combinations(centroids, 2)]
    return float(np.nanmean(dists)), len(valid)


def latent_qc(latents: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    if latents.empty:
        return pd.DataFrame()
    pred_scores = pred[["SubjectID", "fold", "auc_proxy_score"]].drop_duplicates(["SubjectID", "fold"])
    latents = latents.merge(pred_scores, on=["SubjectID", "fold"], how="left")
    mu_cols = [c for c in latents.columns if c.startswith("mu_")]
    rows = []
    for keys, grp in latents.groupby(["run_id", "run_label", "fold"], dropna=False):
        run_id, run_label, fold = keys
        x = grp[mu_cols].to_numpy(dtype=float)
        y_diag = grp["ResearchGroup_Mapped"].astype(str)
        ntr = pd.to_numeric(grp["original_n_TR"], errors="coerce").to_numpy(dtype=float)
        ntr_corr = corr_abs(x, ntr)
        mfr_disp, n_mfr = category_centroid_dispersion(x, grp["Manufacturer"], min_count=3)
        site_disp, n_site = category_centroid_dispersion(x, grp["SiteCode"], min_count=2)
        rows.append(
            {
                "run_id": run_id,
                "run_label": run_label,
                "fold": int(fold),
                "fold_group": "fold4" if int(fold) == 4 else "other_folds",
                "n": int(len(grp)),
                "diagnosis_standardized_centroid_distance": centroid_distance(x, y_diag),
                "latent_variance_mean": float(np.nanmean(np.nanvar(x, axis=0))),
                "active_units_var_gt_1e_2": int((np.nanvar(x, axis=0) > 1e-2).sum()),
                "ntr_max_abs_latent_corr": float(np.nanmax(ntr_corr)) if ntr_corr.size else np.nan,
                "ntr_mean_abs_latent_corr": float(np.nanmean(ntr_corr)) if ntr_corr.size else np.nan,
                "manufacturer_centroid_dispersion": mfr_disp,
                "manufacturer_categories_used": int(n_mfr),
                "sitecode_centroid_dispersion": site_disp,
                "sitecode_categories_used": int(n_site),
            }
        )
    return pd.DataFrame(rows).sort_values(["run_id", "fold"])


def attach_auc_proxy(pred: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    out["auc_proxy_score"] = out["y_score"]
    return out


def scanner_leakage(run_dir: Path, run_id: str, label: str) -> pd.DataFrame:
    rows = []
    for path in sorted(run_dir.glob("fold_*/fold_*scanner_leakage_summary.csv")):
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        row["run_id"] = run_id
        row["run_label"] = label
        row["fold"] = int(path.parent.name.replace("fold_", ""))
        row["scope"] = "test" if "_test_" in path.name else "train_dev"
        row["latent_minus_raw"] = (
            float(row["acc_site_latent"]) - float(row["acc_site_raw"])
            if pd.notna(row.get("acc_site_latent")) and pd.notna(row.get("acc_site_raw"))
            else np.nan
        )
        row["file"] = str(path)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_decision(
    fold_metrics: pd.DataFrame,
    transitions: pd.DataFrame,
    latent: pd.DataFrame,
    leakage: pd.DataFrame,
) -> str:
    primary = fold_metrics[fold_metrics["threshold_strategy"].eq(PRIMARY_THRESHOLD)]
    locked_f4 = primary[(primary["run_id"].eq("locked_140tr")) & (primary["fold"].eq(4))].iloc[0]
    alltr_f4 = primary[(primary["run_id"].eq("all_available_timepoints")) & (primary["fold"].eq(4))].iloc[0]
    alltr_other = primary[(primary["run_id"].eq("all_available_timepoints")) & (~primary["fold"].eq(4))]
    new_errors_f4 = transitions[transitions["fold"].eq(4) & transitions["transition"].eq("new_error_alltr")]
    fixed_f4 = transitions[transitions["fold"].eq(4) & transitions["transition"].eq("fixed_by_alltr")]
    latent_f4 = latent[(latent["run_id"].eq("all_available_timepoints")) & (latent["fold"].eq(4))]
    mfr_test = leakage[(leakage["run_id"].eq("all_available_timepoints")) & (leakage["fold"].eq(4)) & (leakage["scope"].eq("test"))]

    return f"""# Fold 4 Failure Diagnosis

## Decision

The Fold 4 degradation in the exploratory all-timepoints branch is not a pure threshold issue. It is best interpreted as a latent/ranking degradation on an already hard small-sample fold, with site/manufacturer and n_TR confounding risk as important guardrails.

## Evidence

- Locked Fold 4: AUC={locked_f4['auc']:.6f}, PR-AUC={locked_f4['pr_auc']:.6f}, BA={locked_f4['balanced_accuracy']:.6f}, Sens={locked_f4['sensitivity']:.6f}, Spec={locked_f4['specificity']:.6f}, F1={locked_f4['f1']:.6f}.
- All-timepoints Fold 4: AUC={alltr_f4['auc']:.6f}, PR-AUC={alltr_f4['pr_auc']:.6f}, BA={alltr_f4['balanced_accuracy']:.6f}, Sens={alltr_f4['sensitivity']:.6f}, Spec={alltr_f4['specificity']:.6f}, F1={alltr_f4['f1']:.6f}.
- All-timepoints Fold 4 threshold={alltr_f4['threshold']:.6f}; confusion TN={int(alltr_f4['tn'])}, FP={int(alltr_f4['fp'])}, FN={int(alltr_f4['fn'])}, TP={int(alltr_f4['tp'])}.
- All-timepoints non-Fold-4 mean AUC={alltr_other['auc'].mean():.6f}; Fold 4 is the outlier low fold.
- Fold 4 errors fixed by all-timepoints: {len(fixed_f4)}. New Fold 4 errors introduced by all-timepoints: {len(new_errors_f4)}.
- All-timepoints Fold 4 diagnosis latent centroid distance={float(latent_f4['diagnosis_standardized_centroid_distance'].iloc[0]) if not latent_f4.empty else np.nan:.6f}.
- All-timepoints Fold 4 max absolute latent correlation with n_TR={float(latent_f4['ntr_max_abs_latent_corr'].iloc[0]) if not latent_f4.empty else np.nan:.6f}.
- All-timepoints Fold 4 test Manufacturer raw accuracy={float(mfr_test['acc_site_raw'].iloc[0]) if not mfr_test.empty else np.nan:.6f}, latent accuracy={float(mfr_test['acc_site_latent'].iloc[0]) if not mfr_test.empty else np.nan:.6f}.

## Classification

- Threshold issue: partial but not primary. AUC and PR-AUC fall in Fold 4, so ranking degraded before thresholding.
- Site/manufacturer issue: plausible contributor. Manufacturer leakage remains high in all-timepoints latent space, including Fold 4 test leakage.
- n_TR confounding issue: global guardrail. The all-timepoints branch is known to be n_TR diagnosis/manufacturer/site confounded; Fold 4 should not be used to promote this branch even if one threshold metric improves.
- Latent representation issue: yes. Fold 4 rank separation and latent diagnosis separation are poor under all-timepoints.
- Irreducible hard fold: likely. Fold 4 is weak in the locked model and worsens with all-timepoints; the fold has only 19 AD subjects, so a small number of AD score shifts materially changes PR-AUC and sensitivity.

## Recommendation

Do not promote the all-timepoints branch. Keep it as an exploratory confounding-stress-test sensitivity analysis. The locked 140TR v5.1b [1,0,2] model remains the manuscript model.
"""


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    command_log = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dry_run": bool(args.dry_run),
        "locked_run_dir": str(args.locked_run_dir),
        "alltr_run_dir": str(args.alltr_run_dir),
        "metadata": str(args.metadata),
        "training_launched": False,
        "threshold_fitting": False,
        "model_selection": False,
    }

    required = [
        args.locked_readout_dir / "classifier_sweep_predictions.csv",
        args.alltr_readout_dir / "classifier_sweep_predictions.csv",
        args.locked_readout_dir / "classifier_sweep_foldwise_metrics.csv",
        args.alltr_readout_dir / "classifier_sweep_foldwise_metrics.csv",
        args.metadata,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if args.dry_run:
        command_log["missing_required_files"] = missing
        (args.output_dir / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(command_log, indent=2, sort_keys=True))
        return 0
    if missing:
        raise FileNotFoundError("Missing required files: " + json.dumps(missing, indent=2))

    meta = load_metadata(args.metadata)
    locked = augment_predictions(load_predictions(args.locked_readout_dir, "locked_140tr", "Locked v5.1b 140TR"), meta)
    alltr = augment_predictions(
        load_predictions(args.alltr_readout_dir, "all_available_timepoints", "Exploratory all-timepoints"), meta
    )
    pred = pd.concat([locked, alltr], ignore_index=True)
    pred["fold_group"] = np.where(pred["fold"].astype(int).eq(4), "fold4", "other_folds")

    foldwise = pd.concat(
        [
            load_foldwise(args.locked_readout_dir, "locked_140tr", "Locked v5.1b 140TR"),
            load_foldwise(args.alltr_readout_dir, "all_available_timepoints", "Exploratory all-timepoints"),
        ],
        ignore_index=True,
    )
    write_table(fold_metrics_summary(foldwise), args.output_dir, "fold4_vs_other_folds_metrics")
    write_table(score_distribution(pred), args.output_dir, "score_distribution_by_fold_diagnosis")
    primary_foldwise = foldwise[foldwise["threshold_strategy"].eq(PRIMARY_THRESHOLD)].copy()
    write_table(
        primary_foldwise[
            [
                c
                for c in [
                    "run_id",
                    "run_label",
                    "fold",
                    "fold_group",
                    "threshold",
                    "n",
                    "n_cn",
                    "n_ad",
                    "tn",
                    "fp",
                    "fn",
                    "tp",
                    "sensitivity",
                    "specificity",
                    "balanced_accuracy",
                    "f1",
                    "auc",
                    "pr_auc",
                ]
                if c in primary_foldwise.columns
            ]
        ],
        args.output_dir,
        "confusion_by_fold",
    )

    transitions = error_transition(locked, alltr)
    write_table(transitions, args.output_dir, "subject_error_transition")
    write_table(error_transition_summary(transitions), args.output_dir, "subject_error_transition_summary")

    numeric_meta, mfr_counts, site_counts, sex_counts = metadata_summaries(alltr)
    write_table(numeric_meta, args.output_dir, "fold_metadata_ntr_age")
    write_table(mfr_counts, args.output_dir, "fold_metadata_manufacturer_counts")
    write_table(site_counts, args.output_dir, "fold_metadata_sitecode_counts")
    write_table(sex_counts, args.output_dir, "fold_metadata_sex_counts")

    locked_latent = load_latents(args.locked_readout_dir, meta, "locked_140tr", "Locked v5.1b 140TR")
    alltr_latent = load_latents(args.alltr_readout_dir, meta, "all_available_timepoints", "Exploratory all-timepoints")
    latent = pd.concat(
        [
            latent_qc(locked_latent, attach_auc_proxy(locked)),
            latent_qc(alltr_latent, attach_auc_proxy(alltr)),
        ],
        ignore_index=True,
    )
    write_table(latent, args.output_dir, "latent_qc_by_fold")

    leakage = pd.concat(
        [
            scanner_leakage(args.locked_run_dir, "locked_140tr", "Locked v5.1b 140TR"),
            scanner_leakage(args.alltr_run_dir, "all_available_timepoints", "Exploratory all-timepoints"),
        ],
        ignore_index=True,
    )
    write_table(leakage, args.output_dir, "scanner_manufacturer_leakage_by_fold")

    readme = """# ADNI All-Timepoints Fold 4 Failure Audit

This read-only audit compares the exploratory ADNI all-timepoints [1,0,2] FULL 5x5 branch against the locked v5.1b 140TR [1,0,2] model. It uses existing classifier-only Stage B predictions, foldwise metrics, metadata, latent caches, and scanner leakage summaries.

No VAE training, classifier retraining, threshold fitting, or model selection was performed.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    (args.output_dir / "final_failure_diagnosis.md").write_text(
        summarize_decision(primary_foldwise, transitions, latent, leakage), encoding="utf-8"
    )
    (args.output_dir / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote Fold 4 failure audit to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
