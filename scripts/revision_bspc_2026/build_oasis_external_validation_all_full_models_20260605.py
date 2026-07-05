#!/usr/bin/env python3
"""Aggregate existing OASIS external validation artifacts across completed FULL models.

This script is read-only with respect to tensors, metadata, and model artifacts.
It does not run inference, fit thresholds, calibrate on OASIS, or modify any
model output directory. It only consolidates existing frozen-inference results.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


ROOT = Path("results/revision_bspc_2026")
OUT = ROOT / "oasis_external_validation_all_full_models_20260605"
REGISTRY = ROOT / "full5x5_completed_run_registry_followup_plan_20260605" / "completed_full5x5_registry.csv"

PANEL_DIR = ROOT / "oasis_mega_90_90_external_inference_model_panel_20260604"
CH1_PANEL_DIR = ROOT / "ch1only_latent384_beta3p75_oasis_mega_90_90_external_inference_20260605"
BETA3P5_PANEL_DIR = ROOT / "latent384_beta3p5_oasis_mega_90_90_external_inference_20260605"
LATENT448_PANEL_DIR = ROOT / "latent448_beta4p0_oasis_mega_90_90_external_inference_20260605"
MEGA_OLD_DIR = ROOT / "oasis_mega_90cn_90ad_pooled_external_validation_20260531"


REQUESTED_MODELS = [
    {
        "model_id": "ch1only_latent384_beta3p75",
        "display_name": "ch1-only latent384 beta3.75",
        "run_name": "recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5",
        "role": "parsimony_sensitivity",
        "oasis_candidates": ["ch1only_latent384_beta3p75_oof_ecdf"],
        "source_family": "latent384_panel",
    },
    {
        "model_id": "promoted_latent384_beta3p75_ch1_0_2",
        "display_name": "promoted [1,0,2] latent384 beta3.75",
        "run_name": "recover035_latent384_beta3p75_T80_h10000_p560_full5x5",
        "role": "primary",
        "oasis_candidates": ["promoted_beta3p75_oof_ecdf"],
        "source_family": "latent384_panel",
    },
    {
        "model_id": "latent384_beta2p5",
        "display_name": "latent384 beta2.5",
        "run_name": "recover035_latent384_T80_h10000_p560_full5x5",
        "role": "capacity_beta_sensitivity",
        "oasis_candidates": [],
        "source_family": "not_scored",
    },
    {
        "model_id": "latent384_beta4p0",
        "display_name": "latent384 beta4.0",
        "run_name": "recover035_latent384_beta4p0_T80_h10000_p560_full5x5",
        "role": "capacity_beta_sensitivity",
        "oasis_candidates": [],
        "source_family": "not_scored",
    },
    {
        "model_id": "latent384_beta3p5",
        "display_name": "latent384 beta3.5",
        "run_name": "recover035_latent384_beta3p5_T80_h10000_p560_full5x5",
        "role": "local_beta_sensitivity",
        "oasis_candidates": ["latent384_beta3p5_oof_ecdf"],
        "source_family": "beta3p5_panel",
    },
    {
        "model_id": "latent448_beta4p0",
        "display_name": "latent448 beta4.0",
        "run_name": "recover035_latent448_beta4p0_T80_h10000_p560_full5x5",
        "role": "capacity_beta_sensitivity",
        "oasis_candidates": ["latent448_beta4p0_oof_ecdf"],
        "source_family": "latent448_panel",
    },
    {
        "model_id": "latent512_beta3p75",
        "display_name": "latent512 beta3.75",
        "run_name": "recover035_latent512_beta3p75_T80_h10000_p560_full5x5",
        "role": "capacity_beta_sensitivity",
        "oasis_candidates": ["latent512_beta3p75_oof_ecdf"],
        "source_family": "latent384_panel",
    },
    {
        "model_id": "locked_v5_1b_latent256_beta2p5",
        "display_name": "locked v5.1b latent256 beta2.5",
        "run_name": "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5",
        "role": "locked_reference",
        "oasis_candidates": ["primary_v5_1b_horizon4480_classifier_only"],
        "source_family": "legacy_mega_pooled",
    },
    {
        "model_id": "recover035_latent256_beta2p5",
        "display_name": "recover035 latent256 beta2.5",
        "run_name": "adni_v5_1_batch20260514b_ch1_0_2_recover035_full5x5",
        "role": "latent256_recover035_reference",
        "oasis_candidates": ["recover035_oof_logitz"],
        "source_family": "legacy_mega_pooled",
    },
    {
        "model_id": "mfrBalancedVAE_latent384_beta3p75",
        "display_name": "mfrBalancedVAE latent384 beta3.75",
        "run_name": "recover035_latent384_beta3p75_T80_h10000_p560_full5x5_mfrBalancedVAE",
        "role": "deconfounding_sensitivity",
        "oasis_candidates": ["mfrBalancedVAE_beta3p75_oof_ecdf"],
        "source_family": "latent384_panel",
    },
    {
        "model_id": "residualized_mfr_stageB",
        "display_name": "promoted latent residualized by Manufacturer",
        "run_name": "recover035_latent384_beta3p75_T80_h10000_p560_full5x5",
        "role": "classifier_only_harmonization_sensitivity",
        "oasis_candidates": ["promoted_beta3p75_residualized_mfr_oof_ecdf"],
        "source_family": "latent384_panel",
    },
]


def md_write(df: pd.DataFrame, path: Path, max_rows: int | None = None) -> None:
    view = df if max_rows is None else df.head(max_rows)
    try:
        text = view.to_markdown(index=False)
    except Exception:
        text = view.to_string(index=False)
    path.write_text(text + "\n", encoding="utf-8")


def safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def norm_subject_id(row: pd.Series) -> str:
    for col in ("SubjectID", "subject_id", "experiment_id"):
        val = row.get(col)
        if pd.notna(val):
            return str(val)
    return ""


def y_col(df: pd.DataFrame) -> str:
    if "y" in df.columns:
        return "y"
    if "y_true" in df.columns:
        return "y_true"
    raise ValueError("No y/y_true column found")


def metrics_from_predictions(df: pd.DataFrame, score_col: str = "y_score", pred_col: str = "y_pred") -> dict:
    out: dict[str, float | int] = {
        "n": len(df),
        "n_cn": np.nan,
        "n_ad": np.nan,
        "tn": np.nan,
        "fp": np.nan,
        "fn": np.nan,
        "tp": np.nan,
        "auc": np.nan,
        "pr_auc": np.nan,
        "balanced_accuracy": np.nan,
        "sensitivity": np.nan,
        "specificity": np.nan,
        "f1": np.nan,
        "predicted_ad_rate": np.nan,
    }
    if df.empty:
        return out
    yc = y_col(df)
    y = pd.to_numeric(df[yc], errors="coerce")
    score = pd.to_numeric(df.get(score_col), errors="coerce")
    valid = y.notna() & score.notna()
    y = y[valid].astype(int)
    score = score[valid]
    out["n"] = int(valid.sum())
    out["n_cn"] = int((y == 0).sum())
    out["n_ad"] = int((y == 1).sum())
    if y.nunique() == 2:
        out["auc"] = float(roc_auc_score(y, score))
        out["pr_auc"] = float(average_precision_score(y, score))
    if pred_col in df.columns:
        pred = pd.to_numeric(df.loc[valid, pred_col], errors="coerce")
        pred_valid = pred.notna()
        if pred_valid.any():
            yy = y[pred_valid]
            pp = pred[pred_valid].astype(int)
            cm = confusion_matrix(yy, pp, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            out.update(
                {
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),
                    "sensitivity": float(tp / (tp + fn)) if (tp + fn) else np.nan,
                    "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
                    "balanced_accuracy": float(balanced_accuracy_score(yy, pp)),
                    "f1": float(f1_score(yy, pp, zero_division=0)),
                    "predicted_ad_rate": float(pp.mean()),
                }
            )
    return out


def source_paths_for_model(source_family: str) -> list[Path]:
    if source_family == "latent384_panel":
        return [PANEL_DIR / "predictions.csv", CH1_PANEL_DIR / "predictions.csv"]
    if source_family == "beta3p5_panel":
        return [BETA3P5_PANEL_DIR / "predictions.csv"]
    if source_family == "latent448_panel":
        return [LATENT448_PANEL_DIR / "predictions.csv"]
    if source_family == "legacy_mega_pooled":
        return [MEGA_OLD_DIR / "predictions.csv"]
    return []


def legacy_metric_model_ids() -> set[str]:
    threshold = safe_read(MEGA_OLD_DIR / "threshold_metrics.csv")
    if threshold.empty or "adni_model" not in threshold.columns:
        return set()
    out: set[str] = set()
    for spec in REQUESTED_MODELS:
        if spec["source_family"] != "legacy_mega_pooled":
            continue
        if set(spec["oasis_candidates"]) & set(threshold["adni_model"].astype(str)):
            out.add(spec["model_id"])
    return out


def build_prediction_table() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for spec in REQUESTED_MODELS:
        candidates = set(spec["oasis_candidates"])
        if not candidates:
            continue
        for path in source_paths_for_model(spec["source_family"]):
            df = safe_read(path)
            if df.empty:
                continue
            if "candidate" in df.columns:
                mask = df["candidate"].isin(candidates)
            elif "adni_model" in df.columns:
                mask = df["adni_model"].isin(candidates)
            elif "model_label" in df.columns:
                mask = df["model_label"].isin(candidates)
            else:
                continue
            sub = df.loc[mask].copy()
            if sub.empty:
                continue
            sub["model_id"] = spec["model_id"]
            sub["display_name"] = spec["display_name"]
            sub["model_role"] = spec["role"]
            sub["oasis_source_path"] = str(path)
            if "candidate" not in sub.columns:
                sub["candidate"] = sub.get("adni_model", sub.get("model_label", spec["model_id"]))
            if "y" not in sub.columns and "y_true" in sub.columns:
                sub["y"] = sub["y_true"]
            if "SubjectID" not in sub.columns and "subject_id" in sub.columns:
                sub["SubjectID"] = sub["subject_id"]
            frames.append(sub)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True, sort=False)
    # Dedupe promoted rows that are present in both panel sources.
    key_cols = [
        c
        for c in [
            "model_id",
            "build_candidate",
            "prediction_level",
            "fold",
            "SubjectID",
            "candidate",
            "oasis_source_path",
        ]
        if c in merged.columns
    ]
    # Prefer the dedicated ch1 source for ch1-only and the main panel for promoted.
    merged["_source_rank"] = 1
    merged.loc[
        (merged["model_id"] == "promoted_latent384_beta3p75_ch1_0_2")
        & merged["oasis_source_path"].str.contains("oasis_mega_90_90_external_inference_model_panel_20260604"),
        "_source_rank",
    ] = 0
    merged.loc[
        (merged["model_id"] == "ch1only_latent384_beta3p75")
        & merged["oasis_source_path"].str.contains("ch1only_latent384_beta3p75_oasis"),
        "_source_rank",
    ] = 0
    dedupe_cols = [c for c in key_cols if c != "oasis_source_path"]
    merged = merged.sort_values("_source_rank").drop_duplicates(dedupe_cols, keep="first")
    return merged.drop(columns=["_source_rank"])


def age_sex_matched_subjects(meta: pd.DataFrame) -> set[str]:
    if meta.empty or "Age" not in meta.columns or "Sex" not in meta.columns:
        return set()
    base = meta.copy()
    base["_sid"] = base.apply(norm_subject_id, axis=1)
    base["_age"] = pd.to_numeric(base["Age"], errors="coerce")
    base["_sex"] = base["Sex"].astype(str).str.upper().str[0]
    base["_y"] = pd.to_numeric(base[y_col(base)], errors="coerce")
    base = base.dropna(subset=["_age", "_y"])
    selected: set[str] = set()
    for _, batch_df in base.groupby(base.get("source_batch", pd.Series("all", index=base.index)).fillna("all")):
        cn = batch_df[batch_df["_y"] == 0].copy()
        ad = batch_df[batch_df["_y"] == 1].copy().sort_values("_age")
        available = set(cn.index)
        for _, ad_row in ad.iterrows():
            if not available:
                break
            candidates = cn.loc[list(available)].copy()
            candidates["_age_gap"] = (candidates["_age"] - ad_row["_age"]).abs()
            same_sex = candidates[candidates["_sex"] == ad_row["_sex"]]
            pool = same_sex if not same_sex.empty else candidates
            pool = pool[pool["_age_gap"] <= 5.0]
            if pool.empty:
                continue
            cn_idx = pool.sort_values(["_age_gap", "_age"]).index[0]
            available.remove(cn_idx)
            selected.add(str(ad_row["_sid"]))
            selected.add(str(cn.loc[cn_idx, "_sid"]))
    return selected


def subgroup_masks(ensemble: pd.DataFrame) -> dict[str, pd.Series]:
    masks: dict[str, pd.Series] = {}
    masks["OASIS_all_compatible"] = pd.Series(True, index=ensemble.index)
    diag = ensemble.get("diagnosis", pd.Series("", index=ensemble.index)).astype(str)
    masks["OASIS_CN_vs_AD_strict_labels"] = diag.isin(["CN", "AD_DEMENTIA"])
    if "selected_qc_runs" in ensemble.columns:
        runs = pd.to_numeric(ensemble["selected_qc_runs"], errors="coerce")
        masks["OASIS_high_QC_only"] = runs >= 2
    else:
        masks["OASIS_high_QC_only"] = pd.Series(False, index=ensemble.index)
    scanner = ensemble.get("ScannerModel", pd.Series("", index=ensemble.index)).astype(str).str.lower()
    mfr = ensemble.get("Manufacturer", pd.Series("", index=ensemble.index)).astype(str).str.lower()
    masks["OASIS_scanner_or_acquisition_compatible"] = scanner.str.contains("triotim") | mfr.str.contains("siemens")
    # Same matched set for all models/builds, derived from first unique subject metadata only.
    meta_cols = [c for c in ensemble.columns if c not in {"model_id", "display_name", "candidate", "y_score", "y_pred", "threshold"}]
    subject_meta = ensemble[meta_cols].copy()
    subject_meta["_sid"] = subject_meta.apply(norm_subject_id, axis=1)
    subject_meta = subject_meta.drop_duplicates("_sid")
    matched = age_sex_matched_subjects(subject_meta)
    masks["OASIS_ADNI_like_age_matched"] = ensemble.apply(norm_subject_id, axis=1).isin(matched)
    return masks


def metric_rows(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensemble = preds[preds.get("prediction_level", "") == "ensemble_mean_score_majority_vote"].copy()
    rows = []
    conf_rows = []
    subgroup_rows = []
    score_rows = []
    if ensemble.empty:
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    for keys, g in ensemble.groupby(["model_id", "display_name", "model_role", "build_candidate", "candidate"], dropna=False):
        model_id, display_name, role, build, candidate = keys
        primary = metrics_from_predictions(g, "y_score", "y_pred")
        raw = metrics_from_predictions(g, "y_score_raw", "y_pred") if "y_score_raw" in g.columns else {}
        row = {
            "model_id": model_id,
            "display_name": display_name,
            "role": role,
            "build_candidate": build,
            "candidate": candidate,
            "score_type": "ADNI_calibrated_y_score",
            "raw_score_auc": raw.get("auc", np.nan),
            "raw_score_pr_auc": raw.get("pr_auc", np.nan),
            **primary,
        }
        rows.append(row)
        conf_rows.append(
            {
                "model_id": model_id,
                "display_name": display_name,
                "build_candidate": build,
                "candidate": candidate,
                "subgroup": "OASIS_all_compatible",
                **{k: primary[k] for k in ["n", "n_cn", "n_ad", "tn", "fp", "fn", "tp", "sensitivity", "specificity", "balanced_accuracy", "f1"]},
            }
        )
        masks = subgroup_masks(g)
        for subgroup, mask in masks.items():
            sg = g[mask].copy()
            m = metrics_from_predictions(sg, "y_score", "y_pred")
            raw_sg = metrics_from_predictions(sg, "y_score_raw", "y_pred") if "y_score_raw" in sg.columns else {}
            subgroup_rows.append(
                {
                    "model_id": model_id,
                    "display_name": display_name,
                    "role": role,
                    "build_candidate": build,
                    "candidate": candidate,
                    "subgroup": subgroup,
                    "raw_score_auc": raw_sg.get("auc", np.nan),
                    "raw_score_pr_auc": raw_sg.get("pr_auc", np.nan),
                    **m,
                }
            )
        for diag_keys, dg in g.groupby(["diagnosis"], dropna=False):
            score = pd.to_numeric(dg.get("y_score"), errors="coerce")
            raw_score = pd.to_numeric(dg.get("y_score_raw"), errors="coerce") if "y_score_raw" in dg.columns else pd.Series(dtype=float)
            score_rows.append(
                {
                    "model_id": model_id,
                    "display_name": display_name,
                    "build_candidate": build,
                    "candidate": candidate,
                    "diagnosis": diag_keys,
                    "n": int(score.notna().sum()),
                    "score_mean": float(score.mean()) if score.notna().any() else np.nan,
                    "score_sd": float(score.std(ddof=1)) if score.notna().sum() > 1 else np.nan,
                    "score_median": float(score.median()) if score.notna().any() else np.nan,
                    "score_iqr": float(score.quantile(0.75) - score.quantile(0.25)) if score.notna().any() else np.nan,
                    "raw_score_mean": float(raw_score.mean()) if raw_score.notna().any() else np.nan,
                    "raw_score_median": float(raw_score.median()) if raw_score.notna().any() else np.nan,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(subgroup_rows), pd.DataFrame(score_rows), pd.DataFrame(conf_rows)


def error_subjects(preds: pd.DataFrame) -> pd.DataFrame:
    ensemble = preds[preds.get("prediction_level", "") == "ensemble_mean_score_majority_vote"].copy()
    if ensemble.empty:
        return pd.DataFrame()
    y = pd.to_numeric(ensemble.get("y"), errors="coerce")
    yp = pd.to_numeric(ensemble.get("y_pred"), errors="coerce")
    ensemble["error_type"] = np.select(
        [(y == 0) & (yp == 1), (y == 1) & (yp == 0), (y == 0) & (yp == 0), (y == 1) & (yp == 1)],
        ["FP", "FN", "TN", "TP"],
        default="unknown",
    )
    cols = [
        "model_id",
        "display_name",
        "build_candidate",
        "candidate",
        "SubjectID",
        "subject_id",
        "session_id",
        "experiment_id",
        "source_batch",
        "protocol_subset",
        "diagnosis",
        "Age",
        "Sex",
        "Manufacturer",
        "ScannerModel",
        "selected_qc_runs",
        "selected_run_ids",
        "mean_fd_subject",
        "max_fd_subject",
        "y_score_raw",
        "y_score",
        "threshold",
        "y_pred",
        "error_type",
    ]
    cols = [c for c in cols if c in ensemble.columns]
    return ensemble[cols].sort_values(["model_id", "build_candidate", "error_type", "SubjectID"])


def model_registry() -> pd.DataFrame:
    reg = safe_read(REGISTRY)
    rows = []
    preds = build_prediction_table()
    available_models = set(preds["model_id"].unique()) if not preds.empty else set()
    available_models |= legacy_metric_model_ids()
    for spec in REQUESTED_MODELS:
        row = {k: spec[k] for k in ["model_id", "display_name", "run_name", "role", "source_family"]}
        row["oasis_artifact_status"] = "available" if spec["model_id"] in available_models else "missing_existing_oasis_inference_artifact"
        row["requested_oasis_candidates"] = ";".join(spec["oasis_candidates"])
        if not reg.empty:
            hit = reg[reg["run_name"].astype(str) == spec["run_name"]]
            if not hit.empty:
                h = hit.iloc[0]
                for c in [
                    "run_dir",
                    "completion_status",
                    "channel_set_order",
                    "selected_channel_names",
                    "beta_vae",
                    "latent_dim",
                    "oof_ecdf_auc",
                    "oof_ecdf_pr_auc",
                    "oof_ecdf_balanced_accuracy",
                    "oof_ecdf_sensitivity",
                    "oof_ecdf_specificity",
                    "oof_ecdf_f1",
                    "oof_ecdf_philips_cn_fpr",
                    "test_scanner_latent_acc_mean",
                ]:
                    if c in h.index:
                        row[f"adni_{c}" if c.startswith("oof") or c.startswith("test_") else c] = h[c]
        rows.append(row)
    return pd.DataFrame(rows)


def legacy_primary_metric_rows() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    threshold = safe_read(MEGA_OLD_DIR / "threshold_metrics.csv")
    if threshold.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    rows = []
    conf_rows = []
    subgroup_rows = []
    for spec in REQUESTED_MODELS:
        if spec["source_family"] != "legacy_mega_pooled":
            continue
        for candidate in spec["oasis_candidates"]:
            sub = threshold[
                (threshold["adni_model"].astype(str) == candidate)
                & (threshold["threshold_strategy"].astype(str) == "adni_fixed")
                & (threshold["evaluation_subset"].astype(str) == "pooled_90cn_90ad")
            ].copy()
            for _, r in sub.iterrows():
                n = r.get("n")
                pred_rate = (r.get("fp", 0) + r.get("tp", 0)) / n if pd.notna(n) and n else np.nan
                base = {
                    "model_id": spec["model_id"],
                    "display_name": spec["display_name"],
                    "role": spec["role"],
                    "build_candidate": r.get("build_candidate"),
                    "candidate": candidate,
                    "score_type": "legacy_adni_fixed_threshold_metric",
                    "raw_score_auc": np.nan,
                    "raw_score_pr_auc": np.nan,
                    "n": r.get("n"),
                    "n_cn": r.get("n_cn"),
                    "n_ad": r.get("n_ad"),
                    "tn": r.get("tn"),
                    "fp": r.get("fp"),
                    "fn": r.get("fn"),
                    "tp": r.get("tp"),
                    "auc": r.get("auc"),
                    "pr_auc": r.get("pr_auc"),
                    "balanced_accuracy": r.get("balanced_accuracy"),
                    "sensitivity": r.get("sensitivity"),
                    "specificity": r.get("specificity"),
                    "f1": r.get("f1"),
                    "predicted_ad_rate": pred_rate,
                }
                rows.append(base)
                conf_rows.append({**base, "subgroup": "OASIS_all_compatible"})
                subgroup_rows.append({**base, "subgroup": "OASIS_all_compatible"})
    return pd.DataFrame(rows), pd.DataFrame(subgroup_rows), pd.DataFrame(conf_rows)


def adni_vs_oasis(registry: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    if primary.empty:
        return pd.DataFrame()
    cols = [
        "model_id",
        "display_name",
        "adni_oof_ecdf_auc",
        "adni_oof_ecdf_pr_auc",
        "adni_oof_ecdf_balanced_accuracy",
        "adni_oof_ecdf_sensitivity",
        "adni_oof_ecdf_specificity",
        "adni_oof_ecdf_f1",
    ]
    reg_cols = [c for c in cols if c in registry.columns]
    merged = primary.merge(registry[reg_cols], on=["model_id", "display_name"], how="left")
    merged["generalization_drop_auc"] = merged["auc"] - merged.get("adni_oof_ecdf_auc")
    merged["generalization_drop_pr_auc"] = merged["pr_auc"] - merged.get("adni_oof_ecdf_pr_auc")
    merged["generalization_drop_ba"] = merged["balanced_accuracy"] - merged.get("adni_oof_ecdf_balanced_accuracy")
    return merged


def write_readme(out: Path) -> None:
    text = """# OASIS external validation across completed FULL models

This package aggregates existing frozen OASIS inference artifacts only. It does
not train on OASIS, fit OASIS thresholds, fit OASIS calibration models, modify
tensors, modify metadata, or modify model artifacts.

Primary OASIS rows use ensemble-level predictions with ADNI-derived thresholds
and ADNI-derived score calibration when those artifacts were already present.
Raw score AUC/PR-AUC are reported separately from ADNI-calibrated score
AUC/PR-AUC.

Predefined subgroup rules used before metric aggregation:
- OASIS_all_compatible: all ensemble-level compatible OASIS subjects.
- OASIS_ADNI_like_age_matched: deterministic source-batch-stratified
  age/sex nearest-neighbor CN/AD matching using a 5-year age caliper and no
  model scores.
- OASIS_high_QC_only: subjects with at least two selected QC runs.
- OASIS_CN_vs_AD_strict_labels: diagnosis labels CN or AD_DEMENTIA.
- OASIS_scanner_or_acquisition_compatible: Siemens/TrioTim-compatible rows
  when scanner metadata are available.

Models without existing OASIS inference artifacts are retained in the registry
as missing external artifacts rather than scored or inferred here.
"""
    (out / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    registry = model_registry()
    preds = build_prediction_table()
    primary, subgroup, scores, conf = metric_rows(preds)
    legacy_primary, legacy_subgroup, legacy_conf = legacy_primary_metric_rows()
    if not legacy_primary.empty:
        if not primary.empty and "model_id" in primary.columns:
            # The legacy mega package also contains ensemble prediction rows.
            # Use threshold-metric copies only for legacy models lacking
            # compatible ensemble predictions.
            existing = set(primary["model_id"].astype(str))
            legacy_primary = legacy_primary[~legacy_primary["model_id"].astype(str).isin(existing)]
            legacy_subgroup = legacy_subgroup[~legacy_subgroup["model_id"].astype(str).isin(existing)]
            legacy_conf = legacy_conf[~legacy_conf["model_id"].astype(str).isin(existing)]
    if not legacy_primary.empty:
        primary = pd.concat([primary, legacy_primary], ignore_index=True, sort=False)
    if not legacy_subgroup.empty:
        subgroup = pd.concat([subgroup, legacy_subgroup], ignore_index=True, sort=False)
    if not legacy_conf.empty:
        conf = pd.concat([conf, legacy_conf], ignore_index=True, sort=False)
    errors = error_subjects(preds)
    comparison = adni_vs_oasis(registry, primary)

    if args.dry_run:
        print(f"output_dir={out}")
        print(f"registry_rows={len(registry)}")
        print(f"prediction_rows={len(preds)}")
        print(f"primary_metric_rows={len(primary)}")
        return 0

    write_readme(out)
    outputs = {
        "oasis_model_registry": registry,
        "oasis_primary_metrics": primary,
        "oasis_subgroup_metrics": subgroup,
        "oasis_score_distributions": scores,
        "oasis_confusion_by_model": conf,
        "oasis_error_subjects": errors,
        "adni_vs_oasis_comparison": comparison,
    }
    for name, df in outputs.items():
        csv_path = out / f"{name}.csv"
        md_path = out / f"{name}.md"
        df.to_csv(csv_path, index=False)
        md_write(df, md_path, max_rows=120 if name == "oasis_error_subjects" else None)

    interpretation = []
    interpretation.append("# Model selection external-validity interpretation\n")
    interpretation.append("The package uses existing frozen OASIS inference artifacts only. No OASIS model training, threshold fitting, calibration fitting, or model selection was performed.\n")
    if not primary.empty:
        best = primary.sort_values(["build_candidate", "auc", "pr_auc"], ascending=[True, False, False]).groupby("build_candidate").head(1)
        interpretation.append("## Best available OASIS rows by build\n")
        for _, r in best.iterrows():
            interpretation.append(
                f"- {r['build_candidate']}: {r['display_name']} ({r['candidate']}), "
                f"AUC={r['auc']:.6f}, PR-AUC={r['pr_auc']:.6f}, BA={r['balanced_accuracy']:.6f}."
            )
    missing = registry[registry["oasis_artifact_status"] != "available"]
    if not missing.empty:
        interpretation.append("\n## Missing external artifacts\n")
        for _, r in missing.iterrows():
            interpretation.append(f"- {r['display_name']}: no existing OASIS inference artifact was found; not scored here.")
    interpretation.append(
        "\n## Recommendation\n"
        "Keep ADNI model selection anchored to the internal pre-specified ADNI gate. "
        "OASIS remains an external stress test: available rows show moderate and build-dependent transfer, "
        "and missing-model rows should not be treated as external failures. Additional OASIS scoring should be run only with frozen ADNI artifacts and the same no-threshold-fitting guardrails."
    )
    (out / "model_selection_external_validity_interpretation.md").write_text("\n".join(interpretation) + "\n", encoding="utf-8")

    command_log = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__)),
        "output_dir": str(out),
        "inputs": {
            "completed_full5x5_registry": str(REGISTRY),
            "latent384_model_panel_predictions": str(PANEL_DIR / "predictions.csv"),
            "ch1only_model_panel_predictions": str(CH1_PANEL_DIR / "predictions.csv"),
            "beta3p5_model_panel_predictions": str(BETA3P5_PANEL_DIR / "predictions.csv"),
            "latent448_model_panel_predictions": str(LATENT448_PANEL_DIR / "predictions.csv"),
            "legacy_mega_pooled_predictions": str(MEGA_OLD_DIR / "predictions.csv"),
        },
        "guardrails": {
            "no_training": True,
            "no_oasis_threshold_fitting": True,
            "no_oasis_calibration_fitting": True,
            "no_tensor_modification": True,
            "no_metadata_modification": True,
            "no_model_artifact_modification": True,
        },
        "rows": {name: int(len(df)) for name, df in outputs.items()},
    }
    (out / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
    print(f"Wrote OASIS external validation aggregation to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
