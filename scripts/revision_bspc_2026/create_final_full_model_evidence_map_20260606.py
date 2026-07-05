#!/usr/bin/env python3
"""Create the final read-only FULL-model evidence map.

This script aggregates already-generated ADNI and OASIS audit outputs. It does
not train models, score OASIS, fit thresholds/calibrators, or modify model
artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("results/revision_bspc_2026")
OUT = ROOT / "final_full_model_evidence_map_20260606"

FINAL_DECISION = ROOT / "final_model_decision_adni_oasis_manuscript_support_20260605" / "final_model_decision_table.csv"
CAPACITY = ROOT / "final_latent_capacity_beta_rate_distortion_synthesis_20260603" / "capacity_beta_summary.csv"
OASIS_ALL = ROOT / "oasis_external_validation_all_full_models_20260605" / "oasis_primary_metrics.csv"
CH1_FOLLOWUP = ROOT / "ch1only_targeted_followup_completion_promotion_oasis_audit_20260606"
CH12_AUDIT = ROOT / "ch12_latent384_beta3p75_completion_promotion_gate_audit_20260605"
MFR_AUDIT = ROOT / "mfrBalancedVAE_completion_promotion_gate_audit_20260604"
HARM_AUDIT = ROOT / "promoted_beta3p75_stageB_latent_harmonization_by_manufacturer_20260602"

PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
PRIMARY_MODEL = "logreg_l2_original"
PRIMARY_FEATURES = "z_plus_age_sex"


MODEL_ORDER = [
    "promoted_latent384_beta3p75_ch1_0_2",
    "ch1only_latent384_beta3p75",
    "ch1only_latent256_beta3p75",
    "ch1only_latent384_beta3p25",
    "latent384_beta3p5",
    "latent384_beta4p0",
    "latent448_beta4p0",
    "latent512_beta3p75",
    "mfrBalancedVAE_latent384_beta3p75",
    "residualized_mfr_stageB",
    "locked_v5_1b_latent256_beta2p5",
    "recover035_latent256_beta2p5",
    "ch12_latent384_beta3p75",
]

DISPLAY = {
    "promoted_latent384_beta3p75_ch1_0_2": "promoted [1,0,2] latent384 beta3.75",
    "ch1only_latent384_beta3p75": "ch1-only latent384 beta3.75",
    "ch1only_latent256_beta3p75": "ch1-only latent256 beta3.75",
    "ch1only_latent384_beta3p25": "ch1-only latent384 beta3.25",
    "latent384_beta3p5": "latent384 beta3.5",
    "latent384_beta4p0": "latent384 beta4.0",
    "latent448_beta4p0": "latent448 beta4.0",
    "latent512_beta3p75": "latent512 beta3.75",
    "mfrBalancedVAE_latent384_beta3p75": "mfrBalancedVAE latent384 beta3.75",
    "residualized_mfr_stageB": "residualized mfr Stage B",
    "locked_v5_1b_latent256_beta2p5": "locked v5.1b latent256 beta2.5",
    "recover035_latent256_beta2p5": "recover035 latent256 beta2.5",
    "ch12_latent384_beta3p75": "[1,2] latent384 beta3.75",
}

CAPACITY_MAP = {
    "promoted_latent384_beta3p75_ch1_0_2": "latent384_beta3p75_promoted",
    "latent384_beta3p5": "latent384_beta3p5",
    "latent384_beta4p0": "latent384_beta4p0",
    "latent448_beta4p0": "latent448_beta4p0",
    "latent512_beta3p75": "latent512_beta3p75",
    "locked_v5_1b_latent256_beta2p5": "locked_v5_1b_latent256_beta2p5",
    "recover035_latent256_beta2p5": "recover035_latent256_beta2p5",
}

DECISION_DEFAULTS = {
    "promoted_latent384_beta3p75_ch1_0_2": "primary",
    "ch1only_latent384_beta3p75": "parsimony_sensitivity",
    "ch1only_latent256_beta3p75": "reject_not_promoted",
    "ch1only_latent384_beta3p25": "sensitivity_only_not_primary",
    "latent384_beta3p5": "beta_sensitivity",
    "latent384_beta4p0": "beta_sensitivity",
    "latent448_beta4p0": "capacity_sensitivity",
    "latent512_beta3p75": "capacity_sensitivity",
    "mfrBalancedVAE_latent384_beta3p75": "deconfounding_sensitivity",
    "residualized_mfr_stageB": "classifier_only_harmonization_sensitivity",
    "locked_v5_1b_latent256_beta2p5": "reference_not_promoted",
    "recover035_latent256_beta2p5": "reference_not_promoted",
    "ch12_latent384_beta3p75": "channel_pair_sensitivity",
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def scalar(x: Any) -> Any:
    if isinstance(x, pd.Series):
        if x.empty:
            return np.nan
        return x.iloc[0]
    return x


def put(row: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    try:
        if pd.isna(value):
            return
    except TypeError:
        pass
    row[key] = value


def choose(df: pd.DataFrame, **criteria: Any) -> pd.Series | None:
    if df.empty:
        return None
    mask = pd.Series(True, index=df.index)
    for col, value in criteria.items():
        if col not in df.columns:
            return None
        mask &= df[col].astype(str).eq(str(value))
    sub = df[mask]
    if sub.empty:
        return None
    return sub.iloc[0]


def best_metric_row(
    df: pd.DataFrame,
    calib_method: str,
    model_name: str = PRIMARY_MODEL,
    feature_set: str = PRIMARY_FEATURES,
    threshold_strategy: str = PRIMARY_THRESHOLD,
) -> pd.Series | None:
    if df.empty:
        return None
    candidates = [
        dict(model_name=model_name, feature_set=feature_set, calib_method=calib_method, threshold_strategy=threshold_strategy),
        dict(model_name="logreg_l2", readout_feature_set=feature_set, calib_method=calib_method, threshold_strategy=threshold_strategy),
        dict(model_name=model_name, readout_feature_set=feature_set, calib_method=calib_method, threshold_strategy=threshold_strategy),
        dict(calib_method=calib_method, threshold_strategy=threshold_strategy),
    ]
    for crit in candidates:
        row = choose(df, **crit)
        if row is not None:
            return row
    if "calib_method" in df.columns:
        sub = df[df["calib_method"].astype(str).eq(calib_method)]
        if not sub.empty:
            return sub.iloc[0]
    return None


def collect_oof_dir_metrics(model_id: str, row: dict[str, Any], oof_dir: Path) -> None:
    pooled = read_csv(oof_dir / "calib_pooled_metrics.csv")
    if pooled.empty:
        return
    for method in ["raw", "oof_zscore", "oof_logitz", "oof_ecdf", "oof_platt", "oof_isotonic"]:
        r = best_metric_row(pooled, method)
        if r is None:
            continue
        prefix = f"adni_{method}"
        for src, dst in [
            ("auc", "auc"),
            ("pr_auc", "pr_auc"),
            ("balanced_accuracy", "ba"),
            ("sensitivity", "sens"),
            ("specificity", "spec"),
            ("f1", "f1"),
            ("tn", "tn"),
            ("fp", "fp"),
            ("fn", "fn"),
            ("tp", "tp"),
        ]:
            if src in r.index:
                put(row, f"{prefix}_{dst}", r[src])

    fpr = read_csv(oof_dir / "calib_philips_fpr_pooled.csv")
    if not fpr.empty:
        r = best_metric_row(fpr, "oof_ecdf")
        if r is None and "calib_method" in fpr.columns:
            sub = fpr[fpr["calib_method"].astype(str).eq("oof_ecdf")]
            if not sub.empty:
                r = sub.iloc[0]
        if r is not None:
            for col in ["philips_cn_fpr", "philips_cn_fp", "philips_cn_n"]:
                if col in r.index:
                    put(row, col, r[col])


def collect_from_final_decision(model_id: str, row: dict[str, Any], final_df: pd.DataFrame) -> None:
    if final_df.empty:
        return
    m = final_df[final_df["model_id"].astype(str).eq(model_id)] if "model_id" in final_df.columns else pd.DataFrame()
    if m.empty:
        return
    r = m.iloc[0]
    for src, dst in [
        ("display_name", "display_name"),
        ("decision_class", "decision_class"),
        ("final_decision", "final_decision"),
        ("run_name", "run_name"),
        ("run_dir", "run_dir"),
        ("channel_set_order", "channel_set_order"),
        ("selected_channel_names", "selected_channel_names"),
        ("beta_vae", "beta_vae"),
        ("latent_dim", "latent_dim"),
        ("philips_cn_fpr", "philips_cn_fpr"),
        ("scanner_leakage_latent_acc", "scanner_leakage_latent_acc"),
        ("D_val_best_mean", "D_val_best_mean"),
        ("R_val_bits_best_mean", "R_val_bits_best_mean"),
        ("bits_per_latent_dim_best_mean", "bits_per_latent_dim_best_mean"),
        ("beta_KLD_over_D_best_mean", "beta_KLD_over_D_best_mean"),
        ("active_units_mean", "active_units_mean"),
        ("total_correlation_nats_mean", "total_correlation_nats_mean"),
        ("MI_Z_Y_nats_mean", "MI_Z_Y_nats_mean"),
        ("MI_Z_Manufacturer_nats_mean", "MI_Z_Manufacturer_nats_mean"),
        ("MI_Manufacturer_over_MI_Y_mean", "MI_Manufacturer_over_MI_Y_mean"),
        ("oasis_concatenated_auc", "oasis_concatenated_auc"),
        ("oasis_concatenated_pr_auc", "oasis_concatenated_pr_auc"),
        ("oasis_runwise164_auc", "oasis_runwise164_auc"),
        ("oasis_runwise164_pr_auc", "oasis_runwise164_pr_auc"),
        ("oasis_runwise140_auc", "oasis_runwise140_auc"),
        ("oasis_runwise140_pr_auc", "oasis_runwise140_pr_auc"),
    ]:
        if src in r.index:
            put(row, dst, r[src])
    for metric in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]:
        src = f"adni_oof_ecdf_{metric}"
        if src in r.index:
            put(row, f"adni_oof_ecdf_{metric.replace('balanced_accuracy', 'ba')}", r[src])
    for metric in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]:
        src = f"adni_stageb_raw_{metric}"
        if src in r.index:
            put(row, f"adni_raw_{metric.replace('balanced_accuracy', 'ba')}", r[src])


def collect_from_capacity(model_id: str, row: dict[str, Any], cap: pd.DataFrame) -> None:
    if cap.empty:
        return
    cap_id = CAPACITY_MAP.get(model_id, model_id)
    m = cap[cap["run_id"].astype(str).eq(cap_id)] if "run_id" in cap.columns else pd.DataFrame()
    if m.empty:
        return
    r = m.iloc[0]
    for src, dst in [
        ("run_dir", "run_dir"),
        ("channels_to_use", "channel_set_order"),
        ("selected_channel_names", "selected_channel_names"),
        ("latent_dim", "latent_dim"),
        ("beta_vae", "beta_vae"),
        ("D_val_best_mean", "D_val_best_mean"),
        ("R_val_bits_best_mean", "R_val_bits_best_mean"),
        ("bits_per_latent_dim_best_mean", "bits_per_latent_dim_best_mean"),
        ("beta_KLD_over_D_best_mean", "beta_KLD_over_D_best_mean"),
        ("active_units_mean", "active_units_mean"),
        ("total_correlation_nats_mean", "total_correlation_nats_mean"),
        ("MI_Z_Y_nats_mean", "MI_Z_Y_nats_mean"),
        ("MI_Z_Manufacturer_nats_mean", "MI_Z_Manufacturer_nats_mean"),
        ("MI_Manufacturer_over_MI_Y_mean", "MI_Manufacturer_over_MI_Y_mean"),
        ("train_scanner_latent_ba_mean", "scanner_leakage_train_latent_acc"),
        ("test_scanner_latent_ba_mean", "scanner_leakage_latent_acc"),
        ("stageA_logreg_auc_mean", "stageA_logreg_auc"),
        ("stageA_logreg_pr_auc_mean", "stageA_logreg_pr_auc"),
        ("stageA_svm_auc_mean", "stageA_svm_auc"),
        ("stageA_svm_pr_auc_mean", "stageA_svm_pr_auc"),
        ("stageB_oof_logitz_auc", "adni_oof_logitz_auc"),
        ("stageB_oof_logitz_pr_auc", "adni_oof_logitz_pr_auc"),
        ("stageB_oof_logitz_ba", "adni_oof_logitz_ba"),
        ("stageB_oof_logitz_sens", "adni_oof_logitz_sens"),
        ("stageB_oof_logitz_spec", "adni_oof_logitz_spec"),
        ("stageB_oof_logitz_f1", "adni_oof_logitz_f1"),
        ("stageB_oof_ecdf_auc", "adni_oof_ecdf_auc"),
        ("stageB_oof_ecdf_pr_auc", "adni_oof_ecdf_pr_auc"),
        ("stageB_oof_ecdf_ba", "adni_oof_ecdf_ba"),
        ("stageB_oof_ecdf_sens", "adni_oof_ecdf_sens"),
        ("stageB_oof_ecdf_spec", "adni_oof_ecdf_spec"),
        ("stageB_oof_ecdf_f1", "adni_oof_ecdf_f1"),
        ("philips_cn_fpr_oof_ecdf", "philips_cn_fpr"),
        ("oasis_concat_auc", "oasis_concatenated_auc"),
        ("oasis_concat_pr_auc", "oasis_concatenated_pr_auc"),
        ("oasis_runwise164_auc", "oasis_runwise164_auc"),
        ("oasis_runwise164_pr_auc", "oasis_runwise164_pr_auc"),
        ("oasis_runwise140_auc", "oasis_runwise140_auc"),
        ("oasis_runwise140_pr_auc", "oasis_runwise140_pr_auc"),
    ]:
        if src in r.index:
            put(row, dst, r[src])


def collect_from_ch1_followup(model_id: str, row: dict[str, Any]) -> None:
    if model_id not in {"ch1only_latent256_beta3p75", "ch1only_latent384_beta3p25"}:
        return
    primary = read_csv(CH1_FOLLOWUP / "primary_promotion_gate_table.csv")
    src_id = model_id
    m = primary[primary["run_id"].astype(str).eq(src_id)] if not primary.empty else pd.DataFrame()
    if not m.empty:
        r = m.iloc[0]
        put(row, "display_name", r.get("display_name"))
        put(row, "decision_class", r.get("decision_class"))
        for src, dst in [
            ("auc", "adni_oof_ecdf_auc"),
            ("pr_auc", "adni_oof_ecdf_pr_auc"),
            ("balanced_accuracy", "adni_oof_ecdf_ba"),
            ("sensitivity", "adni_oof_ecdf_sens"),
            ("specificity", "adni_oof_ecdf_spec"),
            ("f1", "adni_oof_ecdf_f1"),
            ("philips_cn_fpr", "philips_cn_fpr"),
        ]:
            put(row, dst, r.get(src))
    stageb = read_csv(CH1_FOLLOWUP / "stageb_oof_calibration_metrics.csv")
    m = stageb[stageb["run_id"].astype(str).eq(src_id)] if not stageb.empty else pd.DataFrame()
    for method in ["raw", "oof_zscore", "oof_logitz", "oof_ecdf", "oof_platt", "oof_isotonic"]:
        r = best_metric_row(m, method)
        if r is None:
            continue
        for src, dst in [
            ("auc", "auc"),
            ("pr_auc", "pr_auc"),
            ("balanced_accuracy", "ba"),
            ("sensitivity", "sens"),
            ("specificity", "spec"),
            ("f1", "f1"),
        ]:
            put(row, f"adni_{method}_{dst}", r.get(src))

    stagea = read_csv(CH1_FOLLOWUP / "stagea_summary_metrics.csv")
    sm = stagea[(stagea["run_id"].astype(str).eq(src_id)) & (stagea["level"].astype(str).eq("summary"))] if not stagea.empty else pd.DataFrame()
    for clf in ["logreg", "svm"]:
        sub = sm[sm["classifier"].astype(str).eq(clf)] if not sm.empty else pd.DataFrame()
        if not sub.empty:
            r = sub.iloc[0]
            put(row, f"stageA_{clf}_auc", r.get("auc_final"))
            put(row, f"stageA_{clf}_pr_auc", r.get("pr_auc_final"))

    leak = read_csv(CH1_FOLLOWUP / "scanner_leakage_summary.csv")
    lm = leak[(leak["run_id"].astype(str).eq(src_id)) & (leak["fold"].astype(str).eq("summary"))] if not leak.empty else pd.DataFrame()
    if not lm.empty:
        for split, dst in [("train_dev", "scanner_leakage_train_latent_acc"), ("test", "scanner_leakage_latent_acc")]:
            sub = lm[lm["split"].astype(str).eq(split)]
            if not sub.empty:
                put(row, dst, sub.iloc[0].get("acc_site_latent"))

    rd = read_csv(CH1_FOLLOWUP / "rate_distortion_summary.csv")
    rm = rd[(rd["run_id"].astype(str).eq(src_id)) & (rd["fold"].astype(str).eq("summary")) & (rd["summary_point"].astype(str).eq("best_L_val_betaMax"))] if not rd.empty else pd.DataFrame()
    if not rm.empty:
        r = rm.iloc[0]
        for src, dst in [
            ("latent_dim", "latent_dim"),
            ("beta_vae", "beta_vae"),
            ("D_val", "D_val_best_mean"),
            ("R_val_bits", "R_val_bits_best_mean"),
            ("bits_per_dim", "bits_per_latent_dim_best_mean"),
            ("beta_kld_over_D", "beta_KLD_over_D_best_mean"),
        ]:
            put(row, dst, r.get(src))

    mi = read_csv(CH1_FOLLOWUP / "latent_mi_signal_nuisance_summary.csv")
    mm = mi[(mi["run_id"].astype(str).eq(src_id)) & (mi["fold"].astype(str).eq("summary")) & (mi["split"].astype(str).eq("test"))] if not mi.empty else pd.DataFrame()
    if not mm.empty:
        r = mm.iloc[0]
        for src, dst in [
            ("active_units", "active_units_mean"),
            ("total_correlation_nats", "total_correlation_nats_mean"),
            ("mi_y_sum_nats", "MI_Z_Y_nats_mean"),
            ("mi_manufacturer_sum_nats", "MI_Z_Manufacturer_nats_mean"),
            ("mi_manufacturer_over_y", "MI_Manufacturer_over_MI_Y_mean"),
        ]:
            put(row, dst, r.get(src))

    oasis = read_csv(CH1_FOLLOWUP / "oasis_external_metrics.csv")
    if not oasis.empty:
        o = oasis[oasis["model_id"].astype(str).eq(src_id)] if "model_id" in oasis.columns else pd.DataFrame()
        if o.empty and "candidate" in oasis.columns:
            o = oasis[oasis["candidate"].astype(str).str.contains(src_id, na=False)]
        for build, prefix in [
            ("concatenated_timeseries", "oasis_concatenated"),
            ("runwise164_pilot_parity", "oasis_runwise164"),
            ("runwise_140TR_pilot_parity", "oasis_runwise140"),
            ("runwise140_pilot_parity", "oasis_runwise140"),
        ]:
            sub = o[o["build_candidate"].astype(str).eq(build)]
            if not sub.empty:
                r = sub.iloc[0]
                put(row, f"{prefix}_auc", r.get("auc"))
                put(row, f"{prefix}_pr_auc", r.get("pr_auc"))


def collect_from_ch12(model_id: str, row: dict[str, Any]) -> None:
    if model_id != "ch12_latent384_beta3p75":
        return
    primary = read_csv(CH12_AUDIT / "primary_promotion_gate_table.csv")
    if not primary.empty:
        if "role" in primary.columns:
            candidates = primary[primary["role"].astype(str).str.contains("candidate|ch12", case=False, na=False)]
        else:
            candidates = pd.DataFrame()
        if candidates.empty and "run_id" in primary.columns:
            candidates = primary[primary["run_id"].astype(str).str.contains("ch12|ch1_2|ch1only", case=False, na=False)]
        if candidates.empty and "run_label" in primary.columns:
            candidates = primary[primary["run_label"].astype(str).str.contains("candidate.*ch12|ch12", case=False, na=False)]
        if candidates.empty:
            candidates = primary.tail(1)
        r = candidates.iloc[0]
        for src, dst in [
            ("auc", "adni_oof_ecdf_auc"),
            ("pr_auc", "adni_oof_ecdf_pr_auc"),
            ("balanced_accuracy", "adni_oof_ecdf_ba"),
            ("sensitivity", "adni_oof_ecdf_sens"),
            ("specificity", "adni_oof_ecdf_spec"),
            ("f1", "adni_oof_ecdf_f1"),
            ("philips_cn_fpr", "philips_cn_fpr"),
            ("philips_cn_fpr_primary", "philips_cn_fpr"),
            ("test_latent_scanner_ba_mean", "scanner_leakage_latent_acc"),
            ("D_val_best_mean", "D_val_best_mean"),
            ("R_val_bits_best_mean", "R_val_bits_best_mean"),
            ("R_bits_per_latent_dim_best_mean", "bits_per_latent_dim_best_mean"),
            ("beta_KLD_over_D_best_mean", "beta_KLD_over_D_best_mean"),
            ("MI_Z_Y_nats_mean", "MI_Z_Y_nats_mean"),
            ("MI_Z_Manufacturer_nats_mean", "MI_Z_Manufacturer_nats_mean"),
            ("MI_Manufacturer_over_MI_Y_mean", "MI_Manufacturer_over_MI_Y_mean"),
            ("active_units_mean", "active_units_mean"),
            ("total_correlation_nats_mean", "total_correlation_nats_mean"),
        ]:
            put(row, dst, r.get(src))
    stageb = read_csv(CH12_AUDIT / "stageb_pooled_metrics.csv")
    if not stageb.empty:
        for method in ["raw", "oof_logitz", "oof_ecdf"]:
            r = best_metric_row(stageb, method)
            if r is not None:
                for src, dst in [
                    ("auc", "auc"),
                    ("pr_auc", "pr_auc"),
                    ("balanced_accuracy", "ba"),
                    ("sensitivity", "sens"),
                    ("specificity", "spec"),
                    ("f1", "f1"),
                ]:
                    put(row, f"adni_{method}_{dst}", r.get(src))
    rd = read_csv(CH12_AUDIT / "rate_distortion_summary.csv")
    if not rd.empty:
        r = rd.iloc[0]
        for src, dst in [
            ("D_val", "D_val_best_mean"),
            ("R_val_bits", "R_val_bits_best_mean"),
            ("bits_per_dim", "bits_per_latent_dim_best_mean"),
            ("beta_kld_over_D", "beta_KLD_over_D_best_mean"),
        ]:
            put(row, dst, r.get(src))
    leak = read_csv(CH12_AUDIT / "scanner_leakage_summary.csv")
    if not leak.empty:
        for split, dst in [("train_dev", "scanner_leakage_train_latent_acc"), ("test", "scanner_leakage_latent_acc")]:
            sub = leak[leak["split"].astype(str).eq(split)]
            if not sub.empty:
                put(row, dst, sub.iloc[0].get("acc_site_latent"))
    mi = read_csv(CH12_AUDIT / "latent_mi_signal_nuisance_summary.csv")
    if not mi.empty:
        r = mi.iloc[0]
        for src, dst in [
            ("active_units", "active_units_mean"),
            ("total_correlation_nats", "total_correlation_nats_mean"),
            ("mi_y_sum_nats", "MI_Z_Y_nats_mean"),
            ("mi_manufacturer_sum_nats", "MI_Z_Manufacturer_nats_mean"),
            ("mi_manufacturer_over_y", "MI_Manufacturer_over_MI_Y_mean"),
        ]:
            put(row, dst, r.get(src))


def collect_from_mfr(model_id: str, row: dict[str, Any]) -> None:
    if model_id != "mfrBalancedVAE_latent384_beta3p75":
        return
    pooled = read_csv(MFR_AUDIT / "stageb_oof_all_methods_pooled_primary_model.csv")
    if not pooled.empty:
        for method in ["raw", "oof_zscore", "oof_logitz", "oof_ecdf", "oof_platt", "oof_isotonic"]:
            r = best_metric_row(pooled, method)
            if r is not None:
                for src, dst in [
                    ("auc", "auc"),
                    ("pr_auc", "pr_auc"),
                    ("balanced_accuracy", "ba"),
                    ("sensitivity", "sens"),
                    ("specificity", "spec"),
                    ("f1", "f1"),
                ]:
                    put(row, f"adni_{method}_{dst}", r.get(src))
    fpr = read_csv(MFR_AUDIT / "philips_cn_fpr_promoted_convention.csv")
    if not fpr.empty:
        r = fpr.iloc[0]
        put(row, "philips_cn_fpr", r.get("philips_cn_fpr"))
    leak = read_csv(MFR_AUDIT / "scanner_leakage_summary.csv")
    if not leak.empty:
        for split, dst in [("train_dev", "scanner_leakage_train_latent_acc"), ("test", "scanner_leakage_latent_acc")]:
            sub = leak[leak["split"].astype(str).eq(split)]
            if not sub.empty:
                put(row, dst, sub.iloc[0].get("acc_site_latent"))
    vae = read_csv(MFR_AUDIT / "vae_qc_summary.csv")
    if not vae.empty:
        r = vae.iloc[0]
        for src, dst in [
            ("D_val", "D_val_best_mean"),
            ("R_val_bits", "R_val_bits_best_mean"),
            ("bits_per_dim", "bits_per_latent_dim_best_mean"),
            ("beta_kld_over_D", "beta_KLD_over_D_best_mean"),
            ("active_units", "active_units_mean"),
            ("total_correlation_nats", "total_correlation_nats_mean"),
            ("mi_y_sum_nats", "MI_Z_Y_nats_mean"),
            ("mi_manufacturer_sum_nats", "MI_Z_Manufacturer_nats_mean"),
            ("mi_manufacturer_over_y", "MI_Manufacturer_over_MI_Y_mean"),
        ]:
            put(row, dst, r.get(src))


def collect_from_harmonized(model_id: str, row: dict[str, Any]) -> None:
    if model_id != "residualized_mfr_stageB":
        return
    pooled = read_csv(HARM_AUDIT / "pooled_metrics.csv")
    if pooled.empty:
        return
    preferred = pooled[
        (pooled.get("harmonization_method", pd.Series(dtype=str)).astype(str).str.contains("residual", case=False, na=False))
        & (pooled.get("calib_method", pd.Series(dtype=str)).astype(str).eq("oof_ecdf"))
        & (pooled.get("threshold_strategy", pd.Series(dtype=str)).astype(str).eq(PRIMARY_THRESHOLD))
    ]
    if preferred.empty:
        preferred = pooled[
            (pooled.get("calib_method", pd.Series(dtype=str)).astype(str).eq("oof_ecdf"))
            & (pooled.get("threshold_strategy", pd.Series(dtype=str)).astype(str).eq(PRIMARY_THRESHOLD))
        ]
    if preferred.empty:
        preferred = pooled.tail(1)
    r = preferred.iloc[0]
    for src, dst in [
        ("auc", "adni_oof_ecdf_auc"),
        ("pr_auc", "adni_oof_ecdf_pr_auc"),
        ("balanced_accuracy", "adni_oof_ecdf_ba"),
        ("sensitivity", "adni_oof_ecdf_sens"),
        ("specificity", "adni_oof_ecdf_spec"),
        ("f1", "adni_oof_ecdf_f1"),
        ("philips_cn_fpr", "philips_cn_fpr"),
        ("mean_latent_manufacturer_ba", "scanner_leakage_latent_acc"),
    ]:
        put(row, dst, r.get(src))


def add_oasis_from_all(row: dict[str, Any], oasis: pd.DataFrame, model_id: str) -> None:
    if oasis.empty:
        return
    o = oasis[oasis["model_id"].astype(str).eq(model_id)] if "model_id" in oasis.columns else pd.DataFrame()
    if o.empty:
        return
    for build, prefix in [
        ("concatenated_timeseries", "oasis_concatenated"),
        ("runwise164_pilot_parity", "oasis_runwise164"),
        ("runwise_140TR_pilot_parity", "oasis_runwise140"),
        ("runwise140_pilot_parity", "oasis_runwise140"),
    ]:
        sub = o[o["build_candidate"].astype(str).eq(build)] if "build_candidate" in o.columns else pd.DataFrame()
        if not sub.empty:
            r = sub.iloc[0]
            put(row, f"{prefix}_auc", r.get("auc"))
            put(row, f"{prefix}_pr_auc", r.get("pr_auc"))


def add_stagea_stageb_delta(row: dict[str, Any]) -> None:
    stagea_auc = row.get("stageA_logreg_auc")
    stagea_pr = row.get("stageA_logreg_pr_auc")
    stageb_auc = row.get("adni_oof_ecdf_auc")
    stageb_pr = row.get("adni_oof_ecdf_pr_auc")
    if pd.notna(stagea_auc) and pd.notna(stageb_auc):
        row["stageB_minus_stageA_logreg_auc"] = stageb_auc - stagea_auc
    if pd.notna(stagea_pr) and pd.notna(stageb_pr):
        row["stageB_minus_stageA_logreg_pr_auc"] = stageb_pr - stagea_pr


def to_md(df: pd.DataFrame, path: Path) -> None:
    path.write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")


def rank_pareto(evidence: pd.DataFrame) -> pd.DataFrame:
    cols = ["adni_oof_ecdf_auc", "adni_oof_ecdf_pr_auc", "philips_cn_fpr", "oasis_runwise164_auc", "oasis_runwise140_auc"]
    df = evidence.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    candidates = df[df[cols].notna().any(axis=1)].copy()
    records = []
    for _, a in candidates.iterrows():
        dominated_by = []
        for _, b in candidates.iterrows():
            if a["model_id"] == b["model_id"]:
                continue
            comparisons = []
            strictly_better = False
            for c in cols:
                av = a.get(c)
                bv = b.get(c)
                if pd.isna(av) or pd.isna(bv):
                    continue
                if c == "philips_cn_fpr":
                    comparisons.append(bv <= av)
                    strictly_better |= bool(bv < av)
                else:
                    comparisons.append(bv >= av)
                    strictly_better |= bool(bv > av)
            if comparisons and all(comparisons) and strictly_better:
                dominated_by.append(str(b["model_id"]))
        records.append(
            {
                "model_id": a["model_id"],
                "display_name": a.get("display_name"),
                "decision_class": a.get("decision_class"),
                "adni_oof_ecdf_auc": a.get("adni_oof_ecdf_auc"),
                "adni_oof_ecdf_pr_auc": a.get("adni_oof_ecdf_pr_auc"),
                "philips_cn_fpr": a.get("philips_cn_fpr"),
                "oasis_runwise164_auc": a.get("oasis_runwise164_auc"),
                "oasis_runwise140_auc": a.get("oasis_runwise140_auc"),
                "pareto_non_dominated": len(dominated_by) == 0,
                "dominated_by": "; ".join(dominated_by),
            }
        )
    out = pd.DataFrame(records)
    if not out.empty:
        out = out.sort_values(["pareto_non_dominated", "adni_oof_ecdf_auc", "adni_oof_ecdf_pr_auc"], ascending=[False, False, False])
    return out


def write_text_summaries(evidence: pd.DataFrame, pareto: pd.DataFrame) -> None:
    primary = evidence[evidence["model_id"].eq("promoted_latent384_beta3p75_ch1_0_2")]
    ch1 = evidence[evidence["model_id"].eq("ch1only_latent384_beta3p75")]
    ch1_fups = evidence[evidence["model_id"].isin(["ch1only_latent256_beta3p75", "ch1only_latent384_beta3p25"])]
    cap = evidence[evidence["decision_class"].astype(str).str.contains("capacity", na=False)]
    beta = evidence[evidence["decision_class"].astype(str).str.contains("beta", na=False)]

    def fmt(row: pd.DataFrame, col: str) -> str:
        if row.empty or col not in row.columns or pd.isna(row.iloc[0][col]):
            return "NA"
        return f"{row.iloc[0][col]:.6f}"

    trend = [
        "# Trend Interpretation",
        "",
        "This evidence map is a read-only aggregation of completed FULL 5x5 and classifier-only sensitivity audits. It does not retrain models, refit thresholds, or create new OASIS calibrations.",
        "",
        "## Capacity",
        "Increasing latent capacity beyond the promoted latent384 setting did not reopen model selection. Latent448 beta4.0 and latent512 beta3.75 remained below the promoted ADNI PR-AUC/AUC balance and did not show a decisive OASIS advantage.",
        "",
        "## Beta",
        "The local beta sweep around the promoted beta3.75 did not identify a cleaner primary model. Beta3.5 reduced ADNI OOF-ECDF AUC/PR-AUC relative to the promoted model, while higher-beta/capacity combinations remained sensitivity analyses.",
        "",
        "## Channel Count",
        f"Ch1-only latent384 beta3.75 has the strongest internal ch1-only ranking signal (AUC {fmt(ch1, 'adni_oof_ecdf_auc')}, PR-AUC {fmt(ch1, 'adni_oof_ecdf_pr_auc')}), but it does not displace the promoted multichannel model because BA/F1, Philips CN FPR, and OASIS transfer are not clearly better. The two ch1 follow-ups did not fix that tradeoff.",
        "",
        "## OASIS Transfer",
        "OASIS remains an external stress test rather than a tuning set. The promoted [1,0,2] model retains the strongest overall runwise OASIS profile among primary candidates, while OASIS thresholds remain conservative and source-batch/domain effects remain material.",
    ]
    (OUT / "trend_interpretation.md").write_text("\n".join(trend) + "\n", encoding="utf-8")

    nd = pareto[pareto["pareto_non_dominated"].fillna(False)] if not pareto.empty else pd.DataFrame()
    nd_models = ", ".join(nd["display_name"].dropna().astype(str).tolist()) if not nd.empty else "none"
    recommendation = [
        "# Final Recommendation",
        "",
        "Decision: no model reopens primary selection.",
        "",
        f"The promoted [1,0,2] latent384 beta3.75 model remains the primary manuscript model (ADNI OOF-ECDF AUC {fmt(primary, 'adni_oof_ecdf_auc')}, PR-AUC {fmt(primary, 'adni_oof_ecdf_pr_auc')}).",
        "",
        "The ch1-only latent384 beta3.75 model remains a parsimony sensitivity model, not a primary replacement. The ch1 latent256 beta3.75 and ch1 latent384 beta3.25 follow-ups are not promoted.",
        "",
        "Capacity, beta, deconfounding, residualized Stage B, and locked/recover035 latent256 rows should be reported as sensitivity or negative controls according to their decision_class.",
        "",
        f"Pareto non-dominated rows under the requested objective set: {nd_models}. This Pareto result is descriptive only; it does not override the pre-specified primary-selection guardrails.",
        "",
        "No OASIS-based model selection was performed.",
    ]
    (OUT / "final_recommendation.md").write_text("\n".join(recommendation) + "\n", encoding="utf-8")

    readme = [
        "# Final FULL-Model Evidence Map 20260606",
        "",
        "Read-only aggregation package over completed ADNI FULL 5x5, ch1-only follow-up, and available OASIS external validation artifacts.",
        "",
        "Guardrails honored:",
        "- no VAE training",
        "- no new OASIS scoring beyond existing frozen artifacts",
        "- no threshold or calibration fitting",
        "- no tensor, metadata, ledger, or model artifact modification",
        "",
        "Primary model remains promoted [1,0,2] latent384 beta3.75.",
    ]
    (OUT / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)

    final_df = read_csv(FINAL_DECISION)
    cap = read_csv(CAPACITY)
    oasis = read_csv(OASIS_ALL)

    rows: list[dict[str, Any]] = []
    for model_id in MODEL_ORDER:
        row: dict[str, Any] = {
            "model_id": model_id,
            "display_name": DISPLAY.get(model_id, model_id),
            "decision_class": DECISION_DEFAULTS.get(model_id, "sensitivity"),
        }
        collect_from_final_decision(model_id, row, final_df)
        collect_from_capacity(model_id, row, cap)
        collect_from_ch1_followup(model_id, row)
        collect_from_ch12(model_id, row)
        collect_from_mfr(model_id, row)
        collect_from_harmonized(model_id, row)
        add_oasis_from_all(row, oasis, model_id)

        # Pull OOF dirs for models with standard packages when available.
        oof_dir_map = {
            "promoted_latent384_beta3p75_ch1_0_2": ROOT / "recover035_latent384_beta3p75_stageB_oof_score_calibration",
            "ch1only_latent384_beta3p75": ROOT / "recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5_stageB_oof_score_calibration",
            "latent384_beta3p5": ROOT / "recover035_latent384_beta3p5_stageB_oof_score_calibration",
            "latent448_beta4p0": ROOT / "recover035_latent448_beta4p0_stageB_oof_score_calibration",
            "latent512_beta3p75": ROOT / "recover035_latent512_beta3p75_stageB_oof_score_calibration",
            "ch1only_latent256_beta3p75": ROOT / "ch1only_latent256_beta3p75_T80_h10000_p560_full5x5_stageB_oof_score_calibration",
            "ch1only_latent384_beta3p25": ROOT / "ch1only_latent384_beta3p25_T80_h10000_p560_full5x5_stageB_oof_score_calibration",
        }
        if model_id in oof_dir_map and oof_dir_map[model_id].exists():
            collect_oof_dir_metrics(model_id, row, oof_dir_map[model_id])

        add_stagea_stageb_delta(row)
        rows.append(row)

    evidence = pd.DataFrame(rows)

    # Normalize aliases and ensure requested columns are present.
    for col in [
        "adni_oof_ecdf_auc",
        "adni_oof_ecdf_pr_auc",
        "adni_oof_logitz_auc",
        "adni_oof_logitz_pr_auc",
        "adni_oof_ecdf_ba",
        "adni_oof_ecdf_sens",
        "adni_oof_ecdf_spec",
        "adni_oof_ecdf_f1",
        "philips_cn_fpr",
        "scanner_leakage_train_latent_acc",
        "scanner_leakage_latent_acc",
        "D_val_best_mean",
        "R_val_bits_best_mean",
        "beta_KLD_over_D_best_mean",
        "bits_per_latent_dim_best_mean",
        "active_units_mean",
        "total_correlation_nats_mean",
        "MI_Z_Y_nats_mean",
        "MI_Z_Manufacturer_nats_mean",
        "MI_Manufacturer_over_MI_Y_mean",
        "oasis_concatenated_auc",
        "oasis_concatenated_pr_auc",
        "oasis_runwise164_auc",
        "oasis_runwise164_pr_auc",
        "oasis_runwise140_auc",
        "oasis_runwise140_pr_auc",
    ]:
        if col not in evidence.columns:
            evidence[col] = np.nan

    evidence = evidence[
        [
            "model_id",
            "display_name",
            "decision_class",
            "final_decision",
            "run_name",
            "run_dir",
            "channel_set_order",
            "selected_channel_names",
            "latent_dim",
            "beta_vae",
            "adni_oof_ecdf_auc",
            "adni_oof_ecdf_pr_auc",
            "adni_oof_logitz_auc",
            "adni_oof_logitz_pr_auc",
            "adni_raw_auc",
            "adni_raw_pr_auc",
            "adni_oof_ecdf_ba",
            "adni_oof_ecdf_sens",
            "adni_oof_ecdf_spec",
            "adni_oof_ecdf_f1",
            "philips_cn_fpr",
            "stageA_logreg_auc",
            "stageA_logreg_pr_auc",
            "stageA_svm_auc",
            "stageA_svm_pr_auc",
            "stageB_minus_stageA_logreg_auc",
            "stageB_minus_stageA_logreg_pr_auc",
            "scanner_leakage_train_latent_acc",
            "scanner_leakage_latent_acc",
            "D_val_best_mean",
            "R_val_bits_best_mean",
            "beta_KLD_over_D_best_mean",
            "bits_per_latent_dim_best_mean",
            "active_units_mean",
            "total_correlation_nats_mean",
            "MI_Z_Y_nats_mean",
            "MI_Z_Manufacturer_nats_mean",
            "MI_Manufacturer_over_MI_Y_mean",
            "oasis_concatenated_auc",
            "oasis_concatenated_pr_auc",
            "oasis_runwise164_auc",
            "oasis_runwise164_pr_auc",
            "oasis_runwise140_auc",
            "oasis_runwise140_pr_auc",
        ]
    ]

    evidence.to_csv(OUT / "full_model_evidence_map.csv", index=False)
    to_md(evidence, OUT / "full_model_evidence_map.md")

    pareto = rank_pareto(evidence)
    pareto.to_csv(OUT / "pareto_table.csv", index=False)
    to_md(pareto, OUT / "pareto_table.md")

    optimism_cols = [
        "model_id",
        "display_name",
        "stageA_logreg_auc",
        "stageA_logreg_pr_auc",
        "adni_oof_ecdf_auc",
        "adni_oof_ecdf_pr_auc",
        "stageB_minus_stageA_logreg_auc",
        "stageB_minus_stageA_logreg_pr_auc",
    ]
    optimism = evidence[optimism_cols].copy()
    optimism.to_csv(OUT / "stageA_to_stageB_optimism_table.csv", index=False)
    to_md(optimism, OUT / "stageA_to_stageB_optimism_table.md")

    compact_cols = [
        "model_id",
        "display_name",
        "decision_class",
        "latent_dim",
        "beta_vae",
        "channel_set_order",
        "adni_oof_ecdf_auc",
        "adni_oof_ecdf_pr_auc",
        "adni_oof_logitz_auc",
        "adni_oof_logitz_pr_auc",
        "adni_oof_ecdf_ba",
        "adni_oof_ecdf_sens",
        "adni_oof_ecdf_spec",
        "adni_oof_ecdf_f1",
        "philips_cn_fpr",
        "scanner_leakage_latent_acc",
        "oasis_concatenated_auc",
        "oasis_concatenated_pr_auc",
        "oasis_runwise164_auc",
        "oasis_runwise164_pr_auc",
        "oasis_runwise140_auc",
        "oasis_runwise140_pr_auc",
    ]
    compact = evidence[compact_cols].copy()
    compact.to_csv(OUT / "model_decision_compact_table.csv", index=False)
    to_md(compact, OUT / "model_decision_compact_table.md")

    write_text_summaries(evidence, pareto)

    command_log = {
        "created_at": datetime.now().isoformat(),
        "script": str(Path(__file__)),
        "output_dir": str(OUT),
        "guardrails": [
            "no training",
            "no scoring unless existing frozen OASIS artifacts are already available",
            "no threshold/calibration fitting",
            "no tensor/metadata/model artifact modification",
        ],
        "source_files": {
            "final_decision": str(FINAL_DECISION),
            "capacity_synthesis": str(CAPACITY),
            "oasis_all_models": str(OASIS_ALL),
            "ch1_followup_dir": str(CH1_FOLLOWUP),
            "ch12_audit_dir": str(CH12_AUDIT),
            "mfr_audit_dir": str(MFR_AUDIT),
            "harmonization_audit_dir": str(HARM_AUDIT),
        },
        "outputs": sorted(p.name for p in OUT.iterdir()),
    }
    (OUT / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
