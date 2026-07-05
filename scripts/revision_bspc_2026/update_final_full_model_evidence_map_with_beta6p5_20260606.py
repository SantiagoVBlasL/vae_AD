#!/usr/bin/env python3
"""Create final FULL-model evidence map package with beta6.5 added.

Inputs are existing read-only audit packages:
- final_full_model_evidence_map_20260606
- recover035_latent384_beta6p5_completion_promotion_gate_audit_20260606

The script writes a new derived package only. It does not train, score OASIS,
fit thresholds/calibrators, or modify tensor/metadata/model artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("results/revision_bspc_2026")
BASE = ROOT / "final_full_model_evidence_map_20260606"
BETA6 = ROOT / "recover035_latent384_beta6p5_completion_promotion_gate_audit_20260606"
OUT_DEFAULT = ROOT / "final_full_model_evidence_map_with_beta6p5_20260606"

PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
PRIMARY_MODEL_RAW = "logreg_l2"
PRIMARY_MODEL_OOF = "logreg_l2_original"
PRIMARY_FEATURES = "z_plus_age_sex"
PRIMARY_CALIB = "oof_ecdf"

BETA6_MODEL_ID = "latent384_beta6p5"
BETA6_DISPLAY = "latent384 beta6.5"

BETA_AXIS_ORDER = [
    "latent384_beta3p5",
    "promoted_latent384_beta3p75_ch1_0_2",
    "latent384_beta4p0",
    BETA6_MODEL_ID,
    "ch1only_latent384_beta3p75",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path)


def safe_float(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def put(row: dict[str, Any], key: str, value: Any) -> None:
    try:
        if pd.isna(value):
            return
    except TypeError:
        pass
    row[key] = value


def to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._\n"
    return df.to_markdown(index=False) + "\n"


def write_table(df: pd.DataFrame, stem: str, out: Path) -> None:
    df.to_csv(out / f"{stem}.csv", index=False)
    (out / f"{stem}.md").write_text(to_md(df), encoding="utf-8")


def row_first(df: pd.DataFrame, **criteria: str) -> pd.Series | None:
    if df.empty:
        return None
    mask = pd.Series(True, index=df.index)
    for col, value in criteria.items():
        if col not in df.columns:
            return None
        mask &= df[col].astype(str).eq(str(value))
    rows = df[mask]
    if rows.empty:
        return None
    return rows.iloc[0]


def beta6_primary_row(calib_method: str = PRIMARY_CALIB) -> pd.Series | None:
    df = read_csv(BETA6 / "stageb_primary_rows.csv")
    return row_first(
        df,
        run_id="candidate_latent384_beta6p5",
        model_name=PRIMARY_MODEL_OOF,
        feature_set=PRIMARY_FEATURES,
        calib_method=calib_method,
        threshold_strategy=PRIMARY_THRESHOLD,
    )


def beta6_raw_row() -> pd.Series | None:
    df = read_csv(BETA6 / "stageb_oof_calibration_metrics.csv")
    rows = df[
        df["run_id"].astype(str).eq("candidate_latent384_beta6p5")
        & df["model_name"].astype(str).eq(PRIMARY_MODEL_RAW)
        & df["feature_set"].astype(str).eq(PRIMARY_FEATURES)
        & df["calib_method"].astype(str).eq("raw")
        & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    ]
    if rows.empty:
        return None
    return rows.iloc[0]


def beta6_stagea(model: str) -> pd.Series | None:
    df = read_csv(BETA6 / "stagea_summary_metrics.csv")
    return row_first(df, run_id="candidate_latent384_beta6p5", stageA_model=model)


def beta6_philips_fpr() -> tuple[float, float, float]:
    df = read_csv(BETA6 / "philips_cn_fpr.csv")
    rows = df[
        df["run_id"].astype(str).eq("candidate_latent384_beta6p5")
        & df["manufacturer"].astype(str).str.lower().eq("philips")
        & df["calib_method"].astype(str).eq(PRIMARY_CALIB)
        & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    ]
    if rows.empty:
        return float("nan"), float("nan"), float("nan")
    r = rows.iloc[0]
    return safe_float(r.get("fp_cn_pooled")), safe_float(r.get("n_cn_pooled")), safe_float(r.get("fpr_cn_pooled"))


def summary_row(path: Path, run_id: str, extra_filter: dict[str, str] | None = None) -> pd.Series | None:
    df = read_csv(path)
    rows = df[df["run_id"].astype(str).eq(run_id)]
    if extra_filter:
        for col, value in extra_filter.items():
            rows = rows[rows[col].astype(str).eq(value)] if col in rows.columns else pd.DataFrame()
    if rows.empty:
        return None
    return rows.iloc[0]


def beta6_evidence_row(columns: list[str]) -> dict[str, Any]:
    ecdf = beta6_primary_row("oof_ecdf")
    logitz = beta6_primary_row("oof_logitz")
    raw = beta6_raw_row()
    stagea_logreg = beta6_stagea("logreg")
    stagea_svm = beta6_stagea("svm")
    rd = summary_row(BETA6 / "rate_distortion_summary.csv", "candidate_latent384_beta6p5")
    latent = summary_row(
        BETA6 / "latent_mi_signal_nuisance_summary.csv",
        "candidate_latent384_beta6p5",
        {"split": "trainDev"},
    )
    scanner_test = summary_row(
        BETA6 / "scanner_leakage_summary.csv",
        "candidate_latent384_beta6p5",
        {"split": "test"},
    )
    scanner_train = summary_row(
        BETA6 / "scanner_leakage_summary.csv",
        "candidate_latent384_beta6p5",
        {"split": "trainDev"},
    )
    fp, n_cn, fpr = beta6_philips_fpr()

    row: dict[str, Any] = {col: np.nan for col in columns}
    row.update(
        {
            "model_id": BETA6_MODEL_ID,
            "display_name": BETA6_DISPLAY,
            "decision_class": "beta sensitivity rejected",
            "final_decision": "reject_beta_sensitivity_only",
            "run_name": "recover035_latent384_beta6p5_T80_h10000_p560_full5x5",
            "run_dir": "results/revision_bspc_2026/recover035_latent384_beta6p5_T80_h10000_p560_full5x5",
            "channel_set_order": "[1, 0, 2]",
            "selected_channel_names": '["Pearson_Full_FisherZ_Signed", "Pearson_OMST_GCE_Signed_Weighted", "MI_KNN_Symmetric"]',
            "latent_dim": 384.0,
            "beta_vae": 6.5,
            "oasis_status": "not_scored_adni_gate_failed_optional_sensitivity_only",
        }
    )

    if ecdf is not None:
        for src, dst in [
            ("auc", "adni_oof_ecdf_auc"),
            ("pr_auc", "adni_oof_ecdf_pr_auc"),
            ("balanced_accuracy", "adni_oof_ecdf_ba"),
            ("sensitivity", "adni_oof_ecdf_sens"),
            ("specificity", "adni_oof_ecdf_spec"),
            ("f1", "adni_oof_ecdf_f1"),
        ]:
            put(row, dst, ecdf.get(src))
    if logitz is not None:
        for src, dst in [
            ("auc", "adni_oof_logitz_auc"),
            ("pr_auc", "adni_oof_logitz_pr_auc"),
            ("balanced_accuracy", "adni_oof_logitz_ba"),
            ("sensitivity", "adni_oof_logitz_sens"),
            ("specificity", "adni_oof_logitz_spec"),
            ("f1", "adni_oof_logitz_f1"),
        ]:
            put(row, dst, logitz.get(src))
    if raw is not None:
        for src, dst in [
            ("auc", "adni_raw_auc"),
            ("pr_auc", "adni_raw_pr_auc"),
            ("balanced_accuracy", "adni_raw_ba"),
            ("sensitivity", "adni_raw_sens"),
            ("specificity", "adni_raw_spec"),
            ("f1", "adni_raw_f1"),
        ]:
            put(row, dst, raw.get(src))
    if stagea_logreg is not None:
        put(row, "stageA_logreg_auc", stagea_logreg.get("auc_final_mean"))
        put(row, "stageA_logreg_pr_auc", stagea_logreg.get("pr_auc_final_mean"))
    if stagea_svm is not None:
        put(row, "stageA_svm_auc", stagea_svm.get("auc_final_mean"))
        put(row, "stageA_svm_pr_auc", stagea_svm.get("pr_auc_final_mean"))
    if ecdf is not None and stagea_logreg is not None:
        put(row, "stageB_minus_stageA_logreg_auc", safe_float(ecdf.get("auc")) - safe_float(stagea_logreg.get("auc_final_mean")))
        put(row, "stageB_minus_stageA_logreg_pr_auc", safe_float(ecdf.get("pr_auc")) - safe_float(stagea_logreg.get("pr_auc_final_mean")))
    put(row, "philips_cn_fpr", fpr)
    put(row, "philips_cn_fp", fp)
    put(row, "philips_cn_n", n_cn)

    if scanner_train is not None:
        put(row, "scanner_leakage_train_latent_acc", scanner_train.get("acc_site_latent_mean"))
    if scanner_test is not None:
        put(row, "scanner_leakage_latent_acc", scanner_test.get("acc_site_latent_mean"))
    if rd is not None:
        for src, dst in [
            ("D_val_best_mean", "D_val_best_mean"),
            ("R_val_bits_best_mean", "R_val_bits_best_mean"),
            ("beta_KLD_over_D_best_mean", "beta_KLD_over_D_best_mean"),
            ("bits_per_dim_best_mean", "bits_per_latent_dim_best_mean"),
        ]:
            put(row, dst, rd.get(src))
    if latent is not None:
        for src, dst in [
            ("active_units_mean", "active_units_mean"),
            ("total_correlation_nats_mean", "total_correlation_nats_mean"),
            ("MI_Z_Y_nats_mean", "MI_Z_Y_nats_mean"),
            ("MI_Z_Manufacturer_nats_mean", "MI_Z_Manufacturer_nats_mean"),
            ("MI_Manufacturer_over_MI_Y_mean", "MI_Manufacturer_over_MI_Y_mean"),
        ]:
            put(row, dst, latent.get(src))
    return row


def append_or_replace(df: pd.DataFrame, row: dict[str, Any]) -> pd.DataFrame:
    if "model_id" in df.columns:
        df = df[~df["model_id"].astype(str).eq(str(row["model_id"]))].copy()
    for col in row:
        if col not in df.columns:
            df[col] = np.nan
    for col in df.columns:
        if col not in row:
            row[col] = np.nan
    out = pd.concat([df, pd.DataFrame([row])[df.columns]], ignore_index=True)
    return out


def compact_from_evidence(evidence: pd.DataFrame) -> pd.DataFrame:
    cols = [
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
    for col in cols:
        if col not in evidence.columns:
            evidence[col] = np.nan
    return evidence[cols].copy()


def rank_pareto(evidence: pd.DataFrame) -> pd.DataFrame:
    cols = ["adni_oof_ecdf_auc", "adni_oof_ecdf_pr_auc", "philips_cn_fpr", "oasis_runwise164_auc", "oasis_runwise140_auc"]
    required_core = ["adni_oof_ecdf_auc", "adni_oof_ecdf_pr_auc", "philips_cn_fpr"]
    records: list[dict[str, Any]] = []
    df = evidence.copy()
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
    for _, a in df.iterrows():
        if any(math.isnan(safe_float(a.get(col))) for col in required_core):
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
                    "pareto_non_dominated": False,
                    "dominated_by": "insufficient_core_metrics_for_pareto",
                }
            )
            continue
        dominated_by: list[str] = []
        for _, b in df.iterrows():
            if str(a["model_id"]) == str(b["model_id"]):
                continue
            comparisons: list[bool] = []
            strictly_better = False
            for col in cols:
                av = safe_float(a.get(col))
                bv = safe_float(b.get(col))
                if math.isnan(av) or math.isnan(bv):
                    continue
                if col == "philips_cn_fpr":
                    comparisons.append(bv <= av)
                    strictly_better |= bv < av
                else:
                    comparisons.append(bv >= av)
                    strictly_better |= bv > av
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
    return pd.DataFrame(records).sort_values(
        ["pareto_non_dominated", "adni_oof_ecdf_auc", "adni_oof_ecdf_pr_auc"],
        ascending=[False, False, False],
    )


def beta_axis_table(evidence: pd.DataFrame) -> pd.DataFrame:
    rows = evidence[evidence["model_id"].astype(str).isin(BETA_AXIS_ORDER)].copy()
    order = {mid: i for i, mid in enumerate(BETA_AXIS_ORDER)}
    rows["axis_order"] = rows["model_id"].map(order)
    rows["oasis_availability_status"] = np.where(
        rows[["oasis_concatenated_auc", "oasis_runwise164_auc", "oasis_runwise140_auc"]].notna().any(axis=1),
        "available",
        "not_available_or_not_scored",
    )
    rows.loc[rows["model_id"].eq(BETA6_MODEL_ID), "oasis_availability_status"] = (
        "not_scored_adni_gate_failed_optional_sensitivity_only"
    )
    keep = [
        "axis_order",
        "model_id",
        "display_name",
        "channel_set_order",
        "beta_vae",
        "latent_dim",
        "beta_KLD_over_D_best_mean",
        "adni_oof_ecdf_auc",
        "adni_oof_ecdf_pr_auc",
        "philips_cn_fpr",
        "scanner_leakage_latent_acc",
        "oasis_concatenated_auc",
        "oasis_runwise164_auc",
        "oasis_runwise140_auc",
        "oasis_availability_status",
        "decision_class",
    ]
    for col in keep:
        if col not in rows.columns:
            rows[col] = np.nan
    return rows[keep].sort_values("axis_order")


def axis_csvs(beta_axis: pd.DataFrame) -> dict[str, pd.DataFrame]:
    common = [
        "model_id",
        "display_name",
        "channel_set_order",
        "beta_vae",
        "beta_KLD_over_D_best_mean",
        "oasis_availability_status",
        "decision_class",
    ]
    return {
        "auc_vs_beta_kld_over_D": beta_axis[common + ["adni_oof_ecdf_auc"]].copy(),
        "pr_auc_vs_beta_kld_over_D": beta_axis[common + ["adni_oof_ecdf_pr_auc"]].copy(),
        "philips_cn_fpr_vs_beta_kld_over_D": beta_axis[common + ["philips_cn_fpr"]].copy(),
    }


def adni_vs_oasis_scatter(evidence: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in evidence.iterrows():
        for build, auc_col, pr_col in [
            ("concatenated_timeseries", "oasis_concatenated_auc", "oasis_concatenated_pr_auc"),
            ("runwise164", "oasis_runwise164_auc", "oasis_runwise164_pr_auc"),
            ("runwise140", "oasis_runwise140_auc", "oasis_runwise140_pr_auc"),
        ]:
            rows.append(
                {
                    "model_id": row.get("model_id"),
                    "display_name": row.get("display_name"),
                    "decision_class": row.get("decision_class"),
                    "build": build,
                    "adni_oof_ecdf_auc": row.get("adni_oof_ecdf_auc"),
                    "adni_oof_ecdf_pr_auc": row.get("adni_oof_ecdf_pr_auc"),
                    "oasis_auc": row.get(auc_col),
                    "oasis_pr_auc": row.get(pr_col),
                    "oasis_status": "available" if pd.notna(row.get(auc_col)) else row.get("oasis_status", "not_available"),
                }
            )
    return pd.DataFrame(rows)


def optimism_table(base_opt: pd.DataFrame) -> pd.DataFrame:
    cand = read_csv(BETA6 / "stagea_to_stageb_optimism.csv")
    row = cand[
        cand["run_id"].astype(str).eq("candidate_latent384_beta6p5")
        & cand["stageA_model"].astype(str).eq("logreg")
    ]
    if row.empty:
        return base_opt
    r = row.iloc[0]
    new = {
        "model_id": BETA6_MODEL_ID,
        "display_name": BETA6_DISPLAY,
        "stageA_logreg_auc": r.get("stageA_auc_final_mean"),
        "stageA_logreg_pr_auc": r.get("stageA_pr_auc_final_mean"),
        "adni_oof_ecdf_auc": r.get("stageB_primary_auc"),
        "adni_oof_ecdf_pr_auc": r.get("stageB_primary_pr_auc"),
        "stageB_minus_stageA_logreg_auc": r.get("stageB_minus_stageA_auc"),
        "stageB_minus_stageA_logreg_pr_auc": r.get("stageB_minus_stageA_pr_auc"),
    }
    out = base_opt[~base_opt["model_id"].astype(str).eq(BETA6_MODEL_ID)].copy()
    for col in out.columns:
        if col not in new:
            new[col] = np.nan
    return pd.concat([out, pd.DataFrame([new])[out.columns]], ignore_index=True)


def trend_text(beta_axis: pd.DataFrame) -> str:
    b6 = beta_axis[beta_axis["model_id"].eq(BETA6_MODEL_ID)].iloc[0]
    prom = beta_axis[beta_axis["model_id"].eq("promoted_latent384_beta3p75_ch1_0_2")].iloc[0]
    ch1 = beta_axis[beta_axis["model_id"].eq("ch1only_latent384_beta3p75")].iloc[0]
    beta35 = beta_axis[beta_axis["model_id"].eq("latent384_beta3p5")].iloc[0]
    beta40 = beta_axis[beta_axis["model_id"].eq("latent384_beta4p0")].iloc[0]
    gap = safe_float(ch1["beta_KLD_over_D_best_mean"]) - safe_float(prom["beta_KLD_over_D_best_mean"])
    moved = safe_float(b6["beta_KLD_over_D_best_mean"]) - safe_float(prom["beta_KLD_over_D_best_mean"])
    closed = moved / gap if gap and not math.isnan(gap) else float("nan")
    def fmt(value: Any, digits: int = 6) -> str:
        val = safe_float(value)
        return "not available" if math.isnan(val) else f"{val:.{digits}f}"

    return f"""# Trend Interpretation

This package extends `final_full_model_evidence_map_20260606` by adding the completed beta6.5 post-final exploratory run. It is a read-only aggregation; no training, OASIS scoring, threshold fitting, calibration fitting, tensor modification, metadata modification, or model-artifact modification was performed.

## Capacity
The previous capacity conclusion is unchanged. Latent448 beta4.0 and latent512 beta3.75 remain sensitivity runs and do not reopen primary selection.

## Beta / Effective Regularization
The beta/effective-regularization axis now includes beta3.5, promoted beta3.75, beta4.0, beta6.5, and ch1-only beta3.75.

- beta3.5: beta*KLD/D={fmt(beta35['beta_KLD_over_D_best_mean'])}, AUC={fmt(beta35['adni_oof_ecdf_auc'])}, PR-AUC={fmt(beta35['adni_oof_ecdf_pr_auc'])}.
- promoted beta3.75 [1,0,2]: beta*KLD/D={fmt(prom['beta_KLD_over_D_best_mean'])}, AUC={fmt(prom['adni_oof_ecdf_auc'])}, PR-AUC={fmt(prom['adni_oof_ecdf_pr_auc'])}.
- beta4.0: beta*KLD/D={fmt(beta40['beta_KLD_over_D_best_mean'])}, AUC={fmt(beta40['adni_oof_ecdf_auc'])}, PR-AUC={fmt(beta40['adni_oof_ecdf_pr_auc'])}. OOF-ECDF metrics were not available in the base evidence map for this row.
- beta6.5: beta*KLD/D={fmt(b6['beta_KLD_over_D_best_mean'])}, AUC={fmt(b6['adni_oof_ecdf_auc'])}, PR-AUC={fmt(b6['adni_oof_ecdf_pr_auc'])}, Philips CN FPR={fmt(b6['philips_cn_fpr'], 4)}.
- ch1-only beta3.75: beta*KLD/D={fmt(ch1['beta_KLD_over_D_best_mean'])}, AUC={fmt(ch1['adni_oof_ecdf_auc'])}, PR-AUC={fmt(ch1['adni_oof_ecdf_pr_auc'])}.

Beta6.5 moved the 3-channel model only partway toward the ch1-only effective-regularization regime: it closed approximately {closed:.1%} of the promoted-to-ch1 beta*KLD/D gap. That movement did not improve the primary ADNI ranking metrics; beta6.5 AUC and PR-AUC were lower than the promoted beta3.75 model, and scanner leakage was marginally worse. OASIS scoring was therefore not required for selection and remains optional sensitivity only.

## Channel Count
The channel-count conclusion is unchanged. Ch1-only latent384 beta3.75 has the strongest internal ranking signal, but it remains a parsimony sensitivity model because it does not cleanly improve BA/F1, Philips CN FPR, or OASIS transfer relative to the promoted multichannel model.

## OASIS Transfer
OASIS remains an external stress test, not a tuning set. Beta6.5 was not scored on OASIS because it failed the ADNI promotion gate.
"""


def recommendation_text(evidence: pd.DataFrame, pareto: pd.DataFrame, beta_axis: pd.DataFrame) -> str:
    b6 = beta_axis[beta_axis["model_id"].eq(BETA6_MODEL_ID)].iloc[0]
    nd = pareto[pareto["pareto_non_dominated"].fillna(False)]
    nd_names = ", ".join(nd["display_name"].dropna().astype(str).tolist())
    return f"""# Final Recommendation

Decision: no model reopens primary selection.

The promoted [1,0,2] latent384 beta3.75 model remains the primary manuscript model (ADNI OOF-ECDF AUC 0.795155, PR-AUC 0.573934).

Beta6.5 is added as a post-final beta/effective-regularization sensitivity result. It increased beta*KLD/D to {safe_float(b6['beta_KLD_over_D_best_mean']):.6f}, but this still did not materially approach the ch1-only regime ({safe_float(beta_axis[beta_axis['model_id'].eq('ch1only_latent384_beta3p75')].iloc[0]['beta_KLD_over_D_best_mean']):.6f}) and did not improve ADNI ranking or operating-point metrics. Its OOF-ECDF AUC was {safe_float(b6['adni_oof_ecdf_auc']):.6f}, PR-AUC was {safe_float(b6['adni_oof_ecdf_pr_auc']):.6f}, and Philips CN FPR was {safe_float(b6['philips_cn_fpr']):.4f}. It is classified as `reject_beta_sensitivity_only`.

The ch1-only latent384 beta3.75 model remains a parsimony sensitivity model, not a primary replacement. Capacity, beta, deconfounding, residualized Stage B, and locked/recover035 latent256 rows should be reported as sensitivity or negative controls according to their decision class.

Pareto non-dominated rows under the requested descriptive objective set: {nd_names if nd_names else 'none'}.

No OASIS-based model selection was performed. Beta6.5 was not sent to OASIS because it failed the ADNI promotion gate.
"""


def try_write_pngs(beta_axis: pd.DataFrame, scatter: pd.DataFrame, out: Path) -> list[str]:
    made: list[str] = []
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return made

    plot_specs = [
        ("auc_vs_beta_kld_over_D.png", "adni_oof_ecdf_auc", "ADNI OOF-ECDF AUC"),
        ("pr_auc_vs_beta_kld_over_D.png", "adni_oof_ecdf_pr_auc", "ADNI OOF-ECDF PR-AUC"),
        ("philips_cn_fpr_vs_beta_kld_over_D.png", "philips_cn_fpr", "Philips CN FPR"),
    ]
    for filename, ycol, ylabel in plot_specs:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        data = beta_axis.dropna(subset=["beta_KLD_over_D_best_mean", ycol])
        ax.scatter(data["beta_KLD_over_D_best_mean"], data[ycol], s=48)
        for _, row in data.iterrows():
            ax.annotate(str(row["model_id"]).replace("_", "\n"), (row["beta_KLD_over_D_best_mean"], row[ycol]), fontsize=7)
        ax.set_xlabel("beta*KLD/D")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / filename, dpi=180)
        plt.close(fig)
        made.append(filename)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    data = scatter.dropna(subset=["adni_oof_ecdf_auc", "oasis_auc"])
    for build, group in data.groupby("build"):
        ax.scatter(group["adni_oof_ecdf_auc"], group["oasis_auc"], label=build, s=42)
    ax.set_xlabel("ADNI OOF-ECDF AUC")
    ax.set_ylabel("OASIS AUC")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "adni_vs_oasis_auc_scatter.png", dpi=180)
    plt.close(fig)
    made.append("adni_vs_oasis_auc_scatter.png")
    return made


def main() -> None:
    args = parse_args()
    out = args.output_dir
    required = [
        BASE / "full_model_evidence_map.csv",
        BASE / "model_decision_compact_table.csv",
        BASE / "pareto_table.csv",
        BASE / "stageA_to_stageB_optimism_table.csv",
        BETA6 / "stageb_primary_rows.csv",
        BETA6 / "rate_distortion_summary.csv",
        BETA6 / "primary_promotion_gate_table.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(missing)
    if args.dry_run:
        print("Inputs OK.")
        print("Output:", out)
        return

    out.mkdir(parents=True, exist_ok=True)
    evidence = read_csv(BASE / "full_model_evidence_map.csv")
    beta6_row = beta6_evidence_row(evidence.columns.tolist())
    evidence = append_or_replace(evidence, beta6_row)

    compact = compact_from_evidence(evidence)
    pareto = rank_pareto(evidence)
    base_opt = read_csv(BASE / "stageA_to_stageB_optimism_table.csv")
    optimism = optimism_table(base_opt)
    beta_axis = beta_axis_table(evidence)
    scatter = adni_vs_oasis_scatter(evidence)

    write_table(evidence, "full_model_evidence_map", out)
    write_table(compact, "model_decision_compact_table", out)
    write_table(pareto, "pareto_table", out)
    write_table(optimism, "stageA_to_stageB_optimism_table", out)
    write_table(beta_axis, "beta_effective_regularization_axis", out)
    for stem, df in axis_csvs(beta_axis).items():
        write_table(df, stem, out)
    write_table(scatter, "adni_vs_oasis_scatter", out)

    figure_files = try_write_pngs(beta_axis, scatter, out)
    (out / "trend_interpretation.md").write_text(trend_text(beta_axis), encoding="utf-8")
    (out / "final_recommendation.md").write_text(recommendation_text(evidence, pareto, beta_axis), encoding="utf-8")
    (out / "README.md").write_text(
        """# Final FULL-Model Evidence Map With Beta6.5

This derived package extends `final_full_model_evidence_map_20260606` with the completed beta6.5 post-final exploratory run.

The package is read-only with respect to training artifacts. It performs no training, no OASIS scoring, no threshold/calibration fitting, and no tensor/metadata/model-artifact modification.

New beta-axis outputs:
- `beta_effective_regularization_axis.csv/.md`
- `auc_vs_beta_kld_over_D.csv/.md`
- `pr_auc_vs_beta_kld_over_D.csv/.md`
- `philips_cn_fpr_vs_beta_kld_over_D.csv/.md`
- `adni_vs_oasis_scatter.csv/.md`
""",
        encoding="utf-8",
    )
    command_log = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": "scripts/revision_bspc_2026/update_final_full_model_evidence_map_with_beta6p5_20260606.py",
        "inputs": {
            "base_evidence_map": str(BASE),
            "beta6p5_audit": str(BETA6),
        },
        "output_dir": str(out),
        "guardrails": {
            "no_training": True,
            "no_oasis_scoring": True,
            "no_threshold_or_calibration_fitting": True,
            "no_tensor_modification": True,
            "no_metadata_modification": True,
            "no_model_artifact_modification": True,
        },
        "beta6p5_decision": "reject_beta_sensitivity_only",
        "figures_written": figure_files,
    }
    (out / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
    print(f"Wrote package: {out}")
    print("Figures:", figure_files or "CSV-only outputs")


if __name__ == "__main__":
    main()
