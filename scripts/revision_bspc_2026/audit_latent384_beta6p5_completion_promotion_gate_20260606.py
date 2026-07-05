#!/usr/bin/env python3
"""Read-only completion and promotion-gate audit for latent384 beta6.5.

This audit compares the completed post-final exploratory run
``recover035_latent384_beta6p5_T80_h10000_p560_full5x5`` against the promoted
latent384 beta3.75 reference and nearby capacity/beta/channel references.

It writes summary tables only. It does not train, score OASIS, tune thresholds,
or modify tensors, metadata, ledgers, or model artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/revision_bspc_2026"
OUT_DEFAULT = RESULTS / "recover035_latent384_beta6p5_completion_promotion_gate_audit_20260606"

PRIMARY_MODEL_OOF = "logreg_l2_original"
PRIMARY_MODEL_RAW = "logreg_l2"
PRIMARY_FEATURES = "z_plus_age_sex"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
PRIMARY_CALIB = "oof_ecdf"

PROMOTED_AUC = 0.795155
PROMOTED_PR_AUC = 0.573934
CH1_AUC = 0.800378
PHILIPS_FPR_GATE = 0.4545


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    run_dir: Path
    oof_dir: Path | None
    beta: float
    latent_dim: int
    channel_set: str
    role: str


RUNS = [
    RunSpec(
        "candidate_latent384_beta6p5",
        RESULTS / "recover035_latent384_beta6p5_T80_h10000_p560_full5x5",
        RESULTS / "recover035_latent384_beta6p5_stageB_oof_score_calibration",
        6.5,
        384,
        "[1,0,2]",
        "candidate",
    ),
    RunSpec(
        "promoted_latent384_beta3p75",
        RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5",
        RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration",
        3.75,
        384,
        "[1,0,2]",
        "promoted_reference",
    ),
    RunSpec(
        "ch1only_latent384_beta3p75",
        RESULTS / "recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5",
        RESULTS / "recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5_stageB_oof_score_calibration",
        3.75,
        384,
        "[1]",
        "ch1_reference",
    ),
    RunSpec(
        "latent384_beta3p5",
        RESULTS / "recover035_latent384_beta3p5_T80_h10000_p560_full5x5",
        RESULTS / "recover035_latent384_beta3p5_stageB_oof_score_calibration",
        3.5,
        384,
        "[1,0,2]",
        "beta_reference",
    ),
    RunSpec(
        "latent384_beta4p0",
        RESULTS / "recover035_latent384_beta4p0_T80_h10000_p560_full5x5",
        None,
        4.0,
        384,
        "[1,0,2]",
        "beta_reference",
    ),
    RunSpec(
        "latent448_beta4p0",
        RESULTS / "recover035_latent448_beta4p0_T80_h10000_p560_full5x5",
        RESULTS / "recover035_latent448_beta4p0_stageB_oof_score_calibration",
        4.0,
        448,
        "[1,0,2]",
        "capacity_beta_reference",
    ),
    RunSpec(
        "latent512_beta3p75",
        RESULTS / "recover035_latent512_beta3p75_T80_h10000_p560_full5x5",
        RESULTS / "recover035_latent512_beta3p75_stageB_oof_score_calibration",
        3.75,
        512,
        "[1,0,2]",
        "capacity_reference",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_csv(path: Path, required: bool = False) -> pd.DataFrame:
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


def safe_div(num: float, den: float) -> float:
    if den == 0 or math.isnan(den):
        return float("nan")
    return num / den


def to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._\n"
    try:
        return df.to_markdown(index=False) + "\n"
    except Exception:
        return df.to_string(index=False) + "\n"


def write_table(df: pd.DataFrame, stem: str, out: Path) -> None:
    df.to_csv(out / f"{stem}.csv", index=False)
    (out / f"{stem}.md").write_text(to_md(df), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            "_".join(str(part) for part in col if str(part) and str(part) != "nan").strip("_")
            for col in out.columns
        ]
    return out


def find_stagea_metrics(run_dir: Path) -> Path | None:
    matches = sorted(run_dir.glob("all_folds_metrics_MULTI*.csv"))
    return matches[0] if matches else None


def completion_status(run: RunSpec) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    readout_dir = run.run_dir / "classifier_only_readout"
    for fold in range(1, 6):
        fold_dir = run.run_dir / f"fold_{fold}"
        checks = {
            "fold_dir": fold_dir,
            "vae_model_saved": fold_dir / f"vae_model_fold_{fold}.pt",
            "vae_history": fold_dir / f"vae_train_history_fold_{fold}.joblib",
            "rate_distortion": fold_dir / f"fold_{fold}_rate_distortion.csv",
            "trainDev_latent_info": fold_dir / f"fold_{fold}_trainDev_latent_info_summary.csv",
            "test_latent_info": fold_dir / f"fold_{fold}_test_latent_info_summary.csv",
            "trainDev_scanner_leakage": fold_dir / f"fold_{fold}_scanner_leakage_summary.csv",
            "test_scanner_leakage": fold_dir / f"fold_{fold}_test_scanner_leakage_summary.csv",
            "stageA_logreg_predictions": fold_dir / "test_predictions_logreg.csv",
            "stageA_svm_predictions": fold_dir / "test_predictions_svm.csv",
            "latent_cache_trainDev": readout_dir / "latent_cache" / f"fold_{fold}_trainDev_latent_mu.csv",
            "latent_cache_test": readout_dir / "latent_cache" / f"fold_{fold}_test_latent_mu.csv",
        }
        row: dict[str, Any] = {"run_id": run.run_id, "fold": fold}
        for key, path in checks.items():
            row[key] = path.exists()
        row["classifier_artifacts_present"] = (
            (fold_dir / f"classifier_logreg_final_pipeline_fold_{fold}.joblib").exists()
            and (fold_dir / f"classifier_svm_final_pipeline_fold_{fold}.joblib").exists()
        )
        row["classifier_only_readout_present"] = readout_dir.exists()
        row["oof_score_calibration_present"] = bool(run.oof_dir and run.oof_dir.exists())
        row["fold_complete"] = all(bool(row[key]) for key in checks)
        rows.append(row)
    global_checks = {
        "stageA_metrics": bool(find_stagea_metrics(run.run_dir)),
        "classifier_only_readout": (readout_dir / "classifier_sweep_pooled_metrics.csv").exists(),
        "classifier_only_foldwise": (readout_dir / "classifier_sweep_foldwise_metrics.csv").exists(),
        "oof_pooled": bool(run.oof_dir and (run.oof_dir / "calib_pooled_metrics.csv").exists()),
        "oof_foldwise": bool(run.oof_dir and (run.oof_dir / "calib_foldwise_metrics.csv").exists()),
        "oof_philips_fpr": bool(run.oof_dir and (run.oof_dir / "calib_philips_fpr_pooled.csv").exists()),
    }
    rows.append({"run_id": run.run_id, "fold": "all", **global_checks, "fold_complete": all(global_checks.values())})
    return pd.DataFrame(rows)


def stagea_foldwise(run: RunSpec) -> pd.DataFrame:
    path = find_stagea_metrics(run.run_dir)
    if path is None:
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.insert(0, "run_id", run.run_id)
    df.insert(1, "source_file", rel(path))
    return df


def stagea_summary(stagea: pd.DataFrame) -> pd.DataFrame:
    if stagea.empty:
        return pd.DataFrame()
    model_col = None
    for col in ["actual_classifier_type", "classifier_model", "model", "classifier"]:
        if col in stagea.columns:
            model_col = col
            break
    if model_col is None:
        return pd.DataFrame()
    metric_cols = [
        c
        for c in [
            "auc_raw",
            "pr_auc_raw",
            "auc_final",
            "pr_auc_final",
            "auc",
            "pr_auc",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "f1_score",
        ]
        if c in stagea.columns
    ]
    summary = (
        stagea.groupby(["run_id", model_col], dropna=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary = flatten_columns(summary)
    summary = summary.rename(columns={model_col: "stageA_model"})
    return summary


def raw_stageb_pooled(run: RunSpec) -> pd.DataFrame:
    path = run.run_dir / "classifier_only_readout/classifier_sweep_pooled_metrics.csv"
    df = read_csv(path)
    if df.empty:
        return df
    out = df.copy()
    out.insert(0, "run_id", run.run_id)
    out.insert(1, "source", "classifier_only_raw")
    out["calib_method"] = "raw"
    if "readout_feature_set" in out.columns:
        out = out.rename(columns={"readout_feature_set": "feature_set"})
    return out


def oof_stageb_pooled(run: RunSpec) -> pd.DataFrame:
    if run.oof_dir is None:
        return pd.DataFrame()
    path = run.oof_dir / "calib_pooled_metrics.csv"
    df = read_csv(path)
    if df.empty:
        return df
    out = df.copy()
    out.insert(0, "run_id", run.run_id)
    out.insert(1, "source", "oof_score_calibration")
    return out


def stageb_all_pooled(run: RunSpec) -> pd.DataFrame:
    frames = [raw_stageb_pooled(run), oof_stageb_pooled(run)]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    keep = [
        "run_id",
        "source",
        "model_name",
        "feature_set",
        "calib_method",
        "threshold_strategy",
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
        "predicted_ad_rate",
        "auc",
        "pr_auc",
    ]
    for col in keep:
        if col not in df.columns:
            df[col] = np.nan
    return df[keep]


def primary_oof_rows(pooled: pd.DataFrame) -> pd.DataFrame:
    if pooled.empty:
        return pooled
    rows = pooled[
        pooled["model_name"].astype(str).eq(PRIMARY_MODEL_OOF)
        & pooled["feature_set"].astype(str).eq(PRIMARY_FEATURES)
        & pooled["calib_method"].astype(str).isin(["oof_logitz", PRIMARY_CALIB])
        & pooled["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    ].copy()
    return rows.sort_values(["run_id", "calib_method"])


def focused_stageb_rows(pooled: pd.DataFrame) -> pd.DataFrame:
    if pooled.empty:
        return pooled
    rows = pooled[
        pooled["feature_set"].astype(str).eq(PRIMARY_FEATURES)
        & pooled["calib_method"].astype(str).isin(["raw", "oof_zscore", "oof_logitz", "oof_ecdf", "oof_platt", "oof_isotonic"])
        & pooled["threshold_strategy"].astype(str).isin(
            ["fixed_0p5", "inner_oof_balanced_accuracy", PRIMARY_THRESHOLD, "inner_oof_youden_j"]
        )
        & pooled["model_name"].astype(str).isin([PRIMARY_MODEL_RAW, PRIMARY_MODEL_OOF])
    ].copy()
    calib_order = {"raw": 0, "oof_zscore": 1, "oof_logitz": 2, "oof_ecdf": 3, "oof_platt": 4, "oof_isotonic": 5}
    rows["calib_order"] = rows["calib_method"].map(calib_order).fillna(99)
    return rows.sort_values(["run_id", "calib_order", "threshold_strategy"]).drop(columns=["calib_order"])


def oof_foldwise(run: RunSpec) -> pd.DataFrame:
    if run.oof_dir is None:
        return pd.DataFrame()
    path = run.oof_dir / "calib_foldwise_metrics.csv"
    df = read_csv(path)
    if df.empty:
        return df
    rows = df[
        df["model_name"].astype(str).eq(PRIMARY_MODEL_OOF)
        & df["feature_set"].astype(str).eq(PRIMARY_FEATURES)
        & df["calib_method"].astype(str).isin(["oof_logitz", PRIMARY_CALIB])
        & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    ].copy()
    rows.insert(0, "run_id", run.run_id)
    return rows


def philips_fpr(run: RunSpec) -> pd.DataFrame:
    if run.oof_dir is None:
        return pd.DataFrame()
    path = run.oof_dir / "calib_philips_fpr_pooled.csv"
    df = read_csv(path)
    if df.empty:
        return df
    rows = df[
        df["model_name"].astype(str).eq(PRIMARY_MODEL_OOF)
        & df["feature_set"].astype(str).eq(PRIMARY_FEATURES)
        & df["calib_method"].astype(str).eq(PRIMARY_CALIB)
        & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    ].copy()
    rows.insert(0, "run_id", run.run_id)
    return rows


def scanner_leakage(run: RunSpec) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        for split, file_name in [
            ("trainDev", f"fold_{fold}_scanner_leakage_summary.csv"),
            ("test", f"fold_{fold}_test_scanner_leakage_summary.csv"),
        ]:
            path = run.run_dir / f"fold_{fold}" / file_name
            df = read_csv(path)
            if df.empty:
                rows.append({"run_id": run.run_id, "fold": fold, "split": split, "available": False})
                continue
            rec = df.iloc[0].to_dict()
            rec.update({"run_id": run.run_id, "fold": fold, "split": split, "available": True})
            rec["latent_minus_raw"] = safe_float(rec.get("acc_site_latent")) - safe_float(rec.get("acc_site_raw"))
            rows.append(rec)
    return pd.DataFrame(rows)


def rate_distortion(run: RunSpec) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        path = run.run_dir / f"fold_{fold}" / f"fold_{fold}_rate_distortion.csv"
        df = read_csv(path)
        if df.empty:
            rows.append({"run_id": run.run_id, "fold": fold, "status": "missing"})
            continue
        best_idx = df["L_val_betaMax"].idxmin() if "L_val_betaMax" in df.columns else df.index[-1]
        best = df.loc[best_idx]
        final = df.iloc[-1]
        d_best = safe_float(best.get("D_val"))
        r_nats_best = safe_float(best.get("R_val_nats"))
        rows.append(
            {
                "run_id": run.run_id,
                "fold": fold,
                "beta_vae": run.beta,
                "latent_dim": run.latent_dim,
                "best_epoch": safe_float(best.get("epoch")),
                "final_epoch": safe_float(final.get("epoch")),
                "D_val_best": d_best,
                "R_val_nats_best": r_nats_best,
                "R_val_bits_best": safe_float(best.get("R_val_bits")),
                "bits_per_dim_best": safe_div(safe_float(best.get("R_val_bits")), float(run.latent_dim)),
                "KLD_over_D_best": safe_div(r_nats_best, d_best),
                "beta_KLD_over_D_best": run.beta * safe_div(r_nats_best, d_best),
                "L_val_betaMax_best": safe_float(best.get("L_val_betaMax")),
                "D_val_final": safe_float(final.get("D_val")),
                "R_val_nats_final": safe_float(final.get("R_val_nats")),
                "R_val_bits_final": safe_float(final.get("R_val_bits")),
                "bits_per_dim_final": safe_div(safe_float(final.get("R_val_bits")), float(run.latent_dim)),
                "KLD_over_D_final": safe_div(safe_float(final.get("R_val_nats")), safe_float(final.get("D_val"))),
                "beta_KLD_over_D_final": run.beta
                * safe_div(safe_float(final.get("R_val_nats")), safe_float(final.get("D_val"))),
            }
        )
    return pd.DataFrame(rows)


def latent_info(run: RunSpec) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        for split in ["trainDev", "test"]:
            path = run.run_dir / f"fold_{fold}" / f"fold_{fold}_{split}_latent_info_summary.csv"
            df = read_csv(path)
            if df.empty:
                rows.append({"run_id": run.run_id, "fold": fold, "split": split, "status": "missing"})
                continue
            rec: dict[str, Any] = {"run_id": run.run_id, "fold": fold, "split": split}
            for variable, prefix in [("Y_target", "Y"), ("Manufacturer", "Manufacturer")]:
                row = df[df["variable"].astype(str).eq(variable)]
                if row.empty:
                    continue
                r = row.iloc[0]
                rec[f"MI_Z_{prefix}_nats"] = safe_float(r.get("mi_sum_nats"))
                rec[f"MI_Z_{prefix}_mean_nats"] = safe_float(r.get("mi_mean_nats"))
                if variable == "Y_target":
                    rec["active_units"] = safe_float(r.get("n_active"))
                    rec["frac_active"] = safe_float(r.get("frac_active"))
                    rec["total_correlation_nats"] = safe_float(r.get("total_correlation_nats"))
            rec["MI_Manufacturer_over_MI_Y"] = safe_div(
                rec.get("MI_Z_Manufacturer_nats", float("nan")),
                rec.get("MI_Z_Y_nats", float("nan")),
            )
            rows.append(rec)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, by: list[str], metrics: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    cols = [c for c in metrics if c in df.columns]
    if not cols:
        return pd.DataFrame()
    return flatten_columns(df.groupby(by, dropna=False)[cols].agg(["mean", "std", "min", "max"]).reset_index())


def get_primary_metric(primary: pd.DataFrame, run_id: str, calib: str, metric: str) -> float:
    rows = primary[(primary["run_id"] == run_id) & (primary["calib_method"] == calib)]
    if rows.empty or metric not in rows.columns:
        return float("nan")
    return safe_float(rows.iloc[0][metric])


def get_philips_fpr(philips: pd.DataFrame, run_id: str) -> tuple[float, float, float]:
    rows = philips[(philips["run_id"] == run_id) & (philips["manufacturer"].astype(str).str.lower() == "philips")]
    if rows.empty:
        return float("nan"), float("nan"), float("nan")
    r = rows.iloc[0]
    return safe_float(r.get("fp_cn_pooled")), safe_float(r.get("n_cn_pooled")), safe_float(r.get("fpr_cn_pooled"))


def get_leakage_mean(scanner_summary: pd.DataFrame, run_id: str, split: str = "test") -> float:
    if scanner_summary.empty:
        return float("nan")
    rows = scanner_summary[(scanner_summary["run_id"] == run_id) & (scanner_summary["split"] == split)]
    if rows.empty:
        return float("nan")
    for col in ["acc_site_latent_mean", "acc_site_latent"]:
        if col in rows.columns:
            return safe_float(rows.iloc[0][col])
    return float("nan")


def promotion_gate(primary: pd.DataFrame, philips: pd.DataFrame, scanner_summary: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    cand = "candidate_latent384_beta6p5"
    ref = "promoted_latent384_beta3p75"
    rows: list[dict[str, Any]] = []
    for metric, limit, op, details in [
        ("auc", PROMOTED_AUC, ">", "AUC must exceed promoted 0.795155."),
        ("pr_auc", PROMOTED_PR_AUC, ">=", "PR-AUC must meet promoted 0.573934."),
    ]:
        value = get_primary_metric(primary, cand, PRIMARY_CALIB, metric)
        ref_value = get_primary_metric(primary, ref, PRIMARY_CALIB, metric)
        if op == ">":
            passes = value > limit
        else:
            passes = value >= limit
        rows.append(
            {
                "gate": metric,
                "candidate_value": value,
                "promoted_reference_value": ref_value,
                "required_limit": limit,
                "delta_vs_promoted": value - ref_value,
                "passes": bool(passes),
                "details": details,
            }
        )
    for metric, tolerance in [("balanced_accuracy", 0.01), ("f1", 0.01), ("sensitivity", 0.01)]:
        value = get_primary_metric(primary, cand, PRIMARY_CALIB, metric)
        ref_value = get_primary_metric(primary, ref, PRIMARY_CALIB, metric)
        rows.append(
            {
                "gate": metric,
                "candidate_value": value,
                "promoted_reference_value": ref_value,
                "required_limit": ref_value - tolerance,
                "delta_vs_promoted": value - ref_value,
                "passes": bool(value >= ref_value - tolerance),
                "details": f"Must not be materially worse than promoted; tolerance {-tolerance:+.3f}.",
            }
        )
    cand_fp, cand_n, cand_fpr = get_philips_fpr(philips, cand)
    ref_fp, ref_n, ref_fpr = get_philips_fpr(philips, ref)
    rows.append(
        {
            "gate": "Philips_CN_FPR",
            "candidate_value": cand_fpr,
            "promoted_reference_value": ref_fpr,
            "required_limit": PHILIPS_FPR_GATE,
            "delta_vs_promoted": cand_fpr - ref_fpr,
            "passes": bool(cand_fpr <= PHILIPS_FPR_GATE),
            "candidate_fp_n": f"{int(cand_fp) if not math.isnan(cand_fp) else 'NA'}/{int(cand_n) if not math.isnan(cand_n) else 'NA'}",
            "details": "Philips CN FPR must be <= 0.4545.",
        }
    )
    cand_leak = get_leakage_mean(scanner_summary, cand, "test")
    ref_leak = get_leakage_mean(scanner_summary, ref, "test")
    rows.append(
        {
            "gate": "scanner_leakage_test_latent_acc",
            "candidate_value": cand_leak,
            "promoted_reference_value": ref_leak,
            "required_limit": ref_leak,
            "delta_vs_promoted": cand_leak - ref_leak,
            "passes": bool(cand_leak <= ref_leak) if not (math.isnan(cand_leak) or math.isnan(ref_leak)) else False,
            "details": "Latent scanner/manufacturer leakage must not worsen.",
        }
    )
    gate = pd.DataFrame(rows)
    decision = "promote" if bool(gate["passes"].all()) else "reject_beta_sensitivity_only"
    return gate, decision


def stagea_to_stageb_optimism(stagea_summary_df: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if stagea_summary_df.empty or primary.empty:
        return pd.DataFrame()
    stagea_focus = stagea_summary_df[stagea_summary_df["stageA_model"].astype(str).isin(["logreg", "svm"])].copy()
    primary_ecdf = primary[primary["calib_method"].astype(str).eq(PRIMARY_CALIB)].copy()
    for _, st in stagea_focus.iterrows():
        run_id = st["run_id"]
        pr = primary_ecdf[primary_ecdf["run_id"].eq(run_id)]
        if pr.empty:
            continue
        pr_row = pr.iloc[0]
        rows.append(
            {
                "run_id": run_id,
                "stageA_model": st["stageA_model"],
                "stageA_auc_final_mean": safe_float(st.get("auc_final_mean", st.get("auc_mean"))),
                "stageA_pr_auc_final_mean": safe_float(st.get("pr_auc_final_mean", st.get("pr_auc_mean"))),
                "stageA_ba_mean": safe_float(st.get("balanced_accuracy_mean")),
                "stageA_f1_mean": safe_float(st.get("f1_score_mean")),
                "stageB_primary_auc": safe_float(pr_row.get("auc")),
                "stageB_primary_pr_auc": safe_float(pr_row.get("pr_auc")),
                "stageB_primary_ba": safe_float(pr_row.get("balanced_accuracy")),
                "stageB_primary_f1": safe_float(pr_row.get("f1")),
                "stageB_minus_stageA_auc": safe_float(pr_row.get("auc")) - safe_float(st.get("auc_final_mean", st.get("auc_mean"))),
                "stageB_minus_stageA_pr_auc": safe_float(pr_row.get("pr_auc"))
                - safe_float(st.get("pr_auc_final_mean", st.get("pr_auc_mean"))),
                "stageB_minus_stageA_ba": safe_float(pr_row.get("balanced_accuracy")) - safe_float(st.get("balanced_accuracy_mean")),
                "stageB_minus_stageA_f1": safe_float(pr_row.get("f1")) - safe_float(st.get("f1_score_mean")),
                "note": "Positive delta means Stage B primary OOF-ECDF exceeds Stage A mean fold metric.",
            }
        )
    return pd.DataFrame(rows)


def effective_regularization_text(rd_summary: pd.DataFrame, primary: pd.DataFrame, philips: pd.DataFrame) -> str:
    def mean_for(run_id: str, col: str) -> float:
        row = rd_summary[rd_summary["run_id"] == run_id]
        if row.empty:
            return float("nan")
        mean_col = f"{col}_mean"
        if mean_col in row.columns:
            return safe_float(row.iloc[0][mean_col])
        return safe_float(row.iloc[0].get(col))

    cand_reg = mean_for("candidate_latent384_beta6p5", "beta_KLD_over_D_best")
    prom_reg = mean_for("promoted_latent384_beta3p75", "beta_KLD_over_D_best")
    ch1_reg = mean_for("ch1only_latent384_beta3p75", "beta_KLD_over_D_best")
    cand_auc = get_primary_metric(primary, "candidate_latent384_beta6p5", PRIMARY_CALIB, "auc")
    cand_pr = get_primary_metric(primary, "candidate_latent384_beta6p5", PRIMARY_CALIB, "pr_auc")
    prom_auc = get_primary_metric(primary, "promoted_latent384_beta3p75", PRIMARY_CALIB, "auc")
    prom_pr = get_primary_metric(primary, "promoted_latent384_beta3p75", PRIMARY_CALIB, "pr_auc")
    ch1_auc = get_primary_metric(primary, "ch1only_latent384_beta3p75", PRIMARY_CALIB, "auc")
    ch1_pr = get_primary_metric(primary, "ch1only_latent384_beta3p75", PRIMARY_CALIB, "pr_auc")
    _, _, cand_fpr = get_philips_fpr(philips, "candidate_latent384_beta6p5")
    _, _, prom_fpr = get_philips_fpr(philips, "promoted_latent384_beta3p75")

    if math.isnan(cand_reg) or math.isnan(ch1_reg) or math.isnan(prom_reg):
        approach = "The available rate-distortion files were insufficient to quantify the full effective-regularization comparison."
    else:
        prom_gap = ch1_reg - prom_reg
        cand_gap = ch1_reg - cand_reg
        fraction = safe_div(cand_reg - prom_reg, prom_gap)
        approach = (
            f"Mean best-epoch beta*KLD/D moved from {prom_reg:.6f} in the promoted 3-channel beta3.75 model "
            f"to {cand_reg:.6f} in beta6.5. The ch1-only reference is {ch1_reg:.6f}. "
            f"Relative to the promoted-to-ch1 gap, beta6.5 closes approximately {fraction:.1%} of the distance "
            f"when positive values indicate movement toward ch1-only."
        )

    return f"""# Effective-Regularization Hypothesis Test

Scientific question: did beta6.5 move the promoted 3-channel latent384 model toward the stronger effective-regularization regime observed in ch1-only, and did that improve ADNI/OASIS-relevant performance?

{approach}

Primary ADNI OOF-ECDF performance:
- beta6.5 candidate: AUC={cand_auc:.6f}, PR-AUC={cand_pr:.6f}.
- promoted beta3.75 [1,0,2]: AUC={prom_auc:.6f}, PR-AUC={prom_pr:.6f}.
- ch1-only beta3.75: AUC={ch1_auc:.6f}, PR-AUC={ch1_pr:.6f}.

Philips CN false-positive rate:
- beta6.5 candidate: {cand_fpr:.6f}.
- promoted beta3.75 [1,0,2]: {prom_fpr:.6f}.

Interpretation:
Increasing beta to 6.5 is only useful for promotion if any movement toward the ch1-only beta*KLD/D regime also improves the primary ranking metrics and does not worsen the operating-point and scanner-leakage safeguards. The promotion gate in `primary_promotion_gate_table.csv` is the controlling decision rule. If the AUC/PR-AUC gates fail, OASIS scoring is not required for model selection and should be treated as optional sensitivity only.
"""


def final_decision_text(gate: pd.DataFrame, decision: str, primary: pd.DataFrame) -> str:
    cand_rows = primary[primary["run_id"].eq("candidate_latent384_beta6p5")]
    lines = []
    for _, row in cand_rows.iterrows():
        lines.append(
            f"- {row['calib_method']}: AUC={row['auc']:.6f}, PR-AUC={row['pr_auc']:.6f}, "
            f"BA={row['balanced_accuracy']:.6f}, Sens={row['sensitivity']:.6f}, "
            f"Spec={row['specificity']:.6f}, F1={row['f1']:.6f}."
        )
    failed = gate.loc[~gate["passes"].astype(bool), "gate"].tolist()
    oasis_line = (
        "ADNI promotion gate passed; OASIS scoring may be run as the next external check."
        if decision == "promote"
        else "ADNI promotion gate failed; OASIS is not required for selection and should be optional sensitivity only."
    )
    return f"""# Final Decision

Decision: **{decision}**.

Candidate primary score-harmonized Stage B rows:
{chr(10).join(lines) if lines else "- Primary candidate rows were unavailable."}

Promotion-gate failures:
- {", ".join(failed) if failed else "none"}

Gate definition:
- AUC must exceed promoted 0.795155 and preferably ch1-only 0.800378.
- PR-AUC must meet promoted 0.573934.
- BA/F1/Sensitivity must not be materially worse than promoted.
- Philips CN FPR must be <= 0.4545.
- Scanner leakage must not worsen.

OASIS:
{oasis_line}

Guardrails:
This package is read-only. It performs no VAE training, no OASIS threshold or calibration fitting, no tensor modification, no metadata modification, and no model artifact overwrite.
"""


def main() -> None:
    args = parse_args()
    out = args.output_dir
    candidate = RUNS[0]
    required = [candidate.run_dir, candidate.oof_dir]
    missing = [p for p in required if p is not None and not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required candidate artifacts: {[rel(p) for p in missing]}")

    planned = [
        "completion_status.csv/.md",
        "stagea_summary_metrics.csv/.md",
        "stagea_foldwise_metrics.csv/.md",
        "stageb_oof_calibration_metrics.csv/.md",
        "stageb_primary_foldwise_metrics.csv/.md",
        "stagea_to_stageb_optimism.csv/.md",
        "primary_promotion_gate_table.csv/.md",
        "philips_cn_fpr.csv/.md",
        "scanner_leakage_summary.csv/.md",
        "rate_distortion_summary.csv/.md",
        "latent_mi_signal_nuisance_summary.csv/.md",
        "effective_regularization_hypothesis_test.md",
        "final_decision.md",
        "command_log.json",
    ]
    if args.dry_run:
        print("Required candidate artifacts OK.")
        print("Output:", rel(out))
        for item in planned:
            print("-", item)
        return

    out.mkdir(parents=True, exist_ok=True)

    available_runs = [run for run in RUNS if run.run_dir.exists()]
    status = pd.concat([completion_status(run) for run in available_runs], ignore_index=True)
    write_table(status, "completion_status", out)

    stagea_all = pd.concat([stagea_foldwise(run) for run in available_runs], ignore_index=True)
    write_table(stagea_all, "stagea_foldwise_metrics", out)
    write_table(stagea_summary(stagea_all), "stagea_summary_metrics", out)

    pooled = pd.concat([stageb_all_pooled(run) for run in available_runs], ignore_index=True)
    focused = focused_stageb_rows(pooled)
    write_table(focused, "stageb_oof_calibration_metrics", out)
    primary = primary_oof_rows(pooled)
    write_table(primary, "stageb_primary_rows", out)

    foldwise = pd.concat([oof_foldwise(run) for run in available_runs], ignore_index=True)
    write_table(foldwise, "stageb_primary_foldwise_metrics", out)
    write_table(stagea_to_stageb_optimism(stagea_summary(stagea_all), primary), "stagea_to_stageb_optimism", out)

    philips = pd.concat([philips_fpr(run) for run in available_runs], ignore_index=True)
    write_table(philips, "philips_cn_fpr", out)

    scanner = pd.concat([scanner_leakage(run) for run in available_runs], ignore_index=True)
    scanner_summary = summarize(scanner, ["run_id", "split"], ["acc_site_raw", "acc_site_latent", "latent_minus_raw"])
    write_table(scanner, "scanner_leakage_foldwise", out)
    write_table(scanner_summary, "scanner_leakage_summary", out)

    rd = pd.concat([rate_distortion(run) for run in available_runs], ignore_index=True)
    rd_metrics = [
        "best_epoch",
        "final_epoch",
        "D_val_best",
        "R_val_bits_best",
        "bits_per_dim_best",
        "KLD_over_D_best",
        "beta_KLD_over_D_best",
        "D_val_final",
        "R_val_bits_final",
        "bits_per_dim_final",
        "KLD_over_D_final",
        "beta_KLD_over_D_final",
    ]
    write_table(rd, "rate_distortion_foldwise", out)
    rd_summary = summarize(rd, ["run_id"], rd_metrics)
    write_table(rd_summary, "rate_distortion_summary", out)

    latent = pd.concat([latent_info(run) for run in available_runs], ignore_index=True)
    latent_metrics = [
        "MI_Z_Y_nats",
        "MI_Z_Manufacturer_nats",
        "MI_Manufacturer_over_MI_Y",
        "active_units",
        "frac_active",
        "total_correlation_nats",
    ]
    write_table(latent, "latent_mi_signal_nuisance_foldwise", out)
    latent_summary = summarize(latent, ["run_id", "split"], latent_metrics)
    write_table(latent_summary, "latent_mi_signal_nuisance_summary", out)

    gate, decision = promotion_gate(primary, philips, scanner_summary)
    write_table(gate, "primary_promotion_gate_table", out)

    write_text(out / "effective_regularization_hypothesis_test.md", effective_regularization_text(rd_summary, primary, philips))
    write_text(out / "final_decision.md", final_decision_text(gate, decision, primary))

    readme = f"""# Latent384 Beta6.5 Completion and Promotion-Gate Audit

Candidate: `recover035_latent384_beta6p5_T80_h10000_p560_full5x5`.

Primary row:
- `model_name={PRIMARY_MODEL_OOF}`
- `feature_set={PRIMARY_FEATURES}`
- `calib_method={PRIMARY_CALIB}` plus OOF-logitz companion rows
- `threshold_strategy={PRIMARY_THRESHOLD}`

This package tests whether beta6.5 moved the promoted [1,0,2] latent384 model toward the ch1-only effective-regularization regime and whether that improved ADNI-relevant metrics enough to justify OASIS scoring.

No training, no OASIS threshold/calibration fitting, no tensor modification, no metadata modification, and no model artifact overwrite were performed.
"""
    write_text(out / "README.md", readme)

    command_log = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": rel(Path(__file__)),
        "output_dir": rel(out),
        "read_only_guardrails": {
            "no_training": True,
            "no_oasis_threshold_or_calibration_fitting": True,
            "no_tensor_modification": True,
            "no_metadata_modification": True,
            "no_model_artifact_overwrite": True,
        },
        "available_runs_used": [
            {
                "run_id": run.run_id,
                "run_dir": rel(run.run_dir),
                "oof_dir": rel(run.oof_dir),
                "beta": run.beta,
                "latent_dim": run.latent_dim,
                "channel_set": run.channel_set,
                "role": run.role,
            }
            for run in available_runs
        ],
        "primary_gate": {
            "model_name": PRIMARY_MODEL_OOF,
            "feature_set": PRIMARY_FEATURES,
            "calib_method": PRIMARY_CALIB,
            "threshold_strategy": PRIMARY_THRESHOLD,
        },
        "decision": decision,
        "generated_outputs": planned,
    }
    (out / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
    print(f"Wrote audit package: {rel(out)}")
    print(f"Decision: {decision}")


if __name__ == "__main__":
    main()
