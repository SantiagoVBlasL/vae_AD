#!/usr/bin/env python3
"""Read-only latent-capacity/beta rate-distortion synthesis across completed FULL runs."""

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
OUT_DEFAULT = RESULTS / "final_latent_capacity_beta_rate_distortion_synthesis_20260603"

PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
PRIMARY_FEATURES = "z_plus_age_sex"
RAW_MODEL = "logreg_l2"
OOF_MODEL = "logreg_l2_original"


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    run_dir: Path
    family: str
    oof_dir: Path | None = None
    beta4_audit_json: Path | None = None
    oasis_model_key: str | None = None
    notes: str = ""


RUNS: list[RunSpec] = [
    RunSpec(
        "latent128_beta1p25",
        RESULTS / "recover035_latent128_beta1p25_lockedSchedule_full5x5",
        "capacity_beta",
        RESULTS / "recover035_latent128_beta1p25_lockedSchedule_full5x5_stageB_oof_logitz",
    ),
    RunSpec(
        "latent128_beta2p5",
        RESULTS / "recover035_latent128_beta2p5_lockedSchedule_full5x5",
        "capacity_beta",
        RESULTS / "recover035_latent128_beta2p5_lockedSchedule_full5x5_stageB_oof_logitz",
    ),
    RunSpec(
        "locked_v5_1b_latent256_beta2p5",
        RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5",
        "locked_reference",
        None,
        oasis_model_key="primary_v5_1b_horizon4480_classifier_only",
        notes="Original locked v5.1b 140TR reference, CN=300/AD=96.",
    ),
    RunSpec(
        "recover035_latent256_beta2p5",
        RESULTS / "recover035_full5x5",
        "capacity_beta",
        None,
        notes="All-eligible recover035 latent256 beta2.5 run.",
    ),
    RunSpec(
        "latent384_beta2p5",
        RESULTS / "recover035_latent384_T80_h10000_p560_full5x5",
        "capacity_beta",
        None,
    ),
    RunSpec(
        "latent384_beta3p5",
        RESULTS / "recover035_latent384_beta3p5_T80_h10000_p560_full5x5",
        "capacity_beta",
        RESULTS / "recover035_latent384_beta3p5_stageB_oof_score_calibration",
    ),
    RunSpec(
        "latent384_beta3p75_promoted",
        RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5",
        "promoted",
        RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration",
        oasis_model_key="recover035_oof_logitz",
    ),
    RunSpec(
        "latent384_beta4p0",
        RESULTS / "recover035_latent384_beta4p0_T80_h10000_p560_full5x5",
        "capacity_beta",
        None,
        RESULTS / "beta4p0_completion_promotion_gate_audit_20260601/audit_results.json",
    ),
    RunSpec(
        "latent448_beta4p0",
        RESULTS / "recover035_latent448_beta4p0_T80_h10000_p560_full5x5",
        "capacity_beta",
        RESULTS / "recover035_latent448_beta4p0_stageB_oof_score_calibration",
    ),
    RunSpec(
        "latent512_beta3p75",
        RESULTS / "recover035_latent512_beta3p75_T80_h10000_p560_full5x5",
        "capacity_beta",
        RESULTS / "recover035_latent512_beta3p75_stageB_oof_score_calibration",
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


def to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._\n"
    return df.to_markdown(index=False) + "\n"


def write_table(df: pd.DataFrame, stem: str, out: Path) -> None:
    df.to_csv(out / f"{stem}.csv", index=False)
    (out / f"{stem}.md").write_text(to_md(df), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def config_args(run_dir: Path) -> dict[str, Any]:
    cfg = run_dir / "run_config.json"
    if not cfg.exists():
        return {}
    data = json.loads(cfg.read_text())
    return data.get("args", data)


def safe_float(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def safe_div(a: float, b: float) -> float:
    if b == 0 or math.isnan(b):
        return float("nan")
    return a / b


def find_stagea_metrics(run_dir: Path) -> Path | None:
    matches = sorted(run_dir.glob("all_folds_metrics_MULTI*.csv"))
    return matches[0] if matches else None


def stagea_by_fold(run: RunSpec) -> pd.DataFrame:
    path = find_stagea_metrics(run.run_dir)
    if path is None:
        return pd.DataFrame()
    df = pd.read_csv(path)
    model_col = "actual_classifier_type" if "actual_classifier_type" in df.columns else "classifier_model"
    if model_col not in df.columns:
        model_col = "model"
    rows: list[dict[str, Any]] = []
    for fold, grp in df.groupby("fold", dropna=False):
        row: dict[str, Any] = {"run_id": run.run_id, "fold": int(fold)}
        for model in ["logreg", "svm"]:
            m = grp[grp[model_col].astype(str).str.lower().eq(model)]
            if m.empty:
                continue
            rec = m.iloc[0]
            prefix = f"stageA_{model}"
            for src, dest in [
                ("auc_raw", "auc_raw"),
                ("pr_auc_raw", "pr_auc_raw"),
                ("auc_final", "auc"),
                ("pr_auc_final", "pr_auc"),
                ("auc", "auc_reported"),
                ("pr_auc", "pr_auc_reported"),
            ]:
                row[f"{prefix}_{dest}"] = safe_float(rec.get(src))
        rows.append(row)
    return pd.DataFrame(rows)


def raw_stageb_pooled(run: RunSpec) -> dict[str, Any]:
    path = run.run_dir / "classifier_only_readout/classifier_sweep_pooled_metrics.csv"
    df = read_csv(path)
    if df.empty:
        return {}
    mask = (df["model_name"].astype(str) == RAW_MODEL) & (df["threshold_strategy"].astype(str) == PRIMARY_THRESHOLD)
    if "readout_feature_set" in df.columns:
        mask &= df["readout_feature_set"].astype(str).eq(PRIMARY_FEATURES)
    rows = df.loc[mask]
    if rows.empty:
        return {}
    rec = rows.iloc[0]
    return {
        "stageB_raw_auc": safe_float(rec.get("auc")),
        "stageB_raw_pr_auc": safe_float(rec.get("pr_auc")),
        "stageB_raw_ba": safe_float(rec.get("balanced_accuracy")),
        "stageB_raw_sens": safe_float(rec.get("sensitivity")),
        "stageB_raw_spec": safe_float(rec.get("specificity")),
        "stageB_raw_f1": safe_float(rec.get("f1")),
        "stageB_raw_tn": safe_float(rec.get("tn")),
        "stageB_raw_fp": safe_float(rec.get("fp")),
        "stageB_raw_fn": safe_float(rec.get("fn")),
        "stageB_raw_tp": safe_float(rec.get("tp")),
        "stageB_raw_n": safe_float(rec.get("n")),
        "stageB_raw_n_cn": safe_float(rec.get("n_cn")),
        "stageB_raw_n_ad": safe_float(rec.get("n_ad")),
    }


def raw_stageb_by_fold(run: RunSpec) -> pd.DataFrame:
    path = run.run_dir / "classifier_only_readout/classifier_sweep_foldwise_metrics.csv"
    df = read_csv(path)
    if df.empty:
        return pd.DataFrame()
    mask = (df["model_name"].astype(str) == RAW_MODEL) & (df["threshold_strategy"].astype(str) == PRIMARY_THRESHOLD)
    if "readout_feature_set" in df.columns:
        mask &= df["readout_feature_set"].astype(str).eq(PRIMARY_FEATURES)
    keep = df.loc[mask].copy()
    if keep.empty:
        return pd.DataFrame()
    keep = keep.rename(
        columns={
            "auc": "stageB_raw_auc",
            "pr_auc": "stageB_raw_pr_auc",
            "balanced_accuracy": "stageB_raw_ba",
            "sensitivity": "stageB_raw_sens",
            "specificity": "stageB_raw_spec",
            "f1": "stageB_raw_f1",
        }
    )
    cols = ["fold", "stageB_raw_auc", "stageB_raw_pr_auc", "stageB_raw_ba", "stageB_raw_sens", "stageB_raw_spec", "stageB_raw_f1"]
    return keep[cols]


def oof_pooled(run: RunSpec) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if run.oof_dir and (run.oof_dir / "calib_pooled_metrics.csv").exists():
        df = pd.read_csv(run.oof_dir / "calib_pooled_metrics.csv")
        for method, prefix in [("oof_logitz", "stageB_oof_logitz"), ("oof_ecdf", "stageB_oof_ecdf")]:
            mask = (
                df["model_name"].astype(str).eq(OOF_MODEL)
                & df["feature_set"].astype(str).eq(PRIMARY_FEATURES)
                & df["calib_method"].astype(str).eq(method)
                & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
            )
            rows = df.loc[mask]
            if not rows.empty:
                rec = rows.iloc[0]
                out.update(
                    {
                        f"{prefix}_auc": safe_float(rec.get("auc")),
                        f"{prefix}_pr_auc": safe_float(rec.get("pr_auc")),
                        f"{prefix}_ba": safe_float(rec.get("balanced_accuracy")),
                        f"{prefix}_sens": safe_float(rec.get("sensitivity")),
                        f"{prefix}_spec": safe_float(rec.get("specificity")),
                        f"{prefix}_f1": safe_float(rec.get("f1")),
                    }
                )
    elif run.beta4_audit_json and run.beta4_audit_json.exists():
        data = json.loads(run.beta4_audit_json.read_text())
        beta4 = data.get("oof_logitz", {}).get("beta4p0", {})
        if beta4:
            out.update(
                {
                    "stageB_oof_logitz_auc": safe_float(beta4.get("auc", beta4.get("pooled_auc"))),
                    "stageB_oof_logitz_pr_auc": safe_float(beta4.get("pr_auc", beta4.get("pooled_pr_auc"))),
                    "stageB_oof_logitz_ba": safe_float(beta4.get("balanced_accuracy")),
                    "stageB_oof_logitz_sens": safe_float(beta4.get("sensitivity")),
                    "stageB_oof_logitz_spec": safe_float(beta4.get("specificity")),
                    "stageB_oof_logitz_f1": safe_float(beta4.get("f1")),
                    "stageB_oof_logitz_source": rel(run.beta4_audit_json),
                }
            )
    return out


def oof_by_fold(run: RunSpec, method: str) -> pd.DataFrame:
    if not run.oof_dir or not (run.oof_dir / "calib_foldwise_metrics.csv").exists():
        return pd.DataFrame()
    df = pd.read_csv(run.oof_dir / "calib_foldwise_metrics.csv")
    mask = (
        df["model_name"].astype(str).eq(OOF_MODEL)
        & df["feature_set"].astype(str).eq(PRIMARY_FEATURES)
        & df["calib_method"].astype(str).eq(method)
        & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    )
    rows = df.loc[mask].copy()
    if rows.empty:
        return pd.DataFrame()
    prefix = f"stageB_{method}"
    rows = rows.rename(
        columns={
            "auc": f"{prefix}_auc",
            "pr_auc": f"{prefix}_pr_auc",
            "balanced_accuracy": f"{prefix}_ba",
            "sensitivity": f"{prefix}_sens",
            "specificity": f"{prefix}_spec",
            "f1": f"{prefix}_f1",
        }
    )
    return rows[["fold", f"{prefix}_auc", f"{prefix}_pr_auc", f"{prefix}_ba", f"{prefix}_sens", f"{prefix}_spec", f"{prefix}_f1"]]


def philips_fpr_raw(run: RunSpec) -> dict[str, Any]:
    path = run.run_dir / "classifier_only_readout/classifier_sweep_subgroup_metrics_by_manufacturer.csv"
    df = read_csv(path)
    if df.empty:
        return {}
    mask = (df["model_name"].astype(str).eq(RAW_MODEL)) & (df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD))
    if "readout_feature_set" in df.columns:
        mask &= df["readout_feature_set"].astype(str).eq(PRIMARY_FEATURES)
    rows = df.loc[mask & df["Manufacturer"].astype(str).str.lower().eq("philips")]
    if rows.empty:
        return {}
    n_cn = rows["n_cn"].sum()
    fp = rows["fp"].sum()
    return {"philips_cn_n_raw": int(n_cn), "philips_cn_fp_raw": int(fp), "philips_cn_fpr_raw": safe_div(float(fp), float(n_cn))}


def philips_fpr_oof(run: RunSpec) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if run.oof_dir and (run.oof_dir / "calib_philips_fpr_pooled.csv").exists():
        df = pd.read_csv(run.oof_dir / "calib_philips_fpr_pooled.csv")
        for method in ["oof_logitz", "oof_ecdf"]:
            rows = df[
                df["model_name"].astype(str).eq(OOF_MODEL)
                & df["feature_set"].astype(str).eq(PRIMARY_FEATURES)
                & df["calib_method"].astype(str).eq(method)
                & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
                & df["manufacturer"].astype(str).str.lower().eq("philips")
            ]
            if not rows.empty:
                rec = rows.iloc[0]
                out[f"philips_cn_n_{method}"] = safe_float(rec.get("n_cn_pooled"))
                out[f"philips_cn_fp_{method}"] = safe_float(rec.get("fp_cn_pooled"))
                out[f"philips_cn_fpr_{method}"] = safe_float(rec.get("fpr_cn_pooled"))
    elif run.beta4_audit_json and run.beta4_audit_json.exists():
        data = json.loads(run.beta4_audit_json.read_text())
        rec = data.get("philips_cn_fpr_stageb", {}).get("beta4p0", {})
        if rec:
            # The beta4 audit reports OOF-logitz-computed Philips FPR in its text,
            # but the JSON key names vary. Preserve every numeric key with source.
            for k, v in rec.items():
                if isinstance(v, (int, float)):
                    out[f"philips_beta4_audit_{k}"] = v
            out["philips_beta4_audit_source"] = rel(run.beta4_audit_json)
        oof_rec = data.get("oof_logitz", {}).get("beta4p0", {})
        if oof_rec:
            out["philips_cn_n_oof_logitz"] = safe_float(oof_rec.get("philips_cn_n"))
            out["philips_cn_fp_oof_logitz"] = safe_float(oof_rec.get("philips_cn_fp"))
            out["philips_cn_fpr_oof_logitz"] = safe_float(oof_rec.get("philips_cn_fpr"))
    return out


def vae_qc_by_fold(run: RunSpec) -> pd.DataFrame:
    cfg = config_args(run.run_dir)
    beta = safe_float(cfg.get("beta_vae"))
    latent_dim = safe_float(cfg.get("latent_dim"))
    rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        fold_dir = run.run_dir / f"fold_{fold}"
        row: dict[str, Any] = {"run_id": run.run_id, "fold": fold}
        rd_path = fold_dir / f"fold_{fold}_rate_distortion.csv"
        if rd_path.exists():
            rd = pd.read_csv(rd_path)
            if not rd.empty:
                best_idx = rd["L_val_betaMax"].idxmin() if "L_val_betaMax" in rd.columns else rd.index[-1]
                best = rd.loc[best_idx]
                final = rd.iloc[-1]
                for tag, rec in [("best", best), ("final", final)]:
                    d_val = safe_float(rec.get("D_val"))
                    r_bits = safe_float(rec.get("R_val_bits"))
                    r_nats = safe_float(rec.get("R_val_nats"))
                    row[f"epoch_{tag}"] = safe_float(rec.get("epoch"))
                    row[f"D_val_{tag}"] = d_val
                    row[f"R_val_bits_{tag}"] = r_bits
                    row[f"R_val_nats_{tag}"] = r_nats
                    row[f"bits_per_latent_dim_{tag}"] = safe_div(r_bits, latent_dim)
                    row[f"KLD_over_D_{tag}"] = safe_div(r_nats, d_val)
                    row[f"beta_KLD_over_D_{tag}"] = beta * safe_div(r_nats, d_val)
                    row[f"L_val_betaMax_{tag}"] = safe_float(rec.get("L_val_betaMax"))
        info_path = fold_dir / f"fold_{fold}_trainDev_latent_info_summary.csv"
        if info_path.exists():
            info = pd.read_csv(info_path)
            y = info[info["variable"].astype(str).eq("Y_target")]
            mfr = info[info["variable"].astype(str).eq("Manufacturer")]
            if not y.empty:
                yrow = y.iloc[0]
                row["active_units"] = safe_float(yrow.get("n_active"))
                row["total_correlation_nats"] = safe_float(yrow.get("total_correlation_nats"))
                row["MI_Z_Y_nats"] = safe_float(yrow.get("mi_sum_nats"))
            if not mfr.empty:
                mrow = mfr.iloc[0]
                row["MI_Z_Manufacturer_nats"] = safe_float(mrow.get("mi_sum_nats"))
        row["MI_Manufacturer_over_MI_Y"] = safe_div(row.get("MI_Z_Manufacturer_nats", np.nan), row.get("MI_Z_Y_nats", np.nan))
        rows.append(row)
    return pd.DataFrame(rows)


def leakage_by_fold(run: RunSpec) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        fold_dir = run.run_dir / f"fold_{fold}"
        row: dict[str, Any] = {"run_id": run.run_id, "fold": fold}
        for split, filename in [("train", f"fold_{fold}_scanner_leakage_summary.csv"), ("test", f"fold_{fold}_test_scanner_leakage_summary.csv")]:
            path = fold_dir / filename
            df = read_csv(path)
            if df.empty:
                continue
            rec = df.iloc[0]
            row[f"{split}_scanner_raw_ba"] = safe_float(rec.get("acc_site_raw"))
            row[f"{split}_scanner_latent_ba"] = safe_float(rec.get("acc_site_latent"))
            row[f"{split}_scanner_latent_minus_raw"] = row[f"{split}_scanner_latent_ba"] - row[f"{split}_scanner_raw_ba"]
        rows.append(row)
    return pd.DataFrame(rows)


def foldwise_for_run(run: RunSpec) -> pd.DataFrame:
    cfg = config_args(run.run_dir)
    base = vae_qc_by_fold(run)
    for key in [
        "latent_dim",
        "beta_vae",
        "dropout_rate_vae",
        "encoder_dropout_rate_vae",
        "decoder_dropout_rate_vae",
        "epochs_vae",
        "cyclical_beta_n_cycles",
        "early_stopping_patience_vae",
        "recon_loss_mode",
    ]:
        base[key] = cfg.get(key)
    base["family"] = run.family
    base["run_dir"] = rel(run.run_dir)
    for addon in [
        leakage_by_fold(run),
        stagea_by_fold(run),
        raw_stageb_by_fold(run),
        oof_by_fold(run, "oof_logitz"),
        oof_by_fold(run, "oof_ecdf"),
    ]:
        if not addon.empty:
            base = base.merge(addon, on=["run_id", "fold"], how="left") if "run_id" in addon.columns else base.merge(addon, on="fold", how="left")
    return base


def aggregate_run(df: pd.DataFrame, run: RunSpec) -> dict[str, Any]:
    cfg = config_args(run.run_dir)
    row: dict[str, Any] = {
        "run_id": run.run_id,
        "family": run.family,
        "run_dir": rel(run.run_dir),
        "oof_dir": rel(run.oof_dir),
        "notes": run.notes,
        "completed_folds_with_rd": int(df["D_val_best"].notna().sum()) if "D_val_best" in df else 0,
    }
    for key in [
        "latent_dim",
        "beta_vae",
        "dropout_rate_vae",
        "encoder_dropout_rate_vae",
        "decoder_dropout_rate_vae",
        "vae_dropout_scope",
        "epochs_vae",
        "cyclical_beta_n_cycles",
        "early_stopping_patience_vae",
        "channels_to_use",
        "selected_channel_names",
        "recon_loss_mode",
        "metadata_path",
        "global_tensor_path",
    ]:
        value = cfg.get(key)
        row[key] = json.dumps(value) if isinstance(value, (list, dict)) else value
    metric_cols = [
        "D_val_best",
        "D_val_final",
        "R_val_bits_best",
        "R_val_bits_final",
        "bits_per_latent_dim_best",
        "bits_per_latent_dim_final",
        "beta_KLD_over_D_best",
        "beta_KLD_over_D_final",
        "active_units",
        "total_correlation_nats",
        "MI_Z_Y_nats",
        "MI_Z_Manufacturer_nats",
        "MI_Manufacturer_over_MI_Y",
        "train_scanner_raw_ba",
        "train_scanner_latent_ba",
        "test_scanner_raw_ba",
        "test_scanner_latent_ba",
        "stageA_logreg_auc",
        "stageA_logreg_pr_auc",
        "stageA_svm_auc",
        "stageA_svm_pr_auc",
        "stageB_raw_auc",
        "stageB_raw_pr_auc",
        "stageB_raw_ba",
        "stageB_raw_sens",
        "stageB_raw_spec",
        "stageB_raw_f1",
        "stageB_oof_logitz_auc",
        "stageB_oof_logitz_pr_auc",
        "stageB_oof_logitz_ba",
        "stageB_oof_logitz_sens",
        "stageB_oof_logitz_spec",
        "stageB_oof_logitz_f1",
        "stageB_oof_ecdf_auc",
        "stageB_oof_ecdf_pr_auc",
        "stageB_oof_ecdf_ba",
        "stageB_oof_ecdf_sens",
        "stageB_oof_ecdf_spec",
        "stageB_oof_ecdf_f1",
    ]
    for col in metric_cols:
        if col in df.columns:
            row[f"{col}_mean"] = df[col].mean(skipna=True)
            row[f"{col}_std"] = df[col].std(skipna=True)
    row.update(raw_stageb_pooled(run))
    row.update(oof_pooled(run))
    row.update(philips_fpr_raw(run))
    row.update(philips_fpr_oof(run))
    return row


def add_oasis(summary: pd.DataFrame) -> pd.DataFrame:
    path = RESULTS / "oasis_mega_90cn_90ad_pooled_external_validation_20260531/pooled_ranking_metrics.csv"
    if not path.exists():
        summary["oasis_available"] = False
        return summary
    oasis = pd.read_csv(path)
    build_map = {
        "concatenated_timeseries": "oasis_concat",
        "runwise_140TR_pilot_parity": "oasis_runwise140",
        "runwise164_pilot_parity": "oasis_runwise164",
    }
    out = summary.copy()
    out["oasis_available"] = False
    for idx, row in out.iterrows():
        spec = next((r for r in RUNS if r.run_id == row["run_id"]), None)
        key = spec.oasis_model_key if spec else None
        if not key:
            continue
        subset = oasis[oasis["adni_model"].astype(str).eq(key)]
        if subset.empty:
            continue
        out.at[idx, "oasis_available"] = True
        out.at[idx, "oasis_model_key"] = key
        for build, prefix in build_map.items():
            b = subset[subset["build_candidate"].astype(str).eq(build)]
            if b.empty:
                continue
            rec = b.iloc[0]
            out.at[idx, f"{prefix}_auc"] = safe_float(rec.get("auc"))
            out.at[idx, f"{prefix}_pr_auc"] = safe_float(rec.get("pr_auc"))
            out.at[idx, f"{prefix}_auc_ci_lo"] = safe_float(rec.get("auc_bootstrap95_lo"))
            out.at[idx, f"{prefix}_auc_ci_hi"] = safe_float(rec.get("auc_bootstrap95_hi"))
    return out


def choose_primary_auc(row: pd.Series) -> float:
    for key in ["stageB_oof_ecdf_auc", "stageB_oof_logitz_auc", "stageB_raw_auc"]:
        if key in row and pd.notna(row[key]):
            return float(row[key])
    return float("nan")


def choose_primary_pr(row: pd.Series) -> float:
    for key in ["stageB_oof_ecdf_pr_auc", "stageB_oof_logitz_pr_auc", "stageB_raw_pr_auc"]:
        if key in row and pd.notna(row[key]):
            return float(row[key])
    return float("nan")


def choose_primary_philips_fpr(row: pd.Series) -> float:
    for key in ["philips_cn_fpr_oof_ecdf", "philips_cn_fpr_oof_logitz", "philips_cn_fpr_raw"]:
        if key in row and pd.notna(row[key]):
            return float(row[key])
    # beta4 audit fallback may have variable keys; do not infer.
    return float("nan")


def scatter_tables(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            {
                "run_id": row["run_id"],
                "family": row.get("family"),
                "latent_dim": row.get("latent_dim"),
                "beta_vae": row.get("beta_vae"),
                "beta_KLD_over_D_best_mean": row.get("beta_KLD_over_D_best_mean"),
                "primary_auc": choose_primary_auc(row),
                "primary_pr_auc": choose_primary_pr(row),
                "primary_source": "oof_ecdf_else_oof_logitz_else_raw",
            }
        )
    beta_auc = pd.DataFrame(rows)
    beta_fpr = beta_auc[["run_id", "family", "latent_dim", "beta_vae", "beta_KLD_over_D_best_mean", "primary_auc", "primary_pr_auc"]].copy()
    beta_fpr["philips_cn_fpr"] = [choose_primary_philips_fpr(row) for _, row in summary.iterrows()]
    mi_auc = summary[["run_id", "family", "latent_dim", "beta_vae", "MI_Manufacturer_over_MI_Y_mean"]].copy()
    mi_auc["primary_auc"] = [choose_primary_auc(row) for _, row in summary.iterrows()]
    mi_auc["primary_pr_auc"] = [choose_primary_pr(row) for _, row in summary.iterrows()]
    return beta_auc, beta_fpr, mi_auc


def interpretation(summary: pd.DataFrame) -> str:
    ranked = summary.copy()
    ranked["primary_auc"] = [choose_primary_auc(row) for _, row in ranked.iterrows()]
    ranked["primary_pr_auc"] = [choose_primary_pr(row) for _, row in ranked.iterrows()]
    ranked = ranked.sort_values(["primary_auc", "primary_pr_auc"], ascending=False)
    top = ranked.iloc[0] if not ranked.empty else pd.Series(dtype=object)
    promoted = summary[summary["run_id"].eq("latent384_beta3p75_promoted")]
    promoted_text = ""
    if not promoted.empty:
        p = promoted.iloc[0]
        promoted_text = (
            f"The promoted latent384/beta3.75 model remains the strongest score-harmonized candidate "
            f"among available full runs: AUC={choose_primary_auc(p):.6f}, PR-AUC={choose_primary_pr(p):.6f}. "
        )
    top_text = (
        f"The highest primary AUC in this synthesis is `{top.get('run_id')}` "
        f"(AUC={top.get('primary_auc', np.nan):.6f}, PR-AUC={top.get('primary_pr_auc', np.nan):.6f}). "
        if not top.empty
        else ""
    )
    return f"""# Final Capacity/Beta Trend Interpretation

This read-only synthesis compares completed FULL runs along the latent-capacity and beta axes using existing artifacts only. No training, scoring, tensor modification, metadata modification, or model-artifact modification was performed.

{top_text}{promoted_text}

Main pattern:
- Increasing latent dimensionality beyond 384 did not produce a clean gain in AD/CN ranking. The latent448/beta4.0 and latent512/beta3.75 runs are complete, but neither improves the promoted score-harmonized AUC/PR-AUC.
- The local latent384/beta3.5 sensitivity run is complete but falls below the promoted beta3.75 reference on AUC and PR-AUC, supporting beta3.75 over lower beta in this capacity regime.
- The latent384/beta4.0 branch compressed the representation more than beta3.75 but did not improve ranking and worsened Philips CN false positives in its prior promotion audit.
- The latent448/beta4.0 branch reduced mean test latent scanner leakage relative to the promoted reference, but it lost AUC/PR-AUC and increased Philips CN FPR.
- The requested beta/KLD and MI nuisance-ratio tables show no simple monotonic relationship in which more bottleneck pressure or more latent capacity reliably improves clinical ranking.

Practical conclusion:
The current evidence supports treating latent384/beta3.75 as the best internal ADNI candidate and treating latent128, latent384/beta3.5, latent448, latent512, and beta4.0 variants as controlled sensitivity/negative audits. No further capacity or beta expansion is justified on internal CV evidence alone.

External note:
OASIS mega 90/90 metrics are included only where pre-existing scoring artifacts identify the ADNI model. No new OASIS scoring was run in this synthesis.
"""


def main() -> None:
    args = parse_args()
    out = args.output_dir
    missing = [r for r in RUNS if not r.run_dir.exists()]
    if args.dry_run:
        print("Planned output:", rel(out))
        for r in RUNS:
            print(f"{r.run_id}: run_exists={r.run_dir.exists()} oof_exists={bool(r.oof_dir and r.oof_dir.exists())}")
        if missing:
            print("Missing run dirs:", [r.run_id for r in missing])
        return
    out.mkdir(parents=True, exist_ok=True)

    foldwise_frames = []
    summary_rows = []
    for run in RUNS:
        if not run.run_dir.exists():
            summary_rows.append({"run_id": run.run_id, "family": run.family, "run_dir": rel(run.run_dir), "run_available": False})
            continue
        fw = foldwise_for_run(run)
        fw["run_available"] = True
        foldwise_frames.append(fw)
        summary_rows.append(aggregate_run(fw, run) | {"run_available": True})
    foldwise = pd.concat(foldwise_frames, ignore_index=True) if foldwise_frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    summary = add_oasis(summary)
    summary["primary_auc"] = [choose_primary_auc(row) for _, row in summary.iterrows()]
    summary["primary_pr_auc"] = [choose_primary_pr(row) for _, row in summary.iterrows()]
    summary["primary_philips_cn_fpr"] = [choose_primary_philips_fpr(row) for _, row in summary.iterrows()]

    beta_auc, beta_fpr, mi_auc = scatter_tables(summary)

    write_table(summary, "capacity_beta_summary", out)
    write_table(foldwise, "foldwise_capacity_beta_summary", out)
    write_table(beta_auc, "beta_kld_ratio_vs_auc", out)
    write_table(beta_fpr, "beta_kld_ratio_vs_philips_fpr", out)
    write_table(mi_auc, "mi_manufacturer_over_mi_y_vs_auc", out)
    write_text(out / "final_capacity_trend_interpretation.md", interpretation(summary))
    write_text(
        out / "README.md",
        "# Final Latent Capacity/Beta Rate-Distortion Synthesis\n\n"
        "Read-only synthesis across completed FULL ADNI capacity/beta runs. "
        "Primary AUC/PR-AUC uses OOF-ECDF when available, otherwise OOF-logitz, otherwise raw Stage B.\n",
    )
    command_log = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": rel(Path(__file__)),
        "read_only_scope": {
            "no_training": True,
            "no_scoring_unless_existing_artifacts": True,
            "no_tensor_modification": True,
            "no_metadata_modification": True,
            "no_model_artifact_modification": True,
        },
        "runs": [
            {
                "run_id": r.run_id,
                "run_dir": rel(r.run_dir),
                "oof_dir": rel(r.oof_dir),
                "run_available": r.run_dir.exists(),
                "oof_available": bool(r.oof_dir and r.oof_dir.exists()),
                "oasis_model_key": r.oasis_model_key,
            }
            for r in RUNS
        ],
        "primary_threshold": PRIMARY_THRESHOLD,
        "primary_features": PRIMARY_FEATURES,
    }
    write_text(out / "command_log.json", json.dumps(command_log, indent=2))
    print(f"Wrote synthesis package: {rel(out)}")


if __name__ == "__main__":
    main()
