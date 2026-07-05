#!/usr/bin/env python3
"""Read-only completion and promotion-gate audit for ch12 latent384 beta3.75.

This audit reads existing run outputs only and writes a comparison package.
It does not train, score OASIS, modify tensors, modify metadata, or modify
model artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

OUT = ROOT / "results/revision_bspc_2026/ch12_latent384_beta3p75_completion_promotion_gate_audit_20260605"

RUNS = {
    "candidate_ch12_latent384_beta3p75": {
        "role": "candidate_channel_pair_sensitivity",
        "run_dir": ROOT / "results/revision_bspc_2026/recover035_ch12_latent384_beta3p75_T80_h10000_p560_full5x5",
        "oof_dir": ROOT
        / "results/revision_bspc_2026/recover035_ch12_latent384_beta3p75_T80_h10000_p560_full5x5_stageB_oof_score_calibration",
        "config": ROOT / "configs/runs/adni_v5_1c_recover035_ch12_latent384_beta3p75_T80_h10000_p560_full5x5.json",
    },
    "promoted_ch102_latent384_beta3p75": {
        "role": "promoted_primary_reference",
        "run_dir": ROOT / "results/revision_bspc_2026/recover035_latent384_beta3p75_T80_h10000_p560_full5x5",
        "oof_dir": ROOT / "results/revision_bspc_2026/recover035_latent384_beta3p75_stageB_oof_score_calibration",
        "config": ROOT / "configs/runs/adni_v5_1c_recover035_latent384_beta3p75_T80_h10000_p560_full5x5.json",
    },
    "ch1only_latent384_beta3p75": {
        "role": "parsimonious_channel_sensitivity_reference",
        "run_dir": ROOT / "results/revision_bspc_2026/recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5",
        "oof_dir": ROOT
        / "results/revision_bspc_2026/recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5_stageB_oof_score_calibration",
        "config": ROOT / "configs/runs/adni_v5_1c_recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5.json",
    },
}

PRIMARY_RAW_MODEL = "logreg_l2"
PRIMARY_OOF_MODEL = "logreg_l2_original"
PRIMARY_FEATURE_SET = "z_plus_age_sex"
PRIMARY_CALIB = "oof_ecdf"
SECONDARY_CALIB = "oof_logitz"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"

PROMOTED_AUC = 0.795155
PROMOTED_PR_AUC = 0.573934


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=OUT)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path, required: bool = False) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path)


def to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._\n"
    try:
        return df.to_markdown(index=False) + "\n"
    except Exception:
        return df.to_string(index=False) + "\n"


def write_table(df: pd.DataFrame, out_dir: Path, stem: str) -> None:
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    (out_dir / f"{stem}.md").write_text(to_md(df), encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def safe_float(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def safe_div(num: float, den: float) -> float:
    if den == 0 or math.isnan(num) or math.isnan(den):
        return float("nan")
    return num / den


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def find_stagea_metrics(run_dir: Path) -> Path | None:
    matches = sorted(run_dir.glob("all_folds_metrics_MULTI*.csv"))
    return matches[0] if matches else None


def completion_status(label: str, spec: dict[str, Any]) -> pd.DataFrame:
    run_dir = spec["run_dir"]
    oof_dir = spec["oof_dir"]
    readout_dir = run_dir / "classifier_only_readout"
    rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        fold_dir = run_dir / f"fold_{fold}"
        rows.append(
            {
                "run_label": label,
                "fold": fold,
                "run_dir_exists": run_dir.exists(),
                "fold_dir_exists": fold_dir.exists(),
                "vae_checkpoint": (fold_dir / f"vae_model_fold_{fold}.pt").exists(),
                "vae_history": (fold_dir / f"vae_train_history_fold_{fold}.joblib").exists(),
                "rate_distortion": (fold_dir / f"fold_{fold}_rate_distortion.csv").exists(),
                "stageA_logreg_predictions": (fold_dir / "test_predictions_logreg.csv").exists(),
                "stageA_svm_predictions": (fold_dir / "test_predictions_svm.csv").exists(),
                "stageA_metrics_file": find_stagea_metrics(run_dir) is not None,
                "stageB_readout_dir": readout_dir.exists(),
                "stageB_pooled_metrics": (readout_dir / "classifier_sweep_pooled_metrics.csv").exists(),
                "stageB_foldwise_metrics": (readout_dir / "classifier_sweep_foldwise_metrics.csv").exists(),
                "oof_score_calibration_dir": oof_dir.exists(),
                "oof_pooled_metrics": (oof_dir / "calib_pooled_metrics.csv").exists(),
                "oof_foldwise_metrics": (oof_dir / "calib_foldwise_metrics.csv").exists(),
                "trainDev_latent_info": (fold_dir / f"fold_{fold}_trainDev_latent_info_summary.csv").exists(),
                "test_latent_info": (fold_dir / f"fold_{fold}_test_latent_info_summary.csv").exists(),
                "trainDev_scanner_leakage": (fold_dir / f"fold_{fold}_scanner_leakage_summary.csv").exists(),
                "test_scanner_leakage": (fold_dir / f"fold_{fold}_test_scanner_leakage_summary.csv").exists(),
                "latent_qc_metrics": (fold_dir / "latent_qc_metrics.csv").exists(),
            }
        )
    return pd.DataFrame(rows)


def completion_summary(status: pd.DataFrame) -> pd.DataFrame:
    bool_cols = [c for c in status.columns if c not in {"run_label", "fold"} and status[c].dtype == bool]
    rows: list[dict[str, Any]] = []
    for label, grp in status.groupby("run_label"):
        rows.append(
            {
                "run_label": label,
                "n_folds": int(grp["fold"].nunique()),
                "all_5_folds_completed": bool((grp["fold_dir_exists"] & grp["vae_checkpoint"] & grp["vae_history"]).all())
                and int(grp["fold"].nunique()) == 5,
                "all_stageA_artifacts_present": bool((grp["stageA_logreg_predictions"] & grp["stageA_svm_predictions"]).all())
                and bool(grp["stageA_metrics_file"].all()),
                "stageB_raw_present": bool(grp["stageB_pooled_metrics"].all()),
                "stageB_oof_present": bool(grp["oof_pooled_metrics"].all()),
                "all_qc_artifacts_present": bool(
                    (
                        grp["rate_distortion"]
                        & grp["trainDev_latent_info"]
                        & grp["test_latent_info"]
                        & grp["trainDev_scanner_leakage"]
                        & grp["test_scanner_leakage"]
                    ).all()
                ),
                "all_checked_artifacts_present": bool(grp[bool_cols].all(axis=None)),
            }
        )
    return pd.DataFrame(rows)


def config_inventory() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, spec in RUNS.items():
        cfg = load_config(spec["config"])
        params = cfg.get("parameters", {})
        rows.append(
            {
                "run_label": label,
                "role": spec["role"],
                "config_path": rel(spec["config"]),
                "run_dir": rel(spec["run_dir"]),
                "oof_dir": rel(spec["oof_dir"]),
                "run_name": cfg.get("run_name"),
                "channels_to_use": params.get("channels_to_use"),
                "selected_channel_names": cfg.get("selected_channel_names"),
                "latent_dim": params.get("latent_dim"),
                "beta_vae": params.get("beta_vae"),
                "recon_loss_mode": params.get("recon_loss_mode"),
                "epochs_vae": params.get("epochs_vae"),
                "cyclical_beta_n_cycles": params.get("cyclical_beta_n_cycles"),
                "lr_scheduler_T0": params.get("lr_scheduler_T0"),
                "early_stopping_patience_vae": params.get("early_stopping_patience_vae"),
            }
        )
    return pd.DataFrame(rows)


def stagea_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    foldwise: list[pd.DataFrame] = []
    for label, spec in RUNS.items():
        path = find_stagea_metrics(spec["run_dir"])
        if path is None:
            continue
        df = pd.read_csv(path)
        df.insert(0, "run_label", label)
        df.insert(1, "source_file", rel(path))
        foldwise.append(df)
    if not foldwise:
        return pd.DataFrame(), pd.DataFrame()
    all_df = pd.concat(foldwise, ignore_index=True)
    model_col = "actual_classifier_type"
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
        if c in all_df.columns
    ]
    summary = all_df.groupby(["run_label", model_col], dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join([str(x) for x in c if str(x)]) if isinstance(c, tuple) else str(c) for c in summary.columns]
    summary = summary.rename(columns={f"{model_col}_": model_col})
    return all_df, summary


def stageb_pooled() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    keep = [
        "run_label",
        "readout_type",
        "model_name",
        "feature_set",
        "calib_method",
        "threshold_strategy",
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
        "predicted_ad_rate",
        "auc",
        "pr_auc",
    ]
    for label, spec in RUNS.items():
        raw = read_csv(spec["run_dir"] / "classifier_only_readout/classifier_sweep_pooled_metrics.csv")
        if not raw.empty:
            raw = raw.rename(columns={"readout_feature_set": "feature_set"}).copy()
            raw.insert(0, "run_label", label)
            raw.insert(1, "readout_type", "raw_classifier_only")
            raw["calib_method"] = "raw"
            raw = raw[(raw["model_name"] == PRIMARY_RAW_MODEL) & (raw["feature_set"] == PRIMARY_FEATURE_SET)]
            frames.append(raw)
        oof = read_csv(spec["oof_dir"] / "calib_pooled_metrics.csv")
        if not oof.empty:
            oof = oof.copy()
            oof.insert(0, "run_label", label)
            oof.insert(1, "readout_type", "oof_score_harmonized")
            oof = oof[
                (oof["model_name"] == PRIMARY_OOF_MODEL)
                & (oof["feature_set"] == PRIMARY_FEATURE_SET)
                & (oof["calib_method"].isin([SECONDARY_CALIB, PRIMARY_CALIB]))
            ]
            frames.append(oof)
    if not frames:
        return pd.DataFrame(columns=keep)
    out = pd.concat(frames, ignore_index=True)
    for col in keep:
        if col not in out.columns:
            out[col] = np.nan
    out = out[keep].copy()
    out["primary_promoted_convention"] = (
        (out["model_name"] == PRIMARY_OOF_MODEL)
        & (out["feature_set"] == PRIMARY_FEATURE_SET)
        & (out["calib_method"] == PRIMARY_CALIB)
        & (out["threshold_strategy"] == PRIMARY_THRESHOLD)
    )
    out["primary_raw_inner_oof_target"] = (
        (out["model_name"] == PRIMARY_RAW_MODEL)
        & (out["feature_set"] == PRIMARY_FEATURE_SET)
        & (out["calib_method"] == "raw")
        & (out["threshold_strategy"] == PRIMARY_THRESHOLD)
    )
    return out


def stageb_foldwise() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for label, spec in RUNS.items():
        raw = read_csv(spec["run_dir"] / "classifier_only_readout/classifier_sweep_foldwise_metrics.csv")
        if not raw.empty:
            raw = raw.rename(columns={"readout_feature_set": "feature_set"}).copy()
            raw.insert(0, "run_label", label)
            raw.insert(1, "readout_type", "raw_classifier_only")
            raw["calib_method"] = "raw"
            raw = raw[
                (raw["model_name"] == PRIMARY_RAW_MODEL)
                & (raw["feature_set"] == PRIMARY_FEATURE_SET)
                & (raw["threshold_strategy"] == PRIMARY_THRESHOLD)
            ]
            frames.append(raw)
        oof = read_csv(spec["oof_dir"] / "calib_foldwise_metrics.csv")
        if not oof.empty:
            oof = oof.copy()
            oof.insert(0, "run_label", label)
            oof.insert(1, "readout_type", "oof_score_harmonized")
            oof = oof[
                (oof["model_name"] == PRIMARY_OOF_MODEL)
                & (oof["feature_set"] == PRIMARY_FEATURE_SET)
                & (oof["calib_method"].isin([SECONDARY_CALIB, PRIMARY_CALIB]))
                & (oof["threshold_strategy"] == PRIMARY_THRESHOLD)
            ]
            frames.append(oof)
    if not frames:
        return pd.DataFrame()
    keep = [
        "run_label",
        "readout_type",
        "fold",
        "model_name",
        "feature_set",
        "calib_method",
        "threshold_strategy",
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
    out = pd.concat(frames, ignore_index=True)
    for col in keep:
        if col not in out.columns:
            out[col] = np.nan
    return out[keep]


def philips_fpr() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for label, spec in RUNS.items():
        raw = read_csv(spec["run_dir"] / "classifier_only_readout/classifier_sweep_subgroup_metrics_by_manufacturer.csv")
        if not raw.empty:
            raw = raw.rename(columns={"readout_feature_set": "feature_set", "Manufacturer": "manufacturer"}).copy()
            raw = raw[
                (raw["model_name"] == PRIMARY_RAW_MODEL)
                & (raw["feature_set"] == PRIMARY_FEATURE_SET)
                & (raw["threshold_strategy"] == PRIMARY_THRESHOLD)
                & (raw["manufacturer"].astype(str).eq("Philips"))
            ]
            if not raw.empty:
                pooled = {
                    "run_label": label,
                    "readout_type": "raw_classifier_only",
                    "model_name": PRIMARY_RAW_MODEL,
                    "feature_set": PRIMARY_FEATURE_SET,
                    "calib_method": "raw",
                    "threshold_strategy": PRIMARY_THRESHOLD,
                    "manufacturer": "Philips",
                    "n_cn_pooled": int(raw["n_cn"].sum()),
                    "fp_cn_pooled": int(raw["fp"].sum()),
                }
                pooled["fpr_cn_pooled"] = safe_div(float(pooled["fp_cn_pooled"]), float(pooled["n_cn_pooled"]))
                frames.append(pd.DataFrame([pooled]))
        oof = read_csv(spec["oof_dir"] / "calib_philips_fpr_pooled.csv")
        if not oof.empty:
            oof = oof.copy()
            oof.insert(0, "run_label", label)
            oof.insert(1, "readout_type", "oof_score_harmonized")
            oof = oof[
                (oof["model_name"] == PRIMARY_OOF_MODEL)
                & (oof["feature_set"] == PRIMARY_FEATURE_SET)
                & (oof["calib_method"].isin([SECONDARY_CALIB, PRIMARY_CALIB]))
                & (oof["threshold_strategy"] == PRIMARY_THRESHOLD)
                & (oof["manufacturer"].astype(str).eq("Philips"))
            ]
            frames.append(oof)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_rate_distortion() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for label, spec in RUNS.items():
        cfg = load_config(spec["config"])
        beta = safe_float(cfg.get("parameters", {}).get("beta_vae", 3.75))
        latent_dim = safe_float(cfg.get("parameters", {}).get("latent_dim", np.nan))
        for fold in range(1, 6):
            path = spec["run_dir"] / f"fold_{fold}/fold_{fold}_rate_distortion.csv"
            row: dict[str, Any] = {"run_label": label, "fold": fold, "beta_vae": beta, "latent_dim": latent_dim}
            if path.exists():
                df = pd.read_csv(path)
                if not df.empty:
                    best_idx = df["L_val_betaMax"].idxmin() if "L_val_betaMax" in df.columns else df.index[-1]
                    best = df.loc[best_idx]
                    last = df.iloc[-1]
                    D = safe_float(best.get("D_val"))
                    R_nats = safe_float(best.get("R_val_nats"))
                    R_bits = safe_float(best.get("R_val_bits"))
                    row.update(
                        {
                            "best_epoch": int(best.get("epoch", np.nan)),
                            "final_epoch": int(last.get("epoch", np.nan)),
                            "D_val_best": D,
                            "R_val_nats_best": R_nats,
                            "R_val_bits_best": R_bits,
                            "R_bits_per_latent_dim_best": safe_div(R_bits, latent_dim),
                            "KLD_over_D_best": safe_div(R_nats, D),
                            "beta_KLD_over_D_best": beta * safe_div(R_nats, D),
                            "L_val_betaMax_best": safe_float(best.get("L_val_betaMax")),
                        }
                    )
            rows.append(row)
    foldwise = pd.DataFrame(rows)
    metric_cols = [
        "D_val_best",
        "R_val_bits_best",
        "R_bits_per_latent_dim_best",
        "KLD_over_D_best",
        "beta_KLD_over_D_best",
        "L_val_betaMax_best",
    ]
    summary = foldwise.groupby("run_label", dropna=False)[metric_cols].agg(["mean", "std", "min", "max"]).reset_index()
    summary.columns = ["_".join([str(x) for x in c if str(x)]) if isinstance(c, tuple) else str(c) for c in summary.columns]
    return foldwise, summary


def latent_info() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for label, spec in RUNS.items():
        for fold in range(1, 6):
            for split, name in [
                ("trainDev", f"fold_{fold}_trainDev_latent_info_summary.csv"),
                ("test", f"fold_{fold}_test_latent_info_summary.csv"),
            ]:
                path = spec["run_dir"] / f"fold_{fold}" / name
                if not path.exists():
                    rows.append({"run_label": label, "fold": fold, "split": split, "available": False})
                    continue
                df = pd.read_csv(path)
                y = df[df["variable"].astype(str).eq("Y_target")]
                m = df[df["variable"].astype(str).eq("Manufacturer")]
                rec: dict[str, Any] = {"run_label": label, "fold": fold, "split": split, "available": True}
                if not y.empty:
                    yr = y.iloc[0]
                    rec.update(
                        {
                            "MI_Z_Y_nats": safe_float(yr.get("mi_sum_nats")),
                            "MI_Z_Y_mean_nats": safe_float(yr.get("mi_mean_nats")),
                            "active_units": safe_float(yr.get("n_active")),
                            "frac_active": safe_float(yr.get("frac_active")),
                            "total_correlation_nats": safe_float(yr.get("total_correlation_nats")),
                        }
                    )
                if not m.empty:
                    mr = m.iloc[0]
                    rec.update(
                        {
                            "MI_Z_Manufacturer_nats": safe_float(mr.get("mi_sum_nats")),
                            "MI_Z_Manufacturer_mean_nats": safe_float(mr.get("mi_mean_nats")),
                        }
                    )
                rec["MI_Manufacturer_over_MI_Y"] = safe_div(
                    rec.get("MI_Z_Manufacturer_nats", np.nan), rec.get("MI_Z_Y_nats", np.nan)
                )
                rows.append(rec)
    foldwise = pd.DataFrame(rows)
    metric_cols = [
        "MI_Z_Y_nats",
        "MI_Z_Manufacturer_nats",
        "MI_Manufacturer_over_MI_Y",
        "active_units",
        "frac_active",
        "total_correlation_nats",
    ]
    summary = foldwise.groupby(["run_label", "split"], dropna=False)[metric_cols].agg(["mean", "std", "min", "max"]).reset_index()
    summary.columns = ["_".join([str(x) for x in c if str(x)]) if isinstance(c, tuple) else str(c) for c in summary.columns]
    return foldwise, summary


def scanner_leakage() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for label, spec in RUNS.items():
        for fold in range(1, 6):
            for split, filename in [
                ("trainDev", f"fold_{fold}_scanner_leakage_summary.csv"),
                ("test", f"fold_{fold}_test_scanner_leakage_summary.csv"),
            ]:
                path = spec["run_dir"] / f"fold_{fold}" / filename
                if not path.exists():
                    rows.append({"run_label": label, "fold": fold, "split": split, "available": False})
                    continue
                df = pd.read_csv(path)
                if df.empty:
                    rows.append({"run_label": label, "fold": fold, "split": split, "available": False})
                    continue
                rec = df.iloc[0].to_dict()
                rec.update({"run_label": label, "fold": fold, "split": split, "available": True})
                rec["latent_minus_raw"] = safe_float(rec.get("acc_site_latent")) - safe_float(rec.get("acc_site_raw"))
                rows.append(rec)
    foldwise = pd.DataFrame(rows)
    metric_cols = ["acc_site_raw", "acc_site_latent", "latent_minus_raw"]
    summary = foldwise.groupby(["run_label", "split"], dropna=False)[metric_cols].agg(["mean", "std", "min", "max"]).reset_index()
    summary.columns = ["_".join([str(x) for x in c if str(x)]) if isinstance(c, tuple) else str(c) for c in summary.columns]
    return foldwise, summary


def primary_rows(stageb: pd.DataFrame, philips: pd.DataFrame, leakage_summary: pd.DataFrame, rd_summary: pd.DataFrame, mi_summary: pd.DataFrame) -> pd.DataFrame:
    rows = stageb[
        (
            (stageb["readout_type"].eq("oof_score_harmonized"))
            & (stageb["model_name"].eq(PRIMARY_OOF_MODEL))
            & (stageb["feature_set"].eq(PRIMARY_FEATURE_SET))
            & (stageb["calib_method"].eq(PRIMARY_CALIB))
            & (stageb["threshold_strategy"].eq(PRIMARY_THRESHOLD))
        )
    ].copy()
    rows = rows[
        [
            "run_label",
            "readout_type",
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
            "auc",
            "pr_auc",
        ]
    ]
    phil = philips[
        (philips["readout_type"].eq("oof_score_harmonized"))
        & (philips["model_name"].eq(PRIMARY_OOF_MODEL))
        & (philips["feature_set"].eq(PRIMARY_FEATURE_SET))
        & (philips["calib_method"].eq(PRIMARY_CALIB))
        & (philips["threshold_strategy"].eq(PRIMARY_THRESHOLD))
    ][["run_label", "fpr_cn_pooled", "fp_cn_pooled", "n_cn_pooled"]].rename(
        columns={
            "fpr_cn_pooled": "philips_cn_fpr_primary",
            "fp_cn_pooled": "philips_cn_fp_primary",
            "n_cn_pooled": "philips_cn_n_primary",
        }
    )
    rows = rows.merge(phil, on="run_label", how="left")
    leak = leakage_summary[leakage_summary["split"].eq("test")][["run_label", "acc_site_latent_mean", "acc_site_raw_mean"]].rename(
        columns={"acc_site_latent_mean": "test_latent_scanner_ba_mean", "acc_site_raw_mean": "test_raw_scanner_ba_mean"}
    )
    rows = rows.merge(leak, on="run_label", how="left")
    rd_cols = [
        "run_label",
        "D_val_best_mean",
        "R_val_bits_best_mean",
        "R_bits_per_latent_dim_best_mean",
        "KLD_over_D_best_mean",
        "beta_KLD_over_D_best_mean",
    ]
    rows = rows.merge(rd_summary[rd_cols], on="run_label", how="left")
    mi_test = mi_summary[mi_summary["split"].eq("test")][
        [
            "run_label",
            "MI_Z_Y_nats_mean",
            "MI_Z_Manufacturer_nats_mean",
            "MI_Manufacturer_over_MI_Y_mean",
            "active_units_mean",
            "total_correlation_nats_mean",
        ]
    ]
    rows = rows.merge(mi_test, on="run_label", how="left")
    ref = rows[rows["run_label"].eq("promoted_ch102_latent384_beta3p75")]
    if not ref.empty:
        ref_row = ref.iloc[0]
        for metric in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "philips_cn_fpr_primary", "test_latent_scanner_ba_mean"]:
            rows[f"delta_vs_promoted_{metric}"] = rows[metric] - safe_float(ref_row.get(metric))
    ch1 = rows[rows["run_label"].eq("ch1only_latent384_beta3p75")]
    if not ch1.empty:
        ch1_row = ch1.iloc[0]
        for metric in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "philips_cn_fpr_primary", "test_latent_scanner_ba_mean"]:
            rows[f"delta_vs_ch1only_{metric}"] = rows[metric] - safe_float(ch1_row.get(metric))
    return rows


def beta_regime_decision(rd_summary: pd.DataFrame) -> pd.DataFrame:
    means = rd_summary.set_index("run_label")["beta_KLD_over_D_best_mean"].to_dict()
    candidate = means.get("candidate_ch12_latent384_beta3p75", np.nan)
    promoted = means.get("promoted_ch102_latent384_beta3p75", np.nan)
    ch1 = means.get("ch1only_latent384_beta3p75", np.nan)
    dist_promoted = abs(candidate - promoted)
    dist_ch1 = abs(candidate - ch1)
    closer = "promoted_[1,0,2]" if dist_promoted < dist_ch1 else "ch1_only"
    return pd.DataFrame(
        [
            {
                "candidate_beta_KLD_over_D_mean": candidate,
                "promoted_beta_KLD_over_D_mean": promoted,
                "ch1only_beta_KLD_over_D_mean": ch1,
                "abs_distance_to_promoted": dist_promoted,
                "abs_distance_to_ch1only": dist_ch1,
                "regularization_regime_closer_to": closer,
            }
        ]
    )


def decision_text(primary: pd.DataFrame, regime: pd.DataFrame) -> str:
    cand = primary[primary["run_label"].eq("candidate_ch12_latent384_beta3p75")]
    prom = primary[primary["run_label"].eq("promoted_ch102_latent384_beta3p75")]
    ch1 = primary[primary["run_label"].eq("ch1only_latent384_beta3p75")]
    if cand.empty or prom.empty:
        return "# Final Decision\n\nDecision: `reject`\n\nPrimary rows were missing, so promotion cannot be evaluated.\n"
    c = cand.iloc[0]
    p = prom.iloc[0]
    ch = ch1.iloc[0] if not ch1.empty else pd.Series(dtype=object)
    gate_auc = safe_float(c["auc"]) > PROMOTED_AUC
    gate_pr = safe_float(c["pr_auc"]) >= PROMOTED_PR_AUC
    gate_ba = safe_float(c["balanced_accuracy"]) >= safe_float(p["balanced_accuracy"]) - 0.005
    gate_f1 = safe_float(c["f1"]) >= safe_float(p["f1"]) - 0.005
    gate_sens = safe_float(c["sensitivity"]) >= safe_float(p["sensitivity"]) - 0.005
    gate_philips = safe_float(c["philips_cn_fpr_primary"]) <= safe_float(p["philips_cn_fpr_primary"])
    gate_leak = safe_float(c["test_latent_scanner_ba_mean"]) <= safe_float(p["test_latent_scanner_ba_mean"])
    passes = all([gate_auc, gate_pr, gate_ba, gate_f1, gate_sens, gate_philips, gate_leak])
    if passes:
        decision = "promote"
    else:
        decision = "channel-pair sensitivity only"
    if not gate_auc or not gate_pr:
        decision = "reject"
    closer = regime.iloc[0]["regularization_regime_closer_to"] if not regime.empty else "undetermined"
    lines = [
        "# Final Decision",
        "",
        f"Decision: `{decision}`",
        "",
        "## Primary OOF-ECDF Comparison",
        "",
        f"- Candidate [1,2] AUC={safe_float(c['auc']):.6f}, PR-AUC={safe_float(c['pr_auc']):.6f}, BA={safe_float(c['balanced_accuracy']):.6f}, Sens={safe_float(c['sensitivity']):.6f}, Spec={safe_float(c['specificity']):.6f}, F1={safe_float(c['f1']):.6f}.",
        f"- Promoted [1,0,2] AUC={safe_float(p['auc']):.6f}, PR-AUC={safe_float(p['pr_auc']):.6f}, BA={safe_float(p['balanced_accuracy']):.6f}, Sens={safe_float(p['sensitivity']):.6f}, Spec={safe_float(p['specificity']):.6f}, F1={safe_float(p['f1']):.6f}.",
    ]
    if not ch.empty:
        lines.append(
            f"- ch1-only AUC={safe_float(ch['auc']):.6f}, PR-AUC={safe_float(ch['pr_auc']):.6f}, BA={safe_float(ch['balanced_accuracy']):.6f}, Sens={safe_float(ch['sensitivity']):.6f}, Spec={safe_float(ch['specificity']):.6f}, F1={safe_float(ch['f1']):.6f}."
        )
    lines.extend(
        [
            "",
            "## Promotion Gate",
            "",
            f"- AUC gate (`> {PROMOTED_AUC}`): `{gate_auc}`.",
            f"- PR-AUC gate (`>= {PROMOTED_PR_AUC}`): `{gate_pr}`.",
            f"- BA/F1/Sensitivity not materially worse: BA `{gate_ba}`, F1 `{gate_f1}`, Sens `{gate_sens}`.",
            f"- Philips CN FPR not worse than promoted: `{gate_philips}`.",
            f"- Test latent scanner leakage not worse than promoted: `{gate_leak}`.",
            "",
            "## Regularization Regime",
            "",
            f"- By mean beta*KLD/D, [1,2] beta3.75 is closer to: `{closer}`.",
            "",
            "## Interpretation",
            "",
            "This run is a reviewer-driven channel-pair sensitivity test. It should only displace the promoted model if it improves both AUC and PR-AUC while preserving clinical operating behavior and scanner/manufacturer safeguards. If those gates fail, the correct interpretation is that removing OMST and using Pearson Full + MI-KNN does not provide a cleaner FULL 5x5 result under the matched beta3.75 regime.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    planned = [
        "README.md",
        "completion_status.csv/.md",
        "completion_summary.csv/.md",
        "config_inventory.csv/.md",
        "stagea_foldwise_metrics.csv/.md",
        "stagea_summary_metrics.csv/.md",
        "stageb_pooled_metrics.csv/.md",
        "stageb_foldwise_metrics.csv/.md",
        "philips_cn_fpr.csv/.md",
        "scanner_leakage_foldwise.csv/.md",
        "scanner_leakage_summary.csv/.md",
        "rate_distortion_foldwise.csv/.md",
        "rate_distortion_summary.csv/.md",
        "latent_mi_signal_nuisance_foldwise.csv/.md",
        "latent_mi_signal_nuisance_summary.csv/.md",
        "primary_promotion_gate_table.csv/.md",
        "beta_regularization_regime.csv/.md",
        "final_decision.md",
        "command_log.json",
    ]
    if args.dry_run:
        print("Dry-run OK. Planned outputs:")
        for item in planned:
            print(f"- {out_dir / item}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    config_df = config_inventory()
    status = pd.concat([completion_status(label, spec) for label, spec in RUNS.items()], ignore_index=True)
    status_summary = completion_summary(status)
    stagea_fold, stagea_summary = stagea_tables()
    stageb_pool = stageb_pooled()
    stageb_fold = stageb_foldwise()
    philips = philips_fpr()
    rd_fold, rd_summary = load_rate_distortion()
    mi_fold, mi_summary = latent_info()
    leak_fold, leak_summary = scanner_leakage()
    primary = primary_rows(stageb_pool, philips, leak_summary, rd_summary, mi_summary)
    regime = beta_regime_decision(rd_summary)

    write_table(config_df, out_dir, "config_inventory")
    write_table(status, out_dir, "completion_status")
    write_table(status_summary, out_dir, "completion_summary")
    write_table(stagea_fold, out_dir, "stagea_foldwise_metrics")
    write_table(stagea_summary, out_dir, "stagea_summary_metrics")
    write_table(stageb_pool, out_dir, "stageb_pooled_metrics")
    write_table(stageb_fold, out_dir, "stageb_foldwise_metrics")
    write_table(philips, out_dir, "philips_cn_fpr")
    write_table(leak_fold, out_dir, "scanner_leakage_foldwise")
    write_table(leak_summary, out_dir, "scanner_leakage_summary")
    write_table(rd_fold, out_dir, "rate_distortion_foldwise")
    write_table(rd_summary, out_dir, "rate_distortion_summary")
    write_table(mi_fold, out_dir, "latent_mi_signal_nuisance_foldwise")
    write_table(mi_summary, out_dir, "latent_mi_signal_nuisance_summary")
    write_table(primary, out_dir, "primary_promotion_gate_table")
    write_table(regime, out_dir, "beta_regularization_regime")

    readme = [
        "# ch12 latent384 beta3p75 Completion and Promotion-Gate Audit",
        "",
        "Read-only audit comparing:",
        "",
        "- Candidate `[1,2]` Pearson Full + MI-KNN latent384 beta3.75.",
        "- Promoted `[1,0,2]` latent384 beta3.75.",
        "- ch1-only latent384 beta3.75.",
        "",
        "No training, OASIS scoring, tensor modification, metadata modification, or model artifact modification was performed.",
        "",
        "Primary convention: `logreg_l2_original / z_plus_age_sex / oof_ecdf / inner_oof_target_sens_ge_0p70_max_spec`.",
    ]
    (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    (out_dir / "final_decision.md").write_text(decision_text(primary, regime), encoding="utf-8")
    write_json(
        out_dir / "command_log.json",
        {
            "created_utc": now_utc(),
            "script": rel(Path(__file__)),
            "output_dir": rel(out_dir),
            "runs": {label: {k: rel(v) if isinstance(v, Path) else v for k, v in spec.items()} for label, spec in RUNS.items()},
            "guardrails": [
                "no training",
                "no OASIS scoring unless existing artifacts are already present",
                "no tensor modification",
                "no metadata modification",
                "no model artifact modification",
            ],
        },
    )

    print(f"Audit written to: {out_dir}")
    print(status_summary.to_string(index=False))
    print(primary.to_string(index=False))
    print(regime.to_string(index=False))


if __name__ == "__main__":
    main()
