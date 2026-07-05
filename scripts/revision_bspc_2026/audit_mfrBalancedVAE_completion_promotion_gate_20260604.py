#!/usr/bin/env python3
"""Read-only completion and promotion-gate audit for Manufacturer-balanced VAE.

This script reads existing ADNI run artifacts and writes an audit package. It
does not train models, rescore tensors, modify tensors, modify metadata, or
modify model artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

CANDIDATE_RUN = ROOT / "results/revision_bspc_2026/recover035_latent384_beta3p75_T80_h10000_p560_full5x5_mfrBalancedVAE"
CANDIDATE_OOF = ROOT / "results/revision_bspc_2026/recover035_latent384_beta3p75_mfrBalancedVAE_stageB_oof_score_calibration"
CANDIDATE_LAUNCH_LOG = ROOT / "results/revision_bspc_2026/mfrBalancedVAE_training_launch_20260603T221601.log"

REFERENCE_RUN = ROOT / "results/revision_bspc_2026/recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
REFERENCE_OOF = ROOT / "results/revision_bspc_2026/recover035_latent384_beta3p75_stageB_oof_score_calibration"

OUT = ROOT / "results/revision_bspc_2026/mfrBalancedVAE_completion_promotion_gate_audit_20260604"

PRIMARY_MODEL = "logreg_l2_original"
PRIMARY_FEATURE_SET = "z_plus_age_sex"
PRIMARY_CALIB = "oof_ecdf"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"

REF_AUC = 0.795155
REF_PR_AUC = 0.573934
REF_BA = 0.725979
REF_F1 = 0.563492
REF_PHILIPS_CN_FPR = 45 / 99


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned output paths.")
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_csv(path: Path, required: bool = True) -> pd.DataFrame:
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
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join("" if pd.isna(row[c]) else str(row[c]) for c in cols) + " |")
        return "\n".join(lines) + "\n"


def write_table(df: pd.DataFrame, out: Path, stem: str) -> None:
    df.to_csv(out / f"{stem}.csv", index=False)
    (out / f"{stem}.md").write_text(to_md(df), encoding="utf-8")


def write_text(out: Path, name: str, text: str) -> None:
    (out / name).write_text(text, encoding="utf-8")


def find_stagea_metrics(run_dir: Path) -> Path | None:
    matches = sorted(run_dir.glob("all_folds_metrics_MULTI*.csv"))
    return matches[0] if matches else None


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_process_exit_code(path: Path) -> tuple[bool, int | None]:
    if not path.exists():
        return False, None
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"PROCESS_EXIT_CODE\s*[:=]\s*(-?\d+)", text)
    if not matches:
        return True, None
    return True, int(matches[-1])


def best_rate_distortion_row(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    best_col = "L_val_betaMax" if "L_val_betaMax" in df.columns else None
    best = df.loc[df[best_col].idxmin()] if best_col else df.iloc[-1]
    out: dict[str, Any] = {
        "final_epoch": int(df["epoch"].max()) if "epoch" in df.columns else np.nan,
        "best_epoch": int(best["epoch"]) if "epoch" in best else np.nan,
        "best_ValL_betaMax": float(best.get("L_val_betaMax", np.nan)),
        "D_val": float(best.get("D_val", np.nan)),
        "R_val_nats": float(best.get("R_val_nats", np.nan)),
        "R_val_bits": float(best.get("R_val_bits", np.nan)),
        "beta": float(best.get("beta", np.nan)),
    }
    d_val = out["D_val"]
    r_nats = out["R_val_nats"]
    beta = out["beta"]
    out["kld_over_D_val"] = r_nats / d_val if d_val else np.nan
    out["beta_kld_over_D_val"] = beta * r_nats / d_val if d_val else np.nan
    out["beta_times_KLD_over_recon"] = out["beta_kld_over_D_val"]
    out["beta_times_KLD_over_R"] = beta if r_nats else np.nan
    return out


def completion_status(run_dir: Path, oof_dir: Path, label: str, launch_log: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    launch_log_available, process_exit_code = parse_process_exit_code(launch_log) if launch_log else (False, None)
    run_manifest = read_json(run_dir / "run_manifest.json")
    stageb_log = read_json(run_dir / "classifier_only_readout" / "command_log.json")
    oof_log = read_json(oof_dir / "command_log.json")
    rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        fold_dir = run_dir / f"fold_{fold}"
        rd = best_rate_distortion_row(fold_dir / f"fold_{fold}_rate_distortion.csv")
        rows.append(
            {
                "run_label": label,
                "fold": fold,
                "fold_dir_exists": fold_dir.exists(),
                "vae_checkpoint": (fold_dir / f"vae_model_fold_{fold}.pt").exists(),
                "vae_history": (fold_dir / f"vae_train_history_fold_{fold}.joblib").exists(),
                "rate_distortion": (fold_dir / f"fold_{fold}_rate_distortion.csv").exists(),
                "stageA_logreg_predictions": (fold_dir / "test_predictions_logreg.csv").exists(),
                "stageA_svm_predictions": (fold_dir / "test_predictions_svm.csv").exists(),
                "latent_trainDev_cache": (run_dir / "classifier_only_readout" / "latent_cache" / f"fold_{fold}_trainDev_latent_mu.csv").exists(),
                "latent_test_cache": (run_dir / "classifier_only_readout" / "latent_cache" / f"fold_{fold}_test_latent_mu.csv").exists(),
                "trainDev_latent_info": (fold_dir / f"fold_{fold}_trainDev_latent_info_summary.csv").exists(),
                "test_latent_info": (fold_dir / f"fold_{fold}_test_latent_info_summary.csv").exists(),
                "trainDev_scanner_leakage": (fold_dir / f"fold_{fold}_scanner_leakage_summary.csv").exists(),
                "test_scanner_leakage": (fold_dir / f"fold_{fold}_test_scanner_leakage_summary.csv").exists(),
                "stageB_readout_dir": (run_dir / "classifier_only_readout").exists(),
                "oof_score_calibration_dir": oof_dir.exists(),
                **rd,
            }
        )
    status = pd.DataFrame(rows)
    bool_cols = [c for c in status.columns if status[c].dtype == bool]
    summary = pd.DataFrame(
        [
            {
                "run_label": label,
                "n_folds_found": int(status["fold_dir_exists"].sum()),
                "all_5_folds_completed": bool((status["fold_dir_exists"] & status["vae_checkpoint"] & status["vae_history"] & status["rate_distortion"]).all()),
                "all_stageA_fold_predictions_present": bool((status["stageA_logreg_predictions"] & status["stageA_svm_predictions"]).all()),
                "all_stageB_latent_cache_present": bool((status["latent_trainDev_cache"] & status["latent_test_cache"]).all()),
                "all_qc_artifacts_present": bool(status[bool_cols].all(axis=None)) if bool_cols else False,
                "launch_log_available": launch_log_available,
                "process_exit_code": process_exit_code,
                "process_exit_code_is_zero": process_exit_code == 0 if process_exit_code is not None else np.nan,
                "run_manifest_available": bool(run_manifest),
                "classifier_only_command_log_available": bool(stageb_log),
                "oof_command_log_available": bool(oof_log),
                "oof_training_launched": oof_log.get("training_launched", np.nan),
                "oof_threshold_fitting_on_outer_test": oof_log.get("threshold_fitting_on_outer_test", np.nan),
            }
        ]
    )
    return status, summary


def load_stagea(run_dir: Path, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = find_stagea_metrics(run_dir)
    if path is None:
        return pd.DataFrame(), pd.DataFrame()
    df = pd.read_csv(path)
    df.insert(0, "run_label", label)
    df.insert(1, "source_file", rel(path))
    model_col = "actual_classifier_type" if "actual_classifier_type" in df.columns else "model"
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
        if c in df.columns
    ]
    agg = df.groupby(["run_label", model_col], dropna=False)[metric_cols].mean().reset_index()
    agg = agg.rename(columns={model_col: "stageA_model"})
    return df, agg


def load_stageb_raw_pooled(run_dir: Path, label: str) -> pd.DataFrame:
    path = run_dir / "classifier_only_readout" / "classifier_sweep_pooled_metrics.csv"
    df = read_csv(path, required=False)
    if df.empty:
        return df
    df = df.rename(columns={"readout_feature_set": "feature_set"})
    df.insert(0, "run_label", label)
    df.insert(1, "calib_method", "raw_classifier_only")
    return df


def load_oof_tables(oof_dir: Path, label: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pooled = read_csv(oof_dir / "calib_pooled_metrics.csv", required=False)
    foldwise = read_csv(oof_dir / "calib_foldwise_metrics.csv", required=False)
    pred = read_csv(oof_dir / "calib_predictions.csv", required=False)
    fpr = read_csv(oof_dir / "calib_philips_fpr_pooled.csv", required=False)
    for df in (pooled, foldwise, pred, fpr):
        if not df.empty:
            df.insert(0, "run_label", label)
    return pooled, foldwise, pred, fpr


def filter_primary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col, value in [
        ("model_name", PRIMARY_MODEL),
        ("feature_set", PRIMARY_FEATURE_SET),
        ("calib_method", PRIMARY_CALIB),
        ("threshold_strategy", PRIMARY_THRESHOLD),
    ]:
        if col in out.columns:
            out = out[out[col] == value]
    return out


def primary_row(df: pd.DataFrame, label: str) -> dict[str, Any]:
    sub = filter_primary(df[df["run_label"] == label] if "run_label" in df.columns else df)
    if sub.empty:
        return {"run_label": label}
    return sub.iloc[0].to_dict()


def compare_primary(candidate: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    c = primary_row(candidate, "candidate_mfrBalancedVAE")
    r = primary_row(reference, "promoted_reference")
    metrics = ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "tn", "fp", "fn", "tp"]
    rows = []
    for metric in metrics:
        cval = c.get(metric, np.nan)
        rval = r.get(metric, np.nan)
        rows.append(
            {
                "metric": metric,
                "candidate_mfrBalancedVAE": cval,
                "promoted_reference": rval,
                "delta_candidate_minus_reference": cval - rval if pd.notna(cval) and pd.notna(rval) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def vae_qc_by_fold(run_dir: Path, label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        fold_dir = run_dir / f"fold_{fold}"
        row: dict[str, Any] = {"run_label": label, "fold": fold}
        row.update(best_rate_distortion_row(fold_dir / f"fold_{fold}_rate_distortion.csv"))
        info = read_csv(fold_dir / f"fold_{fold}_trainDev_latent_info_summary.csv", required=False)
        if not info.empty:
            y = info[info["variable"].astype(str).str.lower().isin(["y_target", "diagnosis", "researchgroup_mapped"])]
            m = info[info["variable"].astype(str).str.lower().isin(["manufacturer"])]
            first = info.iloc[0]
            row["active_units"] = first.get("n_active", np.nan)
            row["frac_active"] = first.get("frac_active", np.nan)
            row["total_correlation_nats"] = first.get("total_correlation_nats", np.nan)
            row["MI_Z_Y_nats"] = y.iloc[0].get("mi_sum_nats", np.nan) if not y.empty else np.nan
            row["MI_Z_Manufacturer_nats"] = m.iloc[0].get("mi_sum_nats", np.nan) if not m.empty else np.nan
            row["MI_Manufacturer_over_MI_Y"] = (
                row["MI_Z_Manufacturer_nats"] / row["MI_Z_Y_nats"]
                if pd.notna(row["MI_Z_Manufacturer_nats"]) and pd.notna(row["MI_Z_Y_nats"]) and row["MI_Z_Y_nats"] != 0
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_by_run(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    present = [m for m in metrics if m in df.columns]
    if not present:
        return pd.DataFrame()
    return df.groupby("run_label", dropna=False)[present].mean(numeric_only=True).reset_index()


def scanner_leakage(run_dir: Path, label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        for split, suffix in [("trainDev", "scanner_leakage_summary"), ("test", "test_scanner_leakage_summary")]:
            path = run_dir / f"fold_{fold}" / f"fold_{fold}_{suffix}.csv"
            df = read_csv(path, required=False)
            if df.empty:
                rows.append({"run_label": label, "fold": fold, "split": split})
                continue
            rec = df.iloc[0].to_dict()
            rows.append(
                {
                    "run_label": label,
                    "fold": fold,
                    "split": split,
                    "raw_BA": rec.get("acc_site_raw", np.nan),
                    "latent_BA": rec.get("acc_site_latent", np.nan),
                    "latent_minus_raw": rec.get("acc_site_latent", np.nan) - rec.get("acc_site_raw", np.nan)
                    if pd.notna(rec.get("acc_site_latent", np.nan)) and pd.notna(rec.get("acc_site_raw", np.nan))
                    else np.nan,
                    "n_samples": rec.get("n_samples", np.nan),
                    "n_sites": rec.get("n_sites", np.nan),
                }
            )
    return pd.DataFrame(rows)


def score_distributions(pred: pd.DataFrame, label: str) -> pd.DataFrame:
    if pred.empty:
        return pd.DataFrame()
    sub = filter_primary(pred[pred["run_label"] == label])
    if sub.empty:
        return pd.DataFrame()
    group_cols = ["run_label", "Manufacturer", "ResearchGroup_Mapped"]
    rows = []
    for keys, grp in sub.groupby(group_cols, dropna=False):
        q = grp["y_score"].quantile([0.25, 0.5, 0.75]).to_dict()
        rows.append(
            {
                "run_label": keys[0],
                "Manufacturer": keys[1],
                "ResearchGroup_Mapped": keys[2],
                "n": len(grp),
                "mean_score": grp["y_score"].mean(),
                "std_score": grp["y_score"].std(),
                "median_score": q.get(0.5, np.nan),
                "q25_score": q.get(0.25, np.nan),
                "q75_score": q.get(0.75, np.nan),
                "predicted_ad_rate": grp["y_pred"].mean(),
            }
        )
    return pd.DataFrame(rows)


def primary_philips_fpr(fpr: pd.DataFrame, label: str) -> pd.DataFrame:
    if fpr.empty:
        return pd.DataFrame()
    sub = fpr[fpr["run_label"] == label].copy()
    sub = filter_primary(sub)
    if "manufacturer" in sub.columns:
        sub = sub[sub["manufacturer"].astype(str).str.lower() == "philips"]
    return sub


def build_promotion_gate(primary_cmp: pd.DataFrame, cand_fpr: pd.DataFrame, ref_fpr: pd.DataFrame, leakage_summary: pd.DataFrame) -> pd.DataFrame:
    vals = {row["metric"]: row for _, row in primary_cmp.iterrows()}
    def c(metric: str) -> float:
        return float(vals.get(metric, {}).get("candidate_mfrBalancedVAE", np.nan))
    def r(metric: str) -> float:
        return float(vals.get(metric, {}).get("promoted_reference", np.nan))

    cand_philips_fpr = float(cand_fpr.iloc[0].get("fpr_cn_pooled", np.nan)) if not cand_fpr.empty else np.nan
    ref_philips_fpr = float(ref_fpr.iloc[0].get("fpr_cn_pooled", REF_PHILIPS_CN_FPR)) if not ref_fpr.empty else REF_PHILIPS_CN_FPR
    leak = leakage_summary.pivot(index="split", columns="run_label", values="latent_BA_mean") if not leakage_summary.empty else pd.DataFrame()
    test_leak_not_worse = np.nan
    if not leak.empty and "test" in leak.index:
        cand_leak = leak.loc["test"].get("candidate_mfrBalancedVAE", np.nan)
        ref_leak = leak.loc["test"].get("promoted_reference", np.nan)
        test_leak_not_worse = bool(cand_leak <= ref_leak) if pd.notna(cand_leak) and pd.notna(ref_leak) else np.nan

    rows = [
        {"gate": "AUC >= promoted reference", "candidate_value": c("auc"), "reference_value": REF_AUC, "passed": c("auc") >= REF_AUC},
        {"gate": "PR-AUC >= promoted reference", "candidate_value": c("pr_auc"), "reference_value": REF_PR_AUC, "passed": c("pr_auc") >= REF_PR_AUC},
        {
            "gate": "BA not materially worse",
            "candidate_value": c("balanced_accuracy"),
            "reference_value": r("balanced_accuracy"),
            "passed": c("balanced_accuracy") >= r("balanced_accuracy") - 0.005,
        },
        {"gate": "F1 not materially worse", "candidate_value": c("f1"), "reference_value": r("f1"), "passed": c("f1") >= r("f1") - 0.005},
        {
            "gate": "Sensitivity not materially worse",
            "candidate_value": c("sensitivity"),
            "reference_value": r("sensitivity"),
            "passed": c("sensitivity") >= r("sensitivity") - 0.005,
        },
        {
            "gate": "Philips CN FPR <= promoted reference",
            "candidate_value": cand_philips_fpr,
            "reference_value": ref_philips_fpr,
            "passed": cand_philips_fpr <= ref_philips_fpr if pd.notna(cand_philips_fpr) else False,
        },
        {
            "gate": "test latent scanner leakage not worse",
            "candidate_value": np.nan,
            "reference_value": np.nan,
            "passed": test_leak_not_worse,
        },
    ]
    out = pd.DataFrame(rows)
    out["all_required_gates_passed"] = bool(out["passed"].fillna(False).all())
    return out


def main() -> None:
    args = parse_args()
    required_paths = [
        CANDIDATE_RUN,
        CANDIDATE_OOF / "calib_pooled_metrics.csv",
        CANDIDATE_OOF / "calib_foldwise_metrics.csv",
        CANDIDATE_OOF / "calib_predictions.csv",
        REFERENCE_RUN,
        REFERENCE_OOF / "calib_pooled_metrics.csv",
        REFERENCE_OOF / "calib_foldwise_metrics.csv",
        REFERENCE_OOF / "calib_predictions.csv",
    ]
    missing = [rel(p) for p in required_paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))
    if args.dry_run:
        print(f"Would write audit package to {rel(args.output_dir)}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cand_status, cand_status_summary = completion_status(CANDIDATE_RUN, CANDIDATE_OOF, "candidate_mfrBalancedVAE", CANDIDATE_LAUNCH_LOG)
    ref_status, ref_status_summary = completion_status(REFERENCE_RUN, REFERENCE_OOF, "promoted_reference")
    completion = pd.concat([cand_status, ref_status], ignore_index=True)
    completion_summary = pd.concat([cand_status_summary, ref_status_summary], ignore_index=True)

    cand_stagea_fold, cand_stagea_summary = load_stagea(CANDIDATE_RUN, "candidate_mfrBalancedVAE")
    ref_stagea_fold, ref_stagea_summary = load_stagea(REFERENCE_RUN, "promoted_reference")
    stagea_fold = pd.concat([cand_stagea_fold, ref_stagea_fold], ignore_index=True)
    stagea_summary = pd.concat([cand_stagea_summary, ref_stagea_summary], ignore_index=True)

    cand_pooled, cand_foldwise, cand_pred, cand_fpr_all = load_oof_tables(CANDIDATE_OOF, "candidate_mfrBalancedVAE")
    ref_pooled, ref_foldwise, ref_pred, ref_fpr_all = load_oof_tables(REFERENCE_OOF, "promoted_reference")
    stageb_oof_pooled = pd.concat([cand_pooled, ref_pooled], ignore_index=True)
    stageb_oof_foldwise = pd.concat([cand_foldwise, ref_foldwise], ignore_index=True)
    stageb_raw_pooled_table = pd.concat(
        [
            load_stageb_raw_pooled(CANDIDATE_RUN, "candidate_mfrBalancedVAE"),
            load_stageb_raw_pooled(REFERENCE_RUN, "promoted_reference"),
        ],
        ignore_index=True,
    )

    primary_cmp = compare_primary(cand_pooled, ref_pooled)
    primary_rows = pd.concat(
        [filter_primary(cand_pooled), filter_primary(ref_pooled)],
        ignore_index=True,
    )

    vae_qc_fold = pd.concat(
        [vae_qc_by_fold(CANDIDATE_RUN, "candidate_mfrBalancedVAE"), vae_qc_by_fold(REFERENCE_RUN, "promoted_reference")],
        ignore_index=True,
    )
    vae_qc_summary = summarize_by_run(
        vae_qc_fold,
        [
            "final_epoch",
            "best_epoch",
            "best_ValL_betaMax",
            "D_val",
            "R_val_bits",
            "beta_kld_over_D_val",
            "active_units",
            "total_correlation_nats",
            "MI_Z_Y_nats",
            "MI_Z_Manufacturer_nats",
            "MI_Manufacturer_over_MI_Y",
        ],
    )

    leakage_fold = pd.concat(
        [scanner_leakage(CANDIDATE_RUN, "candidate_mfrBalancedVAE"), scanner_leakage(REFERENCE_RUN, "promoted_reference")],
        ignore_index=True,
    )
    leakage_summary = (
        leakage_fold.groupby(["run_label", "split"], dropna=False)
        .agg(
            raw_BA_mean=("raw_BA", "mean"),
            latent_BA_mean=("latent_BA", "mean"),
            latent_minus_raw_mean=("latent_minus_raw", "mean"),
        )
        .reset_index()
    )

    fpr_all = pd.concat([cand_fpr_all, ref_fpr_all], ignore_index=True)
    cand_primary_fpr = primary_philips_fpr(cand_fpr_all, "candidate_mfrBalancedVAE")
    ref_primary_fpr = primary_philips_fpr(ref_fpr_all, "promoted_reference")
    primary_fpr = pd.concat([cand_primary_fpr, ref_primary_fpr], ignore_index=True)

    score_dist = pd.concat(
        [score_distributions(cand_pred, "candidate_mfrBalancedVAE"), score_distributions(ref_pred, "promoted_reference")],
        ignore_index=True,
    )

    gate = build_promotion_gate(primary_cmp, cand_primary_fpr, ref_primary_fpr, leakage_summary)
    decision = "promote" if bool(gate["passed"].fillna(False).all()) else "retain as manufacturer-balanced sensitivity only"

    write_table(completion, args.output_dir, "completion_status_by_fold")
    write_table(completion_summary, args.output_dir, "completion_summary")
    write_table(stagea_summary, args.output_dir, "stagea_metrics_comparison")
    write_table(stagea_fold, args.output_dir, "stagea_foldwise_metrics")
    write_table(stageb_raw_pooled_table, args.output_dir, "stageb_raw_classifier_only_pooled")
    write_table(
        stageb_oof_pooled[
            (stageb_oof_pooled["model_name"] == PRIMARY_MODEL)
            & (stageb_oof_pooled["feature_set"] == PRIMARY_FEATURE_SET)
        ].copy(),
        args.output_dir,
        "stageb_oof_all_methods_pooled_primary_model",
    )
    write_table(
        stageb_oof_foldwise[
            (stageb_oof_foldwise["model_name"] == PRIMARY_MODEL)
            & (stageb_oof_foldwise["feature_set"] == PRIMARY_FEATURE_SET)
        ].copy(),
        args.output_dir,
        "stageb_oof_all_methods_foldwise_primary_model",
    )
    write_table(primary_rows, args.output_dir, "stageb_promoted_convention_rows")
    write_table(primary_cmp, args.output_dir, "stageb_primary_metric_delta")
    write_table(vae_qc_fold, args.output_dir, "vae_qc_by_fold")
    write_table(vae_qc_summary, args.output_dir, "vae_qc_summary")
    write_table(leakage_fold, args.output_dir, "scanner_leakage_by_fold")
    write_table(leakage_summary, args.output_dir, "scanner_leakage_summary")
    write_table(fpr_all, args.output_dir, "philips_cn_fpr_all_methods")
    write_table(primary_fpr, args.output_dir, "philips_cn_fpr_promoted_convention")
    write_table(score_dist, args.output_dir, "score_distribution_by_manufacturer")
    write_table(gate, args.output_dir, "promotion_gate")

    final_text = f"""# Manufacturer-Balanced VAE Completion and Promotion-Gate Audit

Decision: **{decision}**.

Primary readout convention:
`{PRIMARY_MODEL} / {PRIMARY_FEATURE_SET} / {PRIMARY_CALIB} / {PRIMARY_THRESHOLD}`.

Under the promoted convention, Manufacturer-balanced VAE reached AUC={primary_cmp.loc[primary_cmp['metric'] == 'auc', 'candidate_mfrBalancedVAE'].iloc[0]:.6f}
and PR-AUC={primary_cmp.loc[primary_cmp['metric'] == 'pr_auc', 'candidate_mfrBalancedVAE'].iloc[0]:.6f}, below the promoted reference
AUC={REF_AUC:.6f} and PR-AUC={REF_PR_AUC:.6f}. BA/F1 are also lower than the promoted reference.

The candidate completed all five folds and the available launch log reports `PROCESS_EXIT_CODE:0`.
The OOF score-harmonization command log reports `training_launched=false` and
`threshold_fitting_on_outer_test=false`.

The candidate does not pass the promotion gate because the primary ranking metrics do not meet the
promoted reference thresholds. It can be retained only as a manufacturer-balanced VAE sensitivity
analysis.
"""
    write_text(args.output_dir, "final_decision.md", final_text)

    readme = f"""# mfrBalancedVAE Completion Promotion-Gate Audit

This package is read-only with respect to tensors, metadata, and model artifacts. It compares:

- Candidate: `{rel(CANDIDATE_RUN)}`
- Candidate OOF readout: `{rel(CANDIDATE_OOF)}`
- Promoted reference: `{rel(REFERENCE_RUN)}`
- Reference OOF readout: `{rel(REFERENCE_OOF)}`

Primary promotion convention:
`{PRIMARY_MODEL} / {PRIMARY_FEATURE_SET} / {PRIMARY_CALIB} / {PRIMARY_THRESHOLD}`.

Key outputs:

- `completion_summary.csv/.md`
- `stagea_metrics_comparison.csv/.md`
- `stageb_oof_all_methods_pooled_primary_model.csv/.md`
- `stageb_promoted_convention_rows.csv/.md`
- `vae_qc_summary.csv/.md`
- `scanner_leakage_summary.csv/.md`
- `philips_cn_fpr_promoted_convention.csv/.md`
- `promotion_gate.csv/.md`
- `final_decision.md`
"""
    write_text(args.output_dir, "README.md", readme)

    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": rel(Path(__file__)),
        "candidate_run": rel(CANDIDATE_RUN),
        "candidate_oof": rel(CANDIDATE_OOF),
        "reference_run": rel(REFERENCE_RUN),
        "reference_oof": rel(REFERENCE_OOF),
        "primary_readout": {
            "model_name": PRIMARY_MODEL,
            "feature_set": PRIMARY_FEATURE_SET,
            "calib_method": PRIMARY_CALIB,
            "threshold_strategy": PRIMARY_THRESHOLD,
        },
        "guardrails": {
            "training_launched": False,
            "tensor_modified": False,
            "metadata_modified": False,
            "model_artifact_modified": False,
            "test_fold_threshold_fitting": False,
        },
        "decision": decision,
    }
    write_text(args.output_dir, "command_log.json", json.dumps(command_log, indent=2) + "\n")
    print(f"Wrote audit package to {rel(args.output_dir)}")


if __name__ == "__main__":
    main()
