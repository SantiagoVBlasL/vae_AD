#!/usr/bin/env python3
"""Read-only promotion-gate audit for latent384 beta3.5 vs promoted latent384 beta3.75.

This script only reads existing model outputs and writes an audit package. It does
not train models, score new tensors, modify tensors, or modify model artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

CANDIDATE_RUN = ROOT / "results/revision_bspc_2026/recover035_latent384_beta3p5_T80_h10000_p560_full5x5"
CANDIDATE_OOF = ROOT / "results/revision_bspc_2026/recover035_latent384_beta3p5_stageB_oof_score_calibration"
REFERENCE_RUN = ROOT / "results/revision_bspc_2026/recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
REFERENCE_OOF = ROOT / "results/revision_bspc_2026/recover035_latent384_beta3p75_stageB_oof_score_calibration"
METADATA_PATH = ROOT / "results/revision_bspc_2026/adni_035_metadata_rescue_preflight/patched_metadata_candidate.csv"
OASIS_MEGA = ROOT / "results/revision_bspc_2026/oasis_mega_90cn_90ad_pooled_external_validation_20260531/pooled_ranking_metrics.csv"

OUT = ROOT / "results/revision_bspc_2026/latent384_beta3p5_completion_promotion_gate_audit_20260603"

PRIMARY_MODEL = "logreg_l2_original"
PRIMARY_FEATURES = "z_plus_age_sex"
PRIMARY_CALIB = "oof_ecdf"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"

PROMOTED_AUC = 0.795155
PROMOTED_PR_AUC = 0.573934
PHILIPS_FPR_GATE = 0.444


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned outputs only.")
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


def ensure_out(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._\n"
    try:
        return df.to_markdown(index=False) + "\n"
    except Exception:
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in df.iterrows():
            vals = ["" if pd.isna(row[c]) else str(row[c]) for c in cols]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines) + "\n"


def write_table(df: pd.DataFrame, stem: str, out: Path) -> None:
    df.to_csv(out / f"{stem}.csv", index=False)
    (out / f"{stem}.md").write_text(to_md(df), encoding="utf-8")


def write_text(stem: str, text: str, out: Path) -> None:
    (out / stem).write_text(text, encoding="utf-8")


def as_float(value: Any) -> float:
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


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            "_".join(str(part) for part in col if str(part) and str(part) != "nan").strip("_")
            for col in out.columns
        ]
    return out


def list_existing(paths: Iterable[Path]) -> dict[str, bool]:
    return {rel(path): path.exists() for path in paths}


def find_stagea_metrics(run_dir: Path) -> Path | None:
    matches = sorted(run_dir.glob("all_folds_metrics_MULTI*.csv"))
    return matches[0] if matches else None


def completion_status(run_dir: Path, oof_dir: Path, label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    readout_dir = run_dir / "classifier_only_readout"
    for fold in range(1, 6):
        fold_dir = run_dir / f"fold_{fold}"
        rd = fold_dir / f"fold_{fold}_rate_distortion.csv"
        final_epoch = np.nan
        best_epoch = np.nan
        best_val_l = np.nan
        if rd.exists():
            rdf = pd.read_csv(rd)
            if not rdf.empty:
                final_epoch = int(rdf["epoch"].max())
                if "L_val_betaMax" in rdf.columns:
                    best_idx = rdf["L_val_betaMax"].idxmin()
                    best_epoch = int(rdf.loc[best_idx, "epoch"])
                    best_val_l = float(rdf.loc[best_idx, "L_val_betaMax"])
        rows.append(
            {
                "run_label": label,
                "fold": fold,
                "vae_checkpoint": (fold_dir / f"vae_model_fold_{fold}.pt").exists(),
                "vae_history": (fold_dir / f"vae_train_history_fold_{fold}.joblib").exists(),
                "rate_distortion": rd.exists(),
                "trainDev_latent_info": (fold_dir / f"fold_{fold}_trainDev_latent_info_summary.csv").exists(),
                "test_latent_info": (fold_dir / f"fold_{fold}_test_latent_info_summary.csv").exists(),
                "trainDev_scanner_leakage": (fold_dir / f"fold_{fold}_scanner_leakage_summary.csv").exists(),
                "test_scanner_leakage": (fold_dir / f"fold_{fold}_test_scanner_leakage_summary.csv").exists(),
                "stageA_logreg_predictions": (fold_dir / "test_predictions_logreg.csv").exists(),
                "stageA_svm_predictions": (fold_dir / "test_predictions_svm.csv").exists(),
                "latent_cache_trainDev": (readout_dir / "latent_cache" / f"fold_{fold}_trainDev_latent_mu.csv").exists(),
                "latent_cache_test": (readout_dir / "latent_cache" / f"fold_{fold}_test_latent_mu.csv").exists(),
                "stageB_readout_outputs": readout_dir.exists(),
                "oof_score_harmonization_outputs": oof_dir.exists(),
                "final_epoch_from_rd": final_epoch,
                "best_epoch_from_rd": best_epoch,
                "best_ValL_betaMax_from_rd": best_val_l,
            }
        )
    return pd.DataFrame(rows)


def summarize_completion(status: pd.DataFrame) -> pd.DataFrame:
    bool_cols = [c for c in status.columns if status[c].dtype == bool]
    rows = []
    for label, grp in status.groupby("run_label"):
        all_fold_artifacts = bool(grp[bool_cols].all(axis=None))
        rows.append(
            {
                "run_label": label,
                "n_folds": int(grp["fold"].nunique()),
                "all_5_folds_present": int(grp["fold"].nunique()) == 5,
                "all_required_fold_artifacts_present": all_fold_artifacts,
                "stageB_readout_present_all_folds": bool(grp["stageB_readout_outputs"].all()),
                "oof_score_harmonization_present_all_folds": bool(grp["oof_score_harmonization_outputs"].all()),
            }
        )
    return pd.DataFrame(rows)


def load_stagea(run_dir: Path, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = find_stagea_metrics(run_dir)
    if path is None:
        return pd.DataFrame(), pd.DataFrame()
    df = pd.read_csv(path)
    df.insert(0, "run_label", label)
    df.insert(1, "source_file", rel(path))
    metric_cols = [
        c
        for c in [
            "auc_raw",
            "pr_auc_raw",
            "auc_final",
            "pr_auc_final",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "f1_score",
        ]
        if c in df.columns
    ]
    group_col = "classifier_model" if "classifier_model" in df.columns else "model"
    if group_col not in df.columns:
        group_col = "actual_classifier_type"
    if group_col not in df.columns:
        group_col = "classifier"
    if group_col in df.columns:
        agg = df.groupby(["run_label", group_col], dropna=False)[metric_cols].mean().reset_index()
        agg = agg.rename(columns={group_col: "stageA_model"})
    else:
        agg = pd.DataFrame()
    return df, agg


def normalize_raw_pooled(path: Path, label: str) -> pd.DataFrame:
    df = read_csv(path, required=False)
    if df.empty:
        return df
    out = df.copy()
    out.insert(0, "run_label", label)
    out.insert(1, "readout_source", "raw_classifier_only")
    out["calib_method"] = "raw"
    out = out.rename(columns={"readout_feature_set": "feature_set"})
    return out


def normalize_oof_pooled(path: Path, label: str) -> pd.DataFrame:
    df = read_csv(path, required=False)
    if df.empty:
        return df
    out = df.copy()
    out.insert(0, "run_label", label)
    out.insert(1, "readout_source", "stageB_oof_score_calibration")
    return out


def pooled_metrics(run_dir: Path, oof_dir: Path, label: str) -> pd.DataFrame:
    raw = normalize_raw_pooled(run_dir / "classifier_only_readout/classifier_sweep_pooled_metrics.csv", label)
    oof = normalize_oof_pooled(oof_dir / "calib_pooled_metrics.csv", label)
    keep_cols = [
        "run_label",
        "readout_source",
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
    frames = []
    for df in [raw, oof]:
        if df.empty:
            continue
        for col in keep_cols:
            if col not in df.columns:
                df[col] = np.nan
        frames.append(df[keep_cols])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=keep_cols)


def primary_pooled_row(pooled: pd.DataFrame, label: str) -> pd.Series | None:
    mask = (
        (pooled["run_label"] == label)
        & (pooled["model_name"] == PRIMARY_MODEL)
        & (pooled["feature_set"] == PRIMARY_FEATURES)
        & (pooled["calib_method"] == PRIMARY_CALIB)
        & (pooled["threshold_strategy"] == PRIMARY_THRESHOLD)
    )
    rows = pooled.loc[mask]
    if rows.empty:
        return None
    return rows.iloc[0]


def foldwise_oof(oof_dir: Path, label: str) -> pd.DataFrame:
    df = read_csv(oof_dir / "calib_foldwise_metrics.csv", required=False)
    if df.empty:
        return df
    out = df.copy()
    out.insert(0, "run_label", label)
    return out


def selected_foldwise(foldwise: pd.DataFrame, calib: str = PRIMARY_CALIB) -> pd.DataFrame:
    if foldwise.empty:
        return foldwise
    mask = (
        (foldwise["model_name"] == PRIMARY_MODEL)
        & (foldwise["feature_set"] == PRIMARY_FEATURES)
        & (foldwise["calib_method"] == calib)
        & (foldwise["threshold_strategy"] == PRIMARY_THRESHOLD)
    )
    keep = [
        "run_label",
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
        "best_inner_auc",
        "best_params",
    ]
    rows = foldwise.loc[mask].copy()
    for col in keep:
        if col not in rows.columns:
            rows[col] = np.nan
    return rows[keep]


def foldwise_delta(candidate: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    left = candidate.add_prefix("candidate_")
    right = reference.add_prefix("reference_")
    merged = left.merge(right, left_on="candidate_fold", right_on="reference_fold", how="outer")
    for metric in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]:
        c = f"candidate_{metric}"
        r = f"reference_{metric}"
        if c in merged.columns and r in merged.columns:
            merged[f"delta_{metric}"] = merged[c] - merged[r]
    return merged


def beta_for(label: str) -> float:
    if "beta3p5" in label:
        return 3.5
    return 3.75


def load_vae_qc(run_dir: Path, label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    beta = beta_for(label)
    for fold in range(1, 6):
        fold_dir = run_dir / f"fold_{fold}"
        row: dict[str, Any] = {"run_label": label, "fold": fold, "beta_for_ratio": beta}
        rd_path = fold_dir / f"fold_{fold}_rate_distortion.csv"
        if rd_path.exists():
            rd = pd.read_csv(rd_path)
            if not rd.empty:
                best_idx = rd["L_val_betaMax"].idxmin() if "L_val_betaMax" in rd.columns else rd.index[-1]
                best = rd.loc[best_idx]
                last = rd.iloc[-1]
                row.update(
                    {
                        "rd_epoch_best_ValL": int(best.get("epoch", np.nan)),
                        "rd_final_epoch": int(last.get("epoch", np.nan)),
                        "D_val_best": as_float(best.get("D_val")),
                        "R_val_nats_best": as_float(best.get("R_val_nats")),
                        "R_val_bits_best": as_float(best.get("R_val_bits")),
                        "KLD_over_D_best": safe_div(as_float(best.get("R_val_nats")), as_float(best.get("D_val"))),
                        "beta_KLD_over_D_best": beta * safe_div(as_float(best.get("R_val_nats")), as_float(best.get("D_val"))),
                        "L_val_betaMax_best": as_float(best.get("L_val_betaMax")),
                    }
                )
        info_path = fold_dir / f"fold_{fold}_trainDev_latent_info_summary.csv"
        if info_path.exists():
            info = pd.read_csv(info_path)
            y = info.loc[info.get("variable", pd.Series(dtype=object)).astype(str).eq("Y_target")]
            mfr = info.loc[info.get("variable", pd.Series(dtype=object)).astype(str).eq("Manufacturer")]
            if not y.empty:
                yrow = y.iloc[0]
                row.update(
                    {
                        "MI_Z_Y_nats": as_float(yrow.get("mi_sum_nats")),
                        "active_units": as_float(yrow.get("n_active")),
                        "frac_active": as_float(yrow.get("frac_active")),
                        "total_correlation_nats": as_float(yrow.get("total_correlation_nats")),
                    }
                )
            if not mfr.empty:
                mrow = mfr.iloc[0]
                row.update({"MI_Z_Manufacturer_nats": as_float(mrow.get("mi_sum_nats"))})
        row["MI_Manufacturer_over_MI_Y"] = safe_div(row.get("MI_Z_Manufacturer_nats", np.nan), row.get("MI_Z_Y_nats", np.nan))
        rows.append(row)
    return pd.DataFrame(rows)


def load_scanner_leakage(run_dir: Path, label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        fold_dir = run_dir / f"fold_{fold}"
        for split_name, filename in [
            ("trainDev", f"fold_{fold}_scanner_leakage_summary.csv"),
            ("test", f"fold_{fold}_test_scanner_leakage_summary.csv"),
        ]:
            path = fold_dir / filename
            if not path.exists():
                rows.append({"run_label": label, "fold": fold, "split": split_name, "available": False})
                continue
            df = pd.read_csv(path)
            if df.empty:
                rows.append({"run_label": label, "fold": fold, "split": split_name, "available": False})
                continue
            rec = df.iloc[0].to_dict()
            rec.update({"run_label": label, "fold": fold, "split": split_name, "available": True})
            rec["latent_minus_raw"] = as_float(rec.get("acc_site_latent")) - as_float(rec.get("acc_site_raw"))
            rows.append(rec)
    return pd.DataFrame(rows)


def scanner_leakage_summary(leakage: pd.DataFrame) -> pd.DataFrame:
    if leakage.empty:
        return leakage
    metric_cols = ["acc_site_raw", "acc_site_latent", "latent_minus_raw"]
    for c in metric_cols:
        if c not in leakage.columns:
            leakage[c] = np.nan
    return (
        leakage.groupby(["run_label", "split"], dropna=False)[metric_cols]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )


def load_philips_pooled(oof_dir: Path, label: str) -> pd.DataFrame:
    df = read_csv(oof_dir / "calib_philips_fpr_pooled.csv", required=False)
    if df.empty:
        return df
    rows = df[
        (df["model_name"] == PRIMARY_MODEL)
        & (df["feature_set"] == PRIMARY_FEATURES)
        & (df["calib_method"] == PRIMARY_CALIB)
        & (df["threshold_strategy"] == PRIMARY_THRESHOLD)
    ].copy()
    rows.insert(0, "run_label", label)
    return rows


def primary_predictions(oof_dir: Path, label: str) -> pd.DataFrame:
    df = read_csv(oof_dir / "calib_predictions.csv", required=False)
    if df.empty:
        return df
    rows = df[
        (df["model_name"] == PRIMARY_MODEL)
        & (df["feature_set"] == PRIMARY_FEATURES)
        & (df["calib_method"] == PRIMARY_CALIB)
        & (df["threshold_strategy"] == PRIMARY_THRESHOLD)
    ].copy()
    rows.insert(0, "run_label", label)
    return rows


def philips_by_age_phase_site(pred: pd.DataFrame, label: str) -> pd.DataFrame:
    if pred.empty:
        return pd.DataFrame()
    meta = read_csv(METADATA_PATH, required=False)
    df = pred.copy()
    if not meta.empty:
        drop_cols = [c for c in ["Age", "Sex", "Manufacturer", "ResearchGroup_Mapped"] if c in meta.columns]
        meta_cols = ["SubjectID"] + [c for c in ["Site3", "Phase", "Visit", "source_batch", "n_timepoints_raw"] if c in meta.columns]
        meta = meta[meta_cols].drop_duplicates("SubjectID")
        df = df.merge(meta, on="SubjectID", how="left")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["age_bin"] = pd.cut(
        df["Age"],
        bins=[0, 65, 70, 75, 80, 85, 200],
        labels=["<65", "65-69", "70-74", "75-79", "80-84", "85+"],
        right=False,
    ).astype(str)
    if "Site3" not in df.columns:
        df["Site3"] = "unavailable"
    if "Phase" not in df.columns:
        df["Phase"] = "unavailable"
    if "Visit" not in df.columns:
        df["Visit"] = "unavailable"
    cn_philips = df[(df["y_true"] == 0) & (df["Manufacturer"].astype(str).str.lower() == "philips")].copy()
    if cn_philips.empty:
        return pd.DataFrame(columns=["run_label", "grouping", "group_value", "n_philips_cn", "fp", "fpr"])
    rows: list[dict[str, Any]] = []
    for grouping in ["age_bin", "Site3", "Phase", "Visit"]:
        grouped = cn_philips.groupby(grouping, dropna=False)
        for val, grp in grouped:
            n = len(grp)
            fp = int((grp["y_pred"] == 1).sum())
            rows.append(
                {
                    "run_label": label,
                    "grouping": grouping,
                    "group_value": str(val),
                    "n_philips_cn": n,
                    "fp": fp,
                    "fpr": safe_div(fp, n),
                }
            )
    return pd.DataFrame(rows)


def oasis_available_comparison() -> pd.DataFrame:
    if not OASIS_MEGA.exists():
        return pd.DataFrame(
            [
                {
                    "run_label": "latent384_beta3p5",
                    "status": "not_available",
                    "note": "Mega-OASIS scoring table not found.",
                }
            ]
        )
    df = pd.read_csv(OASIS_MEGA)
    out = df.copy()
    out.insert(0, "status", "available_reference_only")
    out["candidate_latent384_beta3p5_available"] = out["adni_model"].astype(str).str.contains(
        "latent384.*beta3p5|beta3p5", case=False, regex=True
    )
    if not out["candidate_latent384_beta3p5_available"].any():
        note = {
            "status": "not_available_for_candidate",
            "build_candidate": "all",
            "adni_model": "recover035_latent384_beta3p5",
            "prediction_level": "not_scored",
            "analysis_role": "not_available",
            "n": np.nan,
            "n_cn": np.nan,
            "n_ad": np.nan,
            "auc": np.nan,
            "auc_bootstrap95_lo": np.nan,
            "auc_bootstrap95_hi": np.nan,
            "auc_permutation_p_ge_observed": np.nan,
            "pr_auc": np.nan,
            "pr_auc_bootstrap95_lo": np.nan,
            "pr_auc_bootstrap95_hi": np.nan,
            "n_bootstrap": np.nan,
            "n_permutations": np.nan,
            "candidate_latent384_beta3p5_available": False,
        }
        out = pd.concat([out, pd.DataFrame([note])], ignore_index=True)
    return out


def promotion_gate(
    pooled: pd.DataFrame,
    philips: pd.DataFrame,
    leakage_summary_df: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    cand = primary_pooled_row(pooled, "candidate_latent384_beta3p5")
    ref = primary_pooled_row(pooled, "promoted_reference_latent384_beta3p75")
    if cand is None or ref is None:
        rows = [{"gate": "primary_row_available", "passes": False, "details": "Missing candidate or reference primary row."}]
        return pd.DataFrame(rows), "reject"

    def get_metric(row: pd.Series, key: str) -> float:
        return float(row[key]) if key in row and pd.notna(row[key]) else float("nan")

    cand_phil = philips[
        (philips["run_label"] == "candidate_latent384_beta3p5")
        & (philips["manufacturer"].astype(str).str.lower() == "philips")
    ]
    ref_phil = philips[
        (philips["run_label"] == "promoted_reference_latent384_beta3p75")
        & (philips["manufacturer"].astype(str).str.lower() == "philips")
    ]
    cand_phil_fpr = float(cand_phil.iloc[0]["fpr_cn_pooled"]) if not cand_phil.empty else float("nan")
    ref_phil_fpr = float(ref_phil.iloc[0]["fpr_cn_pooled"]) if not ref_phil.empty else float("nan")

    leak = leakage_summary_df.copy()
    cand_test_latent = np.nan
    ref_test_latent = np.nan
    if not leak.empty and isinstance(leak.columns, pd.MultiIndex):
        leak.columns = ["_".join([str(x) for x in c if str(x) != ""]) for c in leak.columns]
    if not leak.empty:
        for run_label, holder in [
            ("candidate_latent384_beta3p5", "cand"),
            ("promoted_reference_latent384_beta3p75", "ref"),
        ]:
            row = leak[(leak.get("run_label", pd.Series(dtype=str)) == run_label) & (leak.get("split", pd.Series(dtype=str)) == "test")]
            val = np.nan
            for col in ["acc_site_latent_mean", "acc_site_latent"]:
                if col in row.columns and not row.empty:
                    val = as_float(row.iloc[0][col])
                    break
            if holder == "cand":
                cand_test_latent = val
            else:
                ref_test_latent = val

    rows = []
    rows.append(
        {
            "gate": "AUC",
            "candidate": get_metric(cand, "auc"),
            "reference_or_limit": PROMOTED_AUC,
            "delta_vs_reference": get_metric(cand, "auc") - get_metric(ref, "auc"),
            "passes": get_metric(cand, "auc") > PROMOTED_AUC,
            "details": "Requires candidate AUC > 0.795155.",
        }
    )
    rows.append(
        {
            "gate": "PR-AUC",
            "candidate": get_metric(cand, "pr_auc"),
            "reference_or_limit": PROMOTED_PR_AUC,
            "delta_vs_reference": get_metric(cand, "pr_auc") - get_metric(ref, "pr_auc"),
            "passes": get_metric(cand, "pr_auc") >= PROMOTED_PR_AUC,
            "details": "Requires candidate PR-AUC >= 0.573934.",
        }
    )
    for metric, tolerance in [("balanced_accuracy", 0.01), ("f1", 0.01), ("sensitivity", 0.01)]:
        delta = get_metric(cand, metric) - get_metric(ref, metric)
        rows.append(
            {
                "gate": metric,
                "candidate": get_metric(cand, metric),
                "reference_or_limit": get_metric(ref, metric),
                "delta_vs_reference": delta,
                "passes": delta >= -tolerance,
                "details": f"Material-worsening screen uses tolerance {-tolerance:+.3f}.",
            }
        )
    rows.append(
        {
            "gate": "Philips_CN_FPR",
            "candidate": cand_phil_fpr,
            "reference_or_limit": PHILIPS_FPR_GATE,
            "delta_vs_reference": cand_phil_fpr - ref_phil_fpr,
            "passes": bool(cand_phil_fpr <= PHILIPS_FPR_GATE) if not math.isnan(cand_phil_fpr) else False,
            "details": "Requires Philips CN FPR <= 44.4%.",
        }
    )
    rows.append(
        {
            "gate": "test_scanner_leakage_latent",
            "candidate": cand_test_latent,
            "reference_or_limit": ref_test_latent,
            "delta_vs_reference": cand_test_latent - ref_test_latent,
            "passes": bool(cand_test_latent <= ref_test_latent) if not (math.isnan(cand_test_latent) or math.isnan(ref_test_latent)) else False,
            "details": "Requires latent Manufacturer leakage not worse than promoted reference.",
        }
    )
    gate_df = pd.DataFrame(rows)
    if bool(gate_df["passes"].all()):
        decision = "promote"
    else:
        # This is a scientifically useful capacity/beta-axis result, but it fails
        # the pre-specified promotion gate.
        decision = "local_beta_sensitivity_only"
    return gate_df, decision


def metric_delta_table(pooled: pd.DataFrame) -> pd.DataFrame:
    focus = pooled[
        (pooled["model_name"].isin([PRIMARY_MODEL, "logreg_l2"]))
        & (pooled["feature_set"].eq(PRIMARY_FEATURES))
        & (pooled["threshold_strategy"].isin(["fixed_0p5", "inner_oof_balanced_accuracy", PRIMARY_THRESHOLD, "inner_oof_youden_j"]))
    ].copy()
    order = {"raw": 0, "oof_logitz": 1, "oof_ecdf": 2}
    focus["calib_order"] = focus["calib_method"].map(order).fillna(9)
    return focus.sort_values(["run_label", "calib_order", "threshold_strategy"]).drop(columns=["calib_order"])


def main() -> None:
    args = parse_args()
    out = args.output_dir
    inputs = [CANDIDATE_RUN, CANDIDATE_OOF, REFERENCE_RUN, REFERENCE_OOF]
    missing = [p for p in inputs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {[rel(p) for p in missing]}")

    planned = [
        "README.md",
        "completion_status.csv/.md",
        "completion_summary.csv/.md",
        "stagea_foldwise_metrics.csv/.md",
        "stagea_summary_metrics.csv/.md",
        "stageb_pooled_metrics_comparison.csv/.md",
        "stageb_primary_foldwise_comparison.csv/.md",
        "stageb_foldwise_delta.csv/.md",
        "vae_qc_comparison.csv/.md",
        "vae_qc_summary.csv/.md",
        "scanner_leakage_comparison.csv/.md",
        "scanner_leakage_summary.csv/.md",
        "philips_cn_fp_comparison.csv/.md",
        "philips_cn_fp_by_age_phase_site.csv/.md",
        "oasis_mega_available_comparison.csv/.md",
        "promotion_gate.csv/.md",
        "final_decision.md",
        "command_log.json",
    ]
    if args.dry_run:
        print("Inputs OK.")
        print("Planned output directory:", rel(out))
        for item in planned:
            print("-", item)
        return

    ensure_out(out)

    command_log: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": rel(Path(__file__)),
        "read_only_scope": {
            "no_training": True,
            "no_scoring": True,
            "no_threshold_fitting": True,
            "no_tensor_modification": True,
            "no_metadata_modification": True,
            "no_model_artifact_modification": True,
        },
        "inputs": {
            "candidate_run": rel(CANDIDATE_RUN),
            "candidate_oof": rel(CANDIDATE_OOF),
            "reference_run": rel(REFERENCE_RUN),
            "reference_oof": rel(REFERENCE_OOF),
            "metadata": rel(METADATA_PATH),
            "oasis_mega": rel(OASIS_MEGA),
        },
        "primary_gate_row": {
            "model_name": PRIMARY_MODEL,
            "feature_set": PRIMARY_FEATURES,
            "calib_method": PRIMARY_CALIB,
            "threshold_strategy": PRIMARY_THRESHOLD,
        },
    }

    cand_status = completion_status(CANDIDATE_RUN, CANDIDATE_OOF, "candidate_latent384_beta3p5")
    ref_status = completion_status(REFERENCE_RUN, REFERENCE_OOF, "promoted_reference_latent384_beta3p75")
    status = pd.concat([cand_status, ref_status], ignore_index=True)
    write_table(status, "completion_status", out)
    write_table(summarize_completion(status), "completion_summary", out)

    cand_stagea, cand_stagea_summary = load_stagea(CANDIDATE_RUN, "candidate_latent384_beta3p5")
    ref_stagea, ref_stagea_summary = load_stagea(REFERENCE_RUN, "promoted_reference_latent384_beta3p75")
    stagea = pd.concat([cand_stagea, ref_stagea], ignore_index=True)
    stagea_summary = pd.concat([cand_stagea_summary, ref_stagea_summary], ignore_index=True)
    write_table(stagea, "stagea_foldwise_metrics", out)
    write_table(stagea_summary, "stagea_summary_metrics", out)

    pooled = pd.concat(
        [
            pooled_metrics(CANDIDATE_RUN, CANDIDATE_OOF, "candidate_latent384_beta3p5"),
            pooled_metrics(REFERENCE_RUN, REFERENCE_OOF, "promoted_reference_latent384_beta3p75"),
        ],
        ignore_index=True,
    )
    pooled_focus = metric_delta_table(pooled)
    write_table(pooled_focus, "stageb_pooled_metrics_comparison", out)

    cand_fold = selected_foldwise(foldwise_oof(CANDIDATE_OOF, "candidate_latent384_beta3p5"), PRIMARY_CALIB)
    ref_fold = selected_foldwise(foldwise_oof(REFERENCE_OOF, "promoted_reference_latent384_beta3p75"), PRIMARY_CALIB)
    foldwise = pd.concat([cand_fold, ref_fold], ignore_index=True)
    write_table(foldwise, "stageb_primary_foldwise_comparison", out)
    write_table(foldwise_delta(cand_fold, ref_fold), "stageb_foldwise_delta", out)

    vae_qc = pd.concat(
        [
            load_vae_qc(CANDIDATE_RUN, "candidate_latent384_beta3p5"),
            load_vae_qc(REFERENCE_RUN, "promoted_reference_latent384_beta3p75"),
        ],
        ignore_index=True,
    )
    write_table(vae_qc, "vae_qc_comparison", out)
    vae_metrics = [
        "D_val_best",
        "R_val_bits_best",
        "KLD_over_D_best",
        "beta_KLD_over_D_best",
        "active_units",
        "total_correlation_nats",
        "MI_Z_Y_nats",
        "MI_Z_Manufacturer_nats",
        "MI_Manufacturer_over_MI_Y",
    ]
    write_table(
        flatten_columns(vae_qc.groupby("run_label", dropna=False)[vae_metrics].agg(["mean", "std", "min", "max"]).reset_index()),
        "vae_qc_summary",
        out,
    )

    scanner = pd.concat(
        [
            load_scanner_leakage(CANDIDATE_RUN, "candidate_latent384_beta3p5"),
            load_scanner_leakage(REFERENCE_RUN, "promoted_reference_latent384_beta3p75"),
        ],
        ignore_index=True,
    )
    write_table(scanner, "scanner_leakage_comparison", out)
    scanner_summary = scanner_leakage_summary(scanner)
    write_table(flatten_columns(scanner_summary), "scanner_leakage_summary", out)

    philips = pd.concat(
        [
            load_philips_pooled(CANDIDATE_OOF, "candidate_latent384_beta3p5"),
            load_philips_pooled(REFERENCE_OOF, "promoted_reference_latent384_beta3p75"),
        ],
        ignore_index=True,
    )
    write_table(philips, "philips_cn_fp_comparison", out)

    cand_pred = primary_predictions(CANDIDATE_OOF, "candidate_latent384_beta3p5")
    ref_pred = primary_predictions(REFERENCE_OOF, "promoted_reference_latent384_beta3p75")
    by_age_site = pd.concat(
        [
            philips_by_age_phase_site(cand_pred, "candidate_latent384_beta3p5"),
            philips_by_age_phase_site(ref_pred, "promoted_reference_latent384_beta3p75"),
        ],
        ignore_index=True,
    )
    write_table(by_age_site, "philips_cn_fp_by_age_phase_site", out)

    oasis = oasis_available_comparison()
    write_table(oasis, "oasis_mega_available_comparison", out)

    gate_df, decision = promotion_gate(pooled, philips, scanner_summary)
    write_table(gate_df, "promotion_gate", out)

    cand_primary = primary_pooled_row(pooled, "candidate_latent384_beta3p5")
    ref_primary = primary_pooled_row(pooled, "promoted_reference_latent384_beta3p75")
    if cand_primary is None or ref_primary is None:
        primary_text = "Primary OOF-ECDF rows were not both available."
    else:
        primary_text = (
            "Primary OOF-ECDF Stage B comparison "
            f"({PRIMARY_MODEL}, {PRIMARY_FEATURES}, {PRIMARY_THRESHOLD}):\n"
            f"- Candidate latent384/beta3.5: AUC={cand_primary['auc']:.6f}, "
            f"PR-AUC={cand_primary['pr_auc']:.6f}, BA={cand_primary['balanced_accuracy']:.6f}, "
            f"Sens={cand_primary['sensitivity']:.6f}, Spec={cand_primary['specificity']:.6f}, "
            f"F1={cand_primary['f1']:.6f}.\n"
            f"- Promoted latent384/beta3.75: AUC={ref_primary['auc']:.6f}, "
            f"PR-AUC={ref_primary['pr_auc']:.6f}, BA={ref_primary['balanced_accuracy']:.6f}, "
            f"Sens={ref_primary['sensitivity']:.6f}, Spec={ref_primary['specificity']:.6f}, "
            f"F1={ref_primary['f1']:.6f}.\n"
        )
    fold_delta = foldwise_delta(cand_fold, ref_fold)
    fold45 = fold_delta[fold_delta["candidate_fold"].isin([4, 5])]
    fold45_lines = []
    for _, row in fold45.iterrows():
        fold45_lines.append(
            f"- Fold {int(row['candidate_fold'])}: delta AUC={row.get('delta_auc', np.nan):+.6f}, "
            f"delta PR-AUC={row.get('delta_pr_auc', np.nan):+.6f}, "
            f"delta BA={row.get('delta_balanced_accuracy', np.nan):+.6f}, "
            f"delta F1={row.get('delta_f1', np.nan):+.6f}."
        )
    failed_gates = gate_df.loc[~gate_df["passes"].astype(bool), "gate"].tolist()
    decision_text = f"""# Final Decision

Decision: **{decision}**.

{primary_text}

Promotion-gate result:
- Failed gates: {", ".join(failed_gates) if failed_gates else "none"}.
- Required gate: AUC > 0.795155 and PR-AUC >= 0.573934, with no material worsening in BA/F1/Sensitivity, Philips CN FPR <= 44.4%, and no worse scanner/manufacturer leakage.

Fold 4 / Fold 5 deltas for the primary OOF-ECDF readout:
{chr(10).join(fold45_lines) if fold45_lines else "- Fold 4/5 primary deltas unavailable."}

Interpretation:
The latent384/beta3.5 candidate is complete and valid as a local beta-axis sensitivity run. Promotion depends only on the pre-specified gate against the promoted latent384/beta3.75 model; if any gate fails, this run should not replace the promoted reference and should be retained as a beta-local sensitivity audit.

OASIS:
Candidate-specific mega-OASIS 90/90 scoring artifacts were not required for this audit. The OASIS table, if present, is descriptive only and is not used in the promotion gate.
"""
    write_text("final_decision.md", decision_text, out)

    readme = f"""# Latent384 Beta3.5 Completion and Promotion-Gate Audit

This package compares `recover035_latent384_beta3p5_T80_h10000_p560_full5x5` against the promoted `recover035_latent384_beta3p75_T80_h10000_p560_full5x5` reference.

Primary promotion-gate row:
- `model_name={PRIMARY_MODEL}`
- `feature_set={PRIMARY_FEATURES}`
- `calib_method={PRIMARY_CALIB}`
- `threshold_strategy={PRIMARY_THRESHOLD}`

The audit is read-only with respect to tensors, metadata, and model artifacts. It performs no training, no scoring, and no threshold fitting.

Main outputs:
- `completion_summary.csv/.md`
- `stageb_pooled_metrics_comparison.csv/.md`
- `stageb_primary_foldwise_comparison.csv/.md`
- `vae_qc_comparison.csv/.md`
- `scanner_leakage_summary.csv/.md`
- `philips_cn_fp_comparison.csv/.md`
- `promotion_gate.csv/.md`
- `final_decision.md`
"""
    write_text("README.md", readme, out)

    command_log["generated_outputs"] = planned
    command_log["decision"] = decision
    command_log["input_existence"] = list_existing([CANDIDATE_RUN, CANDIDATE_OOF, REFERENCE_RUN, REFERENCE_OOF, METADATA_PATH, OASIS_MEGA])
    (out / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
    print(f"Wrote audit package: {rel(out)}")
    print(f"Decision: {decision}")


if __name__ == "__main__":
    main()
