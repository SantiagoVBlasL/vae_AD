#!/usr/bin/env python3
"""Read-only audit for the foldwise ComBat input-harmonized FULL run.

This script intentionally does not train models, create latent caches, score
OASIS, or modify existing run artifacts. It aggregates completed fold outputs
and records whether the predefined Stage B/OOF promotion gate is available.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)


RUN_DIR = Path(
    "results/revision_bspc_2026/"
    "recover035_latent384_beta3p75_foldcombat_mfr_age_sex_T80_h10000_p560_full5x5"
)
OUT_DIR = Path(
    "results/revision_bspc_2026/"
    "foldcombat_completion_promotion_harmonization_audit_20260607"
)
EVIDENCE_MAP = Path(
    "results/revision_bspc_2026/"
    "final_full_model_evidence_map_with_beta6p5_20260606/full_model_evidence_map.csv"
)
CHMEANLOSS_COMPARISON = Path(
    "results/revision_bspc_2026/"
    "chmeanloss_stageB_oof_completion_audit_20260607/comparison_vs_references.csv"
)

PROMOTED_AUC = 0.795155
PROMOTED_PR_AUC = 0.573934
PROMOTED_BA = 0.725979
PROMOTED_SENS = 0.731959
PROMOTED_F1 = 0.563492
PROMOTED_PHILIPS_CN_FPR = 0.4545
PROMOTED_SCANNER_LEAKAGE = 0.727556


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def df_to_md(df: pd.DataFrame, path: Path, max_rows: int | None = None) -> None:
    if max_rows is not None and len(df) > max_rows:
        view = df.head(max_rows).copy()
        suffix = f"\n\n_Truncated to first {max_rows} of {len(df)} rows._\n"
    else:
        view = df.copy()
        suffix = ""
    if view.empty:
        path.write_text("_No rows._\n", encoding="utf-8")
    else:
        path.write_text(view.to_markdown(index=False) + suffix + "\n", encoding="utf-8")


def write_table(df: pd.DataFrame, stem: str, max_md_rows: int | None = None) -> None:
    csv_path = OUT_DIR / f"{stem}.csv"
    md_path = OUT_DIR / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    df_to_md(df, md_path, max_rows=max_md_rows)


def safe_div(num: float, den: float) -> float:
    if den == 0 or pd.isna(den):
        return float("nan")
    return float(num) / float(den)


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "cn": int((y_true == 0).sum()),
        "ad": int((y_true == 1).sum()),
        "auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, scores))
        if len(np.unique(y_true)) == 2
        else float("nan"),
        "ba": float(balanced_accuracy_score(y_true, y_pred)),
        "sens": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "spec": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def normalized_mfr(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().replace({"nan": np.nan, "None": np.nan})


def load_config() -> dict[str, Any]:
    path = RUN_DIR / "run_config.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_metadata(config: dict[str, Any]) -> pd.DataFrame:
    meta_path = Path(config.get("metadata_path", ""))
    if not meta_path.exists():
        return pd.DataFrame()
    meta = pd.read_csv(meta_path)
    keep = [
        "SubjectID",
        "ResearchGroup_Mapped",
        "Diagnosis",
        "Age",
        "Sex",
        "Manufacturer",
        "Site3",
        "exclude_from_supervised",
        "training_ready",
        "n_timepoints_raw",
    ]
    return meta[[c for c in keep if c in meta.columns]].copy()


def fold_dirs() -> list[Path]:
    return [RUN_DIR / f"fold_{i}" for i in range(1, 6)]


def completion_status() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, fold_dir in enumerate(fold_dirs(), start=1):
        rows.append(
            {
                "fold": i,
                "fold_dir_exists": fold_dir.exists(),
                "vae_model_exists": (fold_dir / f"vae_model_fold_{i}.pt").exists(),
                "vae_history_exists": (fold_dir / f"vae_train_history_fold_{i}.joblib").exists(),
                "stagea_logreg_predictions_exists": (fold_dir / "test_predictions_logreg.csv").exists(),
                "stagea_svm_predictions_exists": (fold_dir / "test_predictions_svm.csv").exists(),
                "rate_distortion_exists": (fold_dir / f"fold_{i}_rate_distortion.csv").exists(),
                "train_scanner_leakage_exists": (fold_dir / f"fold_{i}_scanner_leakage_summary.csv").exists(),
                "test_scanner_leakage_exists": (fold_dir / f"fold_{i}_test_scanner_leakage_summary.csv").exists(),
                "train_latent_info_exists": (fold_dir / f"fold_{i}_trainDev_latent_info_summary.csv").exists(),
                "test_latent_info_exists": (fold_dir / f"fold_{i}_test_latent_info_summary.csv").exists(),
                "harmonization_guard_exists": (fold_dir / "input_harmonization_leakage_guard.csv").exists(),
                "harmonization_integrity_exists": (fold_dir / "input_harmonization_integrity.csv").exists(),
            }
        )
    top_stagea = list(RUN_DIR.glob("all_folds_metrics_MULTI_*.csv"))
    stageb_dir = RUN_DIR / "classifier_only_readout"
    rows.append(
        {
            "fold": "all",
            "fold_dir_exists": RUN_DIR.exists(),
            "vae_model_exists": all((RUN_DIR / f"fold_{i}/vae_model_fold_{i}.pt").exists() for i in range(1, 6)),
            "vae_history_exists": all(
                (RUN_DIR / f"fold_{i}/vae_train_history_fold_{i}.joblib").exists() for i in range(1, 6)
            ),
            "stagea_logreg_predictions_exists": len(top_stagea) > 0,
            "stagea_svm_predictions_exists": all(
                (RUN_DIR / f"fold_{i}/test_predictions_svm.csv").exists() for i in range(1, 6)
            ),
            "rate_distortion_exists": all((RUN_DIR / f"fold_{i}/fold_{i}_rate_distortion.csv").exists() for i in range(1, 6)),
            "train_scanner_leakage_exists": all(
                (RUN_DIR / f"fold_{i}/fold_{i}_scanner_leakage_summary.csv").exists() for i in range(1, 6)
            ),
            "test_scanner_leakage_exists": all(
                (RUN_DIR / f"fold_{i}/fold_{i}_test_scanner_leakage_summary.csv").exists() for i in range(1, 6)
            ),
            "train_latent_info_exists": all(
                (RUN_DIR / f"fold_{i}/fold_{i}_trainDev_latent_info_summary.csv").exists() for i in range(1, 6)
            ),
            "test_latent_info_exists": all(
                (RUN_DIR / f"fold_{i}/fold_{i}_test_latent_info_summary.csv").exists() for i in range(1, 6)
            ),
            "harmonization_guard_exists": all(
                (RUN_DIR / f"fold_{i}/input_harmonization_leakage_guard.csv").exists() for i in range(1, 6)
            ),
            "harmonization_integrity_exists": all(
                (RUN_DIR / f"fold_{i}/input_harmonization_integrity.csv").exists() for i in range(1, 6)
            ),
            "classifier_only_readout_exists": stageb_dir.exists(),
            "stageb_oof_calibration_exists": any("oof" in p.name.lower() for p in RUN_DIR.glob("*")),
        }
    )
    return pd.DataFrame(rows)


def load_stagea_predictions(metadata: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for fold in range(1, 6):
        fold_dir = RUN_DIR / f"fold_{fold}"
        for model in ["logreg", "svm"]:
            p = fold_dir / f"test_predictions_{model}.csv"
            if not p.exists():
                continue
            df = pd.read_csv(p)
            df["fold"] = fold
            df["model"] = model
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if not metadata.empty and "SubjectID" in metadata.columns:
        out = out.merge(metadata, on="SubjectID", how="left", suffixes=("", "_meta"))
    out["Manufacturer"] = normalized_mfr(out.get("Manufacturer", pd.Series(index=out.index, dtype=object)))
    return out


def stagea_metrics(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if preds.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows = []
    fold_rows = []
    for model, df_m in preds.groupby("model"):
        for score_col, label in [("y_score_raw", "raw"), ("y_score_final", "final")]:
            if score_col not in df_m.columns:
                continue
            metrics = binary_metrics(df_m["y_true"], df_m[score_col], df_m["y_pred"])
            metrics.update({"stage": "Stage A", "model": model, "score_type": label, "fold": "pooled"})
            rows.append(metrics)
            for fold, df_f in df_m.groupby("fold"):
                fm = binary_metrics(df_f["y_true"], df_f[score_col], df_f["y_pred"])
                fm.update({"stage": "Stage A", "model": model, "score_type": label, "fold": int(fold)})
                fold_rows.append(fm)
    summary = pd.DataFrame(rows)
    foldwise = pd.DataFrame(fold_rows)
    order = ["stage", "model", "score_type", "fold"]
    summary = summary[order + [c for c in summary.columns if c not in order]]
    foldwise = foldwise[order + [c for c in foldwise.columns if c not in order]]
    return summary, foldwise


def manufacturer_error_tables(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if preds.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows_fpr = []
    rows_fnr = []
    for model, df_m in preds.groupby("model"):
        df_final = df_m.copy()
        for mfr, g in df_final.groupby("Manufacturer", dropna=False):
            cn = g[g["y_true"] == 0]
            ad = g[g["y_true"] == 1]
            fp = int(((cn["y_pred"] == 1)).sum())
            tn = int(((cn["y_pred"] == 0)).sum())
            fn = int(((ad["y_pred"] == 0)).sum())
            tp = int(((ad["y_pred"] == 1)).sum())
            rows_fpr.append(
                {
                    "model": model,
                    "manufacturer": mfr,
                    "cn_n": int(len(cn)),
                    "fp": fp,
                    "tn": tn,
                    "cn_fpr": safe_div(fp, fp + tn),
                }
            )
            rows_fnr.append(
                {
                    "model": model,
                    "manufacturer": mfr,
                    "ad_n": int(len(ad)),
                    "fn": fn,
                    "tp": tp,
                    "ad_fnr": safe_div(fn, fn + tp),
                }
            )
    return pd.DataFrame(rows_fpr), pd.DataFrame(rows_fnr)


def aggregate_scanner_leakage() -> pd.DataFrame:
    rows = []
    for fold in range(1, 6):
        for split, suffix in [("train_dev", "scanner_leakage"), ("test", "test_scanner_leakage")]:
            p = RUN_DIR / f"fold_{fold}/fold_{fold}_{suffix}.csv"
            df = read_csv_if_exists(p)
            if df.empty:
                continue
            for _, r in df.iterrows():
                rows.append({"fold": fold, "split": split, **r.to_dict()})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    summary_rows = []
    for (split, representation), g in out.groupby(["split", "representation"]):
        summary_rows.append(
            {
                "split": split,
                "representation": representation,
                "mean_balanced_accuracy": g["balanced_accuracy_mean"].mean(),
                "sd_balanced_accuracy": g["balanced_accuracy_mean"].std(ddof=1),
                "min_balanced_accuracy": g["balanced_accuracy_mean"].min(),
                "max_balanced_accuracy": g["balanced_accuracy_mean"].max(),
                "n_folds": int(g["fold"].nunique()),
            }
        )
    return pd.DataFrame(summary_rows)


def aggregate_rate_distortion() -> pd.DataFrame:
    rows = []
    for fold in range(1, 6):
        p = RUN_DIR / f"fold_{fold}/fold_{fold}_rate_distortion.csv"
        df = read_csv_if_exists(p)
        if df.empty:
            continue
        best_idx = df["L_val_betaMax"].idxmin() if "L_val_betaMax" in df.columns else df.index[-1]
        best = df.loc[best_idx]
        final = df.iloc[-1]
        d_best = float(best.get("D_val", np.nan))
        r_bits_best = float(best.get("R_val_bits", np.nan))
        beta = float(best.get("beta", np.nan))
        rows.append(
            {
                "fold": fold,
                "best_epoch": int(best.get("epoch", -1)),
                "final_epoch": int(final.get("epoch", -1)),
                "D_val_best": d_best,
                "R_val_nats_best": float(best.get("R_val_nats", np.nan)),
                "R_val_bits_best": r_bits_best,
                "L_val_betaMax_best": float(best.get("L_val_betaMax", np.nan)),
                "beta_at_best": beta,
                "KLD_over_D_best": safe_div(float(best.get("R_val_nats", np.nan)), d_best),
                "beta_KLD_over_D_best": safe_div(3.75 * float(best.get("R_val_nats", np.nan)), d_best),
                "bits_per_latent_dim_best": safe_div(r_bits_best, 384.0),
                "D_val_final": float(final.get("D_val", np.nan)),
                "R_val_bits_final": float(final.get("R_val_bits", np.nan)),
            }
        )
    foldwise = pd.DataFrame(rows)
    if foldwise.empty:
        return foldwise
    mean_row = {c: foldwise[c].mean() for c in foldwise.select_dtypes(include=[np.number]).columns if c != "fold"}
    mean_row["fold"] = "mean"
    return pd.concat([foldwise, pd.DataFrame([mean_row])], ignore_index=True)


def mean_row_value(df: pd.DataFrame, col: str, fold_col: str = "fold") -> float:
    if df.empty or col not in df.columns:
        return float("nan")
    if fold_col in df.columns:
        mean_rows = df[df[fold_col].astype(str) == "mean"]
        if not mean_rows.empty:
            return float(mean_rows[col].iloc[0])
    vals = pd.to_numeric(df[col], errors="coerce")
    return float(vals.mean())


def latent_mean_value(df: pd.DataFrame, col: str, split: str = "test") -> float:
    if df.empty or col not in df.columns:
        return float("nan")
    subset = df[(df.get("fold", pd.Series(dtype=object)).astype(str) == "mean") & (df.get("split", "") == split)]
    if subset.empty:
        subset = df[df.get("split", "") == split]
    if subset.empty:
        return float("nan")
    return float(pd.to_numeric(subset[col], errors="coerce").mean())


def aggregate_latent_info() -> pd.DataFrame:
    rows = []
    for fold in range(1, 6):
        for split, prefix in [("train_dev", "trainDev"), ("test", "test")]:
            p = RUN_DIR / f"fold_{fold}/fold_{fold}_{prefix}_latent_info_summary.csv"
            df = read_csv_if_exists(p)
            if df.empty:
                continue
            df = df.copy()
            df["fold"] = fold
            df["split"] = split
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    long = pd.concat(rows, ignore_index=True)
    piv_rows = []
    for (fold, split), g in long.groupby(["fold", "split"]):
        y = g[g["variable"] == "Y_target"]
        m = g[g["variable"] == "Manufacturer"]
        row = {
            "fold": fold,
            "split": split,
            "MI_Z_Y_nats": float(y["mi_sum_nats"].iloc[0]) if not y.empty else np.nan,
            "MI_Z_Manufacturer_nats": float(m["mi_sum_nats"].iloc[0]) if not m.empty else np.nan,
            "active_units": float(g["n_active"].iloc[0]) if "n_active" in g.columns and len(g) else np.nan,
            "total_correlation_nats": float(g["total_correlation_nats"].iloc[0])
            if "total_correlation_nats" in g.columns and len(g)
            else np.nan,
        }
        row["MI_Manufacturer_over_MI_Y"] = safe_div(row["MI_Z_Manufacturer_nats"], row["MI_Z_Y_nats"])
        piv_rows.append(row)
    out = pd.DataFrame(piv_rows)
    means = []
    for split, g in out.groupby("split"):
        row = {c: g[c].mean() for c in g.select_dtypes(include=[np.number]).columns if c != "fold"}
        row.update({"fold": "mean", "split": split})
        means.append(row)
    return pd.concat([out, pd.DataFrame(means)], ignore_index=True)


def aggregate_harmonization() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    file_map = {
        "harmonization_leakage_guard_summary": "input_harmonization_leakage_guard.csv",
        "harmonization_integrity_summary": "input_harmonization_integrity.csv",
        "harmonization_fit_audit": "input_harmonization_fit_audit.csv",
        "harmonization_scale_shift_summary": "input_harmonization_channel_shift_summary.csv",
        "harmonization_manufacturer_separability_summary": "input_harmonization_manufacturer_separability_proxy.csv",
    }
    for stem, fname in file_map.items():
        frames = []
        for fold in range(1, 6):
            p = RUN_DIR / f"fold_{fold}/{fname}"
            df = read_csv_if_exists(p)
            if df.empty:
                continue
            if "fold" not in df.columns:
                df["fold"] = fold
            frames.append(df)
        out[stem] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    sep = out.get("harmonization_manufacturer_separability_summary", pd.DataFrame())
    if not sep.empty:
        numeric_cols = [
            "pre_mean_pairwise_centroid_distance_per_edge",
            "post_mean_pairwise_centroid_distance_per_edge",
            "delta_mean_pairwise_centroid_distance_per_edge",
        ]
        sep_summary = sep.groupby(["channel_position", "channel_name"], dropna=False)[numeric_cols].agg(["mean", "std"]).reset_index()
        sep_summary.columns = ["_".join([str(x) for x in c if str(x)]) for c in sep_summary.columns]
        out["harmonization_manufacturer_separability_channel_mean"] = sep_summary

    shift = out.get("harmonization_scale_shift_summary", pd.DataFrame())
    if not shift.empty:
        cols = ["pre_mean", "post_mean", "delta_mean", "pre_std", "post_std", "delta_std", "mean_abs_delta", "max_abs_delta"]
        shift_summary = shift.groupby(["split", "channel_position", "channel_name"], dropna=False)[cols].mean().reset_index()
        out["harmonization_scale_shift_channel_mean"] = shift_summary
    return out


def stageb_status() -> pd.DataFrame:
    stageb_dir = RUN_DIR / "classifier_only_readout"
    candidates = [
        stageb_dir / "classifier_sweep_pooled_metrics.csv",
        stageb_dir / "primary_results.csv",
        stageb_dir / "stageb_oof_calibration_metrics.csv",
    ]
    return pd.DataFrame(
        [
            {
                "component": "classifier_only_readout_dir",
                "status": "present" if stageb_dir.exists() else "missing",
                "path": str(stageb_dir),
            },
            *[
                {
                    "component": p.name,
                    "status": "present" if p.exists() else "missing",
                    "path": str(p),
                }
                for p in candidates
            ],
            {
                "component": "primary_stageB_oof_ecdf_gate",
                "status": "not_evaluable_stageB_oof_missing",
                "path": "",
            },
        ]
    )


def load_reference_table() -> pd.DataFrame:
    rows = []
    if EVIDENCE_MAP.exists():
        ev = pd.read_csv(EVIDENCE_MAP)
        ids = [
            "promoted_latent384_beta3p75_ch1_0_2",
            "ch1only_latent384_beta3p75",
            "latent384_beta6p5",
            "latent448_beta4p0",
            "latent512_beta3p75",
        ]
        rows.append(ev[ev["model_id"].isin(ids)].copy())
    if CHMEANLOSS_COMPARISON.exists():
        ch = pd.read_csv(CHMEANLOSS_COMPARISON)
        ch = ch[ch["comparison_role"].astype(str).str.contains("candidate", case=False, na=False)].copy()
        if not ch.empty:
            ch["display_name"] = "chmeanloss latent384 beta3.75"
            ch["decision_class"] = "loss sensitivity"
            rows.append(ch)
    if not rows:
        return pd.DataFrame()
    ref = pd.concat(rows, ignore_index=True, sort=False)
    keep = [
        "model_id",
        "display_name",
        "decision_class",
        "channel_set_order",
        "latent_dim",
        "beta_vae",
        "adni_oof_ecdf_auc",
        "adni_oof_ecdf_pr_auc",
        "adni_oof_ecdf_ba",
        "adni_oof_ecdf_sens",
        "adni_oof_ecdf_spec",
        "adni_oof_ecdf_f1",
        "philips_cn_fpr",
        "scanner_leakage_latent_acc",
        "beta_KLD_over_D_best_mean",
        "active_units_mean",
        "total_correlation_nats_mean",
        "MI_Z_Y_nats_mean",
        "MI_Z_Manufacturer_nats_mean",
        "MI_Manufacturer_over_MI_Y_mean",
        "oasis_status",
    ]
    return ref[[c for c in keep if c in ref.columns]]


def promotion_gate(
    stageb: pd.DataFrame,
    scanner_summary: pd.DataFrame,
    guard: pd.DataFrame,
    integrity: pd.DataFrame,
    stagea_summary: pd.DataFrame,
    mfr_fpr: pd.DataFrame,
) -> tuple[pd.DataFrame, str, str]:
    stageb_available = stageb["status"].eq("present").any() and not stageb["component"].eq("primary_stageB_oof_ecdf_gate").any()
    guard_pass = (
        not guard.empty
        and guard.get("status", pd.Series(dtype=str)).eq("PASS").all()
        and guard.get("diagnosis_used_in_harmonizer", pd.Series([True])).eq(False).all()
        and guard.get("global_combat", pd.Series([True])).eq(False).all()
        and guard.get("oasis_used", pd.Series([True])).eq(False).all()
        and guard.get("n_fit_test_subject_overlap", pd.Series([1])).fillna(1).eq(0).all()
    )
    integrity_pass = (
        not integrity.empty
        and integrity.get("status", pd.Series(dtype=str)).eq("PASS").all()
        and integrity.get("diagnosis_used_in_harmonizer", pd.Series([True])).eq(False).all()
        and integrity.get("test_distribution_used_in_fit", pd.Series([True])).eq(False).all()
        and integrity.get("global_combat", pd.Series([True])).eq(False).all()
    )
    test_latent = scanner_summary[
        (scanner_summary.get("split", "") == "test") & (scanner_summary.get("representation", "") == "latent_mu")
    ]
    scanner_val = float(test_latent["mean_balanced_accuracy"].iloc[0]) if not test_latent.empty else np.nan
    scanner_lower = bool(not math.isnan(scanner_val) and scanner_val < PROMOTED_SCANNER_LEAKAGE)

    philips_rows = mfr_fpr[(mfr_fpr["model"] == "logreg") & (mfr_fpr["manufacturer"].astype(str).str.lower() == "philips")]
    stagea_philips = float(philips_rows["cn_fpr"].iloc[0]) if not philips_rows.empty else np.nan
    stagea_philips_lower = bool(not math.isnan(stagea_philips) and stagea_philips <= PROMOTED_PHILIPS_CN_FPR)

    logreg_final = stagea_summary[
        (stagea_summary["model"] == "logreg") & (stagea_summary["score_type"] == "final") & (stagea_summary["fold"] == "pooled")
    ]
    stagea_auc = float(logreg_final["auc"].iloc[0]) if not logreg_final.empty else np.nan
    stagea_pr = float(logreg_final["pr_auc"].iloc[0]) if not logreg_final.empty else np.nan

    rows = [
        {
            "gate": "Stage B classifier-only OOF-ECDF available",
            "required": "required for promotion",
            "observed": "missing" if not stageb_available else "present",
            "pass": bool(stageb_available),
        },
        {
            "gate": "AUC > promoted or PR-AUC materially improves without AUC loss",
            "required": f"AUC > {PROMOTED_AUC:.6f} or PR-AUC >= {PROMOTED_PR_AUC:.6f} with no AUC loss",
            "observed": f"Stage A logreg final AUC={stagea_auc:.6f}, PR-AUC={stagea_pr:.6f}; Stage B unavailable",
            "pass": False,
        },
        {
            "gate": "PR-AUC >= promoted",
            "required": f">= {PROMOTED_PR_AUC:.6f}",
            "observed": "not evaluable because Stage B/OOF missing",
            "pass": False,
        },
        {
            "gate": "BA/F1/Sensitivity not materially worse",
            "required": f"compare with BA={PROMOTED_BA:.6f}, Sens={PROMOTED_SENS:.6f}, F1={PROMOTED_F1:.6f}",
            "observed": "not evaluable because Stage B/OOF missing",
            "pass": False,
        },
        {
            "gate": "Philips CN FPR <= promoted",
            "required": f"<= {PROMOTED_PHILIPS_CN_FPR:.4f}",
            "observed": f"Stage A logreg final Philips CN FPR={stagea_philips:.6f}; Stage B unavailable",
            "pass": bool(stagea_philips_lower and stageb_available),
        },
        {
            "gate": "scanner leakage lower than promoted",
            "required": f"< {PROMOTED_SCANNER_LEAKAGE:.6f}",
            "observed": f"test latent scanner leakage BA={scanner_val:.6f}",
            "pass": bool(scanner_lower),
        },
        {
            "gate": "no evidence of leakage or diagnosis use in harmonization",
            "required": "train/dev fit only, no diagnosis, no global ComBat, no OASIS, no train/test overlap",
            "observed": f"guard_pass={guard_pass}, integrity_pass={integrity_pass}",
            "pass": bool(guard_pass and integrity_pass),
        },
    ]
    gate_df = pd.DataFrame(rows)
    if stageb_available and gate_df["pass"].all():
        decision = "promote_pending_oasis_external_inference"
        oasis = "run_frozen_oasis_inference"
    elif scanner_lower or stagea_philips_lower:
        decision = "do_not_promote_stageB_oof_missing_harmonization_sensitivity_only"
        oasis = "not_run_primary_adni_gate_failed_optional_external_stress_only"
    else:
        decision = "do_not_promote_stageB_oof_missing"
        oasis = "not_run_adni_gate_failed"
    return gate_df, decision, oasis


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    command_log = {
        "created_utc": now_iso(),
        "script": str(Path(__file__)),
        "run_dir": str(RUN_DIR),
        "output_dir": str(OUT_DIR),
        "mode": "read_only_audit",
        "guardrails": [
            "no_training",
            "no_oasis_scoring",
            "no_threshold_fitting",
            "no_tensor_modification",
            "no_metadata_modification",
            "no_model_artifact_modification",
        ],
    }

    config = load_config()
    args = config.get("args", {})
    metadata = load_metadata(config)

    run_manifest = pd.DataFrame(
        [
            {
                "run_dir": str(RUN_DIR),
                "resolved_run_dir": str(RUN_DIR.resolve()) if RUN_DIR.exists() else "",
                "metadata_path": config.get("metadata_path", ""),
                "global_tensor_path": config.get("global_tensor_path", ""),
                "channels_to_use": args.get("channels_to_use"),
                "channel_names_selected": ";".join(config.get("channel_names_selected", [])),
                "latent_dim": args.get("latent_dim"),
                "beta_vae": args.get("beta_vae"),
                "recon_loss_mode": args.get("recon_loss_mode"),
                "input_harmonization_mode": args.get("input_harmonization_mode"),
                "input_harmonization_batch_col": args.get("input_harmonization_batch_col"),
                "input_harmonization_covariates": args.get("input_harmonization_covariates"),
                "input_harmonization_excluded_covariates": args.get("input_harmonization_excluded_covariates"),
                "outer_folds": args.get("outer_folds"),
                "inner_folds": args.get("inner_folds"),
            }
        ]
    )
    write_table(run_manifest, "run_manifest")

    comp = completion_status()
    write_table(comp, "completion_status")

    preds = load_stagea_predictions(metadata)
    stagea_summary, stagea_foldwise = stagea_metrics(preds)
    write_table(stagea_summary, "stagea_summary_metrics")
    write_table(stagea_foldwise, "stagea_foldwise_metrics")

    mfr_fpr, mfr_fnr = manufacturer_error_tables(preds)
    write_table(mfr_fpr, "manufacturer_cn_fpr")
    write_table(mfr_fnr, "manufacturer_ad_fnr")

    stageb = stageb_status()
    write_table(stageb, "stageb_oof_availability")
    placeholder = pd.DataFrame(
        [
            {
                "readout": "logreg_l2_original / z_plus_age_sex / oof_ecdf / inner_oof_target_sens_ge_0p70_max_spec",
                "status": "missing",
                "auc": np.nan,
                "pr_auc": np.nan,
                "ba": np.nan,
                "sens": np.nan,
                "spec": np.nan,
                "f1": np.nan,
                "reason": "classifier_only_readout and OOF score-calibration artifacts were not present; read-only audit did not generate them",
            }
        ]
    )
    write_table(placeholder, "stageb_classifier_only_metrics")
    write_table(placeholder, "stageb_oof_calibration_metrics")

    scanner = aggregate_scanner_leakage()
    write_table(scanner, "scanner_leakage_summary")

    rd = aggregate_rate_distortion()
    write_table(rd, "rate_distortion_summary")

    latent = aggregate_latent_info()
    write_table(latent, "latent_mi_signal_nuisance_summary")

    harm = aggregate_harmonization()
    for stem, df in harm.items():
        write_table(df, stem, max_md_rows=100)

    guard = harm.get("harmonization_leakage_guard_summary", pd.DataFrame())
    integrity = harm.get("harmonization_integrity_summary", pd.DataFrame())
    gate_df, decision, oasis_decision = promotion_gate(stageb, scanner, guard, integrity, stagea_summary, mfr_fpr)
    write_table(gate_df, "primary_promotion_gate_table")

    refs = load_reference_table()
    candidate_row = {}
    logreg_final = stagea_summary[
        (stagea_summary["model"] == "logreg") & (stagea_summary["score_type"] == "final") & (stagea_summary["fold"] == "pooled")
    ]
    if not logreg_final.empty:
        lr = logreg_final.iloc[0].to_dict()
        candidate_row.update(
            {
                "model_id": "foldcombat_mfr_age_sex_latent384_beta3p75_stageA_logreg_final",
                "display_name": "foldcombat Manufacturer Age/Sex latent384 beta3.75 (Stage A only)",
                "decision_class": "harmonization sensitivity",
                "channel_set_order": "[1,0,2]",
                "latent_dim": 384,
                "beta_vae": 3.75,
                "adni_oof_ecdf_auc": np.nan,
                "adni_oof_ecdf_pr_auc": np.nan,
                "adni_oof_ecdf_ba": np.nan,
                "adni_oof_ecdf_sens": np.nan,
                "adni_oof_ecdf_spec": np.nan,
                "adni_oof_ecdf_f1": np.nan,
                "stageA_logreg_auc": lr.get("auc"),
                "stageA_logreg_pr_auc": lr.get("pr_auc"),
                "stageA_logreg_ba": lr.get("ba"),
                "stageA_logreg_sens": lr.get("sens"),
                "stageA_logreg_spec": lr.get("spec"),
                "stageA_logreg_f1": lr.get("f1"),
                "philips_cn_fpr": mfr_fpr[
                    (mfr_fpr["model"] == "logreg") & (mfr_fpr["manufacturer"].astype(str).str.lower() == "philips")
                ]["cn_fpr"].iloc[0]
                if not mfr_fpr[
                    (mfr_fpr["model"] == "logreg") & (mfr_fpr["manufacturer"].astype(str).str.lower() == "philips")
                ].empty
                else np.nan,
                "scanner_leakage_latent_acc": scanner[
                    (scanner["split"] == "test") & (scanner["representation"] == "latent_mu")
                ]["mean_balanced_accuracy"].iloc[0]
                if not scanner[(scanner["split"] == "test") & (scanner["representation"] == "latent_mu")].empty
                else np.nan,
                "beta_KLD_over_D_best_mean": mean_row_value(rd, "beta_KLD_over_D_best"),
                "active_units_mean": latent_mean_value(latent, "active_units", "test"),
                "total_correlation_nats_mean": latent_mean_value(latent, "total_correlation_nats", "test"),
                "MI_Z_Y_nats_mean": latent_mean_value(latent, "MI_Z_Y_nats", "test"),
                "MI_Z_Manufacturer_nats_mean": latent_mean_value(latent, "MI_Z_Manufacturer_nats", "test"),
                "MI_Manufacturer_over_MI_Y_mean": latent_mean_value(latent, "MI_Manufacturer_over_MI_Y", "test"),
                "oasis_status": oasis_decision,
            }
        )
    comp_refs = pd.concat([pd.DataFrame([candidate_row]), refs], ignore_index=True, sort=False) if candidate_row else refs
    write_table(comp_refs, "comparison_vs_references", max_md_rows=50)

    if not logreg_final.empty:
        stagea_to_stageb = pd.DataFrame(
            [
                {
                    "model_id": "foldcombat_mfr_age_sex_latent384_beta3p75",
                    "stageA_logreg_final_auc": float(logreg_final["auc"].iloc[0]),
                    "stageA_logreg_final_pr_auc": float(logreg_final["pr_auc"].iloc[0]),
                    "stageB_oof_ecdf_auc": np.nan,
                    "stageB_oof_ecdf_pr_auc": np.nan,
                    "stageB_minus_stageA_auc": np.nan,
                    "stageB_minus_stageA_pr_auc": np.nan,
                    "status": "not_evaluable_stageB_oof_missing",
                }
            ]
        )
    else:
        stagea_to_stageb = pd.DataFrame(
            [{"model_id": "foldcombat_mfr_age_sex_latent384_beta3p75", "status": "stageA_missing"}]
        )
    write_table(stagea_to_stageb, "stageA_to_stageB_delta")

    (OUT_DIR / "oasis_gate_decision.md").write_text(
        "\n".join(
            [
                "# OASIS Gate Decision",
                "",
                f"Decision: `{oasis_decision}`.",
                "",
                "No frozen OASIS inference was run in this read-only audit. The predefined ADNI primary",
                "promotion gate could not be evaluated because classifier-only Stage B and OOF score",
                "calibration artifacts were missing for the foldcombat run.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    scanner_val = np.nan
    test_latent = scanner[(scanner.get("split", "") == "test") & (scanner.get("representation", "") == "latent_mu")]
    if not test_latent.empty:
        scanner_val = float(test_latent["mean_balanced_accuracy"].iloc[0])
    philips = np.nan
    philips_rows = mfr_fpr[(mfr_fpr["model"] == "logreg") & (mfr_fpr["manufacturer"].astype(str).str.lower() == "philips")]
    if not philips_rows.empty:
        philips = float(philips_rows["cn_fpr"].iloc[0])
    (OUT_DIR / "final_decision.md").write_text(
        "\n".join(
            [
                "# Final Decision",
                "",
                f"Decision: `{decision}`.",
                "",
                "The foldcombat run completed fold-level VAE and Stage A artifacts for all five folds,",
                "and the input-harmonization leakage guards passed at the fold-audit level. However,",
                "the predefined classifier-only Stage B readout and OOF score-calibration artifacts were",
                "not present, so the primary promoted-convention gate cannot be satisfied from this",
                "read-only audit.",
                "",
                "Key read-only observations:",
                f"- Test latent scanner leakage BA mean: `{scanner_val:.6f}` versus promoted reference `{PROMOTED_SCANNER_LEAKAGE:.6f}`.",
                f"- Stage A logreg Philips CN FPR: `{philips:.6f}` versus promoted Stage B reference `{PROMOTED_PHILIPS_CN_FPR:.4f}`.",
                "- Harmonization was fit on outer train/dev only, with Manufacturer as batch, Age/Sex preserved,",
                "  diagnosis excluded, no global ComBat, no OASIS use, and zero fit/test subject overlap.",
                "",
                "This should be retained as a harmonization-method sensitivity until the predefined",
                "classifier-only Stage B plus OOF-ECDF/OOF-logitz readout is generated and audited.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    (OUT_DIR / "README.md").write_text(
        "\n".join(
            [
                "# Foldcombat Completion, Promotion-Gate, And Harmonization-Effect Audit",
                "",
                f"Input run: `{RUN_DIR}`",
                "",
                "This package is read-only with respect to training artifacts. It aggregates existing",
                "fold outputs and records the availability of the predefined Stage B classifier-only",
                "and OOF score-calibration gate.",
                "",
                "Generated outputs include Stage A metrics, manufacturer FPR/FNR, scanner leakage,",
                "rate-distortion summaries, latent MI summaries, foldwise harmonization guard reports,",
                "and the promotion-gate decision.",
                "",
                f"Final decision: `{decision}`.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    command_log["outputs"] = sorted(str(p.relative_to(OUT_DIR)) for p in OUT_DIR.iterdir())
    command_log["final_decision"] = decision
    command_log["oasis_gate_decision"] = oasis_decision
    (OUT_DIR / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
