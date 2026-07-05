#!/usr/bin/env python3
"""Final ADNI model-ranking and deep audit package.

This script is intentionally read-only with respect to tensors, metadata, and
model artifacts. It only reads completed result directories and writes a derived
audit package under results/revision_bspc_2026/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "revision_bspc_2026"
OUT = RESULTS / "final_best_model_deep_audit_and_ranking_20260601"

PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
PRIMARY_MODEL = "logreg_l2"
OOF_MODEL = "logreg_l2_original"
FEATURE = "z_plus_age_sex"


RUNS: dict[str, dict[str, Any]] = {
    "promoted_p015_latent384_beta3p75": {
        "run_dir": RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5",
        "oof_dir": RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration",
        "decision": "promote",
        "notes": "Primary candidate; p=0.15 legacy_all; score-harmonized readout available.",
    },
    "locked_v5_1b_horizon4480": {
        "run_dir": RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5",
        "decision": "superseded_reference",
        "notes": "Previous locked v5.1b [1,0,2] horizon4480 classifier-only readout.",
    },
    "dropout010_global": {
        "run_dir": RESULTS / "recover035_latent384_beta3p75_drop0p10_T80_h10000_p560_full5x5",
        "oof_dir": RESULTS / "drop0p10_oof_logitz_readout_audit_20260531",
        "decision": "reject",
        "notes": "Global dropout_rate_vae 0.15 -> 0.10.",
    },
    "enc015_dec010": {
        "run_dir": RESULTS / "recover035_latent384_beta3p75_encdrop0p15_decdrop0p10_T80_h10000_p560_full5x5",
        "audit_dir": RESULTS / "encdrop0p15_decdrop0p10_completion_comparison_audit_20260531",
        "decision": "reject",
        "notes": "Encoder dropout 0.15 and decoder dropout 0.10.",
    },
    "latent128_beta1p25": {
        "run_dir": RESULTS / "recover035_latent128_beta1p25_lockedSchedule_full5x5",
        "oof_dir": RESULTS / "recover035_latent128_beta1p25_lockedSchedule_full5x5_stageB_oof_logitz",
        "decision": "reject",
        "notes": "Latent_dim 128 with proportional beta 1.25.",
    },
    "latent128_beta2p5": {
        "run_dir": RESULTS / "recover035_latent128_beta2p5_lockedSchedule_full5x5",
        "oof_dir": RESULTS / "recover035_latent128_beta2p5_lockedSchedule_full5x5_stageB_oof_logitz",
        "decision": "reject",
        "notes": "Latent_dim 128 with beta 2.5 capacity control.",
    },
}


NEGATIVE_COMPARISON_FILES: list[tuple[str, Path]] = [
    ("manufacturer_balanced_sampler_full", RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_manufacturer_balanced_sampler_full_5x5_comparison" / "main_model_comparison.csv"),
    ("dropout010_full", RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_dropout010_full_5x5_comparison" / "main_model_comparison.csv"),
    ("dropout020_full", RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_dropout020_full_5x5_comparison" / "main_model_comparison.csv"),
    ("no_decoder_dropout_full", RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_no_decoder_dropout_full_5x5_comparison" / "main_model_comparison.csv"),
    ("beta65_full", RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_beta65_full_5x5_comparison" / "main_model_comparison.csv"),
    ("block_order_norm_act_full", RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_block_order_norm_act_full_5x5_comparison" / "main_model_comparison.csv"),
    ("v5_1c_horizon10000", RESULTS / "adni_v5_1c_horizon10000_cycles125_vs_v5_1b_and_v5_1c4480_comparison" / "main_model_comparison.csv"),
    ("objective_v2_offdiag_channelmean", RESULTS / "adni_v5_1c_objective_v2_offdiag_channelmean_horizon10000_vs_references_comparison" / "main_model_comparison.csv"),
    ("ch1_only_offdiag", RESULTS / "adni_v5_1b_ch1_only_offdiag_channelmean_postmortem" / "primary_decision_table.csv"),
    ("ch1_2_offdiag", RESULTS / "adni_v5_1b_ch1_2_offdiag_channelmean_vs_final_and_ch1_comparison" / "main_model_comparison.csv"),
    ("all_timepoints_exploratory", RESULTS / "adni_all_available_timepoints_ch1_0_2_exploratory_full_5x5_comparison" / "main_model_comparison.csv"),
    ("ultra_regularized_logreg", RESULTS / "adni_v5_1_batch20260514b_ultra_regularized_logreg_readout_audit" / "primary_comparison.csv"),
    ("manufacturer_conditioned_full", RESULTS / "conditional_beta_vae_manufacturer_full5x5_mfrrecovered035_clfpoollocked" / "primary_results.csv"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--dry-run", action="store_true", help="Print planned outputs without writing files.")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame({"read_error": [str(exc)], "source_path": [str(path)]})


def write_table(df: pd.DataFrame, csv_path: Path, md_path: Path | None = None, index: bool = False) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=index)
    if md_path is not None:
        try:
            text = df.to_markdown(index=index)
        except Exception:
            text = df.to_string(index=index)
        md_path.write_text(text + "\n", encoding="utf-8")


def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def first_existing(row: pd.Series, names: Iterable[str], default: Any = np.nan) -> Any:
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    return default


def filter_model_feature(df: pd.DataFrame, oof: bool, threshold_strategy: str | None = PRIMARY_THRESHOLD) -> pd.DataFrame:
    if df.empty:
        return df
    df = norm_cols(df)
    if "model_name" in df.columns:
        wanted_model = OOF_MODEL if oof else PRIMARY_MODEL
        df = df[df["model_name"].astype(str) == wanted_model]
    feature_col = "feature_set" if "feature_set" in df.columns else "readout_feature_set"
    if feature_col in df.columns:
        df = df[df[feature_col].astype(str) == FEATURE]
    if threshold_strategy is not None and "threshold_strategy" in df.columns:
        df = df[df["threshold_strategy"].astype(str) == threshold_strategy]
    return df


def row_to_metric_record(
    row: pd.Series,
    *,
    run_key: str,
    source_kind: str,
    source_path: Path,
    decision: str,
    notes: str,
    scanner: dict[str, Any] | None = None,
    philips: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scanner = scanner or {}
    philips = philips or {}
    return {
        "model_id": run_key,
        "source_kind": source_kind,
        "calib_method": first_existing(row, ["calib_method"], "raw"),
        "threshold_strategy": first_existing(row, ["threshold_strategy"], np.nan),
        "n": first_existing(row, ["n"], np.nan),
        "n_cn": first_existing(row, ["n_cn"], np.nan),
        "n_ad": first_existing(row, ["n_ad"], np.nan),
        "auc": first_existing(row, ["auc", "AUC", "ROC-AUC", "ROC_AUC"], np.nan),
        "pr_auc": first_existing(row, ["pr_auc", "PR-AUC", "PR_AUC"], np.nan),
        "balanced_accuracy": first_existing(row, ["balanced_accuracy", "BA"], np.nan),
        "sensitivity": first_existing(row, ["sensitivity", "Sens"], np.nan),
        "specificity": first_existing(row, ["specificity", "Spec"], np.nan),
        "f1": first_existing(row, ["f1", "f1_score", "F1"], np.nan),
        "tn": first_existing(row, ["tn", "TN"], np.nan),
        "fp": first_existing(row, ["fp", "FP"], np.nan),
        "fn": first_existing(row, ["fn", "FN"], np.nan),
        "tp": first_existing(row, ["tp", "TP"], np.nan),
        "scanner_raw_ba_mean": scanner.get("acc_site_raw_mean", np.nan),
        "scanner_latent_ba_mean": scanner.get("acc_site_latent_mean", np.nan),
        "scanner_latent_minus_raw_mean": scanner.get("latent_minus_raw_mean", np.nan),
        "philips_cn_n": philips.get("philips_cn_n", np.nan),
        "philips_cn_fp": philips.get("philips_cn_fp", np.nan),
        "philips_cn_fpr": philips.get("philips_cn_fpr", np.nan),
        "decision": decision,
        "notes": notes,
        "source_path": str(source_path),
    }


def summarize_scanner_leakage(run_dir: Path) -> dict[str, Any]:
    rows: list[pd.DataFrame] = []
    for fold in range(1, 6):
        for suffix in ["test_scanner_leakage_summary", "scanner_leakage_summary"]:
            p = run_dir / f"fold_{fold}" / f"fold_{fold}_{suffix}.csv"
            df = read_csv(p)
            if not df.empty and "acc_site_raw" in df.columns and "acc_site_latent" in df.columns:
                df = df.copy()
                df["fold"] = fold
                df["source_path"] = str(p)
                rows.append(df)
                break
    if not rows:
        return {}
    all_df = pd.concat(rows, ignore_index=True)
    raw = pd.to_numeric(all_df["acc_site_raw"], errors="coerce")
    latent = pd.to_numeric(all_df["acc_site_latent"], errors="coerce")
    return {
        "acc_site_raw_mean": float(raw.mean()),
        "acc_site_latent_mean": float(latent.mean()),
        "latent_minus_raw_mean": float((latent - raw).mean()),
        "n_leakage_folds": int(len(all_df)),
    }


def summarize_philips_raw(readout_dir: Path) -> dict[str, Any]:
    path = readout_dir / "classifier_sweep_subgroup_metrics_by_manufacturer.csv"
    df = filter_model_feature(read_csv(path), oof=False)
    if df.empty:
        return {}
    manu_col = "Manufacturer" if "Manufacturer" in df.columns else "manufacturer"
    if manu_col not in df.columns:
        return {}
    ph = df[df[manu_col].astype(str).str.lower().eq("philips")]
    if ph.empty:
        return {}
    return {
        "philips_cn_n": float(pd.to_numeric(ph.get("n_cn"), errors="coerce").sum()),
        "philips_cn_fp": float(pd.to_numeric(ph.get("fp"), errors="coerce").sum()),
        "philips_cn_fpr": safe_div(
            pd.to_numeric(ph.get("fp"), errors="coerce").sum(),
            pd.to_numeric(ph.get("n_cn"), errors="coerce").sum(),
        ),
    }


def summarize_philips_oof(oof_dir: Path, method: str) -> dict[str, Any]:
    path = oof_dir / "calib_philips_fpr_pooled.csv"
    df = read_csv(path)
    if df.empty:
        return {}
    df = filter_model_feature(df, oof=True)
    if "calib_method" in df.columns:
        df = df[df["calib_method"].astype(str) == method]
    manu_col = "manufacturer" if "manufacturer" in df.columns else "Manufacturer"
    if manu_col in df.columns:
        df = df[df[manu_col].astype(str).str.lower().eq("philips")]
    if df.empty:
        return {}
    row = df.iloc[0]
    return {
        "philips_cn_n": first_existing(row, ["n_cn_pooled", "n_cn"], np.nan),
        "philips_cn_fp": first_existing(row, ["fp_cn_pooled", "fp"], np.nan),
        "philips_cn_fpr": first_existing(row, ["fpr_cn_pooled"], np.nan),
    }


def safe_div(a: Any, b: Any) -> float:
    try:
        a = float(a)
        b = float(b)
        if b == 0 or math.isnan(b):
            return float("nan")
        return a / b
    except Exception:
        return float("nan")


def collect_core_ranking(threshold_strategy: str | None = PRIMARY_THRESHOLD) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for run_key, meta in RUNS.items():
        run_dir = meta.get("run_dir")
        decision = meta.get("decision", "")
        notes = meta.get("notes", "")
        scanner = summarize_scanner_leakage(run_dir) if isinstance(run_dir, Path) else {}

        if isinstance(run_dir, Path):
            raw_path = run_dir / "classifier_only_readout" / "classifier_sweep_pooled_metrics.csv"
            raw_df = filter_model_feature(read_csv(raw_path), oof=False, threshold_strategy=threshold_strategy)
            if not raw_df.empty:
                philips = summarize_philips_raw(run_dir / "classifier_only_readout")
                for _, row in raw_df.iterrows():
                    records.append(
                        row_to_metric_record(
                            row,
                            run_key=run_key,
                            source_kind="raw_stageb",
                            source_path=raw_path,
                            decision=decision if decision != "promote" else "raw_reference_not_primary",
                            notes=notes,
                            scanner=scanner,
                            philips=philips,
                        )
                    )

        oof_dir = meta.get("oof_dir")
        if isinstance(oof_dir, Path):
            oof_path = oof_dir / "calib_pooled_metrics.csv"
            oof_df = filter_model_feature(read_csv(oof_path), oof=True, threshold_strategy=threshold_strategy)
            if not oof_df.empty:
                # Keep all score-harmonization rows at the primary threshold.
                for _, row in oof_df.iterrows():
                    method = str(row.get("calib_method", "unknown"))
                    if method == "raw":
                        continue
                    philips = summarize_philips_oof(oof_dir, method)
                    row_decision = decision
                    if run_key == "promoted_p015_latent384_beta3p75":
                        row_decision = "promote" if method in {"oof_ecdf", "oof_logitz"} else "sensitivity"
                    records.append(
                        row_to_metric_record(
                            row,
                            run_key=run_key,
                            source_kind="score_harmonized_stageb",
                            source_path=oof_path,
                            decision=row_decision,
                            notes=notes,
                            scanner=scanner,
                            philips=philips,
                        )
                    )

    df = pd.DataFrame(records)
    if not df.empty:
        for col in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.sort_values(["auc", "pr_auc", "balanced_accuracy"], ascending=False, na_position="last")
    return df


def collect_negative_inventory() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    failed = RESULTS / "adni_v5_1_batch20260514b_manuscript_defense_locked_current_model" / "failed_optimization_table.csv"
    failed_df = read_csv(failed)
    if not failed_df.empty:
        failed_df = failed_df.copy()
        failed_df["inventory_source"] = "manuscript_defense_failed_optimization_table"
        failed_df["source_path"] = str(failed)
        return failed_df

    for label, path in NEGATIVE_COMPARISON_FILES:
        exists = path.exists()
        df = read_csv(path)
        row: dict[str, Any] = {
            "candidate_or_audit": label,
            "source_path": str(path),
            "exists": exists,
            "n_rows": int(len(df)) if exists else 0,
        }
        for metric in ["auc", "AUC", "pr_auc", "PR-AUC", "balanced_accuracy", "BA", "f1", "F1"]:
            if metric in df.columns:
                vals = pd.to_numeric(df[metric], errors="coerce")
                row[f"max_{metric}"] = vals.max()
        rows.append(row)
    return pd.DataFrame(rows)


def collect_negative_numeric_summary() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, path in NEGATIVE_COMPARISON_FILES:
        df = read_csv(path)
        if df.empty:
            rows.append({"audit": label, "source_path": str(path), "status": "missing_or_empty"})
            continue
        cols = {c.lower().replace("-", "_"): c for c in df.columns}
        candidate_col = next((df.columns[i] for i, c in enumerate(df.columns) if c.lower() in {"candidate", "model", "model_id", "run", "name"}), None)
        for _, r in df.iterrows():
            rec = {"audit": label, "source_path": str(path), "status": "parsed"}
            if candidate_col:
                rec["candidate"] = r.get(candidate_col)
            for out_name, aliases in {
                "auc": ["auc", "roc_auc"],
                "pr_auc": ["pr_auc", "pr_auc_final"],
                "balanced_accuracy": ["balanced_accuracy", "ba"],
                "sensitivity": ["sensitivity", "sens"],
                "specificity": ["specificity", "spec"],
                "f1": ["f1", "f1_score"],
                "decision": ["decision"],
            }.items():
                for alias in aliases:
                    if alias in cols:
                        rec[out_name] = r.get(cols[alias])
                        break
            rows.append(rec)
    return pd.DataFrame(rows)


def collect_foldwise_from_oof(oof_dir: Path) -> pd.DataFrame:
    path = oof_dir / "calib_foldwise_metrics.csv"
    df = filter_model_feature(read_csv(path), oof=True)
    if df.empty:
        return df
    return df.sort_values(["fold", "calib_method", "threshold_strategy"])


def collect_vae_training_qc(run_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        fdir = run_dir / f"fold_{fold}"
        rd = read_csv(fdir / f"fold_{fold}_rate_distortion.csv")
        rec: dict[str, Any] = {"fold": fold, "run_dir": str(run_dir)}
        if not rd.empty and "epoch" in rd.columns:
            rd = rd.copy()
            rd["epoch"] = pd.to_numeric(rd["epoch"], errors="coerce")
            loss_col = "L_val_betaMax" if "L_val_betaMax" in rd.columns else None
            if loss_col is not None:
                loss = pd.to_numeric(rd[loss_col], errors="coerce")
                idx = loss.idxmin()
                rec.update(
                    {
                        "best_epoch": int(rd.loc[idx, "epoch"]),
                        "final_epoch": int(pd.to_numeric(rd["epoch"], errors="coerce").max()),
                        "epochs_after_best": int(pd.to_numeric(rd["epoch"], errors="coerce").max() - rd.loc[idx, "epoch"]),
                        "best_val_L_betaMax": float(loss.loc[idx]),
                        "best_val_recon_D": float(pd.to_numeric(rd.get("D_val"), errors="coerce").loc[idx]) if "D_val" in rd else np.nan,
                        "best_val_R_nats": float(pd.to_numeric(rd.get("R_val_nats"), errors="coerce").loc[idx]) if "R_val_nats" in rd else np.nan,
                        "best_val_KLD_over_R": safe_div(pd.to_numeric(rd.get("R_val_nats"), errors="coerce").loc[idx] if "R_val_nats" in rd else np.nan, pd.to_numeric(rd.get("D_val"), errors="coerce").loc[idx] if "D_val" in rd else np.nan),
                        "beta_at_best_epoch": float(pd.to_numeric(rd.get("beta"), errors="coerce").loc[idx]) if "beta" in rd else np.nan,
                    }
                )
                rec["best_epoch_in_last_10_percent_observed"] = rec["best_epoch"] >= 0.9 * rec["final_epoch"] if rec["final_epoch"] else np.nan
                if len(rd) >= 100:
                    y = pd.to_numeric(rd[loss_col].tail(100), errors="coerce").to_numpy()
                    x = np.arange(len(y))
                    ok = np.isfinite(y)
                    rec["last100_val_L_betaMax_slope"] = float(np.polyfit(x[ok], y[ok], 1)[0]) if ok.sum() >= 2 else np.nan
                if len(rd) >= 300:
                    y = pd.to_numeric(rd[loss_col].tail(300), errors="coerce").to_numpy()
                    x = np.arange(len(y))
                    ok = np.isfinite(y)
                    rec["last300_val_L_betaMax_slope"] = float(np.polyfit(x[ok], y[ok], 1)[0]) if ok.sum() >= 2 else np.nan
        latent = read_csv(fdir / f"fold_{fold}_test_latent_info_summary.csv")
        if not latent.empty:
            diag = latent[latent.get("variable", "").astype(str).eq("Y_target")] if "variable" in latent.columns else latent.head(1)
            if not diag.empty:
                rec["active_units"] = first_existing(diag.iloc[0], ["n_active"], np.nan)
                rec["frac_active"] = first_existing(diag.iloc[0], ["frac_active"], np.nan)
                rec["total_correlation_nats"] = first_existing(diag.iloc[0], ["total_correlation_nats"], np.nan)
                rec["mi_diagnosis_sum_nats"] = first_existing(diag.iloc[0], ["mi_sum_nats"], np.nan)
        rows.append(rec)
    return pd.DataFrame(rows)


def collect_latent_information(run_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for fold in range(1, 6):
        for split in ["test", "trainDev"]:
            p = run_dir / f"fold_{fold}" / f"fold_{fold}_{split}_latent_info_summary.csv"
            df = read_csv(p)
            if df.empty:
                continue
            df = df.copy()
            df["fold"] = fold
            df["split"] = split
            df["source_path"] = str(p)
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def collect_scanner_leakage_by_fold(run_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for fold in range(1, 6):
        for suffix in ["test_scanner_leakage_summary", "scanner_leakage_summary"]:
            p = run_dir / f"fold_{fold}" / f"fold_{fold}_{suffix}.csv"
            df = read_csv(p)
            if not df.empty:
                df = df.copy()
                df["fold"] = fold
                df["split_source"] = suffix
                df["latent_minus_raw"] = pd.to_numeric(df.get("acc_site_latent"), errors="coerce") - pd.to_numeric(df.get("acc_site_raw"), errors="coerce")
                df["source_path"] = str(p)
                rows.append(df)
                break
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def collect_confusion_from_foldwise(foldwise: pd.DataFrame) -> pd.DataFrame:
    if foldwise.empty:
        return pd.DataFrame()
    cols = [c for c in ["fold", "calib_method", "threshold_strategy", "n", "n_cn", "n_ad", "tn", "fp", "fn", "tp", "sensitivity", "specificity", "balanced_accuracy", "f1", "auc", "pr_auc"] if c in foldwise.columns]
    return foldwise[cols].copy()


def collect_subject_errors(oof_dir: Path, method: str = "oof_ecdf") -> pd.DataFrame:
    df = read_csv(oof_dir / "calib_predictions.csv")
    if df.empty:
        return df
    df = df[
        (df.get("model_name", "").astype(str) == OOF_MODEL)
        & (df.get("feature_set", "").astype(str) == FEATURE)
        & (df.get("calib_method", "").astype(str) == method)
        & (df.get("threshold_strategy", "").astype(str) == PRIMARY_THRESHOLD)
    ].copy()
    if df.empty:
        return df
    df["error_type"] = np.select(
        [
            (df["y_true"] == 0) & (df["y_pred"] == 0),
            (df["y_true"] == 0) & (df["y_pred"] == 1),
            (df["y_true"] == 1) & (df["y_pred"] == 0),
            (df["y_true"] == 1) & (df["y_pred"] == 1),
        ],
        ["TN", "FP", "FN", "TP"],
        default="unknown",
    )
    cols = [
        "SubjectID",
        "ResearchGroup_Mapped",
        "Manufacturer",
        "Age",
        "Sex",
        "fold",
        "threshold",
        "y_true",
        "y_score_raw",
        "y_score",
        "y_pred",
        "error_type",
    ]
    return df[[c for c in cols if c in df.columns]]


def collect_classifier_hyperparameters(oof_dir: Path) -> pd.DataFrame:
    df = read_csv(oof_dir / "calib_meta.csv")
    if df.empty:
        return df
    df = df[
        (df.get("model_name", "").astype(str) == OOF_MODEL)
        & (df.get("feature_set", "").astype(str) == FEATURE)
    ].copy()
    df["selected_C"] = df["best_params"].astype(str).str.extract(r"model__C\"?:\s*([0-9.eE+-]+)").iloc[:, 0]
    return df


def collect_training_curve_manifest(run_dir: Path) -> pd.DataFrame:
    rows = []
    for fold in range(1, 6):
        fdir = run_dir / f"fold_{fold}"
        rows.append(
            {
                "fold": fold,
                "history_joblib": str(fdir / f"vae_train_history_fold_{fold}.joblib"),
                "history_png": str(fdir / f"vae_train_history_fold_{fold}.png"),
                "rate_distortion_csv": str(fdir / f"fold_{fold}_rate_distortion.csv"),
                "history_joblib_exists": (fdir / f"vae_train_history_fold_{fold}.joblib").exists(),
                "history_png_exists": (fdir / f"vae_train_history_fold_{fold}.png").exists(),
                "rate_distortion_exists": (fdir / f"fold_{fold}_rate_distortion.csv").exists(),
            }
        )
    return pd.DataFrame(rows)


def collect_interpretability_readiness(run_dir: Path) -> tuple[pd.DataFrame, str]:
    cache_paths = sorted((run_dir / "classifier_only_readout").glob("latent_cache/*")) if (run_dir / "classifier_only_readout").exists() else []
    shap_paths = [
        p
        for p in sorted(RESULTS.glob("**/*"))
        if re.search(r"(^|[^a-z])shap([^a-z]|$)|shapley", p.name.lower())
    ]
    ig_paths = sorted(RESULTS.glob("**/*integrated*grad*")) + sorted(RESULTS.glob("**/*saliency*"))
    top_dim_rows = []
    for fold in range(1, 6):
        p = run_dir / f"fold_{fold}" / f"fold_{fold}_test_latent_info_summary.csv"
        df = read_csv(p)
        if df.empty or "variable" not in df.columns:
            continue
        for _, row in df.iterrows():
            top_dim_rows.append(
                {
                    "fold": fold,
                    "split": "test",
                    "variable": row.get("variable"),
                    "mi_sum_nats": row.get("mi_sum_nats"),
                    "top_dims": row.get("top_dims"),
                    "source_path": str(p),
                }
            )
    top_dims_df = pd.DataFrame(top_dim_rows)
    md = [
        "# Interpretability Readiness",
        "",
        f"- Latent cache files found under classifier-only readout: {len(cache_paths)}.",
        f"- SHAP-like paths found under revision results: {len(shap_paths)}.",
        f"- IG/saliency-like paths found under revision results: {len(ig_paths)}.",
        "",
        "Interpretability status:",
        "- Latent caches and fold-level latent information summaries are available for post-hoc latent/edge analysis.",
        "- Existing SHAP/IG artifacts were not assumed to be complete unless listed below.",
        "- If manuscript interpretability figures are needed, run a locked post-hoc plan on the promoted p=0.15 folds only: latent-feature SHAP for Stage B and VAE decoder/edge saliency or latent traversal summarized to ROI-edge and Yeo17 network-pair levels.",
        "",
        "SHAP candidates:",
    ]
    md.extend([f"- `{p}`" for p in shap_paths[:50]] or ["- none found"])
    md.append("")
    md.append("IG/saliency candidates:")
    md.extend([f"- `{p}`" for p in ig_paths[:50]] or ["- none found"])
    return top_dims_df, "\n".join(md) + "\n"


def collect_external_summary() -> tuple[pd.DataFrame, str]:
    rows: list[pd.DataFrame] = []
    sources = [
        ("oasis_new_60_60_harmonized_primary", RESULTS / "oasis_next_60cn_60ad_external_scoring_harmonized_horizon4480_20260531" / "primary_metrics.csv"),
        ("oasis_new_60_60_harmonized_locked_test", RESULTS / "oasis_next_60cn_60ad_external_scoring_harmonized_horizon4480_20260531" / "metrics_locked_test.csv"),
        ("oasis_new_60_60_harmonized_calibration", RESULTS / "oasis_next_60cn_60ad_external_scoring_harmonized_horizon4480_20260531" / "metrics_calibration.csv"),
        ("mega_oasis_90_90_pooled", RESULTS / "oasis_mega_90cn_90ad_pooled_external_validation_20260531" / "pooled_ranking_metrics.csv"),
        ("mega_oasis_90_90_threshold", RESULTS / "oasis_mega_90cn_90ad_pooled_external_validation_20260531" / "threshold_metrics.csv"),
        ("pilot_vs_new_existing", RESULTS / "oasis_pilot_vs_new_tensor_scoring_parity_audit_20260531" / "existing_metric_comparison.csv"),
        ("pilot_vs_new_new_rescored", RESULTS / "oasis_pilot_vs_new_tensor_scoring_parity_audit_20260531" / "new_rescored_with_pilot_scorer_metrics.csv"),
        ("pilot_vs_new_pilot_rescored", RESULTS / "oasis_pilot_vs_new_tensor_scoring_parity_audit_20260531" / "pilot_rescored_with_next_scorer_metrics.csv"),
    ]
    for label, path in sources:
        df = read_csv(path)
        if df.empty:
            rows.append(pd.DataFrame([{"external_source": label, "source_path": str(path), "status": "missing_or_empty"}]))
            continue
        df = df.copy()
        df["external_source"] = label
        df["source_path"] = str(path)
        rows.append(df)
    combined = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    md = [
        "# External Stress-Test Summary",
        "",
        "The OASIS external analyses remain stress tests, not model-selection evidence.",
        "",
        "- New OASIS 60/60 calibration/test scoring was harmonized to the horizon4480 classifier-only readout, but transfer remains weaker and threshold-sensitive than ADNI internal CV.",
        "- Mega-OASIS 90/90 pooled analysis is explicitly secondary/exploratory because it pools the earlier pilot batch and the new batch.",
        "- Pilot-vs-new parity audits identified instability across OASIS batches and runwise tensor construction details; this supports external calibration/test rather than further ADNI internal optimization.",
        "- Clinical severity and motion audits should be used as explanatory sensitivity material only; no OASIS-based model promotion or threshold fitting on locked test subjects is endorsed.",
        "",
        "See `external_stress_test_metrics_inventory.csv` for all parsed metric sources.",
    ]
    return combined, "\n".join(md) + "\n"


def write_readme(out_dir: Path, ranking: pd.DataFrame) -> None:
    top = ranking.iloc[0].to_dict() if not ranking.empty else {}
    text = f"""# Final Best Model Deep Audit And Ranking

Generated: {now_iso()}

This package is read-only with respect to tensors, metadata, and model artifacts.
It aggregates completed ADNI and OASIS result artifacts into a final ranking and
deep audit for the current best all-eligible ADNI model.

## Primary Decision

Promoted primary ADNI model:
`recover035_latent384_beta3p75_T80_h10000_p560_full5x5`
with p=0.15 `legacy_all` dropout and score-harmonized Stage B readout.

Top parsed row:

- model_id: `{top.get('model_id', 'NA')}`
- source_kind: `{top.get('source_kind', 'NA')}`
- calibration: `{top.get('calib_method', 'NA')}`
- threshold: `{top.get('threshold_strategy', 'NA')}`
- AUC: {top.get('auc', np.nan)}
- PR-AUC: {top.get('pr_auc', np.nan)}
- BA: {top.get('balanced_accuracy', np.nan)}
- F1: {top.get('f1', np.nan)}

OOF-ECDF and OOF-logitz are both retained in the ranking. The OOF-ECDF row is
the highest parsed score-harmonized row; the OOF-logitz row is nearly tied and
is retained as the requested reference score-harmonization view.

## Guardrails

- No training was launched.
- No tensor, metadata, ledger, or model-output artifact was modified.
- Output files are derived summaries only.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir: Path = args.output_dir
    if args.dry_run:
        print(f"Would write final audit package to {out_dir}")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)

    command_log = {
        "script": str(Path(__file__).resolve()),
        "cwd": str(Path.cwd()),
        "started_at": now_iso(),
        "safety": {
            "training_launched": False,
            "tensor_modified": False,
            "metadata_modified": False,
            "model_artifacts_modified": False,
        },
        "inputs": {k: {kk: str(vv) if isinstance(vv, Path) else vv for kk, vv in v.items()} for k, v in RUNS.items()},
    }

    ranking = collect_core_ranking(threshold_strategy=PRIMARY_THRESHOLD)
    write_table(ranking, out_dir / "model_ranking_table.csv", out_dir / "model_ranking_table.md")
    all_thresholds = collect_core_ranking(threshold_strategy=None)
    write_table(all_thresholds, out_dir / "model_ranking_all_thresholds.csv", out_dir / "model_ranking_all_thresholds.md")

    neg_inventory = collect_negative_inventory()
    write_table(neg_inventory, out_dir / "recorded_negative_audit_inventory.csv", out_dir / "recorded_negative_audit_inventory.md")
    neg_numeric = collect_negative_numeric_summary()
    write_table(neg_numeric, out_dir / "recorded_negative_numeric_metrics.csv", out_dir / "recorded_negative_numeric_metrics.md")

    best_run = RUNS["promoted_p015_latent384_beta3p75"]["run_dir"]
    best_oof = RUNS["promoted_p015_latent384_beta3p75"]["oof_dir"]

    foldwise = collect_foldwise_from_oof(best_oof)
    write_table(foldwise, out_dir / "best_model_foldwise_metrics.csv", out_dir / "best_model_foldwise_metrics.md")
    fold45 = foldwise[foldwise.get("fold", pd.Series(dtype=int)).isin([4, 5])] if not foldwise.empty else pd.DataFrame()
    write_table(fold45, out_dir / "best_model_fold4_fold5_diagnostics.csv", out_dir / "best_model_fold4_fold5_diagnostics.md")

    vae_qc = collect_vae_training_qc(best_run)
    write_table(vae_qc, out_dir / "best_model_vae_training_qc.csv", out_dir / "best_model_vae_training_qc.md")
    curves = collect_training_curve_manifest(best_run)
    write_table(curves, out_dir / "best_model_training_curve_manifest.csv", out_dir / "best_model_training_curve_manifest.md")

    latent = collect_latent_information(best_run)
    write_table(latent, out_dir / "best_model_latent_information.csv", out_dir / "best_model_latent_information.md")
    scanner = collect_scanner_leakage_by_fold(best_run)
    write_table(scanner, out_dir / "best_model_scanner_leakage.csv", out_dir / "best_model_scanner_leakage.md")

    score_scale = read_csv(best_oof / "calib_score_range_by_fold.csv")
    if not score_scale.empty:
        score_scale = score_scale[
            (score_scale.get("model_name", "").astype(str) == OOF_MODEL)
            & (score_scale.get("feature_set", "").astype(str) == FEATURE)
        ].copy()
    write_table(score_scale, out_dir / "best_model_score_scale_audit.csv", out_dir / "best_model_score_scale_audit.md")

    hp = collect_classifier_hyperparameters(best_oof)
    write_table(hp, out_dir / "best_model_classifier_hyperparameters.csv", out_dir / "best_model_classifier_hyperparameters.md")

    conf = collect_confusion_from_foldwise(foldwise)
    write_table(conf, out_dir / "best_model_confusion_matrices.csv", out_dir / "best_model_confusion_matrices.md")

    errors = collect_subject_errors(best_oof, method="oof_ecdf")
    write_table(errors, out_dir / "best_model_subject_errors.csv", out_dir / "best_model_subject_errors.md")

    top_dims, interp_md = collect_interpretability_readiness(best_run)
    write_table(top_dims, out_dir / "best_model_top_latent_dimensions.csv", out_dir / "best_model_top_latent_dimensions.md")
    (out_dir / "interpretability_readiness.md").write_text(interp_md, encoding="utf-8")

    external, external_md = collect_external_summary()
    write_table(external, out_dir / "external_stress_test_metrics_inventory.csv", out_dir / "external_stress_test_metrics_inventory.md")
    (out_dir / "external_stress_test_summary.md").write_text(external_md, encoding="utf-8")

    model_change = [
        "# Final Model Recommendation",
        "",
        "Decision: promote `recover035_latent384_beta3p75_T80_h10000_p560_full5x5` as the current best ADNI internal-CV model.",
        "",
        "Rationale:",
        "- It is the top parsed all-eligible ADNI model by AUC/PR-AUC under score-harmonized Stage B.",
        "- Dropout0.10, encoder0.15/decoder0.10, latent128 beta1.25, latent128 beta2.5, and the broader negative-audit inventory do not cleanly beat it.",
        "- The external OASIS analyses are stress tests and do not justify OASIS-based model selection.",
        "",
        "Remaining limitations:",
        "- Philips CN false positives and GE AD false negatives remain important subgroup error modes.",
        "- Fold 4/Fold 5 diagnostics should be reported as fold-level heterogeneity rather than optimized away.",
        "- Interpretability is ready at the latent-cache level, but edge/network summaries require a separate locked post-hoc interpretability run if not already available.",
    ]
    (out_dir / "final_model_recommendation.md").write_text("\n".join(model_change) + "\n", encoding="utf-8")

    write_readme(out_dir, ranking)
    command_log["finished_at"] = now_iso()
    command_log["outputs"] = sorted(str(p.relative_to(ROOT)) for p in out_dir.glob("*"))
    (out_dir / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote final audit package to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
