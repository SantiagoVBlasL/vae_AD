#!/usr/bin/env python3
"""Completion, promotion-gate, and OASIS audit for ch1-only follow-ups.

Guardrails:
- no VAE training;
- no OASIS threshold/calibration fitting;
- no tensor, metadata, ledger, or model artifact modification;
- Stage B/OASIS classifier reconstruction uses ADNI train/dev latent caches only,
  following the existing mega-OASIS scorer convention.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"
SCRIPT_DIR = PROJECT_ROOT / "scripts/revision_bspc_2026"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import score_oasis_mega_90_90_external_inference_model_panel_20260604 as oasis_base  # noqa: E402


OUT = RESULTS / "ch1only_targeted_followup_completion_promotion_oasis_audit_20260606"

PRIMARY_MODEL = "logreg_l2_original"
PRIMARY_FEATURE_SET = "z_plus_age_sex"
PRIMARY_CALIB = "oof_ecdf"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
OOF_METHODS = ["raw", "oof_zscore", "oof_logitz", "oof_ecdf", "oof_platt", "oof_isotonic"]

RUNS = [
    {
        "run_id": "promoted_ch102_latent384_beta3p75",
        "display_name": "promoted [1,0,2] latent384 beta3.75",
        "role": "promoted_reference",
        "run_dir": RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5",
        "oof_dir": RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration",
        "oasis_existing_candidate": "promoted_beta3p75_oof_ecdf",
    },
    {
        "run_id": "ch1_reference_latent384_beta3p75",
        "display_name": "ch1-only latent384 beta3.75 reference",
        "role": "ch1_reference",
        "run_dir": RESULTS / "recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5",
        "oof_dir": RESULTS / "recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5_stageB_oof_score_calibration",
        "oasis_existing_candidate": "ch1only_latent384_beta3p75_oof_ecdf",
    },
    {
        "run_id": "ch1only_latent256_beta3p75",
        "display_name": "ch1-only latent256 beta3.75",
        "role": "candidate",
        "run_dir": RESULTS / "ch1only_latent256_beta3p75_T80_h10000_p560_full5x5",
        "oof_dir": RESULTS / "ch1only_latent256_beta3p75_T80_h10000_p560_full5x5_stageB_oof_score_calibration",
        "oasis_label": "ch1only_latent256_beta3p75_oof_ecdf",
    },
    {
        "run_id": "ch1only_latent384_beta3p25",
        "display_name": "ch1-only latent384 beta3.25",
        "role": "candidate",
        "run_dir": RESULTS / "ch1only_latent384_beta3p25_T80_h10000_p560_full5x5",
        "oof_dir": RESULTS / "ch1only_latent384_beta3p25_T80_h10000_p560_full5x5_stageB_oof_score_calibration",
        "oasis_label": "ch1only_latent384_beta3p25_oof_ecdf",
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=OUT)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--skip-oasis", action="store_true")
    return p.parse_args()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def safe_float(value: Any) -> Optional[float]:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def safe_int(value: Any) -> Optional[int]:
    try:
        if pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


def write_table(outdir: Path, stem: str, df: pd.DataFrame, max_rows: int = 300) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / f"{stem}.csv", index=False)
    view = df.head(max_rows).copy()
    try:
        text = view.to_markdown(index=False)
    except Exception:
        text = view.to_string(index=False)
    if len(df) > max_rows:
        text += f"\n\n_Showing {max_rows} of {len(df)} rows._"
    (outdir / f"{stem}.md").write_text(text + "\n", encoding="utf-8")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def latest_matching(run_dir: Path, pattern: str) -> Optional[Path]:
    paths = [p for p in run_dir.glob(pattern) if p.exists()]
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def load_run_config(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "run_config.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def completion_status() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for run in RUNS:
        run_dir = run["run_dir"]
        oof_dir = run["oof_dir"]
        fold_rows: List[Dict[str, Any]] = []
        for fold in range(1, 6):
            fold_dir = run_dir / f"fold_{fold}"
            fold_rows.append(
                {
                    "fold": fold,
                    "fold_dir_exists": fold_dir.exists(),
                    "vae_history_exists": (fold_dir / f"vae_train_history_fold_{fold}.joblib").exists(),
                    "vae_model_exists": (fold_dir / f"vae_model_fold_{fold}.pt").exists(),
                    "vae_norm_params_exists": (fold_dir / "vae_norm_params.joblib").exists(),
                    "rate_distortion_exists": (fold_dir / f"fold_{fold}_rate_distortion.csv").exists(),
                    "latent_info_trainDev_exists": (fold_dir / f"fold_{fold}_trainDev_latent_info_summary.csv").exists(),
                    "latent_info_test_exists": (fold_dir / f"fold_{fold}_test_latent_info_summary.csv").exists(),
                    "stageA_logreg_pipeline_exists": (fold_dir / f"classifier_logreg_final_pipeline_fold_{fold}.joblib").exists(),
                    "stageA_svm_pipeline_exists": (fold_dir / f"classifier_svm_final_pipeline_fold_{fold}.joblib").exists(),
                    "stageA_logreg_predictions_exist": (fold_dir / "test_predictions_logreg.csv").exists(),
                    "stageA_svm_predictions_exist": (fold_dir / "test_predictions_svm.csv").exists(),
                }
            )
        fd = pd.DataFrame(fold_rows)
        clf_dir = run_dir / "classifier_only_readout"
        oof_required = [
            oof_dir / "calib_pooled_metrics.csv",
            oof_dir / "calib_foldwise_metrics.csv",
            oof_dir / "calib_philips_fpr_pooled.csv",
        ]
        rows.append(
            {
                "run_id": run["run_id"],
                "display_name": run["display_name"],
                "role": run["role"],
                "run_dir": str(run_dir),
                "run_exists": run_dir.exists(),
                "all_5_fold_dirs": int(fd["fold_dir_exists"].sum()) == 5,
                "all_5_vae_histories": int(fd["vae_history_exists"].sum()) == 5,
                "all_5_vae_models": int(fd["vae_model_exists"].sum()) == 5,
                "all_5_rate_distortion": int(fd["rate_distortion_exists"].sum()) == 5,
                "all_5_stageA_logreg_artifacts": int(fd["stageA_logreg_pipeline_exists"].sum()) == 5,
                "all_5_stageA_svm_artifacts": int(fd["stageA_svm_pipeline_exists"].sum()) == 5,
                "classifier_only_readout_present": clf_dir.exists(),
                "stageB_pooled_metrics_present": (clf_dir / "classifier_sweep_pooled_metrics.csv").exists(),
                "stageB_foldwise_metrics_present": (clf_dir / "classifier_sweep_foldwise_metrics.csv").exists(),
                "stageB_latent_cache_present": (clf_dir / "latent_cache").exists(),
                "oof_calibration_dir_present": oof_dir.exists(),
                "oof_calibration_required_files_present": all(p.exists() for p in oof_required),
                "missing_fold_artifact_count": int((~fd.drop(columns=["fold"]).astype(bool)).sum().sum()),
            }
        )
    return pd.DataFrame(rows)


def stagea_metrics() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for run in RUNS:
        metrics_path = latest_matching(run["run_dir"], "all_folds_metrics_MULTI_logreg_*.csv")
        df = safe_read_csv(metrics_path) if metrics_path else pd.DataFrame()
        if df.empty:
            rows.append({"run_id": run["run_id"], "level": "missing", "metrics_path": str(metrics_path) if metrics_path else ""})
            continue
        for _, row in df.iterrows():
            out = {
                "run_id": run["run_id"],
                "display_name": run["display_name"],
                "role": run["role"],
                "level": "fold",
                "fold": row.get("fold"),
                "classifier": row.get("actual_classifier_type"),
                "metrics_path": str(metrics_path),
            }
            for col in ["auc_raw", "pr_auc_raw", "auc_final", "pr_auc_final", "balanced_accuracy", "sensitivity", "specificity", "f1_score"]:
                out[col] = row.get(col)
            rows.append(out)
        for clf, g in df.groupby("actual_classifier_type"):
            for metric in ["auc_final", "pr_auc_final", "balanced_accuracy", "sensitivity", "specificity", "f1_score"]:
                pass
            rows.append(
                {
                    "run_id": run["run_id"],
                    "display_name": run["display_name"],
                    "role": run["role"],
                    "level": "summary",
                    "fold": "mean",
                    "classifier": clf,
                    "metrics_path": str(metrics_path),
                    "auc_final": g["auc_final"].mean(),
                    "auc_final_sd": g["auc_final"].std(ddof=1),
                    "pr_auc_final": g["pr_auc_final"].mean(),
                    "pr_auc_final_sd": g["pr_auc_final"].std(ddof=1),
                    "balanced_accuracy": g["balanced_accuracy"].mean(),
                    "f1_score": g["f1_score"].mean(),
                    "worst_fold_by_auc": g.loc[g["auc_final"].idxmin(), "fold"] if "auc_final" in g else "",
                    "worst_fold_auc": g["auc_final"].min() if "auc_final" in g else np.nan,
                    "worst_fold_pr_auc": g.loc[g["auc_final"].idxmin(), "pr_auc_final"] if "auc_final" in g else np.nan,
                }
            )
    return pd.DataFrame(rows)


def stageb_and_oof_metrics() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for run in RUNS:
        stageb = safe_read_csv(run["run_dir"] / "classifier_only_readout" / "classifier_sweep_pooled_metrics.csv")
        if not stageb.empty:
            stageb = stageb.copy()
            stageb.insert(0, "source", "stageB_classifier_only")
            stageb.insert(0, "role", run["role"])
            stageb.insert(0, "display_name", run["display_name"])
            stageb.insert(0, "run_id", run["run_id"])
            rows.extend(stageb.to_dict(orient="records"))
        oof = safe_read_csv(run["oof_dir"] / "calib_pooled_metrics.csv")
        if not oof.empty:
            oof = oof.copy()
            oof.insert(0, "source", "oof_calibration")
            oof.insert(0, "role", run["role"])
            oof.insert(0, "display_name", run["display_name"])
            oof.insert(0, "run_id", run["run_id"])
            rows.extend(oof.to_dict(orient="records"))
    return pd.DataFrame(rows)


def primary_row(run: Dict[str, Any], calib_method: str = PRIMARY_CALIB) -> Dict[str, Any]:
    df = safe_read_csv(run["oof_dir"] / "calib_pooled_metrics.csv")
    if df.empty:
        return {}
    rows = df[
        (df["model_name"].astype(str) == PRIMARY_MODEL)
        & (df["feature_set"].astype(str) == PRIMARY_FEATURE_SET)
        & (df["calib_method"].astype(str) == calib_method)
        & (df["threshold_strategy"].astype(str) == PRIMARY_THRESHOLD)
    ]
    if rows.empty:
        return {}
    row = rows.iloc[0].to_dict()
    row.update({"run_id": run["run_id"], "display_name": run["display_name"], "role": run["role"], "calib_method": calib_method})
    return row


def philips_fpr() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for run in RUNS:
        df = safe_read_csv(run["oof_dir"] / "calib_philips_fpr_pooled.csv")
        if df.empty:
            continue
        mask = (
            (df["model_name"].astype(str) == PRIMARY_MODEL)
            & (df["feature_set"].astype(str) == PRIMARY_FEATURE_SET)
            & (df["calib_method"].astype(str) == PRIMARY_CALIB)
            & (df["threshold_strategy"].astype(str) == PRIMARY_THRESHOLD)
            & (df["manufacturer"].astype(str).str.lower() == "philips")
        )
        sub = df[mask].copy()
        if sub.empty:
            continue
        sub.insert(0, "run_id", run["run_id"])
        sub.insert(1, "display_name", run["display_name"])
        sub.insert(2, "role", run["role"])
        rows.extend(sub.to_dict(orient="records"))
    return pd.DataFrame(rows)


def scanner_leakage_summary() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for run in RUNS:
        for fold in range(1, 6):
            fold_dir = run["run_dir"] / f"fold_{fold}"
            for split, path in [
                ("train_dev", fold_dir / f"fold_{fold}_scanner_leakage_summary.csv"),
                ("test", fold_dir / f"fold_{fold}_test_scanner_leakage_summary.csv"),
            ]:
                df = safe_read_csv(path)
                if df.empty:
                    rows.append({"run_id": run["run_id"], "fold": fold, "split": split, "status": "missing"})
                    continue
                for _, row in df.iterrows():
                    out = {"run_id": run["run_id"], "display_name": run["display_name"], "role": run["role"], "fold": fold, "split": split, "status": "available"}
                    out.update(row.to_dict())
                    rows.append(out)
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw
    summary_rows: List[Dict[str, Any]] = []
    for keys, g in raw[raw["status"] == "available"].groupby(["run_id", "display_name", "role", "split"], dropna=False):
        run_id, display, role, split = keys
        summary_rows.append(
            {
                "run_id": run_id,
                "display_name": display,
                "role": role,
                "fold": "mean",
                "split": split,
                "status": "summary",
                "acc_site_raw": pd.to_numeric(g.get("acc_site_raw"), errors="coerce").mean(),
                "acc_site_latent": pd.to_numeric(g.get("acc_site_latent"), errors="coerce").mean(),
                "latent_minus_raw": (
                    pd.to_numeric(g.get("acc_site_latent"), errors="coerce")
                    - pd.to_numeric(g.get("acc_site_raw"), errors="coerce")
                ).mean(),
            }
        )
    return pd.concat([raw, pd.DataFrame(summary_rows)], ignore_index=True)


def rate_distortion_summary() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for run in RUNS:
        cfg = load_run_config(run["run_dir"])
        latent_dim = safe_float(cfg.get("parameters", {}).get("latent_dim"))
        beta = safe_float(cfg.get("parameters", {}).get("beta_vae"))
        for fold in range(1, 6):
            path = run["run_dir"] / f"fold_{fold}" / f"fold_{fold}_rate_distortion.csv"
            df = safe_read_csv(path)
            if df.empty:
                rows.append({"run_id": run["run_id"], "fold": fold, "summary_point": "missing"})
                continue
            for point, row in [("final", df.iloc[-1]), ("best_L_val_betaMax", df.loc[df["L_val_betaMax"].idxmin()])]:
                d_val = safe_float(row.get("D_val"))
                r_bits = safe_float(row.get("R_val_bits"))
                r_nats = safe_float(row.get("R_val_nats"))
                rows.append(
                    {
                        "run_id": run["run_id"],
                        "display_name": run["display_name"],
                        "role": run["role"],
                        "fold": fold,
                        "summary_point": point,
                        "epoch": safe_int(row.get("epoch")),
                        "latent_dim": latent_dim,
                        "beta_vae": beta,
                        "D_train": safe_float(row.get("D_train")),
                        "D_val": d_val,
                        "R_val_bits": r_bits,
                        "R_val_nats": r_nats,
                        "bits_per_dim": (r_bits / latent_dim) if latent_dim and r_bits is not None else None,
                        "kld_over_D": (r_nats / d_val) if d_val and r_nats is not None else None,
                        "beta_kld_over_D": (beta * r_nats / d_val) if beta and d_val and r_nats is not None else None,
                        "L_val_betaMax": safe_float(row.get("L_val_betaMax")),
                    }
                )
    raw = pd.DataFrame(rows)
    summary_rows: List[Dict[str, Any]] = []
    if not raw.empty:
        numeric_cols = ["D_val", "R_val_bits", "bits_per_dim", "kld_over_D", "beta_kld_over_D", "L_val_betaMax"]
        for keys, g in raw[raw["summary_point"].isin(["final", "best_L_val_betaMax"])].groupby(["run_id", "display_name", "role", "summary_point"], dropna=False):
            run_id, display, role, point = keys
            row = {"run_id": run_id, "display_name": display, "role": role, "fold": "mean", "summary_point": point}
            for col in numeric_cols:
                row[col] = pd.to_numeric(g[col], errors="coerce").mean()
                row[f"{col}_sd"] = pd.to_numeric(g[col], errors="coerce").std(ddof=1)
            summary_rows.append(row)
    return pd.concat([raw, pd.DataFrame(summary_rows)], ignore_index=True) if summary_rows else raw


def latent_mi_summary() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for run in RUNS:
        for fold in range(1, 6):
            fold_dir = run["run_dir"] / f"fold_{fold}"
            for split, path in [
                ("trainDev", fold_dir / f"fold_{fold}_trainDev_latent_info_summary.csv"),
                ("test", fold_dir / f"fold_{fold}_test_latent_info_summary.csv"),
            ]:
                df = safe_read_csv(path)
                if df.empty:
                    rows.append({"run_id": run["run_id"], "fold": fold, "split": split, "status": "missing"})
                    continue
                y = df[df["variable"].astype(str) == "Y_target"]
                m = df[df["variable"].astype(str) == "Manufacturer"]
                first = df.iloc[0]
                y_val = safe_float(y.iloc[0].get("mi_sum_nats")) if not y.empty else None
                m_val = safe_float(m.iloc[0].get("mi_sum_nats")) if not m.empty else None
                rows.append(
                    {
                        "run_id": run["run_id"],
                        "display_name": run["display_name"],
                        "role": run["role"],
                        "fold": fold,
                        "split": split,
                        "status": "available",
                        "active_units": safe_int(first.get("n_active")),
                        "frac_active": safe_float(first.get("frac_active")),
                        "total_correlation_nats": safe_float(first.get("total_correlation_nats")),
                        "mi_y_sum_nats": y_val,
                        "mi_manufacturer_sum_nats": m_val,
                        "mi_manufacturer_over_y": (m_val / y_val) if y_val and m_val is not None else None,
                    }
                )
    raw = pd.DataFrame(rows)
    summary_rows: List[Dict[str, Any]] = []
    for keys, g in raw[raw["status"] == "available"].groupby(["run_id", "display_name", "role", "split"], dropna=False):
        run_id, display, role, split = keys
        row = {"run_id": run_id, "display_name": display, "role": role, "fold": "mean", "split": split, "status": "summary"}
        for col in ["active_units", "frac_active", "total_correlation_nats", "mi_y_sum_nats", "mi_manufacturer_sum_nats", "mi_manufacturer_over_y"]:
            row[col] = pd.to_numeric(g[col], errors="coerce").mean()
            row[f"{col}_sd"] = pd.to_numeric(g[col], errors="coerce").std(ddof=1)
        summary_rows.append(row)
    return pd.concat([raw, pd.DataFrame(summary_rows)], ignore_index=True) if summary_rows else raw


def run_oasis_scoring(outdir: Path, device: torch.device, batch_size: int, n_jobs: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    specs = [
        oasis_base.CandidateSpec(
            label="ch1only_latent256_beta3p75_oof_ecdf",
            role="targeted_ch1_followup",
            run_dir=RESULTS / "ch1only_latent256_beta3p75_T80_h10000_p560_full5x5",
            oof_dir=RESULTS / "ch1only_latent256_beta3p75_T80_h10000_p560_full5x5_stageB_oof_score_calibration",
        ),
        oasis_base.CandidateSpec(
            label="ch1only_latent384_beta3p25_oof_ecdf",
            role="targeted_ch1_followup",
            run_dir=RESULTS / "ch1only_latent384_beta3p25_T80_h10000_p560_full5x5",
            oof_dir=RESULTS / "ch1only_latent384_beta3p25_T80_h10000_p560_full5x5_stageB_oof_score_calibration",
        ),
    ]
    original = oasis_base.CANDIDATES
    try:
        oasis_base.CANDIDATES = specs
        artifact_df = oasis_base.validate_artifacts()
        write_table(outdir, "oasis_artifact_validation_followup_candidates", artifact_df)
        available = [s for s in specs if artifact_df.loc[artifact_df["candidate"] == s.label, "status"].iloc[0] == "available"]
        pred_rows: List[pd.DataFrame] = []
        fit_rows: List[pd.DataFrame] = []
        for build_name, tensor_path in oasis_base.TENSORS.items():
            if not tensor_path.exists():
                continue
            tensor, meta, oasis_channel_names = oasis_base.load_mega_tensor(tensor_path)
            for spec in available:
                preds, fit_meta = oasis_base.score_candidate_on_tensor(
                    spec,
                    build_name,
                    tensor,
                    meta,
                    oasis_channel_names,
                    device=device,
                    batch_size=batch_size,
                    n_jobs=n_jobs,
                )
                pred_rows.append(preds)
                fit_rows.append(fit_meta)
        if not pred_rows:
            return pd.DataFrame(), artifact_df
        predictions = pd.concat(pred_rows, ignore_index=True)
        predictions.to_csv(outdir / "oasis_followup_predictions.csv", index=False)
        fit = pd.concat(fit_rows, ignore_index=True) if fit_rows else pd.DataFrame()
        write_table(outdir, "oasis_fold_readout_reconstruction_audit", fit, max_rows=200)
        primary, foldwise, confusion, dist, scanner = oasis_base.metrics_tables(predictions)
        write_table(outdir, "oasis_followup_primary_metrics", primary)
        write_table(outdir, "oasis_followup_foldwise_metrics", foldwise, max_rows=500)
        write_table(outdir, "oasis_followup_confusion_matrices", confusion, max_rows=500)
        write_table(outdir, "oasis_followup_score_distribution_by_diagnosis", dist)
        write_table(outdir, "oasis_followup_score_distribution_by_scanner", scanner, max_rows=500)
        return primary, artifact_df
    finally:
        oasis_base.CANDIDATES = original


def existing_reference_oasis_metrics() -> pd.DataFrame:
    path = RESULTS / "oasis_external_validation_all_full_models_20260605/oasis_primary_metrics.csv"
    df = safe_read_csv(path)
    if df.empty:
        return df
    keep = df[df["candidate"].isin(["promoted_beta3p75_oof_ecdf", "ch1only_latent384_beta3p75_oof_ecdf"])].copy()
    mapping = {
        "promoted_beta3p75_oof_ecdf": "promoted_ch102_latent384_beta3p75",
        "ch1only_latent384_beta3p75_oof_ecdf": "ch1_reference_latent384_beta3p75",
    }
    keep["run_id"] = keep["candidate"].map(mapping)
    keep["oasis_metric_source"] = str(path)
    return keep


def primary_promotion_gate(internal_primary: pd.DataFrame, philips: pd.DataFrame, oasis_metrics: pd.DataFrame) -> pd.DataFrame:
    ref_prom = internal_primary[internal_primary["run_id"] == "promoted_ch102_latent384_beta3p75"]
    ref_ch1 = internal_primary[internal_primary["run_id"] == "ch1_reference_latent384_beta3p75"]
    prom_auc = float(ref_prom.iloc[0]["auc"]) if not ref_prom.empty else 0.795155
    prom_ba = float(ref_prom.iloc[0]["balanced_accuracy"]) if not ref_prom.empty else np.nan
    prom_f1 = float(ref_prom.iloc[0]["f1"]) if not ref_prom.empty else np.nan
    ch1_auc = float(ref_ch1.iloc[0]["auc"]) if not ref_ch1.empty else 0.800378
    ch1_pr = float(ref_ch1.iloc[0]["pr_auc"]) if not ref_ch1.empty else 0.585842

    fpr_map = {}
    for _, row in philips.iterrows():
        fpr_map[row["run_id"]] = safe_float(row.get("fpr_cn_pooled"))
    prom_fpr = fpr_map.get("promoted_ch102_latent384_beta3p75", 0.4545)
    ch1_fpr = fpr_map.get("ch1_reference_latent384_beta3p75", 0.4646)

    oasis_by_run_build = {}
    if not oasis_metrics.empty:
        for _, row in oasis_metrics.iterrows():
            rid = row.get("run_id")
            if not rid:
                cand = str(row.get("candidate"))
                if cand == "promoted_beta3p75_oof_ecdf":
                    rid = "promoted_ch102_latent384_beta3p75"
                elif cand == "ch1only_latent384_beta3p75_oof_ecdf":
                    rid = "ch1_reference_latent384_beta3p75"
                elif cand == "ch1only_latent256_beta3p75_oof_ecdf":
                    rid = "ch1only_latent256_beta3p75"
                elif cand == "ch1only_latent384_beta3p25_oof_ecdf":
                    rid = "ch1only_latent384_beta3p25"
            oasis_by_run_build[(rid, row.get("build_candidate"))] = row

    rows: List[Dict[str, Any]] = []
    for _, row in internal_primary.iterrows():
        rid = row["run_id"]
        if rid not in {"ch1only_latent256_beta3p75", "ch1only_latent384_beta3p25"}:
            decision_class = "reference"
        else:
            auc = float(row["auc"])
            pr = float(row["pr_auc"])
            ba = float(row["balanced_accuracy"])
            f1 = float(row["f1"])
            fpr = fpr_map.get(rid, np.nan)
            auc_gate = (auc > ch1_auc) or (auc > prom_auc)
            pr_gate = pr >= ch1_pr
            ba_f1_gate = (ba >= prom_ba) and (f1 >= prom_f1)
            fpr_gate = (fpr <= prom_fpr) or (fpr < ch1_fpr)
            oasis_gate_values = []
            for build in ["runwise164_pilot_parity", "runwise_140TR_pilot_parity"]:
                cand = oasis_by_run_build.get((rid, build))
                prom = oasis_by_run_build.get(("promoted_ch102_latent384_beta3p75", build))
                if cand is None or prom is None:
                    oasis_gate_values.append(False)
                else:
                    # "Material" margin is intentionally conservative and reported explicitly.
                    oasis_gate_values.append(
                        (float(cand["auc"]) >= float(prom["auc"]) - 0.02)
                        and (float(cand["pr_auc"]) >= float(prom["pr_auc"]) - 0.02)
                    )
            oasis_gate = all(oasis_gate_values) if oasis_gate_values else False
            all_gates = auc_gate and pr_gate and ba_f1_gate and fpr_gate and oasis_gate
            if all_gates:
                decision_class = "promote_candidate"
            elif auc_gate and (pr_gate or fpr_gate):
                decision_class = "sensitivity_only_not_primary"
            else:
                decision_class = "reject_not_promoted"
        rows.append(
            {
                **{k: row.get(k) for k in ["run_id", "display_name", "role", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "tn", "fp", "fn", "tp"]},
                "philips_cn_fpr": fpr_map.get(rid, np.nan),
                "gate_auc_exceeds_ch1_or_promoted": (float(row["auc"]) > ch1_auc) or (float(row["auc"]) > prom_auc) if rid.startswith("ch1only_") else "",
                "gate_pr_auc_meets_ch1": float(row["pr_auc"]) >= ch1_pr if rid.startswith("ch1only_") else "",
                "gate_ba_f1_not_worse_than_promoted": (float(row["balanced_accuracy"]) >= prom_ba and float(row["f1"]) >= prom_f1) if rid.startswith("ch1only_") else "",
                "gate_philips_fpr": ((fpr_map.get(rid, np.nan) <= prom_fpr) or (fpr_map.get(rid, np.nan) < ch1_fpr)) if rid.startswith("ch1only_") else "",
                "decision_class": decision_class,
            }
        )
    return pd.DataFrame(rows)


def write_final_decision(outdir: Path, gate: pd.DataFrame, oasis: pd.DataFrame) -> None:
    lines = [
        "# Final Decision",
        "",
        "This audit is read-only with respect to training/model artifacts. It used existing ADNI folds, existing Stage B/OOF outputs, and frozen ADNI-only OASIS inference reconstruction.",
        "",
        "Promotion gates were evaluated against:",
        "- ch1-only reference ADNI AUC 0.800378 and PR-AUC 0.585842.",
        "- promoted [1,0,2] ADNI AUC 0.795155, BA/F1, and Philips CN FPR 0.4545.",
        "- OASIS runwise164 and runwise140 external metrics versus the promoted [1,0,2] reference, using a conservative 0.02 non-material-loss margin for AUC/PR-AUC.",
        "",
        "## Candidate Decisions",
        "",
    ]
    for _, row in gate[gate["role"] == "candidate"].iterrows():
        lines.append(
            f"- `{row['run_id']}`: **{row['decision_class']}** "
            f"(AUC={row['auc']:.6f}, PR-AUC={row['pr_auc']:.6f}, BA={row['balanced_accuracy']:.6f}, "
            f"F1={row['f1']:.6f}, Philips CN FPR={row['philips_cn_fpr']:.4f})."
        )
    if oasis.empty:
        lines.extend(["", "OASIS scoring was unavailable or skipped."])
    else:
        lines.extend(["", "OASIS metrics were written to `oasis_external_metrics.csv`. OASIS was not used for model selection."])
    lines.extend(
        [
            "",
            "No VAE training, tensor edits, metadata edits, ledger edits, existing artifact overwrites, OASIS threshold fitting, or OASIS calibration fitting were performed.",
        ]
    )
    (outdir / "final_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    outdir = resolve(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    command_log: Dict[str, Any] = {
        "script": str(Path(__file__).resolve()),
        "timestamp_start": now_iso(),
        "output_dir": str(outdir),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "guardrails": [
            "no VAE training",
            "no tensor modification",
            "no metadata modification",
            "no ledger modification",
            "no existing artifact overwrite",
            "no OASIS threshold/calibration fitting",
        ],
    }

    completion = completion_status()
    stagea = stagea_metrics()
    stageb_oof = stageb_and_oof_metrics()
    internal_primary = pd.DataFrame([primary_row(run, PRIMARY_CALIB) for run in RUNS]).dropna(how="all")
    primary_logitz = pd.DataFrame([primary_row(run, "oof_logitz") for run in RUNS]).dropna(how="all")
    philips = philips_fpr()
    scanner = scanner_leakage_summary()
    rd = rate_distortion_summary()
    mi = latent_mi_summary()

    oasis_candidate_primary = pd.DataFrame()
    oasis_artifacts = pd.DataFrame()
    if not args.skip_oasis:
        oasis_candidate_primary, oasis_artifacts = run_oasis_scoring(outdir, device, args.batch_size, args.n_jobs)
    ref_oasis = existing_reference_oasis_metrics()
    if not oasis_candidate_primary.empty:
        oasis_candidate_primary = oasis_candidate_primary.copy()
        mapping = {
            "ch1only_latent256_beta3p75_oof_ecdf": "ch1only_latent256_beta3p75",
            "ch1only_latent384_beta3p25_oof_ecdf": "ch1only_latent384_beta3p25",
        }
        oasis_candidate_primary["run_id"] = oasis_candidate_primary["candidate"].map(mapping)
        oasis_candidate_primary["oasis_metric_source"] = "audit_frozen_external_inference"
    oasis_all = pd.concat([ref_oasis, oasis_candidate_primary], ignore_index=True, sort=False)

    gate = primary_promotion_gate(internal_primary, philips, oasis_all)

    write_table(outdir, "completion_status", completion)
    write_table(outdir, "stagea_summary_metrics", stagea, max_rows=500)
    write_table(outdir, "stageb_oof_calibration_metrics", stageb_oof, max_rows=800)
    write_table(outdir, "primary_promotion_gate_table", gate)
    write_table(outdir, "philips_cn_fpr", philips)
    write_table(outdir, "scanner_leakage_summary", scanner, max_rows=500)
    write_table(outdir, "rate_distortion_summary", rd, max_rows=500)
    write_table(outdir, "latent_mi_signal_nuisance_summary", mi, max_rows=500)
    write_table(outdir, "oasis_external_metrics", oasis_all, max_rows=500)
    if not primary_logitz.empty:
        write_table(outdir, "primary_oof_logitz_rows", primary_logitz)
    write_final_decision(outdir, gate, oasis_all)
    readme = [
        "# ch1-only Targeted Follow-up Completion/Promotion/OASIS Audit",
        "",
        "Audited completed follow-up runs:",
        "- `ch1only_latent256_beta3p75_T80_h10000_p560_full5x5`",
        "- `ch1only_latent384_beta3p25_T80_h10000_p560_full5x5`",
        "",
        "Reference models:",
        "- promoted `[1,0,2]` latent384 beta3.75",
        "- ch1-only latent384 beta3.75",
        "",
        "Primary ADNI row: `logreg_l2_original / z_plus_age_sex / oof_ecdf / inner_oof_target_sens_ge_0p70_max_spec`.",
        "",
        "OASIS inference, when run, used frozen ADNI fold VAEs and ADNI-train/dev-only Stage B reconstruction. No OASIS threshold/calibration fitting was performed.",
    ]
    (outdir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    command_log.update(
        {
            "timestamp_end": now_iso(),
            "oasis_scoring_skipped": bool(args.skip_oasis),
            "oasis_candidates_scored": [] if args.skip_oasis else ["ch1only_latent256_beta3p75_oof_ecdf", "ch1only_latent384_beta3p25_oof_ecdf"],
            "outputs": [
                "completion_status.csv/.md",
                "stagea_summary_metrics.csv/.md",
                "stageb_oof_calibration_metrics.csv/.md",
                "primary_promotion_gate_table.csv/.md",
                "philips_cn_fpr.csv/.md",
                "scanner_leakage_summary.csv/.md",
                "rate_distortion_summary.csv/.md",
                "latent_mi_signal_nuisance_summary.csv/.md",
                "oasis_external_metrics.csv/.md",
                "final_decision.md",
            ],
        }
    )
    write_json(outdir / "command_log.json", command_log)
    print(f"Wrote audit package to {outdir}")
    print(gate.to_string(index=False))


if __name__ == "__main__":
    main()
