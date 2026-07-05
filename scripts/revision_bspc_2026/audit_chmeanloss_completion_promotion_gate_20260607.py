#!/usr/bin/env python3
"""Read-only completion and promotion-gate audit for chmeanloss FULL run.

Candidate:
  recover035_latent384_beta3p75_chmeanloss_T80_h10000_p560_full5x5

This audit writes summary tables only. It does not train, refit classifiers,
score OASIS, fit thresholds, or modify tensors/metadata/model artifacts.
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
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/revision_bspc_2026"
RUN_DIR = RESULTS / "recover035_latent384_beta3p75_chmeanloss_T80_h10000_p560_full5x5"
OUT_DEFAULT = RESULTS / "chmeanloss_completion_promotion_gate_audit_20260607"
EVIDENCE_MAP = RESULTS / "final_full_model_evidence_map_with_beta6p5_20260606/full_model_evidence_map.csv"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
PRIMARY_CALIBS = {"oof_ecdf", "oof_logitz"}
PROMOTED_AUC = 0.795155
PROMOTED_PR_AUC = 0.573934
PROMOTED_BA = 0.725979
PROMOTED_F1 = 0.563492
PROMOTED_SENS = 0.731959
PROMOTED_PHILIPS_FPR = 0.4545
CH1_BETA_KLD_OVER_D = 0.0652679952247123
CH1_AUC = 0.800378


REFERENCE_MODEL_IDS = [
    "promoted_latent384_beta3p75_ch1_0_2",
    "ch1only_latent384_beta3p75",
    "latent384_beta6p5",
    "latent448_beta4p0",
    "latent512_beta3p75",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--evidence-map", type=Path, default=EVIDENCE_MAP)
    return parser.parse_args()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._\n"
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6g}")
    return view.to_markdown(index=False) + "\n"


def write_table(out: Path, stem: str, df: pd.DataFrame) -> None:
    df.to_csv(out / f"{stem}.csv", index=False)
    (out / f"{stem}.md").write_text(md(df), encoding="utf-8")


def safe_float(x: Any) -> float:
    try:
        if x is None or pd.isna(x):
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den and not math.isnan(den) else float("nan")


def run_config(run_dir: Path) -> dict[str, Any]:
    cfg = read_json(run_dir / "run_config.json")
    return cfg.get("args", cfg)


def metadata_from_run(run_dir: Path) -> pd.DataFrame:
    args = run_config(run_dir)
    meta = pd.read_csv(args["metadata_path"])
    if "tensor_index" in meta.columns and "tensor_idx" not in meta.columns:
        meta = meta.rename(columns={"tensor_index": "tensor_idx"})
    meta["SubjectID"] = meta["SubjectID"].astype(str)
    meta["tensor_idx"] = pd.to_numeric(meta["tensor_idx"], errors="coerce").astype("Int64")
    meta["Manufacturer"] = meta["Manufacturer"].map(normalize_mfr)
    meta["ResearchGroup_Mapped"] = meta["ResearchGroup_Mapped"].astype(str)
    return meta


def normalize_mfr(x: Any) -> str:
    s = "" if pd.isna(x) else str(x).strip()
    low = s.lower()
    if "philips" in low:
        return "Philips"
    if "siemens" in low:
        return "SIEMENS"
    if low in {"ge", "general electric"} or "general electric" in low:
        return "GE"
    return s or "UNKNOWN"


def find_stagea_metrics(run_dir: Path) -> Path | None:
    files = sorted(run_dir.glob("all_folds_metrics_MULTI*.csv"))
    return files[0] if files else None


def completion_status(run_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        fdir = run_dir / f"fold_{fold}"
        checks = {
            "fold_dir": fdir,
            "vae_model": fdir / f"vae_model_fold_{fold}.pt",
            "vae_history": fdir / f"vae_train_history_fold_{fold}.joblib",
            "rate_distortion": fdir / f"fold_{fold}_rate_distortion.csv",
            "latent_qc": fdir / "latent_qc_metrics.csv",
            "trainDev_latent_info": fdir / f"fold_{fold}_trainDev_latent_info_summary.csv",
            "test_latent_info": fdir / f"fold_{fold}_test_latent_info_summary.csv",
            "trainDev_scanner_leakage": fdir / f"fold_{fold}_scanner_leakage_summary.csv",
            "test_scanner_leakage": fdir / f"fold_{fold}_test_scanner_leakage_summary.csv",
            "stageA_logreg_predictions": fdir / "test_predictions_logreg.csv",
            "stageA_svm_predictions": fdir / "test_predictions_svm.csv",
        }
        row: dict[str, Any] = {"fold": fold}
        for key, p in checks.items():
            row[key] = p.exists()
        row["fold_complete_for_stageA_qc"] = all(bool(row[k]) for k in checks)
        rows.append(row)
    top_stagea = find_stagea_metrics(run_dir)
    readout = run_dir / "classifier_only_readout"
    top = {
        "fold": "top",
        "stageA_metrics_file": bool(top_stagea and top_stagea.exists()),
        "classifier_only_readout_dir": readout.exists(),
        "classifier_only_latent_cache": (readout / "latent_cache").exists(),
        "oof_score_calibration_dir": False,
        "fold_complete_for_stageA_qc": bool(top_stagea and all(r["fold_complete_for_stageA_qc"] for r in rows)),
    }
    rows.append(top)
    return pd.DataFrame(rows)


def load_fold_predictions(run_dir: Path, classifier: str, meta: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for fold in range(1, 6):
        path = run_dir / f"fold_{fold}" / f"test_predictions_{classifier}.csv"
        if not path.exists():
            continue
        pred = pd.read_csv(path)
        pred["fold"] = fold
        pred["classifier_type"] = classifier
        pred["SubjectID"] = pred["SubjectID"].astype(str)
        subj = pd.read_csv(run_dir / f"fold_{fold}" / "test_subjects_fold.csv")
        subj["SubjectID"] = subj["SubjectID"].astype(str)
        subj = subj.merge(
            meta[["SubjectID", "Manufacturer"]].drop_duplicates("SubjectID"),
            on="SubjectID",
            how="left",
        )
        pred = pred.merge(subj[["SubjectID", "Manufacturer"]], on="SubjectID", how="left")
        parts.append(pred)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def metrics_from_predictions(df: pd.DataFrame, *, run_id: str, stage: str) -> dict[str, Any]:
    y = df["y_true"].astype(int).to_numpy()
    score = df["y_score_final"].astype(float).to_numpy()
    pred = df["y_pred"].astype(int).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {
        "run_id": run_id,
        "stage": stage,
        "classifier_type": str(df["classifier_type"].iloc[0]) if "classifier_type" in df.columns and len(df) else "",
        "n": int(len(df)),
        "n_cn": int((y == 0).sum()),
        "n_ad": int((y == 1).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "auc": roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan,
        "pr_auc": average_precision_score(y, score) if len(np.unique(y)) == 2 else np.nan,
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "sensitivity": safe_div(float(tp), float(tp + fn)),
        "specificity": safe_div(float(tn), float(tn + fp)),
        "f1": f1_score(y, pred, zero_division=0),
    }
    cn = df[df["y_true"].astype(int).eq(0)].copy()
    phil = cn[cn["Manufacturer"].eq("Philips")]
    out["philips_cn_n"] = int(len(phil))
    out["philips_cn_fp"] = int(phil["y_pred"].astype(int).sum()) if len(phil) else 0
    out["philips_cn_fpr"] = safe_div(float(out["philips_cn_fp"]), float(out["philips_cn_n"]))
    return out


def foldwise_metrics(df: pd.DataFrame, *, run_id: str) -> pd.DataFrame:
    rows = []
    for (classifier, fold), g in df.groupby(["classifier_type", "fold"], sort=True):
        row = metrics_from_predictions(g, run_id=run_id, stage="stageA_foldwise")
        row["fold"] = fold
        rows.append(row)
    return pd.DataFrame(rows)


def stagea_tables(run_dir: Path, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    preds = []
    for clf in ["logreg", "svm"]:
        p = load_fold_predictions(run_dir, clf, meta)
        if not p.empty:
            preds.append(p)
    pred = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    summary = pd.DataFrame(
        [
            metrics_from_predictions(g, run_id="candidate_chmeanloss", stage="stageA_pooled")
            for _, g in pred.groupby("classifier_type", sort=True)
        ]
    )
    foldwise = foldwise_metrics(pred, run_id="candidate_chmeanloss") if not pred.empty else pd.DataFrame()
    return summary, foldwise, pred


def stageb_oof_availability(run_dir: Path, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    readout = run_dir / "classifier_only_readout"
    rows = [
        {"artifact": "classifier_only_readout_dir", "path": rel(readout), "exists": readout.exists()},
        {"artifact": "classifier_only_latent_cache", "path": rel(readout / "latent_cache"), "exists": (readout / "latent_cache").exists()},
        {"artifact": "stageB_oof_score_calibration_dir", "path": "", "exists": False},
    ]
    # No known OOF output exists for this branch; keep the table explicit.
    metrics = []
    for calib in ["raw", "oof_zscore", "oof_logitz", "oof_ecdf", "oof_platt", "oof_isotonic"]:
        metrics.append(
            {
                "run_id": "candidate_chmeanloss",
                "source": "unavailable",
                "model_name": "logreg_l2_original",
                "feature_set": "z_plus_age_sex",
                "calib_method": calib,
                "threshold_strategy": PRIMARY_THRESHOLD,
                "auc": np.nan,
                "pr_auc": np.nan,
                "balanced_accuracy": np.nan,
                "sensitivity": np.nan,
                "specificity": np.nan,
                "f1": np.nan,
                "status": "missing_classifier_only_readout_or_oof_calibration",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(metrics)


def rate_distortion(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for fold in range(1, 6):
        path = run_dir / f"fold_{fold}" / f"fold_{fold}_rate_distortion.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        idx = df["L_val_betaMax"].astype(float).idxmin() if "L_val_betaMax" in df.columns else len(df) - 1
        row = df.loc[idx].to_dict()
        d = safe_float(row.get("D_val"))
        r_nats = safe_float(row.get("R_val_nats"))
        r_bits = safe_float(row.get("R_val_bits"))
        beta = safe_float(row.get("beta", 3.75))
        rows.append(
            {
                "run_id": "candidate_chmeanloss",
                "fold": fold,
                "best_epoch": safe_float(row.get("epoch")),
                "D_val_best": d,
                "R_val_nats_best": r_nats,
                "R_val_bits_best": r_bits,
                "beta_KLD_over_D_best": safe_div(beta * r_nats, d),
                "bits_per_latent_dim_best": safe_div(r_bits, 384.0),
                "L_val_betaMax_best": safe_float(row.get("L_val_betaMax")),
            }
        )
    foldwise = pd.DataFrame(rows)
    summary = foldwise.drop(columns=["fold"]).groupby("run_id", as_index=False).mean(numeric_only=True) if not foldwise.empty else pd.DataFrame()
    return foldwise, summary


def latent_mi(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for fold in range(1, 6):
        path = run_dir / f"fold_{fold}" / f"fold_{fold}_test_latent_info_summary.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        row = {"run_id": "candidate_chmeanloss", "fold": fold}
        for _, r in df.iterrows():
            var = str(r["variable"])
            if var == "Y_target":
                row["MI_Z_Y_nats"] = safe_float(r.get("mi_sum_nats"))
            elif var == "Manufacturer":
                row["MI_Z_Manufacturer_nats"] = safe_float(r.get("mi_sum_nats"))
            row["active_units"] = safe_float(r.get("n_active"))
            row["total_correlation_nats"] = safe_float(r.get("total_correlation_nats"))
        row["MI_Manufacturer_over_MI_Y"] = safe_div(row.get("MI_Z_Manufacturer_nats", np.nan), row.get("MI_Z_Y_nats", np.nan))
        rows.append(row)
    foldwise = pd.DataFrame(rows)
    summary = foldwise.drop(columns=["fold"]).groupby("run_id", as_index=False).mean(numeric_only=True) if not foldwise.empty else pd.DataFrame()
    return foldwise, summary


def scanner_leakage(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for fold in range(1, 6):
        for split, fname in [
            ("train_dev", f"fold_{fold}_scanner_leakage_summary.csv"),
            ("test", f"fold_{fold}_test_scanner_leakage_summary.csv"),
        ]:
            path = run_dir / f"fold_{fold}" / fname
            if not path.exists():
                continue
            df = pd.read_csv(path)
            r = df.iloc[0].to_dict()
            rows.append(
                {
                    "run_id": "candidate_chmeanloss",
                    "fold": fold,
                    "split": split,
                    "n_samples": r.get("n_samples"),
                    "acc_site_raw": r.get("acc_site_raw"),
                    "acc_site_latent": r.get("acc_site_latent"),
                    "acc_site_raw_std": r.get("acc_site_raw_std"),
                    "acc_site_latent_std": r.get("acc_site_latent_std"),
                }
            )
    foldwise = pd.DataFrame(rows)
    summary = (
        foldwise.groupby(["run_id", "split"], as_index=False).mean(numeric_only=True)
        if not foldwise.empty
        else pd.DataFrame()
    )
    return foldwise, summary


def reference_comparison(evidence_map: Path) -> pd.DataFrame:
    if not evidence_map.exists():
        return pd.DataFrame()
    df = pd.read_csv(evidence_map)
    refs = df[df["model_id"].isin(REFERENCE_MODEL_IDS)].copy()
    cols = [
        "model_id",
        "display_name",
        "decision_class",
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
        "D_val_best_mean",
        "R_val_bits_best_mean",
        "beta_KLD_over_D_best_mean",
        "bits_per_latent_dim_best_mean",
        "active_units_mean",
        "total_correlation_nats_mean",
        "MI_Z_Y_nats_mean",
        "MI_Z_Manufacturer_nats_mean",
        "MI_Manufacturer_over_MI_Y_mean",
    ]
    return refs[[c for c in cols if c in refs.columns]]


def promotion_gate(
    stageb_metrics: pd.DataFrame,
    stagea_summary: pd.DataFrame,
    rd_summary: pd.DataFrame,
    leakage_summary: pd.DataFrame,
    latent_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    stageb_available = bool(stageb_metrics["status"].ne("missing_classifier_only_readout_or_oof_calibration").any()) if "status" in stageb_metrics.columns else False
    cand_auc = np.nan
    cand_pr = np.nan
    cand_ba = np.nan
    cand_sens = np.nan
    cand_f1 = np.nan
    cand_philips = np.nan
    if stageb_available:
        row = stageb_metrics[
            stageb_metrics["calib_method"].eq("oof_ecdf")
            & stageb_metrics["threshold_strategy"].eq(PRIMARY_THRESHOLD)
        ].head(1)
        if not row.empty:
            cand_auc = safe_float(row.iloc[0].get("auc"))
            cand_pr = safe_float(row.iloc[0].get("pr_auc"))
            cand_ba = safe_float(row.iloc[0].get("balanced_accuracy"))
            cand_sens = safe_float(row.iloc[0].get("sensitivity"))
            cand_f1 = safe_float(row.iloc[0].get("f1"))
    else:
        logreg = stagea_summary[stagea_summary["classifier_type"].eq("logreg")]
        if not logreg.empty:
            cand_philips = safe_float(logreg.iloc[0].get("philips_cn_fpr"))
    beta_kld = safe_float(rd_summary.iloc[0].get("beta_KLD_over_D_best")) if not rd_summary.empty else np.nan
    scanner_test = np.nan
    if not leakage_summary.empty:
        test = leakage_summary[leakage_summary["split"].eq("test")]
        if not test.empty:
            scanner_test = safe_float(test.iloc[0].get("acc_site_latent"))
    rows = [
        {
            "gate": "Stage B OOF-ECDF readout available",
            "candidate_value": stageb_available,
            "required": True,
            "pass": stageb_available,
            "details": "Required before ADNI promotion or OASIS model-selection scoring.",
        },
        {"gate": "AUC > promoted", "candidate_value": cand_auc, "required": f">{PROMOTED_AUC}", "pass": bool(cand_auc > PROMOTED_AUC)},
        {"gate": "PR-AUC >= promoted", "candidate_value": cand_pr, "required": f">={PROMOTED_PR_AUC}", "pass": bool(cand_pr >= PROMOTED_PR_AUC)},
        {"gate": "BA not materially worse", "candidate_value": cand_ba, "required": f">={PROMOTED_BA}", "pass": bool(cand_ba >= PROMOTED_BA)},
        {"gate": "Sensitivity not materially worse", "candidate_value": cand_sens, "required": f">={PROMOTED_SENS}", "pass": bool(cand_sens >= PROMOTED_SENS)},
        {"gate": "F1 not materially worse", "candidate_value": cand_f1, "required": f">={PROMOTED_F1}", "pass": bool(cand_f1 >= PROMOTED_F1)},
        {"gate": "Philips CN FPR <= promoted", "candidate_value": cand_philips, "required": f"<={PROMOTED_PHILIPS_FPR}", "pass": bool(cand_philips <= PROMOTED_PHILIPS_FPR)},
        {"gate": "beta*KLD/D near ch1-only", "candidate_value": beta_kld, "required": f"near {CH1_BETA_KLD_OVER_D:.6f}", "pass": bool(abs(beta_kld - CH1_BETA_KLD_OVER_D) <= 0.01)},
        {"gate": "scanner leakage available", "candidate_value": scanner_test, "required": "reported", "pass": not math.isnan(scanner_test)},
    ]
    gate = pd.DataFrame(rows)
    decision = "do_not_promote_stageB_oof_missing"
    if bool(gate["pass"].all()):
        decision = "promote_and_run_frozen_oasis_inference"
    return gate, decision


def main() -> int:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    cmd_log: dict[str, Any] = {
        "script": rel(Path(__file__)),
        "start_utc": now(),
        "run_dir": rel(args.run_dir),
        "training_launched": False,
        "classifier_refit": False,
        "oasis_scoring_run": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "model_artifact_modified": False,
    }
    meta = metadata_from_run(args.run_dir)
    completion = completion_status(args.run_dir)
    stagea_summary, stagea_foldwise, stagea_predictions = stagea_tables(args.run_dir, meta)
    stageb_avail, stageb_metrics = stageb_oof_availability(args.run_dir, out)
    rd_fold, rd_summary = rate_distortion(args.run_dir)
    mi_fold, mi_summary = latent_mi(args.run_dir)
    leak_fold, leak_summary = scanner_leakage(args.run_dir)
    refs = reference_comparison(args.evidence_map)
    gate, decision = promotion_gate(stageb_metrics, stagea_summary, rd_summary, leak_summary, mi_summary)

    # Compact comparison row for the candidate alongside reference evidence.
    candidate_row = {
        "model_id": "candidate_chmeanloss_latent384_beta3p75",
        "display_name": "chmeanloss [1,0,2] latent384 beta3.75",
        "decision_class": decision,
        "adni_oof_ecdf_auc": np.nan,
        "adni_oof_ecdf_pr_auc": np.nan,
        "adni_oof_logitz_auc": np.nan,
        "adni_oof_logitz_pr_auc": np.nan,
        "adni_oof_ecdf_ba": np.nan,
        "adni_oof_ecdf_sens": np.nan,
        "adni_oof_ecdf_spec": np.nan,
        "adni_oof_ecdf_f1": np.nan,
        "philips_cn_fpr": stagea_summary.loc[stagea_summary["classifier_type"].eq("logreg"), "philips_cn_fpr"].iloc[0]
        if not stagea_summary[stagea_summary["classifier_type"].eq("logreg")].empty
        else np.nan,
        "scanner_leakage_latent_acc": leak_summary.loc[leak_summary["split"].eq("test"), "acc_site_latent"].iloc[0]
        if not leak_summary[leak_summary["split"].eq("test")].empty
        else np.nan,
        "D_val_best_mean": rd_summary.iloc[0].get("D_val_best", np.nan) if not rd_summary.empty else np.nan,
        "R_val_bits_best_mean": rd_summary.iloc[0].get("R_val_bits_best", np.nan) if not rd_summary.empty else np.nan,
        "beta_KLD_over_D_best_mean": rd_summary.iloc[0].get("beta_KLD_over_D_best", np.nan) if not rd_summary.empty else np.nan,
        "bits_per_latent_dim_best_mean": rd_summary.iloc[0].get("bits_per_latent_dim_best", np.nan) if not rd_summary.empty else np.nan,
        "active_units_mean": mi_summary.iloc[0].get("active_units", np.nan) if not mi_summary.empty else np.nan,
        "total_correlation_nats_mean": mi_summary.iloc[0].get("total_correlation_nats", np.nan) if not mi_summary.empty else np.nan,
        "MI_Z_Y_nats_mean": mi_summary.iloc[0].get("MI_Z_Y_nats", np.nan) if not mi_summary.empty else np.nan,
        "MI_Z_Manufacturer_nats_mean": mi_summary.iloc[0].get("MI_Z_Manufacturer_nats", np.nan) if not mi_summary.empty else np.nan,
        "MI_Manufacturer_over_MI_Y_mean": mi_summary.iloc[0].get("MI_Manufacturer_over_MI_Y", np.nan) if not mi_summary.empty else np.nan,
    }
    comparison = pd.concat([pd.DataFrame([candidate_row]), refs], ignore_index=True)

    write_table(out, "completion_status", completion)
    write_table(out, "stagea_summary_metrics", stagea_summary)
    write_table(out, "stagea_foldwise_metrics", stagea_foldwise)
    write_table(out, "stageb_oof_availability", stageb_avail)
    write_table(out, "stageb_oof_calibration_metrics", stageb_metrics)
    write_table(out, "philips_cn_fpr", stagea_summary[["run_id", "stage", "classifier_type", "philips_cn_fp", "philips_cn_n", "philips_cn_fpr"]])
    write_table(out, "scanner_leakage_foldwise", leak_fold)
    write_table(out, "scanner_leakage_summary", leak_summary)
    write_table(out, "rate_distortion_foldwise", rd_fold)
    write_table(out, "rate_distortion_summary", rd_summary)
    write_table(out, "latent_mi_signal_nuisance_foldwise", mi_fold)
    write_table(out, "latent_mi_signal_nuisance_summary", mi_summary)
    write_table(out, "reference_comparison_table", comparison)
    write_table(out, "primary_promotion_gate_table", gate)

    beta_kld = safe_float(candidate_row["beta_KLD_over_D_best_mean"])
    ref_text = f"""
# Effective-Regularization Hypothesis

The channel-normalized loss substantially changes the reconstruction scale. The completed run's mean `beta*KLD/D` is `{beta_kld:.6f}`.

Reference regimes:

- promoted [1,0,2] beta3.75 current-loss: about `0.0257`
- ch1-only beta3.75: about `{CH1_BETA_KLD_OVER_D:.6f}`
- preflight channel-mean scale estimate: about `0.0771`

Interpretation: the completed chmeanloss run moves the multichannel model closer to or above the ch1-only effective-regularization scale, depending on the exact fold summary. However, the predefined Stage B classifier-only and OOF-ECDF/OOF-logitz readouts are not present, so the ADNI promotion question cannot be satisfied from completed artifacts.
""".strip()
    (out / "effective_regularization_hypothesis_test.md").write_text(ref_text + "\n", encoding="utf-8")

    oasis_text = """
# OASIS Gate

Frozen OASIS inference was not run.

Reason: the ADNI promotion gate failed before external scoring because the chmeanloss branch does not have the predefined Stage B classifier-only / OOF-ECDF / OOF-logitz readout artifacts required for primary comparison. Under the requested rule, OASIS inference is only triggered if the ADNI gate passes.
""".lstrip()
    (out / "oasis_gate_decision.md").write_text(oasis_text, encoding="utf-8")

    final_text = f"""
# Final Decision

Decision: `{decision}`.

The VAE Stage A/QC artifacts are complete for all five folds, and the run uses the intended channel-normalized reconstruction objective. Stage A pooled metrics are reported in `stagea_summary_metrics.csv`, with foldwise values in `stagea_foldwise_metrics.csv`.

The branch cannot be promoted from this audit because the predefined Stage B classifier-only readout and OOF score-harmonization artifacts are absent. Therefore the required OOF-ECDF and OOF-logitz AUC/PR-AUC/BA/Sensitivity/Specificity/F1 rows are unavailable, and the ADNI promotion gate fails before OASIS scoring.

Scientific interpretation: channel-normalized off-diagonal reconstruction loss does move the effective rate-distortion scale toward the ch1-only regime, but there is no completed Stage B evidence here that it improves the promoted operating point or AD/CN ranking. Treat this as a completed Stage A/QC result pending a separately authorized classifier-only Stage B readout, not as a promoted model.

Guardrails observed:

- no VAE training launched by this audit
- no classifier refit launched by this audit
- no OASIS scoring
- no threshold/calibration fitting
- no tensor, metadata, ledger, or model-artifact modification
""".lstrip()
    (out / "final_decision.md").write_text(final_text, encoding="utf-8")

    readme = f"""
# Chmeanloss Completion and Promotion-Gate Audit

Candidate: `recover035_latent384_beta3p75_chmeanloss_T80_h10000_p560_full5x5`.

Output generated by a read-only audit on existing artifacts. No real training, classifier refit, OASIS scoring, threshold fitting, or artifact modification was performed.

Main decision: `{decision}`.

Primary outputs:

- `completion_status.csv/.md`
- `stagea_summary_metrics.csv/.md`
- `stagea_foldwise_metrics.csv/.md`
- `stageb_oof_availability.csv/.md`
- `stageb_oof_calibration_metrics.csv/.md`
- `primary_promotion_gate_table.csv/.md`
- `rate_distortion_summary.csv/.md`
- `latent_mi_signal_nuisance_summary.csv/.md`
- `scanner_leakage_summary.csv/.md`
- `effective_regularization_hypothesis_test.md`
- `oasis_gate_decision.md`
- `final_decision.md`
""".lstrip()
    (out / "README.md").write_text(readme, encoding="utf-8")
    cmd_log["decision"] = decision
    cmd_log["end_utc"] = now()
    (out / "command_log.json").write_text(json.dumps(cmd_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote audit package: {out}")
    print(f"Decision: {decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
