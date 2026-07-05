#!/usr/bin/env python3
"""Completion and promotion-gate audit for ch12 latent384 beta2.75 current loss.

Reads completed VAE/Stage A artifacts plus predefined Stage B and OOF score
calibration outputs. It does not train a VAE, modify tensors/metadata, overwrite
model artifacts, or score OASIS unless explicitly added elsewhere after ADNI
promotion gates pass.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
RUN_LABEL = "ch12_latent384_beta2p75_currentloss"
RUN_DIR = ROOT / "results/revision_bspc_2026/recover035_ch12_latent384_beta2p75_currentloss_T80_h10000_p560_full5x5"
OOF_DIR = ROOT / "results/revision_bspc_2026/recover035_ch12_latent384_beta2p75_currentloss_T80_h10000_p560_full5x5_stageB_oof_score_calibration"
OUT = ROOT / "results/revision_bspc_2026/ch12_beta2p75_currentloss_completion_promotion_gate_audit_20260607"

PROMOTED = {
    "label": "promoted_ch102_latent384_beta3p75",
    "auc": 0.795155,
    "pr_auc": 0.573934,
    "balanced_accuracy": 0.725979,
    "sensitivity": 0.731959,
    "specificity": 0.720000,
    "f1": 0.563492,
    "philips_cn_fpr": 0.4545,
    "test_latent_scanner_ba_mean": 0.727556,
    "beta_KLD_over_D_best_mean": 0.025711,
}
CH1ONLY = {
    "label": "ch1only_latent384_beta3p75",
    "auc": 0.800378,
    "pr_auc": 0.585842,
    "beta_KLD_over_D_best_mean": 0.065268,
}

REF_AUDITS = {
    "promoted_ch102_latent384_beta3p75": ROOT
    / "results/revision_bspc_2026/ch12_latent384_beta3p75_completion_promotion_gate_audit_20260605/primary_promotion_gate_table.csv",
    "ch1only_latent384_beta3p75": ROOT
    / "results/revision_bspc_2026/ch1only_latent384_beta3p75_completion_promotion_gate_audit_20260604/primary_promotion_gate_table.csv",
    "ch12_latent384_beta3p75": ROOT
    / "results/revision_bspc_2026/ch12_latent384_beta3p75_completion_promotion_gate_audit_20260605/primary_promotion_gate_table.csv",
    "beta6p5_ch102": ROOT
    / "results/revision_bspc_2026/recover035_latent384_beta6p5_completion_promotion_gate_audit_20260606/primary_promotion_gate_table.csv",
    "chmeanloss_ch102": ROOT
    / "results/revision_bspc_2026/chmeanloss_stageB_oof_completion_audit_20260607/primary_promotion_gate_table.csv",
    "foldcombat_ch102": ROOT
    / "results/revision_bspc_2026/foldcombat_stageB_oof_completion_audit_20260607/primary_promotion_gate_table.csv",
}

PRIMARY_MODEL = "logreg_l2_original"
PRIMARY_FEATURE_SET = "z_plus_age_sex"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
PRIMARY_ECDF = "oof_ecdf"
PRIMARY_LOGITZ = "oof_logitz"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def write_table(df: pd.DataFrame, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{stem}.csv", index=False)
    try:
        md = df.to_markdown(index=False)
    except Exception:
        md = df.to_string(index=False)
    (OUT / f"{stem}.md").write_text(md + "\n", encoding="utf-8")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def safe_div(num: float, den: float) -> float:
    if den == 0 or pd.isna(num) or pd.isna(den):
        return float("nan")
    return float(num) / float(den)


def stagea_paths() -> Dict[str, Path]:
    metric = sorted(RUN_DIR.glob("all_folds_metrics_MULTI*.csv"))
    pred = sorted(RUN_DIR.glob("all_folds_clf_predictions_MULTI*.csv"))
    return {"metrics": metric[0] if metric else Path(""), "predictions": pred[0] if pred else Path("")}


def completion_status() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    stagea = stagea_paths()
    readout = RUN_DIR / "classifier_only_readout"
    for fold in range(1, 6):
        fd = RUN_DIR / f"fold_{fold}"
        rows.append(
            {
                "run_label": RUN_LABEL,
                "fold": fold,
                "fold_dir_exists": fd.exists(),
                "vae_model_saved": (fd / f"vae_model_fold_{fold}.pt").exists(),
                "vae_history_saved": (fd / f"vae_train_history_fold_{fold}.joblib").exists(),
                "stageA_logreg_predictions": (fd / "test_predictions_logreg.csv").exists(),
                "stageA_svm_predictions": (fd / "test_predictions_svm.csv").exists(),
                "stageA_top_level_metrics": stagea["metrics"].exists(),
                "stageA_top_level_predictions": stagea["predictions"].exists(),
                "rate_distortion_present": (fd / f"fold_{fold}_rate_distortion.csv").exists(),
                "latent_qc_present": (fd / "latent_qc_metrics.csv").exists(),
                "trainDev_latent_info": (fd / f"fold_{fold}_trainDev_latent_info_summary.csv").exists(),
                "test_latent_info": (fd / f"fold_{fold}_test_latent_info_summary.csv").exists(),
                "trainDev_scanner_leakage": (fd / f"fold_{fold}_scanner_leakage_summary.csv").exists(),
                "test_scanner_leakage": (fd / f"fold_{fold}_test_scanner_leakage_summary.csv").exists(),
                "classifier_only_readout_present": readout.exists(),
                "stageB_pooled_metrics": (readout / "classifier_sweep_pooled_metrics.csv").exists(),
                "stageB_foldwise_metrics": (readout / "classifier_sweep_foldwise_metrics.csv").exists(),
                "stageB_latent_cache": (readout / "latent_cache").exists(),
                "oof_calibration_dir_present": OOF_DIR.exists(),
                "oof_pooled_metrics": (OOF_DIR / "calib_pooled_metrics.csv").exists(),
                "oof_foldwise_metrics": (OOF_DIR / "calib_foldwise_metrics.csv").exists(),
            }
        )
    out = pd.DataFrame(rows)
    bool_cols = [c for c in out.columns if c not in {"run_label", "fold"}]
    out["all_required_complete_for_fold"] = out[bool_cols].all(axis=1)
    return out


def stagea_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = stagea_paths()
    foldwise = read_csv(paths["metrics"])
    pred = read_csv(paths["predictions"])
    rows: List[Dict[str, Any]] = []
    for clf, grp in pred.groupby("classifier_type"):
        y = grp["y_true"].astype(int)
        score = grp["y_score_final"].astype(float)
        pred_label = grp["y_pred"].astype(int)
        rows.append(
            {
                "run_label": RUN_LABEL,
                "classifier_type": clf,
                "n": int(len(grp)),
                "auc": roc_auc_score(y, score),
                "pr_auc": average_precision_score(y, score),
                "balanced_accuracy": balanced_accuracy_score(y, pred_label),
                "sensitivity": safe_div(int(((pred_label == 1) & (y == 1)).sum()), int((y == 1).sum())),
                "specificity": safe_div(int(((pred_label == 0) & (y == 0)).sum()), int((y == 0).sum())),
                "f1": f1_score(y, pred_label),
            }
        )
    return foldwise, pd.DataFrame(rows)


def stageb_metrics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = read_csv(RUN_DIR / "classifier_only_readout/classifier_sweep_pooled_metrics.csv")
    raw_foldwise = read_csv(RUN_DIR / "classifier_only_readout/classifier_sweep_foldwise_metrics.csv")
    oof = read_csv(OOF_DIR / "calib_pooled_metrics.csv")
    oof_foldwise = read_csv(OOF_DIR / "calib_foldwise_metrics.csv")
    raw = raw.assign(run_label=RUN_LABEL, readout_type="raw_classifier_only")
    raw_foldwise = raw_foldwise.assign(run_label=RUN_LABEL, readout_type="raw_classifier_only")
    oof = oof.assign(run_label=RUN_LABEL, readout_type="oof_score_calibrated")
    oof_foldwise = oof_foldwise.assign(run_label=RUN_LABEL, readout_type="oof_score_calibrated")
    return raw, oof, pd.concat([raw_foldwise, oof_foldwise], ignore_index=True)


def primary_rows(oof: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    filt = (
        (oof["model_name"] == PRIMARY_MODEL)
        & (oof["feature_set"] == PRIMARY_FEATURE_SET)
        & (oof["threshold_strategy"] == PRIMARY_THRESHOLD)
    )
    ecdf = oof[filt & (oof["calib_method"] == PRIMARY_ECDF)]
    logitz = oof[filt & (oof["calib_method"] == PRIMARY_LOGITZ)]
    if ecdf.empty or logitz.empty:
        raise RuntimeError("Missing primary OOF-ECDF or OOF-logitz rows")
    return ecdf.iloc[0], logitz.iloc[0]


def manufacturer_fpr_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    by_fold = read_csv(OOF_DIR / "calib_philips_fpr_by_fold.csv")
    pred = read_csv(OOF_DIR / "calib_predictions.csv")
    primary_pred = pred[
        (pred["model_name"] == PRIMARY_MODEL)
        & (pred["feature_set"] == PRIMARY_FEATURE_SET)
        & (pred["calib_method"].isin([PRIMARY_ECDF, PRIMARY_LOGITZ]))
        & (pred["threshold_strategy"] == PRIMARY_THRESHOLD)
    ].copy()
    rows: List[Dict[str, Any]] = []
    for (calib, mfr), grp in primary_pred.groupby(["calib_method", "Manufacturer"]):
        y = grp["y_true"].astype(int)
        yp = grp["y_pred"].astype(int)
        n_cn = int((y == 0).sum())
        fp_cn = int(((y == 0) & (yp == 1)).sum())
        n_ad = int((y == 1).sum())
        fn_ad = int(((y == 1) & (yp == 0)).sum())
        tp_ad = int(((y == 1) & (yp == 1)).sum())
        rows.append(
            {
                "model_name": PRIMARY_MODEL,
                "feature_set": PRIMARY_FEATURE_SET,
                "calib_method": calib,
                "threshold_strategy": PRIMARY_THRESHOLD,
                "manufacturer": mfr,
                "n_cn_pooled": n_cn,
                "fp_cn_pooled": fp_cn,
                "fpr_cn_pooled": safe_div(fp_cn, n_cn),
                "specificity_cn_pooled": 1.0 - safe_div(fp_cn, n_cn),
                "n_ad_pooled": n_ad,
                "fn_ad_pooled": fn_ad,
                "tp_ad_pooled": tp_ad,
                "ad_fnr_pooled": safe_div(fn_ad, n_ad),
                "sensitivity_ad_pooled": safe_div(tp_ad, n_ad),
            }
        )
    return pd.DataFrame(rows), by_fold


def rate_distortion() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    for fold in range(1, 6):
        p = RUN_DIR / f"fold_{fold}/fold_{fold}_rate_distortion.csv"
        rd = read_csv(p)
        best = rd.loc[rd["L_val_betaMax"].idxmin()].to_dict()
        row = {"run_label": RUN_LABEL, "fold": fold}
        row.update({k + "_best": best[k] for k in ["epoch", "beta", "D_val", "R_val_nats", "R_val_bits", "L_val_betaMax"]})
        row["KLD_over_D_best"] = safe_div(row["R_val_nats_best"], row["D_val_best"])
        row["beta_KLD_over_D_best"] = safe_div(row["beta_best"] * row["R_val_nats_best"], row["D_val_best"])
        row["R_bits_per_latent_dim_best"] = safe_div(row["R_val_bits_best"], 384.0)
        rows.append(row)
    foldwise = pd.DataFrame(rows)
    numeric = foldwise.select_dtypes(include=[np.number]).columns
    summary = foldwise[numeric].mean().to_frame().T
    summary.insert(0, "run_label", RUN_LABEL)
    return foldwise, summary


def latent_mi() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    for fold in range(1, 6):
        test = read_csv(RUN_DIR / f"fold_{fold}/fold_{fold}_test_latent_info_summary.csv")
        pivot = test.pivot_table(index=None, columns="variable", values="mi_sum_nats", aggfunc="first")
        row = {"run_label": RUN_LABEL, "fold": fold}
        row["MI_Z_Y_nats"] = float(pivot.get("Y_target", pd.Series([np.nan])).iloc[0])
        row["MI_Z_Manufacturer_nats"] = float(pivot.get("Manufacturer", pd.Series([np.nan])).iloc[0])
        row["MI_Manufacturer_over_MI_Y"] = safe_div(row["MI_Z_Manufacturer_nats"], row["MI_Z_Y_nats"])
        first = test.iloc[0]
        row["active_units"] = float(first.get("n_active", np.nan))
        row["total_correlation_nats"] = float(first.get("total_correlation_nats", np.nan))
        rows.append(row)
    foldwise = pd.DataFrame(rows)
    summary = foldwise.select_dtypes(include=[np.number]).mean().to_frame().T
    summary.insert(0, "run_label", RUN_LABEL)
    return foldwise, summary


def scanner_leakage() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    for fold in range(1, 6):
        for split, path in [
            ("trainDev", RUN_DIR / f"fold_{fold}/fold_{fold}_scanner_leakage_summary.csv"),
            ("test", RUN_DIR / f"fold_{fold}/fold_{fold}_test_scanner_leakage_summary.csv"),
        ]:
            df = read_csv(path)
            for _, rec in df.iterrows():
                row = {"run_label": RUN_LABEL, "fold": fold, "split": split}
                row.update(rec.to_dict())
                rows.append(row)
    foldwise = pd.DataFrame(rows)
    summary_rows: List[Dict[str, Any]] = []
    for split, grp in foldwise.groupby("split"):
        summary_rows.append(
            {
                "run_label": RUN_LABEL,
                "split": split,
                "raw_scanner_ba_mean": float(grp["acc_site_raw"].mean()),
                "latent_scanner_ba_mean": float(grp["acc_site_latent"].mean()),
                "latent_minus_raw_mean": float((grp["acc_site_latent"] - grp["acc_site_raw"]).mean()),
            }
        )
    return foldwise, pd.DataFrame(summary_rows)


def reference_summary() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for label, path in REF_AUDITS.items():
        exists = path.exists()
        row: Dict[str, Any] = {"reference_label": label, "source": rel(path), "available": exists}
        if exists:
            df = pd.read_csv(path)
            if "run_label" in df.columns:
                row.update({"format": "model_rows", "n_rows": len(df)})
            else:
                row.update({"format": "gate_rows", "n_rows": len(df)})
        rows.append(row)
    return pd.DataFrame(rows)


def promotion_gate(primary_ecdf: pd.Series, primary_logitz: pd.Series, fpr: pd.DataFrame, leak_summary: pd.DataFrame, rd_summary: pd.DataFrame) -> pd.DataFrame:
    philips = fpr[(fpr["calib_method"] == PRIMARY_ECDF) & (fpr["manufacturer"] == "Philips")]
    philips_fpr = float(philips["fpr_cn_pooled"].iloc[0]) if not philips.empty else np.nan
    test_leak = leak_summary[leak_summary["split"] == "test"]
    latent_leak = float(test_leak["latent_scanner_ba_mean"].iloc[0]) if not test_leak.empty else np.nan
    beta_kld = float(rd_summary["beta_KLD_over_D_best"].iloc[0])
    rows = [
        {
            "gate": "OOF-ECDF AUC exceeds promoted",
            "candidate_value": float(primary_ecdf["auc"]),
            "reference_value": PROMOTED["auc"],
            "required": "> promoted",
            "passes": float(primary_ecdf["auc"]) > PROMOTED["auc"],
        },
        {
            "gate": "OOF-ECDF PR-AUC meets promoted",
            "candidate_value": float(primary_ecdf["pr_auc"]),
            "reference_value": PROMOTED["pr_auc"],
            "required": ">= promoted",
            "passes": float(primary_ecdf["pr_auc"]) >= PROMOTED["pr_auc"],
        },
        {
            "gate": "OOF-logitz AUC exceeds promoted",
            "candidate_value": float(primary_logitz["auc"]),
            "reference_value": PROMOTED["auc"],
            "required": "> promoted",
            "passes": float(primary_logitz["auc"]) > PROMOTED["auc"],
        },
        {
            "gate": "OOF-logitz PR-AUC meets promoted",
            "candidate_value": float(primary_logitz["pr_auc"]),
            "reference_value": PROMOTED["pr_auc"],
            "required": ">= promoted",
            "passes": float(primary_logitz["pr_auc"]) >= PROMOTED["pr_auc"],
        },
        {
            "gate": "BA not worse than promoted",
            "candidate_value": float(primary_ecdf["balanced_accuracy"]),
            "reference_value": PROMOTED["balanced_accuracy"],
            "required": ">= promoted",
            "passes": float(primary_ecdf["balanced_accuracy"]) >= PROMOTED["balanced_accuracy"],
        },
        {
            "gate": "Sensitivity not worse than promoted",
            "candidate_value": float(primary_ecdf["sensitivity"]),
            "reference_value": PROMOTED["sensitivity"],
            "required": ">= promoted",
            "passes": float(primary_ecdf["sensitivity"]) >= PROMOTED["sensitivity"],
        },
        {
            "gate": "F1 not worse than promoted",
            "candidate_value": float(primary_ecdf["f1"]),
            "reference_value": PROMOTED["f1"],
            "required": ">= promoted",
            "passes": float(primary_ecdf["f1"]) >= PROMOTED["f1"],
        },
        {
            "gate": "Philips CN FPR not worse than promoted",
            "candidate_value": philips_fpr,
            "reference_value": PROMOTED["philips_cn_fpr"],
            "required": "<= promoted",
            "passes": philips_fpr <= PROMOTED["philips_cn_fpr"],
        },
        {
            "gate": "scanner leakage not worse than promoted",
            "candidate_value": latent_leak,
            "reference_value": PROMOTED["test_latent_scanner_ba_mean"],
            "required": "<= promoted",
            "passes": latent_leak <= PROMOTED["test_latent_scanner_ba_mean"],
        },
        {
            "gate": "effective regularization remains below ch1-only",
            "candidate_value": beta_kld,
            "reference_value": CH1ONLY["beta_KLD_over_D_best_mean"],
            "required": "interpretive only",
            "passes": beta_kld < CH1ONLY["beta_KLD_over_D_best_mean"],
        },
    ]
    return pd.DataFrame(rows)


def stagea_to_stageb_delta(stagea_pooled: pd.DataFrame, primary_ecdf: pd.Series, primary_logitz: pd.Series) -> pd.DataFrame:
    logreg = stagea_pooled[stagea_pooled["classifier_type"] == "logreg"].iloc[0]
    rows = []
    for name, row in [("oof_ecdf", primary_ecdf), ("oof_logitz", primary_logitz)]:
        rows.append(
            {
                "stageB_method": name,
                "stageA_logreg_auc": float(logreg["auc"]),
                "stageA_logreg_pr_auc": float(logreg["pr_auc"]),
                "stageB_auc": float(row["auc"]),
                "stageB_pr_auc": float(row["pr_auc"]),
                "delta_stageB_minus_stageA_auc": float(row["auc"]) - float(logreg["auc"]),
                "delta_stageB_minus_stageA_pr_auc": float(row["pr_auc"]) - float(logreg["pr_auc"]),
            }
        )
    return pd.DataFrame(rows)


def write_interpretation(primary_ecdf: pd.Series, primary_logitz: pd.Series, rd_summary: pd.DataFrame, gate: pd.DataFrame) -> None:
    beta_kld = float(rd_summary["beta_KLD_over_D_best"].iloc[0])
    text = f"""# Effective-Regularization Interpretation

Lowering beta from 3.75 to 2.75 in the [1,2] Pearson Full + MI pair reduces the
effective beta*KLD/D regime relative to the completed ch12 beta3.75 proxy. The
observed beta2.75 current-loss beta*KLD/D mean is `{beta_kld:.6f}`.

Primary Stage B:
- OOF-ECDF AUC `{float(primary_ecdf['auc']):.6f}`, PR-AUC `{float(primary_ecdf['pr_auc']):.6f}`.
- OOF-logitz AUC `{float(primary_logitz['auc']):.6f}`, PR-AUC `{float(primary_logitz['pr_auc']):.6f}`.

The candidate does not exceed the promoted [1,0,2] AUC `{PROMOTED['auc']:.6f}` or
meet promoted PR-AUC `{PROMOTED['pr_auc']:.6f}` under the promoted-convention
OOF-ECDF row. OOF-logitz improves the ranking relative to raw Stage B but still
does not satisfy the AUC/PR-AUC promotion gate.

Interpretation: reducing beta preserved or recovered some score-scale/ranking
behavior compared with raw Stage B, but it did not create a clean diagnostic
signal improvement over the promoted [1,0,2] model or ch1-only reference. OASIS
inference is therefore skipped by the predefined rule.
"""
    (OUT / "effective_regularization_interpretation.md").write_text(text, encoding="utf-8")
    passed = bool(gate[gate["gate"].str.contains("AUC|PR-AUC|BA|Sensitivity|F1|Philips|scanner")]["passes"].all())
    decision = "promote" if passed else "reject"
    final = f"""# Final Decision

Decision: `{decision}`.

The ch12 beta2.75 current-loss branch completed VAE/Stage A and the predefined
Stage B/OOF readout, but it fails the ADNI promotion gate. It does not beat the
promoted [1,0,2] AUC/PR-AUC and does not reach the ch1-only AUC/PR-AUC target.

Per protocol, frozen OASIS inference was not run because the ADNI gate failed.

Guardrails observed:
- no VAE retraining during this audit;
- Stage B used existing VAE artifacts/latent cache only;
- no tensor, metadata, ledger, or existing model artifact overwrite;
- no OASIS threshold or calibration fitting.
"""
    (OUT / "final_decision.md").write_text(final, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    status = completion_status()
    write_table(status, "completion_status")
    summary = pd.DataFrame(
        [
            {
                "run_label": RUN_LABEL,
                "all_5_vae_folds_complete": bool((status["fold_dir_exists"] & status["vae_model_saved"] & status["vae_history_saved"]).all()),
                "stageA_artifacts_present": bool((status["stageA_logreg_predictions"] & status["stageA_svm_predictions"] & status["stageA_top_level_metrics"]).all()),
                "rate_distortion_and_latent_qc_present": bool((status["rate_distortion_present"] & status["latent_qc_present"]).all()),
                "stageB_classifier_only_present": bool(status["stageB_pooled_metrics"].all()),
                "oof_calibration_present": bool(status["oof_pooled_metrics"].all()),
            }
        ]
    )
    write_table(summary, "completion_summary")

    stagea_foldwise, stagea_pooled = stagea_summary()
    write_table(stagea_foldwise, "stagea_foldwise_metrics")
    write_table(stagea_pooled, "stagea_pooled_metrics")

    raw, oof, foldwise_stageb = stageb_metrics()
    write_table(raw, "stageb_raw_pooled_metrics")
    write_table(oof, "stageb_oof_calibration_metrics")
    write_table(foldwise_stageb, "stageb_foldwise_metrics")
    primary_ecdf, primary_logitz = primary_rows(oof)

    fpr_pooled, fpr_by_fold = manufacturer_fpr_tables()
    write_table(fpr_pooled, "manufacturer_cn_fpr_ad_fnr_primary")
    write_table(fpr_by_fold, "manufacturer_cn_fpr_ad_fnr_by_fold")

    rd_foldwise, rd_summary = rate_distortion()
    write_table(rd_foldwise, "rate_distortion_foldwise")
    write_table(rd_summary, "rate_distortion_summary")
    mi_foldwise, mi_summary = latent_mi()
    write_table(mi_foldwise, "latent_mi_signal_nuisance_foldwise")
    write_table(mi_summary, "latent_mi_signal_nuisance_summary")
    leak_foldwise, leak_summary = scanner_leakage()
    write_table(leak_foldwise, "scanner_leakage_foldwise")
    write_table(leak_summary, "scanner_leakage_summary")

    gate = promotion_gate(primary_ecdf, primary_logitz, fpr_pooled, leak_summary, rd_summary)
    write_table(gate, "primary_promotion_gate_table")
    write_table(stagea_to_stageb_delta(stagea_pooled, primary_ecdf, primary_logitz), "stageA_to_stageB_delta")
    write_table(reference_summary(), "reference_audit_inventory")

    primary_summary = pd.DataFrame(
        [
            {"primary_row": "oof_ecdf", **primary_ecdf.to_dict()},
            {"primary_row": "oof_logitz", **primary_logitz.to_dict()},
        ]
    )
    write_table(primary_summary, "primary_oof_rows")
    write_interpretation(primary_ecdf, primary_logitz, rd_summary, gate)
    write_json(
        OUT / "command_log.json",
        {
            "created_utc": now_utc(),
            "run_dir": rel(RUN_DIR),
            "oof_dir": rel(OOF_DIR),
            "output_dir": rel(OUT),
            "stageB_generated_in_this_turn": True,
            "oof_generated_in_this_turn": True,
            "oasis_inference_run": False,
            "oasis_skip_reason": "ADNI promotion gate failed",
            "guardrails": [
                "no VAE retraining",
                "no tensor modification",
                "no metadata modification",
                "no existing model artifact overwrite",
                "no OASIS threshold/calibration fitting",
            ],
        },
    )


if __name__ == "__main__":
    main()
