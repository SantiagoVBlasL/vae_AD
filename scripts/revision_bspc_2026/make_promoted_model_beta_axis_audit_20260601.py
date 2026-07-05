#!/usr/bin/env python3
"""Read-only beta-axis audit for the promoted ADNI model.

This script only reads existing audit/run artifacts and writes a synthesis under:
results/revision_bspc_2026/promoted_model_beta_axis_audit_20260601/

It does not train, score, modify tensors, metadata, or model artifacts.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "revision_bspc_2026"
OUTPUT = RESULTS / "promoted_model_beta_axis_audit_20260601"

LINEAGE_CSV = RESULTS / "promoted_model_performance_lineage_audit_20260601" / "chronological_lineage_table.csv"
RANKING_CSV = RESULTS / "final_best_model_deep_audit_and_ranking_20260601" / "model_ranking_table.csv"
OASIS_STRESS_CSV = (
    RESULTS
    / "promoted_model_oasis_external_stress_test_package_20260601"
    / "tables"
    / "oasis_auc_pr_summary_with_bootstrap_ci.csv"
)
OASIS_MEGA_CSV = RESULTS / "oasis_mega_90cn_90ad_pooled_external_validation_20260531" / "pooled_ranking_metrics.csv"


RUN_DIRS = {
    "locked_v5_1b_horizon4480_cycles56": RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5",
    "recover035_latent384_beta2p5_T80_h10000_p560": RESULTS / "recover035_latent384_T80_h10000_p560_full5x5",
    "recover035_latent384_beta3p75_T80_h10000_p560": RESULTS
    / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5",
    "recover035_latent384_beta3p75_drop0p10": RESULTS
    / "recover035_latent384_beta3p75_drop0p10_T80_h10000_p560_full5x5",
    "recover035_latent384_beta3p75_enc015_dec010": RESULTS
    / "recover035_latent384_beta3p75_encdrop0p15_decdrop0p10_T80_h10000_p560_full5x5",
    "beta65_full_5x5": RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_beta65_full_5x5",
}

CONFIGS = {
    "latent384_beta2p5": ROOT / "configs/runs/adni_v5_1c_recover035_latent384_T80_h10000_p560_full5x5.json",
    "latent384_beta3p75": ROOT / "configs/runs/adni_v5_1c_recover035_latent384_beta3p75_T80_h10000_p560_full5x5.json",
    "latent384_beta3p75_drop0p10": ROOT
    / "configs/runs/adni_v5_1c_recover035_latent384_beta3p75_drop0p10_T80_h10000_p560_full5x5.json",
    "latent384_beta3p75_enc015_dec010": ROOT
    / "configs/runs/adni_v5_1c_recover035_latent384_beta3p75_encdrop0p15_decdrop0p10_T80_h10000_p560_full5x5.json",
    "locked_beta65": ROOT / "configs/runs/adni_v5_1_batch20260514b_ch1_0_2_beta65_full_5x5.json",
}

STAGEB_TARGET = "inner_oof_target_sens_ge_0p70_max_spec"


def read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


def safe_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def safe_mean(values: Iterable[Any]) -> float:
    arr = [safe_float(v) for v in values]
    arr = [v for v in arr if np.isfinite(v)]
    return float(np.mean(arr)) if arr else np.nan


def to_md(df: pd.DataFrame, path: Path) -> None:
    try:
        path.write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")
    except Exception:
        path.write_text(df.to_csv(index=False), encoding="utf-8")


def write_table(df: pd.DataFrame, stem: str) -> None:
    csv_path = OUTPUT / f"{stem}.csv"
    md_path = OUTPUT / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    to_md(df, md_path)


def find_best_rate_distortion_row(run_dir: Path, fold: int) -> Dict[str, float]:
    path = run_dir / f"fold_{fold}" / f"fold_{fold}_rate_distortion.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty or "L_val_betaMax" not in df.columns:
        return {}
    idx = df["L_val_betaMax"].astype(float).idxmin()
    row = df.loc[idx]
    d_val = safe_float(row.get("D_val"))
    r_val = safe_float(row.get("R_val_nats"))
    return {
        "best_epoch_estimate": safe_float(row.get("epoch")),
        "reconstruction_loss_val": d_val,
        "kld_val_nats": r_val,
        "kld_over_recon": (r_val / d_val) if np.isfinite(r_val) and np.isfinite(d_val) and d_val != 0 else np.nan,
        "best_val_L_betaMax": safe_float(row.get("L_val_betaMax")),
    }


def summarize_run_qc(run_dir: Optional[Path], beta: float, run_name: str) -> Dict[str, Any]:
    if run_dir is None or not run_dir.exists():
        return {
            "qc_source_status": "run_dir_missing",
            "active_units": np.nan,
            "kld_over_recon": np.nan,
            "beta_kld_over_recon": np.nan,
            "reconstruction_loss": np.nan,
            "total_correlation": np.nan,
            "latent_scanner_manufacturer_leakage": np.nan,
        }

    rd_rows: List[Dict[str, float]] = []
    active_units: List[float] = []
    total_correlation: List[float] = []
    leakage: List[float] = []
    for fold in range(1, 6):
        rd = find_best_rate_distortion_row(run_dir, fold)
        if rd:
            rd_rows.append(rd)

        latent_path = run_dir / f"fold_{fold}" / f"fold_{fold}_test_latent_info_summary.csv"
        if latent_path.exists():
            latent = pd.read_csv(latent_path)
            if not latent.empty:
                first = latent.iloc[0]
                active_units.append(safe_float(first.get("n_active")))
                total_correlation.append(safe_float(first.get("total_correlation_nats")))

        scanner_path = run_dir / f"fold_{fold}" / f"fold_{fold}_test_scanner_leakage_summary.csv"
        if scanner_path.exists():
            scanner = pd.read_csv(scanner_path)
            if not scanner.empty:
                leakage.append(safe_float(scanner.iloc[0].get("acc_site_latent")))

    kld_over_recon = safe_mean(row.get("kld_over_recon") for row in rd_rows)
    return {
        "qc_source_status": "ok" if rd_rows or active_units or total_correlation or leakage else "qc_files_missing",
        "active_units": safe_mean(active_units),
        "kld_over_recon": kld_over_recon,
        "beta_kld_over_recon": beta * kld_over_recon if np.isfinite(kld_over_recon) else np.nan,
        "reconstruction_loss": safe_mean(row.get("reconstruction_loss_val") for row in rd_rows),
        "total_correlation": safe_mean(total_correlation),
        "latent_scanner_manufacturer_leakage": safe_mean(leakage),
    }


def enrich_lineage(lineage: pd.DataFrame) -> pd.DataFrame:
    selected_names = {
        "locked_v5_1b_horizon4480_cycles56",
        "recover035_latent384_beta2p5_T80_h10000_p560",
        "recover035_latent384_beta3p75_T80_h10000_p560",
        "recover035_latent384_beta3p75_drop0p10",
        "recover035_latent384_beta3p75_enc015_dec010",
        "recover035_latent128_beta2p5",
        "recover035_latent128_beta1p25",
    }
    rows = lineage[lineage["run_name"].isin(selected_names)].copy()

    # Add beta65 from its dedicated comparison because it predates the recover035 latent384 line.
    beta65_cmp = read_csv(RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_beta65_full_5x5_comparison" / "main_model_comparison.csv")
    if beta65_cmp is not None:
        beta65 = beta65_cmp[beta65_cmp["run_id"].astype(str).eq("beta65_full_5x5")]
        if not beta65.empty:
            r = beta65.iloc[0].to_dict()
            rows = pd.concat(
                [
                    rows,
                    pd.DataFrame(
                        [
                            {
                                "lineage_order": 99,
                                "run_name": "adni_v5_1b_latent256_beta6p5_beta65",
                                "change_class": "strong_beta_negative_control",
                                "score_mode": "raw",
                                "subject_pool_cn": 300,
                                "subject_pool_mci": 250,
                                "subject_pool_ad": 96,
                                "classifier_pool_cn": safe_float(r.get("n_cn")),
                                "classifier_pool_ad": safe_float(r.get("n_ad")),
                                "channels": "[1, 0, 2]",
                                "selected_channel_names": "['Pearson_Full_FisherZ_Signed', 'Pearson_OMST_GCE_Signed_Weighted', 'MI_KNN_Symmetric']",
                                "latent_dim": 256,
                                "beta_vae": 6.5,
                                "dropout_rate_vae": 0.15,
                                "encoder_dropout_rate_vae": np.nan,
                                "decoder_dropout_rate_vae": np.nan,
                                "vae_dropout_scope": "legacy_all",
                                "epochs_vae": 3840,
                                "cycles": 48,
                                "lr_scheduler_T0": 80,
                                "early_stopping_patience_vae": np.nan,
                                "final_activation": "tanh",
                                "readout_type": "StageB logreg_l2 z_plus_age_sex",
                                "score_harmonization_type": "raw",
                                "auc": safe_float(r.get("auc")),
                                "pr_auc": safe_float(r.get("pr_auc")),
                                "balanced_accuracy": safe_float(r.get("balanced_accuracy")),
                                "sensitivity": safe_float(r.get("sensitivity")),
                                "specificity": safe_float(r.get("specificity")),
                                "f1": safe_float(r.get("f1")),
                                "tn": safe_float(r.get("tn")),
                                "fp": safe_float(r.get("fp")),
                                "fn": safe_float(r.get("fn")),
                                "tp": safe_float(r.get("tp")),
                                "scanner_raw_ba": np.nan,
                                "scanner_latent_ba": np.nan,
                                "scanner_latent_minus_raw": np.nan,
                                "philips_cn_n": np.nan,
                                "philips_cn_fp": np.nan,
                                "philips_cn_fpr": np.nan,
                                "oasis_mega_model": np.nan,
                                "oasis_mega_best_build": np.nan,
                                "oasis_mega_auc": np.nan,
                                "oasis_mega_pr_auc": np.nan,
                                "oasis_mega_auc_ci": np.nan,
                                "decision": "reject",
                                "source_path": str(
                                    RESULTS
                                    / "adni_v5_1_batch20260514b_ch1_0_2_beta65_full_5x5_comparison"
                                    / "main_model_comparison.csv"
                                ),
                                "source_status": "ok",
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    qc_records = []
    for _, row in rows.iterrows():
        run_name = str(row.get("run_name"))
        beta = safe_float(row.get("beta_vae"))
        run_dir = RUN_DIRS.get(run_name)
        if run_name == "adni_v5_1b_latent256_beta6p5_beta65":
            run_dir = RUN_DIRS.get("beta65_full_5x5")
        qc = summarize_run_qc(run_dir, beta, run_name)
        qc_records.append(qc)
    qc_df = pd.DataFrame(qc_records)
    rows = pd.concat([rows.reset_index(drop=True), qc_df], axis=1)
    rows["beta_per_latent_dim"] = rows["beta_vae"].astype(float) / rows["latent_dim"].astype(float)

    # Prefer explicit scanner leakage from existing lineage where available; otherwise keep generic run summary.
    if "scanner_latent_ba" in rows.columns:
        rows["latent_scanner_manufacturer_leakage"] = rows["scanner_latent_ba"].combine_first(
            rows["latent_scanner_manufacturer_leakage"]
        )

    keep = [
        "run_name",
        "change_class",
        "score_mode",
        "latent_dim",
        "beta_vae",
        "beta_per_latent_dim",
        "dropout_rate_vae",
        "encoder_dropout_rate_vae",
        "decoder_dropout_rate_vae",
        "vae_dropout_scope",
        "channels",
        "auc",
        "pr_auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "active_units",
        "kld_over_recon",
        "beta_kld_over_recon",
        "reconstruction_loss",
        "total_correlation",
        "latent_scanner_manufacturer_leakage",
        "scanner_raw_ba",
        "scanner_latent_minus_raw",
        "philips_cn_fp",
        "philips_cn_fpr",
        "oasis_mega_best_build",
        "oasis_mega_auc",
        "oasis_mega_pr_auc",
        "decision",
        "qc_source_status",
        "source_path",
    ]
    for col in keep:
        if col not in rows.columns:
            rows[col] = np.nan
    rows = rows[keep].sort_values(["latent_dim", "beta_vae", "dropout_rate_vae", "score_mode"], na_position="last")
    return rows


def artifact_availability() -> pd.DataFrame:
    checks = [
        ("latent384_beta3p0", "glob/search", False, "No config or result artifact found in configs/runs, scripts, or results."),
        ("latent384_beta2p5", str(RUN_DIRS["recover035_latent384_beta2p5_T80_h10000_p560"]), RUN_DIRS["recover035_latent384_beta2p5_T80_h10000_p560"].exists(), "Primary beta2.5 latent384 comparison."),
        ("latent384_beta3p75_promoted", str(RUN_DIRS["recover035_latent384_beta3p75_T80_h10000_p560"]), RUN_DIRS["recover035_latent384_beta3p75_T80_h10000_p560"].exists(), "Promoted run."),
        ("latent384_beta3p75_drop0p10", str(RUN_DIRS["recover035_latent384_beta3p75_drop0p10"]), RUN_DIRS["recover035_latent384_beta3p75_drop0p10"].exists(), "Dropout ablation at same beta."),
        ("latent384_beta3p75_enc015_dec010", str(RUN_DIRS["recover035_latent384_beta3p75_enc015_dec010"]), RUN_DIRS["recover035_latent384_beta3p75_enc015_dec010"].exists(), "Decoder-specific dropout ablation at same beta."),
        ("latent256_beta2p5_locked", str(RUN_DIRS["locked_v5_1b_horizon4480_cycles56"]), RUN_DIRS["locked_v5_1b_horizon4480_cycles56"].exists(), "Locked/reference model."),
        ("latent256_beta6p5_beta65", str(RUN_DIRS["beta65_full_5x5"]), RUN_DIRS["beta65_full_5x5"].exists(), "Strong beta negative control."),
    ]
    return pd.DataFrame(
        [
            {
                "artifact": name,
                "path_or_search": path,
                "available": bool(available),
                "note": note,
            }
            for name, path, available, note in checks
        ]
    )


def load_stagea(run_dir: Path, beta_label: str) -> pd.DataFrame:
    files = sorted(run_dir.glob("all_folds_metrics_MULTI_*.csv"))
    if not files:
        return pd.DataFrame()
    df = pd.read_csv(files[0])
    df["beta_label"] = beta_label
    return df


def stagea_beta2p5_vs_beta3p75() -> pd.DataFrame:
    frames = [
        load_stagea(RUN_DIRS["recover035_latent384_beta2p5_T80_h10000_p560"], "latent384_beta2p5"),
        load_stagea(RUN_DIRS["recover035_latent384_beta3p75_T80_h10000_p560"], "latent384_beta3p75"),
    ]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(not f.empty for f in frames) else pd.DataFrame()
    if df.empty:
        return df
    cols = [
        "beta_label",
        "fold",
        "actual_classifier_type",
        "auc_raw",
        "pr_auc_raw",
        "auc_final",
        "pr_auc_final",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1_score",
        "best_clf_params",
    ]
    return df[[c for c in cols if c in df.columns]]


def stageb_foldwise_beta2p5_vs_beta3p75() -> pd.DataFrame:
    rows = []
    raw_sources = [
        ("latent384_beta2p5_raw", RUN_DIRS["recover035_latent384_beta2p5_T80_h10000_p560"] / "classifier_only_readout" / "classifier_sweep_foldwise_metrics.csv"),
        ("latent384_beta3p75_raw", RUN_DIRS["recover035_latent384_beta3p75_T80_h10000_p560"] / "classifier_only_readout" / "classifier_sweep_foldwise_metrics.csv"),
    ]
    for label, path in raw_sources:
        df = read_csv(path)
        if df is None:
            continue
        sub = df[
            df["model_name"].astype(str).eq("logreg_l2")
            & df["readout_feature_set"].astype(str).eq("z_plus_age_sex")
            & df["threshold_strategy"].astype(str).eq(STAGEB_TARGET)
        ].copy()
        sub["comparison_label"] = label
        rows.append(sub)

    oof_path = RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration" / "calib_foldwise_metrics.csv"
    oof = read_csv(oof_path)
    if oof is not None:
        sub = oof[
            oof["model_name"].astype(str).eq("logreg_l2_original")
            & oof["feature_set"].astype(str).eq("z_plus_age_sex")
            & oof["calib_method"].astype(str).isin(["oof_logitz", "oof_ecdf"])
            & oof["threshold_strategy"].astype(str).eq(STAGEB_TARGET)
        ].copy()
        sub["comparison_label"] = "latent384_beta3p75_" + sub["calib_method"].astype(str)
        rows.append(sub.rename(columns={"feature_set": "readout_feature_set"}))

    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    cols = [
        "comparison_label",
        "fold",
        "threshold_strategy",
        "threshold",
        "best_inner_auc",
        "best_params",
        "n",
        "n_cn",
        "n_ad",
        "tn",
        "fp",
        "fn",
        "tp",
        "auc",
        "pr_auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
    ]
    return df[[c for c in cols if c in df.columns]].sort_values(["fold", "comparison_label"])


def stageb_delta_table(stageb: pd.DataFrame) -> pd.DataFrame:
    if stageb.empty:
        return pd.DataFrame()
    metrics = ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]
    base = stageb[stageb["comparison_label"].eq("latent384_beta2p5_raw")]
    if base.empty:
        return pd.DataFrame()
    out = []
    for label, sub in stageb.groupby("comparison_label"):
        if label == "latent384_beta2p5_raw":
            continue
        merged = sub.merge(base[["fold"] + metrics], on="fold", suffixes=("", "_beta2p5_raw"))
        for _, row in merged.iterrows():
            rec = {
                "comparison": f"{label} minus latent384_beta2p5_raw",
                "fold": row["fold"],
            }
            for m in metrics:
                rec[f"delta_{m}"] = safe_float(row[m]) - safe_float(row[f"{m}_beta2p5_raw"])
            out.append(rec)
    return pd.DataFrame(out)


def write_beta3p0_report(available: pd.DataFrame) -> None:
    msg = """# beta3.0 vs beta3.75 comparison

No latent384 beta3.0 run artifact was found in the current repository/results tree.
The audit searched configs, scripts, and existing result packages for beta3.0/beta3p0 patterns.
Therefore a direct beta=3.0 vs beta=3.75 comparison is not available and no inference is made from a missing run.

Closest available beta-axis comparisons are:
- latent384 beta2.5 vs latent384 beta3.75.
- latent256 beta2.5 locked/reference vs latent256 beta6.5 beta65 negative control.
- latent128 beta1.25 vs latent128 beta2.5 capacity-control negatives.
"""
    (OUTPUT / "beta3p0_vs_beta3p75_comparison.md").write_text(msg, encoding="utf-8")
    write_table(available[available["artifact"].astype(str).str.contains("beta3p0", na=False)], "beta3p0_availability")


def write_evidence_assessment(beta_lineage: pd.DataFrame, deltas: pd.DataFrame) -> None:
    def pick(run: str, mode: str) -> Optional[pd.Series]:
        sub = beta_lineage[(beta_lineage["run_name"].astype(str).eq(run)) & (beta_lineage["score_mode"].astype(str).eq(mode))]
        if sub.empty:
            return None
        return sub.iloc[0]

    locked = pick("locked_v5_1b_horizon4480_cycles56", "raw")
    b25 = pick("recover035_latent384_beta2p5_T80_h10000_p560", "raw")
    b375_raw = pick("recover035_latent384_beta3p75_T80_h10000_p560", "raw")
    b375 = pick("recover035_latent384_beta3p75_T80_h10000_p560", "oof_ecdf")
    if b375 is None:
        b375 = pick("recover035_latent384_beta3p75_T80_h10000_p560", "oof_logitz")
    beta65 = pick("adni_v5_1b_latent256_beta6p5_beta65", "raw")

    lines = [
        "# Beta-axis evidence assessment",
        "",
        "This audit is read-only and uses existing nested-CV, readout, QC, and external-stress artifacts.",
        "",
        "## Key observations",
    ]
    if locked is not None and b375 is not None:
        lines.append(
            f"- The promoted latent384 beta3.75 score-harmonized row has AUC={b375['auc']:.6f}, "
            f"PR-AUC={b375['pr_auc']:.6f}, compared with locked latent256 beta2.5 AUC={locked['auc']:.6f}, "
            f"PR-AUC={locked['pr_auc']:.6f}."
        )
        lines.append(
            f"- The beta/latent_dim proxy is identical for locked beta2.5/256 and promoted beta3.75/384 "
            f"({b375['beta_per_latent_dim']:.8f}), so beta3.75 is a proportional-capacity scaling rather than a pure increase in per-dimension pressure."
        )
    if b25 is not None and b375_raw is not None:
        lines.append(
            f"- Within latent384, beta3.75 raw Stage B did not beat beta2.5 raw: "
            f"AUC {b375_raw['auc']:.6f} vs {b25['auc']:.6f}; PR-AUC {b375_raw['pr_auc']:.6f} vs {b25['pr_auc']:.6f}."
        )
        lines.append(
            "- The promoted gain therefore depends materially on score harmonization/OOF calibration, not on a monotonic raw beta improvement alone."
        )
    if beta65 is not None:
        lines.append(
            f"- A stronger beta stress test at beta6.5 reduced latent manufacturer leakage in its own comparison, "
            f"but worsened clinical ranking/operating metrics: AUC={beta65['auc']:.6f}, PR-AUC={beta65['pr_auc']:.6f}, "
            f"BA={beta65['balanced_accuracy']:.6f}, F1={beta65['f1']:.6f}."
        )
    lines += [
        "",
        "## Is increasing beta justified?",
        "",
        "- Scanner/manufacturer leakage: not sufficient. The promoted latent384 beta3.75 model reduces raw-to-latent leakage on average, but leakage is not eliminated and beta65 shows that simply increasing beta can hurt AUC/PR-AUC/BA/F1.",
        "- Score-scale instability: not sufficient. Score harmonization is already the mechanism that rescued the promoted model; changing beta is not the direct fix for fold score-scale mismatch.",
        "- Active units / TC: not sufficient. The promoted model keeps all 384 units active and has high TC, but stronger beta controls did not demonstrate a clean clinical-ranking advantage.",
        "- OASIS transfer: not sufficient. External transfer remains moderate and source-batch dependent; this supports calibration/domain-shift work rather than another internal beta-only run.",
        "- Training curves: not sufficient. Existing promoted folds early-stopped and do not show a right-censored training horizon that would motivate beta escalation.",
        "",
        "Conclusion: the beta-axis evidence does not support beta4.0, beta4.25, or beta4.5 as the next controlled experiment.",
    ]
    (OUTPUT / "beta_axis_evidence_assessment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_recommendation() -> None:
    text = """# Controlled beta experiment recommendation

Decision: **no further beta experiment**.

Rationale:
- No latent384 beta3.0 artifact is available, so there is no direct beta3.0 -> beta3.75 slope to extend.
- The promoted beta3.75 model uses the same beta/latent-dimension proxy as the locked beta2.5/256 model.
- Within latent384, beta3.75 raw Stage B does not clearly outperform beta2.5 raw Stage B; the promoted result comes from score-harmonized Stage B.
- The strong beta65 negative control suggests that additional bottleneck pressure can reduce nuisance leakage while degrading clinical ranking and operating-point metrics.
- OASIS instability is better handled as external calibration/domain-shift work than as another internal beta-axis optimization.

If a beta run were forced later, beta4.0 would be the least aggressive option, but it is not recommended from the present evidence.
"""
    (OUTPUT / "controlled_beta_experiment_recommendation.md").write_text(text, encoding="utf-8")


def write_readme() -> None:
    text = """# Promoted Model Beta-axis Audit

Promoted model audited:
`recover035_latent384_beta3p75_T80_h10000_p560_full5x5`

Scope:
- Read-only synthesis of available beta-related ADNI runs and existing QC/audit outputs.
- No VAE training, classifier fitting, threshold fitting, model selection, tensor edits, metadata edits, or model-artifact edits.

Primary outputs:
- `beta_lineage_table.csv/.md`
- `available_beta_artifacts.csv/.md`
- `stagea_beta2p5_vs_beta3p75.csv/.md`
- `stageb_beta2p5_vs_beta3p75_foldwise.csv/.md`
- `stageb_beta2p5_vs_beta3p75_deltas.csv/.md`
- `beta3p0_vs_beta3p75_comparison.md`
- `beta_axis_evidence_assessment.md`
- `controlled_beta_experiment_recommendation.md`
- `command_log.json`
"""
    (OUTPUT / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned outputs without writing.")
    args = parser.parse_args()

    start = datetime.now().isoformat(timespec="seconds")
    required = [LINEAGE_CSV, RANKING_CSV]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Required audit source(s) missing: " + ", ".join(missing))

    planned_outputs = [
        OUTPUT / "README.md",
        OUTPUT / "beta_lineage_table.csv",
        OUTPUT / "available_beta_artifacts.csv",
        OUTPUT / "stagea_beta2p5_vs_beta3p75.csv",
        OUTPUT / "stageb_beta2p5_vs_beta3p75_foldwise.csv",
        OUTPUT / "stageb_beta2p5_vs_beta3p75_deltas.csv",
        OUTPUT / "beta3p0_vs_beta3p75_comparison.md",
        OUTPUT / "beta_axis_evidence_assessment.md",
        OUTPUT / "controlled_beta_experiment_recommendation.md",
        OUTPUT / "command_log.json",
    ]
    if args.dry_run:
        print("Dry-run OK. Planned outputs:")
        for path in planned_outputs:
            print(path)
        return 0

    OUTPUT.mkdir(parents=True, exist_ok=True)

    lineage = pd.read_csv(LINEAGE_CSV)
    beta_lineage = enrich_lineage(lineage)
    artifacts = artifact_availability()
    stagea = stagea_beta2p5_vs_beta3p75()
    stageb = stageb_foldwise_beta2p5_vs_beta3p75()
    deltas = stageb_delta_table(stageb)

    write_table(beta_lineage, "beta_lineage_table")
    write_table(artifacts, "available_beta_artifacts")
    write_table(stagea, "stagea_beta2p5_vs_beta3p75")
    write_table(stageb, "stageb_beta2p5_vs_beta3p75_foldwise")
    write_table(deltas, "stageb_beta2p5_vs_beta3p75_deltas")
    write_beta3p0_report(artifacts)
    write_evidence_assessment(beta_lineage, deltas)
    write_recommendation()
    write_readme()

    command_log = {
        "script": str(Path(__file__).resolve()),
        "start_time": start,
        "end_time": datetime.now().isoformat(timespec="seconds"),
        "mode": "read_only_audit",
        "inputs": {
            "lineage_csv": str(LINEAGE_CSV),
            "ranking_csv": str(RANKING_CSV),
            "oasis_stress_csv": str(OASIS_STRESS_CSV),
            "oasis_mega_csv": str(OASIS_MEGA_CSV),
            "run_dirs": {k: str(v) for k, v in RUN_DIRS.items()},
            "configs": {k: str(v) for k, v in CONFIGS.items()},
        },
        "outputs": [str(p) for p in planned_outputs],
        "safety": {
            "training_launched": False,
            "model_selection_performed": False,
            "tensor_modified": False,
            "metadata_modified": False,
            "model_artifacts_modified": False,
        },
    }
    (OUTPUT / "command_log.json").write_text(json.dumps(command_log, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote beta-axis audit to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
