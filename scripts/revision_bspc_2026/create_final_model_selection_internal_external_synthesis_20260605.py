#!/usr/bin/env python3
"""Create final model-selection synthesis from existing ADNI and OASIS artifacts.

Read-only with respect to model/data artifacts. Writes only the synthesis package.
No training, scoring, threshold fitting, or calibration fitting is performed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("results/revision_bspc_2026")
OUT = ROOT / "final_model_selection_internal_external_synthesis_20260605"

PRIMARY_MODEL = "logreg_l2_original"
PRIMARY_FEATURE_SET = "z_plus_age_sex"
PRIMARY_CALIB = "oof_ecdf"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
TARGET_BUILDS = ["concatenated_timeseries", "runwise_140TR_pilot_parity", "runwise164_pilot_parity"]


DIRECT_MODELS = [
    {
        "model_id": "promoted_latent384_beta3p75_ch1_0_2",
        "display_name": "Promoted [1,0,2] latent384 beta3.75",
        "role_flag": "primary",
        "decision": "promote",
        "run_dir": ROOT / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5",
        "oof_dir": ROOT / "recover035_latent384_beta3p75_stageB_oof_score_calibration",
        "channels": "[1,0,2]",
        "latent_dim": 384,
        "beta_vae": 3.75,
        "adni_source_type": "direct_oof_ecdf",
    },
    {
        "model_id": "ch1only_latent384_beta3p75",
        "display_name": "ch1-only latent384 beta3.75",
        "role_flag": "parsimony sensitivity",
        "decision": "sensitivity_only",
        "run_dir": ROOT / "recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5",
        "oof_dir": ROOT / "recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5_stageB_oof_score_calibration",
        "channels": "[1]",
        "latent_dim": 384,
        "beta_vae": 3.75,
        "adni_source_type": "direct_oof_ecdf",
    },
    {
        "model_id": "mfrBalancedVAE_latent384_beta3p75",
        "display_name": "Manufacturer-balanced VAE latent384 beta3.75",
        "role_flag": "deconfounding sensitivity",
        "decision": "sensitivity_only",
        "run_dir": ROOT / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5_mfrBalancedVAE",
        "oof_dir": ROOT / "recover035_latent384_beta3p75_mfrBalancedVAE_stageB_oof_score_calibration",
        "channels": "[1,0,2]",
        "latent_dim": 384,
        "beta_vae": 3.75,
        "adni_source_type": "direct_oof_ecdf",
    },
    {
        "model_id": "latent512_beta3p75",
        "display_name": "latent512 beta3.75",
        "role_flag": "capacity/beta sensitivity",
        "decision": "rejected",
        "run_dir": ROOT / "recover035_latent512_beta3p75_T80_h10000_p560_full5x5",
        "oof_dir": ROOT / "recover035_latent512_beta3p75_stageB_oof_score_calibration",
        "channels": "[1,0,2]",
        "latent_dim": 512,
        "beta_vae": 3.75,
        "adni_source_type": "direct_oof_ecdf",
    },
    {
        "model_id": "latent448_beta4p0",
        "display_name": "latent448 beta4.0",
        "role_flag": "capacity/beta sensitivity",
        "decision": "rejected",
        "run_dir": ROOT / "recover035_latent448_beta4p0_T80_h10000_p560_full5x5",
        "oof_dir": ROOT / "recover035_latent448_beta4p0_stageB_oof_score_calibration",
        "channels": "[1,0,2]",
        "latent_dim": 448,
        "beta_vae": 4.0,
        "adni_source_type": "direct_oof_ecdf",
    },
    {
        "model_id": "latent384_beta3p5",
        "display_name": "latent384 beta3.5",
        "role_flag": "capacity/beta sensitivity",
        "decision": "rejected",
        "run_dir": ROOT / "recover035_latent384_beta3p5_T80_h10000_p560_full5x5",
        "oof_dir": ROOT / "recover035_latent384_beta3p5_stageB_oof_score_calibration",
        "channels": "[1,0,2]",
        "latent_dim": 384,
        "beta_vae": 3.5,
        "adni_source_type": "direct_oof_ecdf",
    },
]


CAPACITY_MODELS = [
    {
        "model_id": "latent384_beta4p0",
        "capacity_run_id": "latent384_beta4p0",
        "display_name": "latent384 beta4.0",
        "role_flag": "capacity/beta sensitivity",
        "decision": "rejected",
    },
    {
        "model_id": "locked_v5_1b_latent256_beta2p5",
        "capacity_run_id": "locked_v5_1b_latent256_beta2p5",
        "display_name": "locked v5.1b latent256 beta2.5",
        "role_flag": "capacity/beta sensitivity",
        "decision": "rejected",
    },
    {
        "model_id": "recover035_latent256_beta2p5",
        "capacity_run_id": "recover035_latent256_beta2p5",
        "display_name": "recover035 latent256 beta2.5",
        "role_flag": "capacity/beta sensitivity",
        "decision": "rejected",
    },
]

OPTIONAL_CAPACITY_MODELS = [
    {
        "model_id": "latent128_beta1p25",
        "capacity_run_id": "latent128_beta1p25",
        "display_name": "latent128 beta1.25",
        "role_flag": "capacity/beta sensitivity",
        "decision": "rejected",
    },
    {
        "model_id": "latent128_beta2p5",
        "capacity_run_id": "latent128_beta2p5",
        "display_name": "latent128 beta2.5",
        "role_flag": "capacity/beta sensitivity",
        "decision": "rejected",
    },
]


OASIS_SOURCES = [
    {
        "path": ROOT / "oasis_mega_90_90_external_inference_model_panel_20260604" / "primary_metrics.csv",
        "map": {
            "promoted_beta3p75_oof_ecdf": "promoted_latent384_beta3p75_ch1_0_2",
            "mfrBalancedVAE_beta3p75_oof_ecdf": "mfrBalancedVAE_latent384_beta3p75",
            "latent512_beta3p75_oof_ecdf": "latent512_beta3p75",
            "promoted_beta3p75_residualized_mfr_oof_ecdf": "residualized_mfr_stageB",
        },
    },
    {
        "path": ROOT / "ch1only_latent384_beta3p75_oasis_mega_90_90_external_inference_20260605" / "primary_metrics.csv",
        "map": {
            "ch1only_latent384_beta3p75_oof_ecdf": "ch1only_latent384_beta3p75",
        },
    },
]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def md_table(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df.empty:
        return "_No rows._\n"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6g}")
    try:
        text = view.to_markdown(index=False)
    except Exception:
        text = view.to_string(index=False)
    if len(df) > max_rows:
        text += f"\n\n_Showing {max_rows} of {len(df)} rows._"
    return text + "\n"


def write_table(stem: str, df: pd.DataFrame, max_rows: int = 200) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{stem}.csv", index=False)
    (OUT / f"{stem}.md").write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


def scanner_leakage_mean(run_dir: Path) -> float:
    vals: list[float] = []
    for fold in range(1, 6):
        path = run_dir / f"fold_{fold}" / f"fold_{fold}_test_scanner_leakage_summary.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "acc_site_latent" in df.columns and not df.empty:
            vals.append(float(df["acc_site_latent"].iloc[0]))
    return float(np.mean(vals)) if vals else np.nan


def direct_oof_row(spec: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model_id": spec["model_id"],
        "display_name": spec["display_name"],
        "role_flag": spec["role_flag"],
        "decision": spec["decision"],
        "channels": spec.get("channels", ""),
        "latent_dim": spec.get("latent_dim"),
        "beta_vae": spec.get("beta_vae"),
        "adni_source_type": spec.get("adni_source_type", "direct_oof_ecdf"),
        "adni_source_path": str(spec["oof_dir"]),
    }
    path = spec["oof_dir"] / "calib_pooled_metrics.csv"
    if not path.exists():
        row["artifact_status"] = "missing_calib_pooled_metrics"
        return row
    df = pd.read_csv(path)
    match = df[
        (df["model_name"] == PRIMARY_MODEL)
        & (df["feature_set"] == PRIMARY_FEATURE_SET)
        & (df["calib_method"] == PRIMARY_CALIB)
        & (df["threshold_strategy"] == PRIMARY_THRESHOLD)
    ]
    if len(match) != 1:
        row["artifact_status"] = f"primary_row_count_{len(match)}"
        return row
    m = match.iloc[0]
    row.update(
        {
            "artifact_status": "available",
            "n": m.get("n"),
            "n_cn": m.get("n_cn"),
            "n_ad": m.get("n_ad"),
            "tn": m.get("tn"),
            "fp": m.get("fp"),
            "fn": m.get("fn"),
            "tp": m.get("tp"),
            "auc": m.get("auc"),
            "pr_auc": m.get("pr_auc"),
            "balanced_accuracy": m.get("balanced_accuracy"),
            "sensitivity": m.get("sensitivity"),
            "specificity": m.get("specificity"),
            "f1": m.get("f1"),
            "predicted_ad_rate": m.get("predicted_ad_rate"),
            "scanner_leakage_latent_ba": scanner_leakage_mean(spec["run_dir"]),
        }
    )
    fpr_path = spec["oof_dir"] / "calib_philips_fpr_pooled.csv"
    if fpr_path.exists():
        fpr = pd.read_csv(fpr_path)
        fpr_match = fpr[
            (fpr["model_name"] == PRIMARY_MODEL)
            & (fpr["feature_set"] == PRIMARY_FEATURE_SET)
            & (fpr["calib_method"] == PRIMARY_CALIB)
            & (fpr["threshold_strategy"] == PRIMARY_THRESHOLD)
            & (fpr["manufacturer"].astype(str).str.lower() == "philips")
        ]
        if len(fpr_match) == 1:
            r = fpr_match.iloc[0]
            row["philips_cn_n"] = r.get("n_cn_pooled")
            row["philips_cn_fp"] = r.get("fp_cn_pooled")
            row["philips_cn_fpr"] = r.get("fpr_cn_pooled")
    return row


def capacity_rows() -> pd.DataFrame:
    path = ROOT / "final_latent_capacity_beta_rate_distortion_synthesis_20260603" / "capacity_beta_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    cap = pd.read_csv(path)
    wanted = CAPACITY_MODELS + OPTIONAL_CAPACITY_MODELS
    rows: list[dict[str, Any]] = []
    for spec in wanted:
        sub = cap[cap["run_id"] == spec["capacity_run_id"]]
        if sub.empty:
            rows.append(
                {
                    "model_id": spec["model_id"],
                    "display_name": spec["display_name"],
                    "role_flag": spec["role_flag"],
                    "decision": spec["decision"],
                    "artifact_status": "missing_capacity_summary_row",
                    "adni_source_type": "capacity_beta_summary",
                    "adni_source_path": str(path),
                }
            )
            continue
        r = sub.iloc[0]
        if pd.notna(r.get("stageB_oof_ecdf_auc")):
            auc = r.get("stageB_oof_ecdf_auc")
            pr = r.get("stageB_oof_ecdf_pr_auc")
            ba = r.get("stageB_oof_ecdf_ba")
            sens = r.get("stageB_oof_ecdf_sens")
            specv = r.get("stageB_oof_ecdf_spec")
            f1 = r.get("stageB_oof_ecdf_f1")
            fpr = r.get("philips_cn_fpr_oof_ecdf")
            source_type = "capacity_summary_oof_ecdf"
        elif pd.notna(r.get("stageB_oof_logitz_auc")):
            auc = r.get("stageB_oof_logitz_auc")
            pr = r.get("stageB_oof_logitz_pr_auc")
            ba = r.get("stageB_oof_logitz_ba")
            sens = r.get("stageB_oof_logitz_sens")
            specv = r.get("stageB_oof_logitz_spec")
            f1 = r.get("stageB_oof_logitz_f1")
            fpr = r.get("philips_cn_fpr_oof_logitz")
            source_type = "capacity_summary_oof_logitz"
        else:
            auc = r.get("stageB_raw_auc")
            pr = r.get("stageB_raw_pr_auc")
            ba = r.get("stageB_raw_ba")
            sens = r.get("stageB_raw_sens")
            specv = r.get("stageB_raw_spec")
            f1 = r.get("stageB_raw_f1")
            fpr = r.get("philips_cn_fpr_raw")
            source_type = "capacity_summary_raw"
        rows.append(
            {
                "model_id": spec["model_id"],
                "display_name": spec["display_name"],
                "role_flag": spec["role_flag"],
                "decision": spec["decision"],
                "channels": r.get("channels_to_use"),
                "latent_dim": r.get("latent_dim"),
                "beta_vae": r.get("beta_vae"),
                "artifact_status": "available",
                "n": r.get("stageB_raw_n"),
                "n_cn": r.get("stageB_raw_n_cn"),
                "n_ad": r.get("stageB_raw_n_ad"),
                "tn": r.get("stageB_raw_tn"),
                "fp": r.get("stageB_raw_fp"),
                "fn": r.get("stageB_raw_fn"),
                "tp": r.get("stageB_raw_tp"),
                "auc": auc,
                "pr_auc": pr,
                "balanced_accuracy": ba,
                "sensitivity": sens,
                "specificity": specv,
                "f1": f1,
                "philips_cn_fpr": fpr if pd.notna(fpr) else r.get("primary_philips_cn_fpr"),
                "philips_cn_n": r.get("philips_cn_n_oof_ecdf") if pd.notna(r.get("philips_cn_n_oof_ecdf")) else r.get("philips_cn_n_raw"),
                "philips_cn_fp": r.get("philips_cn_fp_oof_ecdf") if pd.notna(r.get("philips_cn_fp_oof_ecdf")) else r.get("philips_cn_fp_raw"),
                "scanner_leakage_latent_ba": r.get("test_scanner_latent_ba_mean"),
                "adni_source_type": source_type,
                "adni_source_path": str(path),
            }
        )
    return pd.DataFrame(rows)


def residualized_rows() -> pd.DataFrame:
    path = ROOT / "promoted_beta3p75_stageB_latent_harmonization_by_manufacturer_20260602" / "pooled_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    sub = df[
        (df["harmonization_method"] == "residualize_mfr_preserve_age_sex")
        & (df["model_name"] == PRIMARY_MODEL)
        & (df["feature_set"] == PRIMARY_FEATURE_SET)
        & (df["calib_method"] == PRIMARY_CALIB)
        & (df["threshold_strategy"] == PRIMARY_THRESHOLD)
    ]
    if sub.empty:
        return pd.DataFrame()
    r = sub.iloc[0]
    return pd.DataFrame(
        [
            {
                "model_id": "residualized_mfr_stageB",
                "display_name": "Stage B manufacturer-residualized latent sensitivity",
                "role_flag": "deconfounding sensitivity",
                "decision": "sensitivity_only",
                "channels": "[1,0,2]",
                "latent_dim": 384,
                "beta_vae": 3.75,
                "artifact_status": "available",
                "n": r.get("n"),
                "n_cn": r.get("n_cn"),
                "n_ad": r.get("n_ad"),
                "tn": r.get("tn"),
                "fp": r.get("fp"),
                "fn": r.get("fn"),
                "tp": r.get("tp"),
                "auc": r.get("auc"),
                "pr_auc": r.get("pr_auc"),
                "balanced_accuracy": r.get("balanced_accuracy"),
                "sensitivity": r.get("sensitivity"),
                "specificity": r.get("specificity"),
                "f1": r.get("f1"),
                "predicted_ad_rate": r.get("predicted_ad_rate"),
                "philips_cn_n": r.get("philips_cn_n"),
                "philips_cn_fp": r.get("philips_cn_fp"),
                "philips_cn_fpr": r.get("philips_cn_fpr"),
                "scanner_leakage_latent_ba": r.get("mean_latent_manufacturer_ba"),
                "adni_source_type": "latent_harmonization_sensitivity",
                "adni_source_path": str(path),
            }
        ]
    )


def build_adni_table() -> pd.DataFrame:
    rows = [direct_oof_row(spec) for spec in DIRECT_MODELS]
    frames = [pd.DataFrame(rows), capacity_rows(), residualized_rows()]
    out = pd.concat([f for f in frames if not f.empty], ignore_index=True, sort=False)
    preferred_order = [
        "promoted_latent384_beta3p75_ch1_0_2",
        "ch1only_latent384_beta3p75",
        "mfrBalancedVAE_latent384_beta3p75",
        "latent512_beta3p75",
        "latent448_beta4p0",
        "latent384_beta3p5",
        "latent384_beta4p0",
        "locked_v5_1b_latent256_beta2p5",
        "recover035_latent256_beta2p5",
        "residualized_mfr_stageB",
        "latent128_beta1p25",
        "latent128_beta2p5",
    ]
    out["sort_order"] = out["model_id"].map({m: i for i, m in enumerate(preferred_order)}).fillna(999).astype(int)
    return out.sort_values(["sort_order", "model_id"]).drop(columns=["sort_order"])


def load_oasis_table(adni: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for src in OASIS_SOURCES:
        if not src["path"].exists():
            continue
        df = pd.read_csv(src["path"])
        df = df[df["build_candidate"].isin(TARGET_BUILDS)].copy()
        df["model_id"] = df["candidate"].map(src["map"])
        df = df[df["model_id"].notna()].copy()
        df["oasis_source_path"] = str(src["path"])
        frames.append(df)
    oasis = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if oasis.empty:
        return oasis
    meta_cols = ["model_id", "display_name", "role_flag", "decision", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]
    meta = adni[[c for c in meta_cols if c in adni.columns]].copy()
    meta = meta.rename(columns={c: f"adni_{c}" for c in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]})
    oasis = oasis.merge(meta, on="model_id", how="left")
    for metric in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]:
        oasis[f"generalization_drop_{metric}"] = oasis[metric] - oasis[f"adni_{metric}"]
    keep = [
        "model_id",
        "display_name",
        "role_flag",
        "decision",
        "build_candidate",
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
        "predicted_ad_rate",
        "adni_auc",
        "adni_pr_auc",
        "adni_balanced_accuracy",
        "generalization_drop_auc",
        "generalization_drop_pr_auc",
        "generalization_drop_balanced_accuracy",
        "oasis_source_path",
    ]
    return oasis[[c for c in keep if c in oasis.columns]].sort_values(["model_id", "build_candidate"])


def build_oasis_pivot(oasis: pd.DataFrame) -> pd.DataFrame:
    if oasis.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for model_id, g in oasis.groupby("model_id", dropna=False):
        row: dict[str, Any] = {
            "model_id": model_id,
            "display_name": g["display_name"].dropna().iloc[0] if g["display_name"].notna().any() else model_id,
            "role_flag": g["role_flag"].dropna().iloc[0] if g["role_flag"].notna().any() else "",
            "decision": g["decision"].dropna().iloc[0] if g["decision"].notna().any() else "",
        }
        for _, r in g.iterrows():
            b = r["build_candidate"]
            row[f"{b}_auc"] = r["auc"]
            row[f"{b}_pr_auc"] = r["pr_auc"]
            row[f"{b}_ba"] = r["balanced_accuracy"]
            row[f"{b}_sens"] = r["sensitivity"]
            row[f"{b}_spec"] = r["specificity"]
            row[f"{b}_f1"] = r["f1"]
            row[f"{b}_auc_drop"] = r["generalization_drop_auc"]
            row[f"{b}_pr_auc_drop"] = r["generalization_drop_pr_auc"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_availability_table(adni: pd.DataFrame, oasis: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in adni.iterrows():
        sub = oasis[oasis["model_id"] == r["model_id"]] if not oasis.empty else pd.DataFrame()
        rows.append(
            {
                "model_id": r["model_id"],
                "display_name": r["display_name"],
                "adni_available": r.get("artifact_status") == "available",
                "adni_source_type": r.get("adni_source_type"),
                "oasis_available": not sub.empty,
                "oasis_builds_available": ",".join(sorted(sub["build_candidate"].dropna().unique().tolist())) if not sub.empty else "",
                "notes": "No existing mega-OASIS artifact found; no scoring performed." if sub.empty else "",
            }
        )
    return pd.DataFrame(rows)


def write_interpretation(adni: pd.DataFrame, oasis: pd.DataFrame, pivot: pd.DataFrame) -> None:
    p = adni[adni["model_id"] == "promoted_latent384_beta3p75_ch1_0_2"].iloc[0]
    ch1 = adni[adni["model_id"] == "ch1only_latent384_beta3p75"].iloc[0]
    lines = [
        "# Final Model-Selection Interpretation",
        "",
        "The final primary ADNI model remains **recover035 [1,0,2] latent384 beta3.75** with the score-harmonized OOF-ECDF Stage B readout.",
        "",
        "## Why [1,0,2] Remains Primary",
        "",
        f"The promoted [1,0,2] model has ADNI AUC={p['auc']:.6f}, PR-AUC={p['pr_auc']:.6f}, BA={p['balanced_accuracy']:.6f}, sensitivity={p['sensitivity']:.6f}, specificity={p['specificity']:.6f}, and F1={p['f1']:.6f}. It is the internally promoted model because it remains the best balanced tradeoff across ranking, thresholded clinical operating behavior, scanner leakage, and Philips CN false-positive behavior among the controlled FULL runs.",
        "",
        "## Why ch1-only Is Sensitivity Only",
        "",
        f"The ch1-only branch improves ADNI AUC/PR-AUC numerically (AUC={ch1['auc']:.6f}, PR-AUC={ch1['pr_auc']:.6f}) but worsens specificity/F1 and Philips CN FPR internally. On mega-OASIS it does not beat the promoted [1,0,2] model on AUC or PR-AUC for any focus build. Therefore it is useful as a parsimonious/channel sensitivity model, not a co-primary model.",
        "",
        "## Deconfounding and Capacity/Beta Sensitivities",
        "",
        "Manufacturer-balanced VAE and manufacturer-residualized Stage B sensitivities reduce or target nuisance structure in different ways, but neither cleanly improves the internal promoted operating point. Latent512, latent448/beta4.0, beta3.5, beta4.0, and latent256/latent128 comparisons do not provide a controlled improvement over the promoted latent384 beta3.75 setting.",
        "",
        "## OASIS External Stress Test",
        "",
        "Mega-OASIS is a moderate external stress test rather than a model-selection set. The promoted model shows above-chance ranking on the runwise pilot-parity OASIS builds, but external transfer remains weaker than ADNI internal CV and varies by run-handling protocol. This supports reporting external transfer as moderate/unstable and motivates calibration/harmonization work, without invalidating the internal ADNI model.",
        "",
        "## Final Direction",
        "",
        "No additional internal FULL tuning is scientifically justified from this synthesis. The next defensible step is the pre-specified OASIS calibration/test protocol and reviewer-facing reporting of the controlled negative and sensitivity branches.",
    ]
    (OUT / "final_manuscript_ready_interpretation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    rec = [
        "# Final Recommendation",
        "",
        "- **Primary model:** promoted [1,0,2] latent384 beta3.75.",
        "- **Sensitivity models to report:** ch1-only parsimony model, Manufacturer-balanced/deconfounding sensitivities, and selected capacity/beta negative controls.",
        "- **Rejected/non-promoted:** ch1-only as primary, mfrBalancedVAE as primary, latent512, latent448 beta4.0, latent384 beta3.5/beta4.0, latent256 beta2.5 locked/recover035, and residualized Stage B as primary.",
        "- **External interpretation:** OASIS supports a moderate but unstable external ranking signal; it is not used for post-hoc model promotion.",
        "- **Next step:** locked OASIS calibration/test, not more internal FULL tuning.",
    ]
    (OUT / "final_recommendation.md").write_text("\n".join(rec) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    command_log = {
        "timestamp_start": now(),
        "argv": sys.argv,
        "guardrails": [
            "no training",
            "no scoring beyond reading existing OASIS artifacts",
            "no threshold fitting",
            "no calibration fitting",
            "no tensor/metadata/model artifact modification",
        ],
    }

    if args.dry_run:
        command_log["dry_run"] = True
        command_log["timestamp_end"] = now()
        (OUT / "command_log.json").write_text(json.dumps(command_log, indent=2) + "\n")
        print("dry_run_ok")
        return 0

    adni = build_adni_table()
    oasis = load_oasis_table(adni)
    oasis_pivot = build_oasis_pivot(oasis)
    availability = build_availability_table(adni, oasis)

    write_table("adni_primary_metrics", adni, max_rows=200)
    write_table("oasis_external_metrics", oasis, max_rows=300)
    write_table("adni_to_oasis_generalization_drop", oasis, max_rows=300)
    write_table("oasis_metrics_wide_by_model", oasis_pivot, max_rows=200)
    write_table("model_artifact_availability", availability, max_rows=200)
    write_interpretation(adni, oasis, oasis_pivot)

    readme = [
        "# Final Model-Selection Internal/External Synthesis",
        "",
        f"Generated: {now()}",
        "",
        "This package consolidates existing ADNI internal metrics and existing OASIS mega 90CN/90AD external metrics across the promoted model and controlled sensitivity branches.",
        "",
        "No training, scoring, threshold fitting, calibration fitting, tensor modification, metadata modification, or model artifact modification was performed.",
        "",
        "Primary convention for score-harmonized models: `logreg_l2_original / z_plus_age_sex / oof_ecdf / inner_oof_target_sens_ge_0p70_max_spec`.",
    ]
    (OUT / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    command_log["timestamp_end"] = now()
    command_log["dry_run"] = False
    command_log["outputs"] = sorted(p.name for p in OUT.iterdir())
    (OUT / "command_log.json").write_text(json.dumps(command_log, indent=2, default=str) + "\n")
    print(f"output_dir={OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
