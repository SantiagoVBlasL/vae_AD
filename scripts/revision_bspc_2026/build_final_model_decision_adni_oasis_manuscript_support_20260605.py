#!/usr/bin/env python3
"""Final read-only ADNI/OASIS model-decision and manuscript-support audit."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("results/revision_bspc_2026")
OUT = ROOT / "final_model_decision_adni_oasis_manuscript_support_20260605"
REGISTRY = ROOT / "full5x5_completed_run_registry_followup_plan_20260605" / "completed_full5x5_registry.csv"
CAPACITY = ROOT / "final_latent_capacity_beta_rate_distortion_synthesis_20260603" / "capacity_beta_summary.csv"
OASIS = ROOT / "oasis_external_validation_all_full_models_20260605"
HARMONIZATION = ROOT / "promoted_beta3p75_stageB_latent_harmonization_by_manufacturer_20260602"


MODEL_SPECS = [
    {
        "model_id": "promoted_latent384_beta3p75_ch1_0_2",
        "display_name": "promoted [1,0,2] latent384 beta3.75",
        "run_name": "recover035_latent384_beta3p75_T80_h10000_p560_full5x5",
        "capacity_run_id": "latent384_beta3p75_promoted",
        "decision_class": "primary final model",
        "final_decision": "primary",
        "rationale_short": "Best overall internally promoted multichannel model with balanced ADNI metrics and strongest available runwise OASIS transfer among primary candidates.",
    },
    {
        "model_id": "ch1only_latent384_beta3p75",
        "display_name": "ch1-only latent384 beta3.75",
        "run_name": "recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5",
        "capacity_run_id": None,
        "decision_class": "parsimony sensitivity",
        "final_decision": "not_promoted",
        "rationale_short": "Higher ADNI AUC/PR-AUC, but worse BA/F1, Philips CN FPR, and OASIS transfer than promoted [1,0,2].",
    },
    {
        "model_id": "latent384_beta3p5",
        "display_name": "latent384 beta3.5",
        "run_name": "recover035_latent384_beta3p5_T80_h10000_p560_full5x5",
        "capacity_run_id": "latent384_beta3p5",
        "decision_class": "beta sensitivity",
        "final_decision": "not_promoted",
        "rationale_short": "Lower ADNI AUC/PR-AUC/BA/F1 than beta3.75 and weaker OASIS transfer.",
    },
    {
        "model_id": "latent384_beta4p0",
        "display_name": "latent384 beta4.0",
        "run_name": "recover035_latent384_beta4p0_T80_h10000_p560_full5x5",
        "capacity_run_id": "latent384_beta4p0",
        "decision_class": "beta sensitivity",
        "final_decision": "not_promoted",
        "rationale_short": "Raw internal readout was below promoted beta3.75 and OOF-ECDF/OASIS artifacts were not available.",
    },
    {
        "model_id": "latent448_beta4p0",
        "display_name": "latent448 beta4.0",
        "run_name": "recover035_latent448_beta4p0_T80_h10000_p560_full5x5",
        "capacity_run_id": "latent448_beta4p0",
        "decision_class": "capacity sensitivity",
        "final_decision": "not_promoted",
        "rationale_short": "Lower ADNI OOF-ECDF AUC/PR-AUC than promoted and no OASIS improvement over promoted runwise builds.",
    },
    {
        "model_id": "latent512_beta3p75",
        "display_name": "latent512 beta3.75",
        "run_name": "recover035_latent512_beta3p75_T80_h10000_p560_full5x5",
        "capacity_run_id": "latent512_beta3p75",
        "decision_class": "capacity sensitivity",
        "final_decision": "not_promoted",
        "rationale_short": "Larger latent capacity reduced ADNI AUC/PR-AUC relative to promoted and did not improve OASIS transfer.",
    },
    {
        "model_id": "locked_v5_1b_latent256_beta2p5",
        "display_name": "locked v5.1b latent256 beta2.5",
        "run_name": "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5",
        "capacity_run_id": "locked_v5_1b_latent256_beta2p5",
        "decision_class": "reject / not promoted",
        "final_decision": "superseded_reference",
        "rationale_short": "Historical locked reference; superseded internally by recover035 latent384 beta3.75.",
    },
    {
        "model_id": "recover035_latent256_beta2p5",
        "display_name": "recover035 latent256 beta2.5",
        "run_name": "adni_v5_1_batch20260514b_ch1_0_2_recover035_full5x5",
        "capacity_run_id": "recover035_latent256_beta2p5",
        "decision_class": "reject / not promoted",
        "final_decision": "superseded_reference",
        "rationale_short": "Recovered-metadata latent256 reference; weaker internal performance than latent384 beta3.75.",
    },
    {
        "model_id": "mfrBalancedVAE_latent384_beta3p75",
        "display_name": "mfrBalancedVAE latent384 beta3.75",
        "run_name": "recover035_latent384_beta3p75_T80_h10000_p560_full5x5_mfrBalancedVAE",
        "capacity_run_id": None,
        "decision_class": "deconfounding sensitivity",
        "final_decision": "not_promoted",
        "rationale_short": "Manufacturer-balanced VAE did not meet ADNI promotion gates; retained as deconfounding sensitivity.",
    },
    {
        "model_id": "residualized_mfr_stageB",
        "display_name": "promoted latent residualized by Manufacturer",
        "run_name": "recover035_latent384_beta3p75_T80_h10000_p560_full5x5",
        "capacity_run_id": "latent384_beta3p75_promoted",
        "decision_class": "classifier-only harmonization sensitivity",
        "final_decision": "not_promoted",
        "rationale_short": "Post-hoc classifier-only harmonization sensitivity; OASIS advantage on concatenated build is not a model-selection criterion.",
    },
]


def md_write(df: pd.DataFrame, path: Path, max_rows: int | None = None) -> None:
    view = df if max_rows is None else df.head(max_rows)
    try:
        text = view.to_markdown(index=False)
    except Exception:
        text = view.to_string(index=False)
    path.write_text(text + "\n", encoding="utf-8")


def safe_read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def first_row(df: pd.DataFrame) -> pd.Series | None:
    return None if df.empty else df.iloc[0]


def get_value(row: pd.Series | None, key: str, default: Any = np.nan) -> Any:
    if row is None or key not in row.index:
        return default
    return row[key]


def oasis_wide(oasis_primary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if oasis_primary.empty:
        return pd.DataFrame()
    for model_id, g in oasis_primary.groupby("model_id", dropna=False):
        row: dict[str, Any] = {"model_id": model_id}
        for build, prefix in [
            ("concatenated_timeseries", "oasis_concatenated"),
            ("runwise164_pilot_parity", "oasis_runwise164"),
            ("runwise_140TR_pilot_parity", "oasis_runwise140"),
        ]:
            b = g[g["build_candidate"] == build]
            if b.empty:
                continue
            r = b.iloc[0]
            for m in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "predicted_ad_rate"]:
                row[f"{prefix}_{m}"] = r.get(m, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def residualized_adni_metrics() -> dict[str, Any]:
    path = HARMONIZATION / "pooled_metrics.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    rows = df[
        (df.get("harmonization_method", "") == "residualize_mfr_preserve_age_sex")
        & (df.get("model_name", "") == "logreg_l2_original")
        & (df.get("feature_set", "") == "z_plus_age_sex")
        & (df.get("calib_method", "") == "oof_ecdf")
        & (df.get("threshold_strategy", "") == "inner_oof_target_sens_ge_0p70_max_spec")
    ]
    if rows.empty:
        return {}
    r = rows.iloc[0]
    out = {
        "adni_oof_ecdf_auc": r.get("auc"),
        "adni_oof_ecdf_pr_auc": r.get("pr_auc"),
        "adni_oof_ecdf_balanced_accuracy": r.get("balanced_accuracy"),
        "adni_oof_ecdf_sensitivity": r.get("sensitivity"),
        "adni_oof_ecdf_specificity": r.get("specificity"),
        "adni_oof_ecdf_f1": r.get("f1"),
        "adni_oof_ecdf_philips_cn_fpr": r.get("philips_cn_fpr"),
        "adni_metric_source": "classifier_only_residualized_oof_ecdf",
    }
    if "mean_latent_manufacturer_ba" in r.index:
        out["adni_test_scanner_latent_acc_mean"] = r.get("mean_latent_manufacturer_ba")
    return out


def build_decision_table() -> pd.DataFrame:
    registry = safe_read(REGISTRY)
    capacity = safe_read(CAPACITY)
    oasis_registry = safe_read(OASIS / "oasis_model_registry.csv")
    oasis_primary = safe_read(OASIS / "oasis_primary_metrics.csv")
    ow = oasis_wide(oasis_primary)
    residualized = residualized_adni_metrics()

    rows: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        reg_row = first_row(registry[registry["run_name"].astype(str) == spec["run_name"]]) if not registry.empty else None
        cap_row = None
        if spec.get("capacity_run_id") and not capacity.empty:
            cap_row = first_row(capacity[capacity["run_id"].astype(str) == spec["capacity_run_id"]])
        oasis_reg_row = first_row(oasis_registry[oasis_registry["model_id"].astype(str) == spec["model_id"]]) if not oasis_registry.empty else None
        oasis_row = first_row(ow[ow["model_id"].astype(str) == spec["model_id"]]) if not ow.empty else None

        row: dict[str, Any] = {
            "model_id": spec["model_id"],
            "display_name": spec["display_name"],
            "decision_class": spec["decision_class"],
            "final_decision": spec["final_decision"],
            "rationale_short": spec["rationale_short"],
            "run_name": spec["run_name"],
            "run_dir": get_value(reg_row, "run_dir"),
            "completion_status": get_value(reg_row, "completion_status"),
            "channel_set_order": get_value(reg_row, "channel_set_order", get_value(cap_row, "channels_to_use")),
            "selected_channel_names": get_value(reg_row, "selected_channel_names", get_value(cap_row, "selected_channel_names")),
            "beta_vae": get_value(reg_row, "beta_vae", get_value(cap_row, "beta_vae")),
            "latent_dim": get_value(reg_row, "latent_dim", get_value(cap_row, "latent_dim")),
            "adni_metric_source": "oof_ecdf" if pd.notna(get_value(reg_row, "oof_ecdf_auc")) else "oof_ecdf_not_available",
            "adni_oof_ecdf_auc": get_value(reg_row, "oof_ecdf_auc"),
            "adni_oof_ecdf_pr_auc": get_value(reg_row, "oof_ecdf_pr_auc"),
            "adni_oof_ecdf_balanced_accuracy": get_value(reg_row, "oof_ecdf_balanced_accuracy"),
            "adni_oof_ecdf_sensitivity": get_value(reg_row, "oof_ecdf_sensitivity"),
            "adni_oof_ecdf_specificity": get_value(reg_row, "oof_ecdf_specificity"),
            "adni_oof_ecdf_f1": get_value(reg_row, "oof_ecdf_f1"),
            "philips_cn_fpr": get_value(reg_row, "oof_ecdf_philips_cn_fpr"),
            "scanner_leakage_latent_acc": get_value(reg_row, "test_scanner_latent_acc_mean"),
            "adni_stageb_raw_auc": get_value(reg_row, "stageB_raw_auc"),
            "adni_stageb_raw_pr_auc": get_value(reg_row, "stageB_raw_pr_auc"),
            "adni_stageb_raw_balanced_accuracy": get_value(reg_row, "stageB_raw_balanced_accuracy"),
            "adni_stageb_raw_sensitivity": get_value(reg_row, "stageB_raw_sensitivity"),
            "adni_stageb_raw_specificity": get_value(reg_row, "stageB_raw_specificity"),
            "adni_stageb_raw_f1": get_value(reg_row, "stageB_raw_f1"),
            "oasis_artifact_status": get_value(oasis_reg_row, "oasis_artifact_status"),
            "oasis_source_family": get_value(oasis_reg_row, "source_family"),
        }
        # Prefer capacity synthesis rate-distortion where available; fall back to registry.
        row.update(
            {
                "D_val_best_mean": get_value(cap_row, "D_val_best_mean", get_value(reg_row, "D_val_mean")),
                "R_val_bits_best_mean": get_value(cap_row, "R_val_bits_best_mean", get_value(reg_row, "R_val_bits_mean")),
                "bits_per_latent_dim_best_mean": get_value(cap_row, "bits_per_latent_dim_best_mean", get_value(reg_row, "R_bits_per_latent_dim_mean")),
                "beta_KLD_over_D_best_mean": get_value(cap_row, "beta_KLD_over_D_best_mean", get_value(reg_row, "beta_KLD_over_D_mean")),
                "active_units_mean": get_value(cap_row, "active_units_mean", get_value(reg_row, "active_units_mean")),
                "total_correlation_nats_mean": get_value(cap_row, "total_correlation_nats_mean", get_value(reg_row, "total_correlation_nats_mean")),
                "MI_Z_Y_nats_mean": get_value(cap_row, "MI_Z_Y_nats_mean", get_value(reg_row, "MI_Z_Y_nats_mean")),
                "MI_Z_Manufacturer_nats_mean": get_value(cap_row, "MI_Z_Manufacturer_nats_mean", get_value(reg_row, "MI_Z_Manufacturer_nats_mean")),
                "MI_Manufacturer_over_MI_Y_mean": get_value(cap_row, "MI_Manufacturer_over_MI_Y_mean", get_value(reg_row, "MI_Manufacturer_over_MI_Y_mean")),
            }
        )
        if spec["model_id"] == "residualized_mfr_stageB":
            row.update(residualized)
            # Rate-distortion belongs to the unchanged promoted VAE.
            row["rationale_short"] += " VAE rate-distortion is identical to the promoted model; only Stage B features were transformed."
        if oasis_row is not None:
            for k, v in oasis_row.items():
                if k != "model_id":
                    row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)


def comparison_table(decision: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "model_id",
        "display_name",
        "decision_class",
        "final_decision",
        "adni_oof_ecdf_auc",
        "adni_oof_ecdf_pr_auc",
        "adni_oof_ecdf_balanced_accuracy",
        "adni_oof_ecdf_sensitivity",
        "adni_oof_ecdf_specificity",
        "adni_oof_ecdf_f1",
        "philips_cn_fpr",
        "scanner_leakage_latent_acc",
        "oasis_concatenated_auc",
        "oasis_concatenated_pr_auc",
        "oasis_runwise164_auc",
        "oasis_runwise164_pr_auc",
        "oasis_runwise140_auc",
        "oasis_runwise140_pr_auc",
    ]
    return decision[[c for c in keep if c in decision.columns]].copy()


def fmt(x: Any, digits: int = 3) -> str:
    try:
        if pd.isna(x):
            return "not available"
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def write_texts(out: Path, decision: pd.DataFrame) -> None:
    p = decision[decision["model_id"] == "promoted_latent384_beta3p75_ch1_0_2"].iloc[0]
    ch1 = decision[decision["model_id"] == "ch1only_latent384_beta3p75"].iloc[0]
    b35 = decision[decision["model_id"] == "latent384_beta3p5"].iloc[0]
    l512 = decision[decision["model_id"] == "latent512_beta3p75"].iloc[0]
    l448 = decision[decision["model_id"] == "latent448_beta4p0"].iloc[0]

    threshold_text = f"""# OASIS Threshold-Shift Interpretation

OASIS was evaluated as an external stress test using frozen ADNI-trained VAEs,
classifiers, OOF score transformations, and ADNI-derived thresholds. No OASIS
threshold fitting or calibration fitting was performed in this package.

For the promoted model, the ADNI-derived threshold is conservative on OASIS:
- concatenated_timeseries predicted AD rate {fmt(p.get('oasis_concatenated_predicted_ad_rate'))}, sensitivity {fmt(p.get('oasis_concatenated_sensitivity'))}, specificity {fmt(p.get('oasis_concatenated_specificity'))};
- runwise164_pilot_parity predicted AD rate {fmt(p.get('oasis_runwise164_predicted_ad_rate'))}, sensitivity {fmt(p.get('oasis_runwise164_sensitivity'))}, specificity {fmt(p.get('oasis_runwise164_specificity'))};
- runwise_140TR_pilot_parity predicted AD rate {fmt(p.get('oasis_runwise140_predicted_ad_rate'))}, sensitivity {fmt(p.get('oasis_runwise140_sensitivity'))}, specificity {fmt(p.get('oasis_runwise140_specificity'))}.

This pattern indicates threshold transfer and score-distribution shift rather
than a valid opportunity for OASIS-based model selection. OASIS-calibrated
thresholds should only be estimated in a pre-specified external calibration
subset and then evaluated on a locked external test subset.
"""
    (out / "oasis_threshold_shift_interpretation.md").write_text(threshold_text, encoding="utf-8")

    results_para = f"""The final ADNI model was the recover035 [1,0,2] latent384 beta3.75 beta-VAE with the OOF-ECDF logreg_l2 readout. It achieved ADNI OOF AUC={fmt(p['adni_oof_ecdf_auc'], 6)}, PR-AUC={fmt(p['adni_oof_ecdf_pr_auc'], 6)}, balanced accuracy={fmt(p['adni_oof_ecdf_balanced_accuracy'], 6)}, sensitivity={fmt(p['adni_oof_ecdf_sensitivity'], 6)}, specificity={fmt(p['adni_oof_ecdf_specificity'], 6)}, and F1={fmt(p['adni_oof_ecdf_f1'], 6)}. The ch1-only parsimonious model had higher ADNI AUC/PR-AUC (AUC={fmt(ch1['adni_oof_ecdf_auc'], 6)}, PR-AUC={fmt(ch1['adni_oof_ecdf_pr_auc'], 6)}) but lower BA/F1 and a higher Philips CN false-positive rate, so it was retained as a sensitivity model. Beta/capacity variants did not provide a clean improvement: beta3.5 reduced ADNI AUC/PR-AUC, latent448 beta4.0 and latent512 beta3.75 remained below the promoted model, and beta4.0 lacked a promoted-convention OOF-ECDF/OASIS package. On OASIS, the promoted model showed build-dependent external ranking signal, strongest on runwise164_pilot_parity (AUC={fmt(p.get('oasis_runwise164_auc'), 6)}, PR-AUC={fmt(p.get('oasis_runwise164_pr_auc'), 6)}), supporting its use as an external stress-test result rather than a tuning criterion.
"""
    (out / "manuscript_results_paragraph.md").write_text(results_para, encoding="utf-8")

    limitations_para = """The OASIS analyses were external stress tests and were not used for ADNI model selection. All OASIS scores used frozen ADNI-derived preprocessing, fold VAEs, classifier readouts, score transformations, and thresholds; no OASIS labels were used to train, calibrate, or select a model in this package. The transferred ADNI thresholds produced conservative OASIS operating points with low predicted AD rates and low sensitivity in some builds, indicating score-distribution and threshold-transfer shift. Because several sensitivity models lack matched OASIS artifacts or matched OOF-ECDF readouts, absence of an OASIS row is treated as missing evidence rather than failure. Future external validation should use a pre-specified OASIS calibration/test protocol with threshold calibration performed only on the calibration subset and locked evaluation on a held-out test subset.
"""
    (out / "manuscript_limitations_paragraph.md").write_text(limitations_para, encoding="utf-8")

    reviewer_para = f"""We evaluated scanner and external-cohort robustness using a prespecified hierarchy. The primary ADNI model remains the [1,0,2] latent384 beta3.75 model because it provides the best balanced internal decision profile among the promoted-convention candidates: AUC={fmt(p['adni_oof_ecdf_auc'], 6)}, PR-AUC={fmt(p['adni_oof_ecdf_pr_auc'], 6)}, BA={fmt(p['adni_oof_ecdf_balanced_accuracy'], 6)}, F1={fmt(p['adni_oof_ecdf_f1'], 6)}, Philips CN FPR={fmt(p['philips_cn_fpr'], 4)}, and latent scanner leakage={fmt(p['scanner_leakage_latent_acc'], 4)}. The ch1-only model is reported as a parsimony sensitivity because, despite higher AUC/PR-AUC, it worsened the operating-point profile and transferred less well to OASIS runwise builds. Manufacturer-balanced and residualized readouts are reported only as deconfounding or classifier-only harmonization sensitivities. OASIS results are interpreted as external stress tests: they reveal moderate, build-dependent transfer and conservative ADNI threshold behavior, but they were not used for model selection or threshold fitting.
"""
    (out / "reviewer_response_scanner_oasis_paragraph.md").write_text(reviewer_para, encoding="utf-8")

    interp = f"""# Final Interpretation

## Primary Model
The primary final model remains `recover035_latent384_beta3p75_T80_h10000_p560_full5x5` with channels [1,0,2]. It has the strongest promoted-convention overall profile: ADNI OOF-ECDF AUC={fmt(p['adni_oof_ecdf_auc'], 6)}, PR-AUC={fmt(p['adni_oof_ecdf_pr_auc'], 6)}, BA={fmt(p['adni_oof_ecdf_balanced_accuracy'], 6)}, F1={fmt(p['adni_oof_ecdf_f1'], 6)}, Philips CN FPR={fmt(p['philips_cn_fpr'], 4)}, and latent scanner leakage={fmt(p['scanner_leakage_latent_acc'], 4)}.

## Why ch1-only is not primary
The ch1-only model is useful as a parsimonious sensitivity and has higher ADNI AUC/PR-AUC, but it does not improve the decision operating point: BA and F1 are lower than the promoted model, Philips CN FPR is higher, and OASIS transfer is weaker across the primary runwise builds. Therefore it should not replace the multichannel [1,0,2] model.

## Why beta/capacity variants are not promoted
Beta3.5 underperforms beta3.75 internally (AUC={fmt(b35['adni_oof_ecdf_auc'], 6)}, PR-AUC={fmt(b35['adni_oof_ecdf_pr_auc'], 6)}) and externally. Latent448 beta4.0 and latent512 beta3.75 do not exceed the promoted model on ADNI OOF-ECDF and do not deliver an OASIS advantage. Latent384 beta4.0 and beta2.5 lack matched OOF-ECDF/OASIS artifacts in the completed packages and their raw readouts were weaker than the promoted model.

## OASIS interpretation
OASIS is treated as an external stress test, not a tuning set. The available OASIS results show moderate, build-dependent ranking signal, with the promoted model best among primary candidates on runwise164 and runwise140. ADNI-derived thresholds are conservative on OASIS and yield low sensitivity/high specificity operating points, so threshold transfer should be handled only in a separately locked external calibration/test protocol.
"""
    (out / "final_interpretation.md").write_text(interp, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    decision = build_decision_table()
    comparison = comparison_table(decision)

    if args.dry_run:
        print(f"output_dir={out}")
        print(f"decision_rows={len(decision)}")
        print(f"comparison_rows={len(comparison)}")
        print("models=" + ",".join(decision["model_id"].astype(str)))
        return 0

    decision.to_csv(out / "final_model_decision_table.csv", index=False)
    md_write(decision, out / "final_model_decision_table.md")
    comparison.to_csv(out / "adni_oasis_final_comparison.csv", index=False)
    md_write(comparison, out / "adni_oasis_final_comparison.md")
    write_texts(out, decision)

    readme = """# Final Model Decision ADNI/OASIS Manuscript Support

Read-only aggregation over completed ADNI FULL 5x5 registry, latent capacity
rate-distortion synthesis, and OASIS external-validation packages. This package
does not train models, fit OASIS thresholds, fit OASIS calibration, perform model
selection on OASIS, or modify tensors/metadata/model artifacts.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")

    log = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__)),
        "output_dir": str(out),
        "inputs": {
            "completed_full5x5_registry": str(REGISTRY),
            "capacity_beta_summary": str(CAPACITY),
            "oasis_external_validation_all_full_models": str(OASIS),
            "residualized_harmonization_package": str(HARMONIZATION),
        },
        "guardrails": {
            "no_training": True,
            "no_new_oasis_threshold_fitting": True,
            "no_new_oasis_calibration_fitting": True,
            "no_model_selection_based_on_oasis": True,
            "no_tensor_metadata_ledger_model_artifact_modification": True,
        },
        "outputs": [
            "final_model_decision_table.csv",
            "final_model_decision_table.md",
            "adni_oasis_final_comparison.csv",
            "adni_oasis_final_comparison.md",
            "oasis_threshold_shift_interpretation.md",
            "manuscript_results_paragraph.md",
            "manuscript_limitations_paragraph.md",
            "reviewer_response_scanner_oasis_paragraph.md",
        ],
        "n_decision_rows": int(len(decision)),
    }
    (out / "command_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"Wrote final model-decision package to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
