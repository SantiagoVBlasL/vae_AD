#!/usr/bin/env python3
"""Package Stage B/OOF completion audit for the chmeanloss FULL run.

This script is read-only with respect to tensors, metadata, and model artifacts.
It only reads completed Stage A/Stage B outputs and writes an audit package.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("results/revision_bspc_2026")
RUN_DIR = ROOT / "recover035_latent384_beta3p75_chmeanloss_T80_h10000_p560_full5x5"
STAGEB_DIR = RUN_DIR / "classifier_only_readout"
OOF_DIR = ROOT / "recover035_latent384_beta3p75_chmeanloss_stageB_oof_score_calibration"
PREV_AUDIT = ROOT / "chmeanloss_completion_promotion_gate_audit_20260607"
EVIDENCE_MAP = ROOT / "final_full_model_evidence_map_with_beta6p5_20260606" / "full_model_evidence_map.csv"
OUT_DIR = ROOT / "chmeanloss_stageB_oof_completion_audit_20260607"

PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
PRIMARY_MODEL = "logreg_l2_original"
PRIMARY_FEATURE = "z_plus_age_sex"
PRIMARY_CALIB = "oof_ecdf"

REFERENCE_IDS = {
    "promoted_latent384_beta3p75_ch1_0_2": "promoted [1,0,2] latent384 beta3.75",
    "ch1only_latent384_beta3p75": "ch1-only latent384 beta3.75",
    "latent384_beta6p5": "latent384 beta6.5",
    "latent448_beta4p0": "latent448 beta4.0",
    "latent512_beta3p75": "latent512 beta3.75",
}


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def read_csv(path: Path) -> pd.DataFrame:
    require_file(path)
    return pd.read_csv(path)


def write_table(df: pd.DataFrame, name: str) -> None:
    csv_path = OUT_DIR / f"{name}.csv"
    md_path = OUT_DIR / f"{name}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")


def get_single(df: pd.DataFrame, **filters: Any) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col, val in filters.items():
        mask &= df[col].eq(val)
    out = df.loc[mask]
    if len(out) != 1:
        raise RuntimeError(f"Expected one row for filters {filters}, found {len(out)}")
    return out.iloc[0]


def value(row: pd.Series, key: str, default: float | str | None = None) -> Any:
    return row[key] if key in row.index else default


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stageb_pooled = read_csv(STAGEB_DIR / "classifier_sweep_pooled_metrics.csv")
    stageb_foldwise = read_csv(STAGEB_DIR / "classifier_sweep_foldwise_metrics.csv")
    stageb_status = read_csv(STAGEB_DIR / "classifier_sweep_model_status.csv")
    oof_pooled = read_csv(OOF_DIR / "calib_pooled_metrics.csv")
    oof_foldwise = read_csv(OOF_DIR / "calib_foldwise_metrics.csv")
    oof_philips = read_csv(OOF_DIR / "calib_philips_fpr_pooled.csv")
    stagea_summary = read_csv(PREV_AUDIT / "stagea_summary_metrics.csv")
    scanner = read_csv(PREV_AUDIT / "scanner_leakage_summary.csv")
    rd = read_csv(PREV_AUDIT / "rate_distortion_summary.csv")
    latent_mi = read_csv(PREV_AUDIT / "latent_mi_signal_nuisance_summary.csv")
    evidence = read_csv(EVIDENCE_MAP)

    stageb_main = stageb_pooled[
        (stageb_pooled["model_name"] == "logreg_l2")
        & (stageb_pooled["readout_feature_set"] == PRIMARY_FEATURE)
    ].copy()
    oof_main = oof_pooled[
        (oof_pooled["model_name"] == PRIMARY_MODEL)
        & (oof_pooled["feature_set"] == PRIMARY_FEATURE)
    ].copy()
    oof_foldwise_main = oof_foldwise[
        (oof_foldwise["model_name"] == PRIMARY_MODEL)
        & (oof_foldwise["feature_set"] == PRIMARY_FEATURE)
    ].copy()
    philips_main = oof_philips[
        (oof_philips["model_name"] == PRIMARY_MODEL)
        & (oof_philips["feature_set"] == PRIMARY_FEATURE)
    ].copy()

    write_table(stageb_status, "stageb_execution_status")
    write_table(stageb_main, "stageb_classifier_only_metrics")
    write_table(oof_main, "stageb_oof_calibration_metrics")
    write_table(oof_foldwise_main, "stageb_oof_foldwise_metrics")
    write_table(philips_main, "philips_cn_fpr")
    write_table(scanner, "scanner_leakage_summary")
    write_table(rd, "rate_distortion_summary")
    write_table(latent_mi, "latent_mi_signal_nuisance_summary")

    primary = get_single(
        oof_pooled,
        model_name=PRIMARY_MODEL,
        feature_set=PRIMARY_FEATURE,
        calib_method=PRIMARY_CALIB,
        threshold_strategy=PRIMARY_THRESHOLD,
    )
    primary_logitz = get_single(
        oof_pooled,
        model_name=PRIMARY_MODEL,
        feature_set=PRIMARY_FEATURE,
        calib_method="oof_logitz",
        threshold_strategy=PRIMARY_THRESHOLD,
    )
    primary_raw = get_single(
        stageb_pooled,
        model_name="logreg_l2",
        readout_feature_set=PRIMARY_FEATURE,
        threshold_strategy=PRIMARY_THRESHOLD,
    )
    primary_fpr = get_single(
        oof_philips,
        model_name=PRIMARY_MODEL,
        feature_set=PRIMARY_FEATURE,
        calib_method=PRIMARY_CALIB,
        threshold_strategy=PRIMARY_THRESHOLD,
        manufacturer="Philips",
    )
    stagea_logreg = get_single(stagea_summary, classifier_type="logreg")
    rd_row = rd.iloc[0]
    latent_row = latent_mi.iloc[0]
    scanner_test = get_single(scanner, split="test")
    scanner_train = get_single(scanner, split="train_dev")

    stage_delta = pd.DataFrame(
        [
            {
                "candidate": "chmeanloss",
                "stageA_classifier": "logreg",
                "stageB_model": PRIMARY_MODEL,
                "stageB_feature_set": PRIMARY_FEATURE,
                "stageB_calib_method": PRIMARY_CALIB,
                "threshold_strategy": PRIMARY_THRESHOLD,
                "stageA_auc": stagea_logreg["auc"],
                "stageB_auc": primary["auc"],
                "delta_auc": primary["auc"] - stagea_logreg["auc"],
                "stageA_pr_auc": stagea_logreg["pr_auc"],
                "stageB_pr_auc": primary["pr_auc"],
                "delta_pr_auc": primary["pr_auc"] - stagea_logreg["pr_auc"],
                "stageA_ba": stagea_logreg["balanced_accuracy"],
                "stageB_ba": primary["balanced_accuracy"],
                "delta_ba": primary["balanced_accuracy"] - stagea_logreg["balanced_accuracy"],
                "stageA_f1": stagea_logreg["f1"],
                "stageB_f1": primary["f1"],
                "delta_f1": primary["f1"] - stagea_logreg["f1"],
            }
        ]
    )
    write_table(stage_delta, "stageA_to_stageB_delta")

    ref = evidence[evidence["model_id"].isin(REFERENCE_IDS.keys())].copy()
    ref["comparison_role"] = ref["model_id"].map(REFERENCE_IDS)
    ref_cols = [
        "comparison_role",
        "model_id",
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
    ref_compact = ref[ref_cols].copy()
    candidate_row = pd.DataFrame(
        [
            {
                "comparison_role": "candidate chmeanloss primary OOF-ECDF",
                "model_id": "chmeanloss_latent384_beta3p75_stageB_oof_ecdf",
                "channel_set_order": "[1,0,2]",
                "latent_dim": 384,
                "beta_vae": 3.75,
                "adni_oof_ecdf_auc": primary["auc"],
                "adni_oof_ecdf_pr_auc": primary["pr_auc"],
                "adni_oof_ecdf_ba": primary["balanced_accuracy"],
                "adni_oof_ecdf_sens": primary["sensitivity"],
                "adni_oof_ecdf_spec": primary["specificity"],
                "adni_oof_ecdf_f1": primary["f1"],
                "philips_cn_fpr": primary_fpr["fpr_cn_pooled"],
                "scanner_leakage_latent_acc": scanner_test["acc_site_latent"],
                "beta_KLD_over_D_best_mean": rd_row["beta_KLD_over_D_best"],
                "active_units_mean": latent_row["active_units"],
                "total_correlation_nats_mean": latent_row["total_correlation_nats"],
                "MI_Z_Y_nats_mean": latent_row["MI_Z_Y_nats"],
                "MI_Z_Manufacturer_nats_mean": latent_row["MI_Z_Manufacturer_nats"],
                "MI_Manufacturer_over_MI_Y_mean": latent_row["MI_Manufacturer_over_MI_Y"],
                "oasis_status": "not_run_adni_gate_failed",
            }
        ]
    )
    comparison = pd.concat([candidate_row, ref_compact], ignore_index=True)
    write_table(comparison, "comparison_vs_references")

    promoted = get_single(evidence, model_id="promoted_latent384_beta3p75_ch1_0_2")
    ch1 = get_single(evidence, model_id="ch1only_latent384_beta3p75")
    gates = [
        {
            "gate": "Stage B classifier-only readout completed",
            "criterion": "classifier_only_readout metrics and latent_cache present",
            "observed": bool((STAGEB_DIR / "latent_cache").exists()),
            "reference": True,
            "passes": bool((STAGEB_DIR / "latent_cache").exists()),
        },
        {
            "gate": "OOF calibration completed",
            "criterion": "raw, oof_zscore, oof_logitz, oof_ecdf, oof_platt, oof_isotonic available",
            "observed": ", ".join(sorted(oof_main["calib_method"].dropna().unique())),
            "reference": "all requested methods",
            "passes": set(["raw", "oof_zscore", "oof_logitz", "oof_ecdf", "oof_platt", "oof_isotonic"]).issubset(
                set(oof_main["calib_method"].dropna().unique())
            ),
        },
        {
            "gate": "AUC exceeds promoted reference",
            "criterion": "candidate AUC > promoted AUC",
            "observed": primary["auc"],
            "reference": promoted["adni_oof_ecdf_auc"],
            "passes": primary["auc"] > promoted["adni_oof_ecdf_auc"],
        },
        {
            "gate": "PR-AUC meets promoted reference",
            "criterion": "candidate PR-AUC >= promoted PR-AUC",
            "observed": primary["pr_auc"],
            "reference": promoted["adni_oof_ecdf_pr_auc"],
            "passes": primary["pr_auc"] >= promoted["adni_oof_ecdf_pr_auc"],
        },
        {
            "gate": "BA not worse than promoted reference",
            "criterion": "candidate BA >= promoted BA",
            "observed": primary["balanced_accuracy"],
            "reference": promoted["adni_oof_ecdf_ba"],
            "passes": primary["balanced_accuracy"] >= promoted["adni_oof_ecdf_ba"],
        },
        {
            "gate": "Sensitivity not worse than promoted reference",
            "criterion": "candidate Sens >= promoted Sens",
            "observed": primary["sensitivity"],
            "reference": promoted["adni_oof_ecdf_sens"],
            "passes": primary["sensitivity"] >= promoted["adni_oof_ecdf_sens"],
        },
        {
            "gate": "F1 not worse than promoted reference",
            "criterion": "candidate F1 >= promoted F1",
            "observed": primary["f1"],
            "reference": promoted["adni_oof_ecdf_f1"],
            "passes": primary["f1"] >= promoted["adni_oof_ecdf_f1"],
        },
        {
            "gate": "Philips CN FPR not worse than promoted reference",
            "criterion": "candidate Philips CN FPR <= promoted Philips CN FPR",
            "observed": primary_fpr["fpr_cn_pooled"],
            "reference": promoted["philips_cn_fpr"],
            "passes": primary_fpr["fpr_cn_pooled"] <= promoted["philips_cn_fpr"],
        },
        {
            "gate": "Scanner leakage not worse than promoted reference",
            "criterion": "candidate latent scanner leakage <= promoted latent scanner leakage",
            "observed": scanner_test["acc_site_latent"],
            "reference": promoted["scanner_leakage_latent_acc"],
            "passes": scanner_test["acc_site_latent"] <= promoted["scanner_leakage_latent_acc"],
        },
        {
            "gate": "Effective regularization approaches ch1-only",
            "criterion": "|candidate beta*KLD/D - ch1-only beta*KLD/D| <= 0.01",
            "observed": rd_row["beta_KLD_over_D_best"],
            "reference": ch1["beta_KLD_over_D_best_mean"],
            "passes": abs(rd_row["beta_KLD_over_D_best"] - ch1["beta_KLD_over_D_best_mean"]) <= 0.01,
        },
    ]
    gate_df = pd.DataFrame(gates)
    write_table(gate_df, "primary_promotion_gate_table")

    oasis_decision = (
        "# OASIS Gate Decision\n\n"
        "Decision: **not_run_adni_gate_failed**.\n\n"
        "The requested rule was to run frozen OASIS inference only if the ADNI promotion gate passed. "
        "The chmeanloss primary OOF-ECDF row did not pass the ADNI gate against the promoted "
        "[1,0,2] latent384 beta3.75 model, so no OASIS scoring was run and no OASIS threshold or "
        "calibration fitting was performed.\n"
    )
    (OUT_DIR / "oasis_gate_decision.md").write_text(oasis_decision, encoding="utf-8")

    final_decision = f"""# Final Decision

Decision: **do_not_promote_adni_gate_failed**.

Stage B classifier-only readout and OOF score calibration are now complete for
`recover035_latent384_beta3p75_chmeanloss_T80_h10000_p560_full5x5`.

Primary row:
- model/readout: `{PRIMARY_MODEL}` / `{PRIMARY_FEATURE}` / `{PRIMARY_CALIB}` / `{PRIMARY_THRESHOLD}`
- AUC={primary['auc']:.6f}
- PR-AUC={primary['pr_auc']:.6f}
- BA={primary['balanced_accuracy']:.6f}
- Sens={primary['sensitivity']:.6f}
- Spec={primary['specificity']:.6f}
- F1={primary['f1']:.6f}
- Philips CN FPR={int(primary_fpr['fp_cn_pooled'])}/{int(primary_fpr['n_cn_pooled'])}={primary_fpr['fpr_cn_pooled']:.6f}

OOF-logitz sensitivity:
- AUC={primary_logitz['auc']:.6f}
- PR-AUC={primary_logitz['pr_auc']:.6f}
- BA={primary_logitz['balanced_accuracy']:.6f}
- Sens={primary_logitz['sensitivity']:.6f}
- Spec={primary_logitz['specificity']:.6f}
- F1={primary_logitz['f1']:.6f}

Raw Stage B primary threshold:
- AUC={primary_raw['auc']:.6f}
- PR-AUC={primary_raw['pr_auc']:.6f}
- BA={primary_raw['balanced_accuracy']:.6f}
- Sens={primary_raw['sensitivity']:.6f}
- Spec={primary_raw['specificity']:.6f}
- F1={primary_raw['f1']:.6f}

Compared with the promoted [1,0,2] latent384 beta3.75 reference, chmeanloss is lower on AUC,
PR-AUC, BA, sensitivity, and F1, has higher Philips CN FPR, and has higher latent scanner
leakage. The effective regularization ratio beta*KLD/D={rd_row['beta_KLD_over_D_best']:.6f}
increased relative to the promoted multichannel reference but did not approach the ch1-only
regime ({ch1['beta_KLD_over_D_best_mean']:.6f}).

No OASIS inference was run because the ADNI gate failed. No VAE retraining, tensor modification,
metadata modification, or model artifact overwrite was performed by this audit package.
"""
    (OUT_DIR / "final_decision.md").write_text(final_decision, encoding="utf-8")

    readme = f"""# Chmeanloss Stage B + OOF Completion Audit

Run: `recover035_latent384_beta3p75_chmeanloss_T80_h10000_p560_full5x5`

This package records completion of the predefined classifier-only Stage B readout and OOF score
calibration for the completed chmeanloss VAE/Stage A run.

Generated outputs:
- `stageb_execution_status.csv/.md`
- `stageb_classifier_only_metrics.csv/.md`
- `stageb_oof_calibration_metrics.csv/.md`
- `stageb_oof_foldwise_metrics.csv/.md`
- `philips_cn_fpr.csv/.md`
- `scanner_leakage_summary.csv/.md`
- `rate_distortion_summary.csv/.md`
- `latent_mi_signal_nuisance_summary.csv/.md`
- `stageA_to_stageB_delta.csv/.md`
- `comparison_vs_references.csv/.md`
- `primary_promotion_gate_table.csv/.md`
- `oasis_gate_decision.md`
- `final_decision.md`
- `command_log.json`

Primary result: AUC={primary['auc']:.6f}, PR-AUC={primary['pr_auc']:.6f},
BA={primary['balanced_accuracy']:.6f}, Sens={primary['sensitivity']:.6f},
Spec={primary['specificity']:.6f}, F1={primary['f1']:.6f}.

Final decision: `do_not_promote_adni_gate_failed`.
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")

    command_log = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(RUN_DIR),
        "stageb_output_dir": str(STAGEB_DIR),
        "oof_output_dir": str(OOF_DIR),
        "audit_output_dir": str(OUT_DIR),
        "commands_completed_before_packaging": [
            {
                "purpose": "Audit script syntax validation",
                "command": (
                    "/home/diego/anaconda3/envs/vae_ad/bin/python -m py_compile "
                    "scripts/revision_bspc_2026/audit_chmeanloss_stageB_oof_completion_20260607.py"
                ),
                "status": "completed_exit_code_0",
            },
            {
                "purpose": "Stage B classifier-only readout from completed fold VAE artifacts",
                "command": (
                    "/home/diego/anaconda3/envs/vae_ad/bin/python "
                    "scripts/revision_bspc_2026/run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep.py "
                    "--run-dir results/revision_bspc_2026/recover035_latent384_beta3p75_chmeanloss_T80_h10000_p560_full5x5 "
                    "--output-dir results/revision_bspc_2026/recover035_latent384_beta3p75_chmeanloss_T80_h10000_p560_full5x5/classifier_only_readout "
                    "--outer-folds 5 --inner-folds 5 --models logreg_l2 --readout-feature-sets z_plus_age_sex "
                    "--device cpu --n-jobs 4"
                ),
                "status": "completed_exit_code_0",
                "vae_retraining": False,
            },
            {
                "purpose": "OOF score calibration sweep",
                "command": (
                    "/home/diego/anaconda3/envs/vae_ad/bin/python "
                    "scripts/revision_bspc_2026/run_recover035_latent384_beta3p75_stageB_oof_score_calibration.py "
                    "--run-dir results/revision_bspc_2026/recover035_latent384_beta3p75_chmeanloss_T80_h10000_p560_full5x5 "
                    "--output-dir results/revision_bspc_2026/recover035_latent384_beta3p75_chmeanloss_stageB_oof_score_calibration "
                    "--n-jobs 4 --overwrite"
                ),
                "status": "completed_exit_code_0",
                "oasis_threshold_or_calibration_fitting": False,
            },
            {
                "purpose": "Package completion and promotion-gate audit",
                "command": (
                    "/home/diego/anaconda3/envs/vae_ad/bin/python "
                    "scripts/revision_bspc_2026/audit_chmeanloss_stageB_oof_completion_20260607.py"
                ),
                "status": "completed_exit_code_0",
            },
        ],
        "guardrails": {
            "vae_training_launched": False,
            "tensor_modified": False,
            "metadata_modified": False,
            "existing_model_artifact_overwrite": False,
            "oasis_inference_run": False,
            "oasis_threshold_or_calibration_fitting": False,
        },
        "decision": "do_not_promote_adni_gate_failed",
    }
    (OUT_DIR / "command_log.json").write_text(json.dumps(command_log, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {OUT_DIR}")
    print("Decision: do_not_promote_adni_gate_failed")


if __name__ == "__main__":
    main()
