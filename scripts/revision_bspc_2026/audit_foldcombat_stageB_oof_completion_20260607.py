#!/usr/bin/env python3
"""Package the foldcombat Stage B + OOF score-calibration completion audit.

Reads completed VAE/Stage A, classifier-only Stage B, and OOF calibration
artifacts. It writes an audit package only; it does not retrain VAE, refit
thresholds on outer tests, score OASIS, or modify tensors/metadata/model
artifacts.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("results/revision_bspc_2026")
RUN_DIR = ROOT / "recover035_latent384_beta3p75_foldcombat_mfr_age_sex_T80_h10000_p560_full5x5"
STAGEB_DIR = RUN_DIR / "classifier_only_readout"
OOF_DIR = ROOT / "recover035_latent384_beta3p75_foldcombat_stageB_oof_score_calibration"
PREV_AUDIT = ROOT / "foldcombat_completion_promotion_harmonization_audit_20260607"
EVIDENCE_MAP = ROOT / "final_full_model_evidence_map_with_beta6p5_20260606/full_model_evidence_map.csv"
CHMEANLOSS_COMPARISON = ROOT / "chmeanloss_stageB_oof_completion_audit_20260607/comparison_vs_references.csv"
OUT_DIR = ROOT / "foldcombat_stageB_oof_completion_audit_20260607"

PRIMARY_MODEL = "logreg_l2_original"
PRIMARY_FEATURE = "z_plus_age_sex"
PRIMARY_CALIB = "oof_ecdf"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"

PROMOTED = {
    "auc": 0.795155,
    "pr_auc": 0.573934,
    "balanced_accuracy": 0.725979,
    "sensitivity": 0.731959,
    "f1": 0.563492,
    "philips_cn_fpr": 0.4545,
    "scanner_leakage_latent_acc": 0.727556,
}

REFERENCE_IDS = {
    "promoted_latent384_beta3p75_ch1_0_2": "promoted [1,0,2] latent384 beta3.75",
    "ch1only_latent384_beta3p75": "ch1-only latent384 beta3.75",
    "latent384_beta6p5": "latent384 beta6.5",
    "latent448_beta4p0": "latent448 beta4.0",
    "latent512_beta3p75": "latent512 beta3.75",
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def maybe_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def md_table(df: pd.DataFrame, max_rows: int = 120) -> str:
    if df.empty:
        return "_No rows._\n"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6g}")
    text = view.to_markdown(index=False)
    if len(df) > max_rows:
        text += f"\n\n_Showing {max_rows} of {len(df)} rows._"
    return text + "\n"


def write_table(df: pd.DataFrame, stem: str, max_rows: int = 120) -> None:
    df.to_csv(OUT_DIR / f"{stem}.csv", index=False)
    (OUT_DIR / f"{stem}.md").write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


def get_single(df: pd.DataFrame, **filters: Any) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col, val in filters.items():
        mask &= df[col].eq(val)
    sub = df.loc[mask]
    if len(sub) != 1:
        raise RuntimeError(f"Expected one row for {filters}, found {len(sub)}")
    return sub.iloc[0]


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else float("nan")


def completion_status() -> pd.DataFrame:
    rows = []
    for fold in range(1, 6):
        rows.append(
            {
                "fold": fold,
                "vae_model": (RUN_DIR / f"fold_{fold}/vae_model_fold_{fold}.pt").exists(),
                "stagea_logreg": (RUN_DIR / f"fold_{fold}/test_predictions_logreg.csv").exists(),
                "stagea_svm": (RUN_DIR / f"fold_{fold}/test_predictions_svm.csv").exists(),
                "latent_cache_trainDev": (STAGEB_DIR / f"latent_cache/fold_{fold}_trainDev_latent_mu.csv").exists(),
                "latent_cache_test": (STAGEB_DIR / f"latent_cache/fold_{fold}_test_latent_mu.csv").exists(),
                "harmonization_guard": (RUN_DIR / f"fold_{fold}/input_harmonization_leakage_guard.csv").exists(),
            }
        )
    rows.append(
        {
            "fold": "all",
            "vae_model": all((RUN_DIR / f"fold_{fold}/vae_model_fold_{fold}.pt").exists() for fold in range(1, 6)),
            "stagea_logreg": all((RUN_DIR / f"fold_{fold}/test_predictions_logreg.csv").exists() for fold in range(1, 6)),
            "stagea_svm": all((RUN_DIR / f"fold_{fold}/test_predictions_svm.csv").exists() for fold in range(1, 6)),
            "latent_cache_trainDev": all(
                (STAGEB_DIR / f"latent_cache/fold_{fold}_trainDev_latent_mu.csv").exists() for fold in range(1, 6)
            ),
            "latent_cache_test": all(
                (STAGEB_DIR / f"latent_cache/fold_{fold}_test_latent_mu.csv").exists() for fold in range(1, 6)
            ),
            "harmonization_guard": all(
                (RUN_DIR / f"fold_{fold}/input_harmonization_leakage_guard.csv").exists() for fold in range(1, 6)
            ),
            "classifier_only_readout": STAGEB_DIR.exists(),
            "oof_score_calibration": OOF_DIR.exists(),
        }
    )
    return pd.DataFrame(rows)


def manufacturer_tables(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = preds[
        (preds["model_name"] == PRIMARY_MODEL)
        & (preds["feature_set"] == PRIMARY_FEATURE)
        & (preds["calib_method"] == PRIMARY_CALIB)
        & (preds["threshold_strategy"] == PRIMARY_THRESHOLD)
    ].copy()
    fpr_rows = []
    fnr_rows = []
    for mfr, g in sub.groupby("Manufacturer", dropna=False):
        cn = g[g["y_true"] == 0]
        ad = g[g["y_true"] == 1]
        fp = int((cn["y_pred"] == 1).sum())
        tn = int((cn["y_pred"] == 0).sum())
        fn = int((ad["y_pred"] == 0).sum())
        tp = int((ad["y_pred"] == 1).sum())
        fpr_rows.append({"manufacturer": mfr, "cn_n": len(cn), "fp": fp, "tn": tn, "cn_fpr": safe_div(fp, fp + tn)})
        fnr_rows.append({"manufacturer": mfr, "ad_n": len(ad), "fn": fn, "tp": tp, "ad_fnr": safe_div(fn, fn + tp)})
    return pd.DataFrame(fpr_rows), pd.DataFrame(fnr_rows)


def load_refs() -> pd.DataFrame:
    refs = []
    if EVIDENCE_MAP.exists():
        ev = read_csv(EVIDENCE_MAP)
        ref = ev[ev["model_id"].isin(REFERENCE_IDS)].copy()
        ref["comparison_role"] = ref["model_id"].map(REFERENCE_IDS)
        refs.append(ref)
    if CHMEANLOSS_COMPARISON.exists():
        ch = read_csv(CHMEANLOSS_COMPARISON)
        cand = ch[ch["comparison_role"].astype(str).str.contains("candidate", case=False, na=False)].copy()
        if not cand.empty:
            cand["comparison_role"] = "chmeanloss latent384 beta3.75"
            refs.append(cand)
    return pd.concat(refs, ignore_index=True, sort=False) if refs else pd.DataFrame()


def read_prev(stem: str) -> pd.DataFrame:
    return maybe_csv(PREV_AUDIT / f"{stem}.csv")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stagea_summary = read_prev("stagea_summary_metrics")
    stageb_pooled = read_csv(STAGEB_DIR / "classifier_sweep_pooled_metrics.csv")
    stageb_foldwise = read_csv(STAGEB_DIR / "classifier_sweep_foldwise_metrics.csv")
    stageb_status = read_csv(STAGEB_DIR / "classifier_sweep_model_status.csv")
    stageb_subgroups = read_csv(STAGEB_DIR / "classifier_sweep_subgroup_metrics_by_manufacturer.csv")
    oof_pooled = read_csv(OOF_DIR / "calib_pooled_metrics.csv")
    oof_foldwise = read_csv(OOF_DIR / "calib_foldwise_metrics.csv")
    oof_preds = read_csv(OOF_DIR / "calib_predictions.csv")
    oof_philips = read_csv(OOF_DIR / "calib_philips_fpr_pooled.csv")
    score_ranges = read_csv(OOF_DIR / "calib_score_range_by_fold.csv")
    scanner = read_prev("scanner_leakage_summary")
    rd = read_prev("rate_distortion_summary")
    latent_mi = read_prev("latent_mi_signal_nuisance_summary")
    harm_guard = read_prev("harmonization_leakage_guard_summary")
    harm_integrity = read_prev("harmonization_integrity_summary")
    harm_sep = read_prev("harmonization_manufacturer_separability_channel_mean")
    harm_shift = read_prev("harmonization_scale_shift_channel_mean")

    write_table(completion_status(), "completion_status")
    write_table(stagea_summary, "stagea_summary_metrics")
    write_table(stageb_status, "stageb_execution_status")
    write_table(stageb_pooled, "stageb_classifier_only_metrics")
    write_table(stageb_foldwise, "stageb_classifier_only_foldwise_metrics", max_rows=240)
    write_table(oof_pooled, "stageb_oof_calibration_metrics", max_rows=160)
    write_table(oof_foldwise, "stageb_oof_foldwise_metrics", max_rows=240)
    write_table(score_ranges, "fold_score_scale_audit", max_rows=180)
    write_table(stageb_subgroups, "stageb_subgroup_metrics_by_manufacturer", max_rows=120)
    write_table(scanner, "scanner_leakage_summary")
    write_table(rd, "rate_distortion_summary")
    write_table(latent_mi, "latent_mi_signal_nuisance_summary")
    write_table(harm_guard, "harmonization_leakage_guard_summary")
    write_table(harm_integrity, "harmonization_integrity_summary")
    write_table(harm_sep, "harmonization_manufacturer_separability_summary")
    write_table(harm_shift, "harmonization_scale_shift_summary")

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
    raw_primary = get_single(
        stageb_pooled,
        model_name="logreg_l2",
        readout_feature_set=PRIMARY_FEATURE,
        threshold_strategy=PRIMARY_THRESHOLD,
    )
    stagea_logreg = (
        stagea_summary[
            (stagea_summary.get("model", stagea_summary.get("classifier_type")) == "logreg")
            & (stagea_summary.get("score_type", pd.Series(["final"] * len(stagea_summary))) == "final")
        ].iloc[0]
        if not stagea_summary.empty
        else pd.Series(dtype=object)
    )

    mfr_fpr, mfr_fnr = manufacturer_tables(oof_preds)
    write_table(mfr_fpr, "manufacturer_cn_fpr")
    write_table(mfr_fnr, "manufacturer_ad_fnr")

    philips_fpr = float(mfr_fpr.loc[mfr_fpr["manufacturer"].eq("Philips"), "cn_fpr"].iloc[0])
    test_latent = scanner[(scanner["split"] == "test") & (scanner["representation"] == "latent_mu")]
    scanner_test = float(test_latent["mean_balanced_accuracy"].iloc[0]) if not test_latent.empty else np.nan

    gate_rows = [
        {
            "gate": "AUC > promoted or PR-AUC materially improves without AUC loss",
            "required": f"AUC > {PROMOTED['auc']:.6f} or PR-AUC improves with no AUC loss",
            "observed": f"OOF-ECDF AUC={primary['auc']:.6f}, PR-AUC={primary['pr_auc']:.6f}",
            "pass": bool(primary["auc"] > PROMOTED["auc"]),
        },
        {
            "gate": "PR-AUC >= promoted",
            "required": f">= {PROMOTED['pr_auc']:.6f}",
            "observed": f"{primary['pr_auc']:.6f}",
            "pass": bool(primary["pr_auc"] >= PROMOTED["pr_auc"]),
        },
        {
            "gate": "BA/F1/Sensitivity not materially worse",
            "required": f"BA {PROMOTED['balanced_accuracy']:.6f}, Sens {PROMOTED['sensitivity']:.6f}, F1 {PROMOTED['f1']:.6f}",
            "observed": (
                f"BA={primary['balanced_accuracy']:.6f}, Sens={primary['sensitivity']:.6f}, "
                f"F1={primary['f1']:.6f}"
            ),
            "pass": bool(
                primary["balanced_accuracy"] >= PROMOTED["balanced_accuracy"] - 0.005
                and primary["sensitivity"] >= PROMOTED["sensitivity"] - 0.005
                and primary["f1"] >= PROMOTED["f1"] - 0.005
            ),
        },
        {
            "gate": "Philips CN FPR <= promoted",
            "required": f"<= {PROMOTED['philips_cn_fpr']:.4f}",
            "observed": f"{philips_fpr:.6f}",
            "pass": bool(philips_fpr <= PROMOTED["philips_cn_fpr"]),
        },
        {
            "gate": "scanner leakage lower than promoted",
            "required": f"< {PROMOTED['scanner_leakage_latent_acc']:.6f}",
            "observed": f"{scanner_test:.6f}",
            "pass": bool(scanner_test < PROMOTED["scanner_leakage_latent_acc"]),
        },
        {
            "gate": "no leakage or diagnosis use in harmonization",
            "required": "all fold guard/integrity rows PASS, diagnosis excluded, no global ComBat, no OASIS",
            "observed": (
                f"guard_status={';'.join(harm_guard.get('status', pd.Series(dtype=str)).astype(str).unique())}; "
                f"integrity_status={';'.join(harm_integrity.get('status', pd.Series(dtype=str)).astype(str).unique())}"
            ),
            "pass": bool(
                not harm_guard.empty
                and not harm_integrity.empty
                and harm_guard.get("status", pd.Series(dtype=str)).eq("PASS").all()
                and harm_integrity.get("status", pd.Series(dtype=str)).eq("PASS").all()
                and harm_guard.get("diagnosis_used_in_harmonizer", pd.Series([True])).eq(False).all()
                and harm_guard.get("global_combat", pd.Series([True])).eq(False).all()
                and harm_guard.get("oasis_used", pd.Series([True])).eq(False).all()
            ),
        },
    ]
    gates = pd.DataFrame(gate_rows)
    write_table(gates, "primary_promotion_gate_table")

    stage_delta = pd.DataFrame(
        [
            {
                "candidate": "foldcombat",
                "stageA_logreg_final_auc": stagea_logreg.get("auc", np.nan),
                "stageB_raw_auc": raw_primary["auc"],
                "stageB_oof_ecdf_auc": primary["auc"],
                "stageB_oof_logitz_auc": primary_logitz["auc"],
                "stageA_to_oof_ecdf_delta_auc": primary["auc"] - stagea_logreg.get("auc", np.nan),
                "stageA_logreg_final_pr_auc": stagea_logreg.get("pr_auc", np.nan),
                "stageB_raw_pr_auc": raw_primary["pr_auc"],
                "stageB_oof_ecdf_pr_auc": primary["pr_auc"],
                "stageB_oof_logitz_pr_auc": primary_logitz["pr_auc"],
                "stageA_to_oof_ecdf_delta_pr_auc": primary["pr_auc"] - stagea_logreg.get("pr_auc", np.nan),
            }
        ]
    )
    write_table(stage_delta, "stageA_to_stageB_delta")

    refs = load_refs()
    rd_mean = rd[rd["fold"].astype(str).eq("mean")].iloc[0] if not rd.empty else pd.Series(dtype=object)
    latent_mean = (
        latent_mi[(latent_mi["fold"].astype(str).eq("mean")) & (latent_mi["split"].eq("test"))].iloc[0]
        if not latent_mi.empty
        else pd.Series(dtype=object)
    )
    candidate = pd.DataFrame(
        [
            {
                "comparison_role": "candidate foldcombat primary OOF-ECDF",
                "model_id": "foldcombat_mfr_age_sex_latent384_beta3p75_stageB_oof_ecdf",
                "channel_set_order": "[1,0,2]",
                "latent_dim": 384,
                "beta_vae": 3.75,
                "adni_oof_ecdf_auc": primary["auc"],
                "adni_oof_ecdf_pr_auc": primary["pr_auc"],
                "adni_oof_ecdf_ba": primary["balanced_accuracy"],
                "adni_oof_ecdf_sens": primary["sensitivity"],
                "adni_oof_ecdf_spec": primary["specificity"],
                "adni_oof_ecdf_f1": primary["f1"],
                "adni_oof_logitz_auc": primary_logitz["auc"],
                "adni_oof_logitz_pr_auc": primary_logitz["pr_auc"],
                "philips_cn_fpr": philips_fpr,
                "scanner_leakage_latent_acc": scanner_test,
                "beta_KLD_over_D_best_mean": rd_mean.get("beta_KLD_over_D_best", np.nan),
                "active_units_mean": latent_mean.get("active_units", np.nan),
                "total_correlation_nats_mean": latent_mean.get("total_correlation_nats", np.nan),
                "MI_Z_Y_nats_mean": latent_mean.get("MI_Z_Y_nats", np.nan),
                "MI_Z_Manufacturer_nats_mean": latent_mean.get("MI_Z_Manufacturer_nats", np.nan),
                "MI_Manufacturer_over_MI_Y_mean": latent_mean.get("MI_Manufacturer_over_MI_Y", np.nan),
                "oasis_status": "not_run_adni_gate_failed_harmonization_sensitivity_only",
            }
        ]
    )
    keep = list(candidate.columns)
    if not refs.empty:
        refs = refs[[c for c in keep if c in refs.columns] + [c for c in refs.columns if c not in keep]]
    comparison = pd.concat([candidate, refs], ignore_index=True, sort=False)
    write_table(comparison, "comparison_vs_references")

    best_primary = oof_pooled[
        (oof_pooled["feature_set"] == PRIMARY_FEATURE) & (oof_pooled["threshold_strategy"] == PRIMARY_THRESHOLD)
    ].sort_values("auc", ascending=False)
    write_table(best_primary, "primary_threshold_oof_ranked_metrics")

    decision = "do_not_promote_harmonization_sensitivity_only"
    oasis_decision = "not_run_adni_gate_failed"
    (OUT_DIR / "oasis_gate_decision.md").write_text(
        "\n".join(
            [
                "# OASIS Gate Decision",
                "",
                f"Decision: `{oasis_decision}`.",
                "",
                "Frozen OASIS inference was not run. The ADNI promoted-convention OOF-ECDF gate failed:",
                f"AUC `{primary['auc']:.6f}` < promoted `{PROMOTED['auc']:.6f}`, PR-AUC `{primary['pr_auc']:.6f}` < promoted `{PROMOTED['pr_auc']:.6f}`,",
                f"and Philips CN FPR `{philips_fpr:.6f}` > promoted `{PROMOTED['philips_cn_fpr']:.4f}`.",
                "",
                "The scanner leakage reduction is retained as a harmonization sensitivity finding, not as a model-selection basis.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (OUT_DIR / "final_decision.md").write_text(
        "\n".join(
            [
                "# Final Decision",
                "",
                f"Decision: `{decision}`.",
                "",
                "The foldwise input ComBat branch completed Stage B classifier-only readout and OOF score calibration.",
                "It materially reduced latent scanner/manufacturer leakage, but did not preserve the promoted model's",
                "ADNI ranking metrics and did not reduce Philips CN FPR under the promoted OOF-ECDF target-sensitivity threshold.",
                "",
                "Primary promoted-convention row:",
                f"- AUC `{primary['auc']:.6f}`",
                f"- PR-AUC `{primary['pr_auc']:.6f}`",
                f"- BA `{primary['balanced_accuracy']:.6f}`",
                f"- Sens `{primary['sensitivity']:.6f}`",
                f"- Spec `{primary['specificity']:.6f}`",
                f"- F1 `{primary['f1']:.6f}`",
                f"- Philips CN FPR `{philips_fpr:.6f}`",
                f"- test latent scanner leakage BA `{scanner_test:.6f}`",
                "",
                "Interpretation: foldwise ComBat succeeded as an acquisition-nuisance reduction intervention,",
                "but the downstream AD/CN readout trade-off is not promotable. Keep this as a harmonization",
                "sensitivity/deconfounding audit only.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (OUT_DIR / "README.md").write_text(
        "\n".join(
            [
                "# Foldcombat Stage B + OOF Completion Audit",
                "",
                f"Input run: `{RUN_DIR}`",
                f"Stage B readout: `{STAGEB_DIR}`",
                f"OOF calibration: `{OOF_DIR}`",
                "",
                "Scope: classifier-only Stage B and OOF score calibration on completed foldcombat latent caches.",
                "No VAE retraining, no tensor/metadata edits, no OASIS threshold fitting, and no OASIS inference for selection.",
                "",
                f"Final decision: `{decision}`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (OUT_DIR / "command_log.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "run_dir": str(RUN_DIR),
                "stageb_dir": str(STAGEB_DIR),
                "oof_dir": str(OOF_DIR),
                "output_dir": str(OUT_DIR),
                "stageb_command": [
                    "/home/diego/anaconda3/envs/vae_ad/bin/python",
                    "scripts/revision_bspc_2026/run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep.py",
                    "--run-dir",
                    str(RUN_DIR),
                    "--output-dir",
                    str(STAGEB_DIR),
                    "--outer-folds",
                    "5",
                    "--inner-folds",
                    "5",
                    "--models",
                    "logreg_l2",
                    "--readout-feature-sets",
                    "z_plus_age_sex",
                    "--reuse-latent-cache",
                    "--device",
                    "cpu",
                    "--n-jobs",
                    "4",
                ],
                "oof_command": [
                    "/home/diego/anaconda3/envs/vae_ad/bin/python",
                    "scripts/revision_bspc_2026/run_recover035_latent384_beta3p75_stageB_oof_score_calibration.py",
                    "--run-dir",
                    str(RUN_DIR),
                    "--output-dir",
                    str(OOF_DIR),
                    "--n-jobs",
                    "4",
                ],
                "audit_script": str(Path(__file__)),
                "final_decision": decision,
                "oasis_gate_decision": oasis_decision,
                "guardrails": [
                    "no_vae_retraining",
                    "no_tensor_modification",
                    "no_metadata_modification",
                    "no_model_artifact_overwrite",
                    "no_oasis_threshold_or_calibration_fitting",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
