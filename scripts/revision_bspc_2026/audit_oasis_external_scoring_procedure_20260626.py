#!/usr/bin/env python3
"""Read-only audit of final-model OASIS external scoring procedure."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


ROOT = Path("/home/diego/proyectos/vae_AD")
RESULTS = ROOT / "results/revision_bspc_2026"
OUT_DIR = RESULTS / "oasis_external_scoring_procedure_audit_20260626"

SCORING_DIR = RESULTS / "oasis_mega_90_90_external_inference_model_panel_20260604"
SCORING_SCRIPT = ROOT / "scripts/revision_bspc_2026/score_oasis_mega_90_90_external_inference_model_panel_20260604.py"
STRESS_PACKAGE = RESULTS / "promoted_model_oasis_external_stress_test_package_20260601"

CANDIDATE = "promoted_beta3p75_oof_ecdf"
FINAL_RUN = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
FINAL_OOF_DIR = RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration"
PRIMARY_MODEL = "logreg_l2_original"
PRIMARY_FEATURE_SET = "z_plus_age_sex"
PRIMARY_CALIB = "oof_ecdf"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
BUILDS = ["concatenated_timeseries", "runwise_140TR_pilot_parity", "runwise164_pilot_parity"]


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else np.nan


def binary_metrics(y: pd.Series, score: pd.Series, pred: pd.Series) -> dict[str, Any]:
    yy = y.astype(int).to_numpy()
    ss = score.astype(float).to_numpy()
    pp = pred.astype(int).to_numpy()
    tn, fp, fn, tp = confusion_matrix(yy, pp, labels=[0, 1]).ravel()
    sens = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)
    return {
        "n": int(len(yy)),
        "n_cn": int((yy == 0).sum()),
        "n_ad": int((yy == 1).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": float(np.nanmean([sens, spec])),
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
        "auc": float(roc_auc_score(yy, ss)) if len(np.unique(yy)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(yy, ss)) if len(np.unique(yy)) == 2 else np.nan,
        "predicted_ad_rate": float(pp.mean()) if len(pp) else np.nan,
    }


def md_table(df: pd.DataFrame, title: str | None = None) -> str:
    lines: list[str] = []
    if title:
        lines += [f"# {title}", ""]
    if df.empty:
        lines.append("_No rows._")
    else:
        lines.append(df.to_markdown(index=False, floatfmt=".6g"))
    lines.append("")
    return "\n".join(lines)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def path_record(kind: str, label: str, path: Path, note: str = "") -> dict[str, Any]:
    return {
        "artifact_type": kind,
        "label": label,
        "path": str(path),
        "exists": bool(path.exists()),
        "note": note,
    }


def build_artifact_inventory() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append(path_record("scoring_script", "final_mega_oasis_panel_script", SCORING_SCRIPT, "Defines the final OASIS scoring rule."))
    rows.append(path_record("scoring_output_dir", "final_mega_oasis_panel", SCORING_DIR, "Contains final selected model OASIS predictions and metrics."))
    rows.append(path_record("summary_package", "promoted_external_stress_test_package", STRESS_PACKAGE, "Manuscript-oriented summaries derived from external OASIS artifacts."))
    rows.append(path_record("adni_run_dir", "final_selected_adni_run", FINAL_RUN))
    rows.append(path_record("adni_oof_calibration_dir", "final_selected_stageB_oof_calibration", FINAL_OOF_DIR))
    for f in ["predictions.csv", "primary_metrics.csv", "foldwise_metrics.csv", "fold_readout_reconstruction_audit.csv", "tensor_artifact_validation.csv", "artifact_validation.csv", "command_log.json"]:
        rows.append(path_record("oasis_output_file", f, SCORING_DIR / f))
    for f in ["calib_foldwise_metrics.csv", "calib_pooled_metrics.csv", "calib_predictions.csv"]:
        rows.append(path_record("adni_oof_calibration_file", f, FINAL_OOF_DIR / f))
    rows.append(path_record("adni_run_config", "run_config.json", FINAL_RUN / "run_config.json"))
    rows.append(path_record("adni_classifier_latent_cache", "classifier_only_readout/latent_cache", FINAL_RUN / "classifier_only_readout/latent_cache"))
    for fold in range(1, 6):
        rows.append(path_record("adni_fold_vae_checkpoint", f"fold_{fold}_vae_model", FINAL_RUN / f"fold_{fold}" / f"vae_model_fold_{fold}.pt"))
        rows.append(path_record("adni_fold_normalization", f"fold_{fold}_vae_norm_params", FINAL_RUN / f"fold_{fold}" / "vae_norm_params.joblib"))
        rows.append(path_record("adni_fold_train_dev_latents", f"fold_{fold}_trainDev_latent_mu", FINAL_RUN / "classifier_only_readout" / "latent_cache" / f"fold_{fold}_trainDev_latent_mu.csv"))
    tensor_validation = pd.read_csv(SCORING_DIR / "tensor_artifact_validation.csv")
    for _, r in tensor_validation.iterrows():
        rows.append(path_record("oasis_tensor", str(r["build_candidate"]), Path(str(r["tensor_path"]))))
    return pd.DataFrame(rows)


def prediction_files_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, path, role in [
        ("final_panel_predictions", SCORING_DIR / "predictions.csv", "fold_model plus ensemble_mean_score_majority_vote predictions for 3 OASIS builds and candidate panel"),
        ("final_panel_primary_metrics", SCORING_DIR / "primary_metrics.csv", "ensemble metrics for each build/candidate"),
        ("final_panel_foldwise_metrics", SCORING_DIR / "foldwise_metrics.csv", "per-ADNI-fold OASIS metrics"),
        ("stress_package_auc_pr_summary", STRESS_PACKAGE / "tables/oasis_auc_pr_summary_with_bootstrap_ci.csv", "manuscript stress-test summary, derived/secondary"),
        ("stress_package_metric_context", STRESS_PACKAGE / "tables/adni_and_oasis_metric_context.csv", "ADNI/OASIS context summary, derived/secondary"),
    ]:
        info: dict[str, Any] = path_record("prediction_or_metric_file", label, path, role)
        if path.exists() and path.suffix == ".csv":
            try:
                df = pd.read_csv(path)
                info["rows"] = len(df)
                info["columns"] = "|".join(df.columns.astype(str).tolist())
                if "candidate" in df.columns:
                    info["contains_final_candidate"] = bool(df["candidate"].astype(str).eq(CANDIDATE).any())
                elif "model_label" in df.columns:
                    info["contains_final_candidate"] = bool(df["model_label"].astype(str).str.contains("recover035|promoted", case=False, regex=True).any())
                else:
                    info["contains_final_candidate"] = ""
            except Exception as exc:
                info["read_error"] = str(exc)
        rows.append(info)
    return pd.DataFrame(rows)


def metric_reconstruction() -> pd.DataFrame:
    preds = pd.read_csv(SCORING_DIR / "predictions.csv")
    saved = pd.read_csv(SCORING_DIR / "primary_metrics.csv")
    rows: list[dict[str, Any]] = []
    subset = preds[preds["candidate"].astype(str).eq(CANDIDATE)].copy()
    if subset.empty:
        raise RuntimeError(f"No predictions found for {CANDIDATE}")
    for (build, level, fold), g in subset.groupby(["build_candidate", "prediction_level", "fold"], dropna=False):
        if level == "fold_model" or level == "ensemble_mean_score_majority_vote":
            metrics = binary_metrics(g["y"], g["y_score"], g["y_pred"])
            row = {
                "build_candidate": build,
                "candidate": CANDIDATE,
                "prediction_level": level,
                "fold": fold,
                **metrics,
            }
            saved_row = saved[
                saved["build_candidate"].astype(str).eq(str(build))
                & saved["candidate"].astype(str).eq(CANDIDATE)
                & saved["prediction_level"].astype(str).eq(str(level))
                & saved["fold"].astype(str).eq(str(fold))
            ]
            if len(saved_row) == 1:
                s = saved_row.iloc[0]
                for col in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]:
                    row[f"saved_{col}"] = float(s[col])
                    row[f"delta_{col}"] = float(row[col] - s[col])
                row["saved_metric_match"] = bool(all(abs(row.get(f"delta_{c}", 0.0)) < 1e-12 for c in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]))
            else:
                row["saved_metric_match"] = "not_in_primary_metrics"
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["build_candidate", "prediction_level", "fold"]).reset_index(drop=True)


def procedure_text(metrics: pd.DataFrame, artifact_df: pd.DataFrame) -> str:
    ens = metrics[metrics["prediction_level"].eq("ensemble_mean_score_majority_vote")]
    ens = ens[ens["build_candidate"].isin(BUILDS)].copy()
    lines = [
        "# OASIS External-Validation Scoring Procedure Audit",
        "",
        "## Scope",
        "",
        f"Selected ADNI pipeline: `{FINAL_RUN.name}`.",
        f"Audited OASIS output package: `{SCORING_DIR}`.",
        "",
        "## Final OASIS Outputs Located",
        "",
        "The final selected model appears in the 90CN/90AD mega-OASIS panel as `promoted_beta3p75_oof_ecdf`. "
        "That panel includes three OASIS tensor builds: `concatenated_timeseries`, `runwise_140TR_pilot_parity`, and `runwise164_pilot_parity`.",
        "",
        "## ADNI Fold Artifacts Used",
        "",
        "- VAE encoders/checkpoints: `fold_*/vae_model_fold_*.pt` from the selected ADNI run.",
        "- Channel/input normalization: `fold_*/vae_norm_params.joblib`, applied to OASIS tensors before encoding.",
        "- Classifier/readout data: ADNI `classifier_only_readout/latent_cache/fold_*_trainDev_latent_mu.csv`.",
        "- Score transform/calibration: fold-local OOF-ECDF was reconstructed from ADNI train/dev inner-OOF scores, using `interp` mode for the final selected non-harmonized readout.",
        "- Thresholds: fold-specific ADNI thresholds from `calib_foldwise_metrics.csv` for `logreg_l2_original / z_plus_age_sex / oof_ecdf / inner_oof_target_sens_ge_0p70_max_spec`.",
        "",
        "No serialized Stage B estimator was used for this final OASIS panel. The scorer deterministically reconstructed each fold's logistic readout from ADNI train/dev latent caches using the fixed C grid `[0.001, 0.01, 0.1, 1.0]`, 5-fold inner CV, class weighting, and the same feature set `mu + Age + Sex`. This reconstruction happened in memory and did not use OASIS labels.",
        "",
        "## Scoring Rule",
        "",
        "Each OASIS subject was scored by all five ADNI fold-specific VAE/readout pipelines. The saved prediction table contains both per-fold rows (`prediction_level=fold_model`) and an ensemble row (`prediction_level=ensemble_mean_score_majority_vote`). The ensemble score is the arithmetic mean of the five fold OOF-ECDF scores. The ensemble class label is a majority vote over the five fold threshold decisions, with positive votes >=3 labeled AD.",
        "",
        "## Metric Computation",
        "",
        "Final OASIS ROC-AUC and PR-AUC were computed from the ensemble `y_score` values and OASIS labels. Threshold metrics were computed from the ensemble majority-vote `y_pred`. OASIS labels were used only at this final metric-computation stage.",
        "",
        "## Reconstructed Final Ensemble Metrics",
        "",
        ens[["build_candidate", "n", "n_cn", "n_ad", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "tn", "fp", "fn", "tp"]].to_markdown(index=False, floatfmt=".6g"),
        "",
    ]
    return "\n".join(lines) + "\n"


def interpretation_text(metrics: pd.DataFrame) -> str:
    ens = metrics[metrics["prediction_level"].eq("ensemble_mean_score_majority_vote")].copy()
    lines = [
        "# Audit Interpretation",
        "",
        "Status: `read_only_complete`.",
        "",
        "The final selected ADNI model was externally scored on OASIS by applying frozen ADNI fold-specific VAE checkpoints, fold-specific ADNI normalization, and ADNI-only Stage B readout reconstruction. OASIS was not used for VAE training, feature scaling, classifier fitting, OOF-ECDF calibration, threshold selection, hyperparameter selection, or model selection.",
        "",
        "The reported external AUC/PR-AUC values are ensemble-ranking metrics: each OASIS subject receives five fold-specific scores and the final score is their mean. This differs from a single-fold external scorer and is consistent with the 5-fold ADNI training protocol.",
        "",
        "The runwise builds should be interpreted as external tensor-construction stress tests. The strongest final-model OASIS ranking in the located final panel is `runwise164_pilot_parity`; this does not imply OASIS-based model selection because the ADNI model and thresholds were locked before OASIS scoring.",
        "",
        "## Final selected model ensemble rows",
        "",
        ens[["build_candidate", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]].to_markdown(index=False, floatfmt=".6g"),
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_df = build_artifact_inventory()
    pred_files = prediction_files_table()
    metrics = metric_reconstruction()

    artifact_df.to_csv(OUT_DIR / "oasis_artifact_inventory.csv", index=False)
    pred_files.to_csv(OUT_DIR / "oasis_prediction_files.csv", index=False)
    metrics.to_csv(OUT_DIR / "oasis_metric_reconstruction.csv", index=False)

    (OUT_DIR / "oasis_scoring_procedure.md").write_text(procedure_text(metrics, artifact_df), encoding="utf-8")
    (OUT_DIR / "audit_interpretation.md").write_text(interpretation_text(metrics), encoding="utf-8")

    log = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "output_dir": str(OUT_DIR),
        "inputs": {
            "scoring_dir": str(SCORING_DIR),
            "scoring_script": str(SCORING_SCRIPT),
            "final_run": str(FINAL_RUN),
            "final_oof_dir": str(FINAL_OOF_DIR),
        },
        "candidate_audited": CANDIDATE,
        "primary_convention": {
            "model_name": PRIMARY_MODEL,
            "feature_set": PRIMARY_FEATURE_SET,
            "calib_method": PRIMARY_CALIB,
            "threshold_strategy": PRIMARY_THRESHOLD,
        },
        "guardrails": {
            "did_train_models": False,
            "did_modify_tensors": False,
            "did_modify_metadata": False,
            "did_modify_configs": False,
            "did_modify_predictions": False,
            "did_modify_ledgers": False,
            "did_modify_manuscript": False,
            "did_run_oasis_scoring": False,
        },
        "outputs": [
            "oasis_scoring_procedure.md",
            "oasis_artifact_inventory.csv",
            "oasis_prediction_files.csv",
            "oasis_metric_reconstruction.csv",
            "audit_interpretation.md",
            "command_log.txt",
        ],
    }
    (OUT_DIR / "command_log.txt").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_DIR}")
    print(metrics[metrics["prediction_level"].eq("ensemble_mean_score_majority_vote")][["build_candidate", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
