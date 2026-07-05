#!/usr/bin/env python3
"""Audit OASIS run handling and final subject-level scoring for BSPC revision."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


ROOT = Path("/home/diego/proyectos/vae_AD")
RESULTS = ROOT / "results/revision_bspc_2026"
OUT = RESULTS / "oasis_run_handling_and_subject_level_audit_20260626"

MEGA_DIR = RESULTS / "oasis_mega_90cn_90ad_pooled_external_validation_20260531"
SCORING_DIR = RESULTS / "oasis_mega_90_90_external_inference_model_panel_20260604"
SCORING_SCRIPT = ROOT / "scripts/revision_bspc_2026/score_oasis_mega_90_90_external_inference_model_panel_20260604.py"
MEGA_BUILD_SCRIPT = ROOT / "scripts/revision_bspc_2026/build_oasis_mega_90cn_90ad_pooled_external_validation_20260531.py"
RUNWISE_NEW_SCRIPT = ROOT / "scripts/revision_bspc_2026/build_oasis_next_60cn_60ad_pilot_parity_runwise_tensors_20260531.py"
PILOT_PREFLIGHT_SCRIPT = ROOT / "scripts/revision_bspc_2026/prepare_oasis_tanda_20260525_connectome_build_preflight.py"

CANDIDATE = "promoted_beta3p75_oof_ecdf"
BUILDS = ["concatenated_timeseries", "runwise_140TR_pilot_parity", "runwise164_pilot_parity"]


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


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
        "auc": float(roc_auc_score(yy, ss)),
        "pr_auc": float(average_precision_score(yy, ss)),
        "balanced_accuracy": float(np.nanmean([sens, spec])),
        "sensitivity": sens,
        "specificity": spec,
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def unique_sessions(df: pd.DataFrame) -> int:
    return int(df[["subject_id", "session_id"]].drop_duplicates().shape[0])


def run_count(df: pd.DataFrame) -> int:
    return int(pd.to_numeric(df["selected_qc_runs"], errors="coerce").fillna(0).sum())


def count_dict(df: pd.DataFrame, y_col: str = "y") -> dict[str, int]:
    y = pd.to_numeric(df[y_col], errors="coerce")
    return {"n_cn": int((y == 0).sum()), "n_ad": int((y == 1).sum())}


def tensor_counts(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as zf:
        subject_ids = pd.Series(np.asarray(zf["subject_ids"]).astype(str))
        session_ids = pd.Series(np.asarray(zf["session_ids"]).astype(str))
        diagnosis = pd.Series(np.asarray(zf["diagnosis"]).astype(str))
        return {
            "n_rows": int(len(subject_ids)),
            "n_subjects": int(subject_ids.nunique()),
            "n_sessions": int(pd.DataFrame({"s": subject_ids, "sess": session_ids}).drop_duplicates().shape[0]),
            "n_cn": int(diagnosis.eq("CN").sum()),
            "n_ad": int(diagnosis.eq("AD_DEMENTIA").sum()),
            "shape": "x".join(map(str, zf["global_tensor_data"].shape)),
        }


def build_data_flow(manifest: pd.DataFrame, tensor_manifest: pd.DataFrame, preds: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    manifest_runs = run_count(manifest)
    for build in BUILDS:
        tensor_path = Path(tensor_manifest.loc[tensor_manifest["build_candidate"].eq(build), "tensor_path"].iloc[0])
        tcounts = tensor_counts(tensor_path)
        note = tensor_manifest.loc[tensor_manifest["build_candidate"].eq(build), "construction_note"].iloc[0]
        rows.append(
            {
                "build_candidate": build,
                "stage": "selected_subject_session_manifest",
                "input_file": str(MEGA_DIR / "mega_manifest.csv"),
                "output_file": str(MEGA_DIR / "mega_manifest.csv"),
                "unit_of_analysis": "subject_session_experiment_row",
                "n_rows": int(len(manifest)),
                "n_subjects": int(manifest["subject_id"].nunique()),
                "n_cn": int(manifest["diagnosis"].eq("CN").sum()),
                "n_ad": int(manifest["diagnosis"].eq("AD_DEMENTIA").sum()),
                "n_sessions": unique_sessions(manifest),
                "n_runs_or_scans": manifest_runs,
                "notes": "Common OASIS 90CN/90AD manifest; selected_qc_runs records how many rs-fMRI runs were used before tensor collapse.",
            }
        )
        rows.append(
            {
                "build_candidate": build,
                "stage": "tensor_construction",
                "input_file": str(tensor_manifest.loc[tensor_manifest["build_candidate"].eq(build), "pilot_source_tensor"].iloc[0])
                + " ; "
                + str(tensor_manifest.loc[tensor_manifest["build_candidate"].eq(build), "new_source_tensor"].iloc[0]),
                "output_file": str(tensor_path),
                "unit_of_analysis": "one_tensor_row_per_subject_session",
                "n_rows": tcounts["n_rows"],
                "n_subjects": tcounts["n_subjects"],
                "n_cn": tcounts["n_cn"],
                "n_ad": tcounts["n_ad"],
                "n_sessions": tcounts["n_sessions"],
                "n_runs_or_scans": manifest_runs,
                "notes": f"{note} Tensor shape {tcounts['shape']}. Runs are not independent tensor rows.",
            }
        )
        sub_fold = preds[
            preds["build_candidate"].eq(build)
            & preds["candidate"].eq(CANDIDATE)
            & preds["prediction_level"].eq("fold_model")
        ]
        rows.append(
            {
                "build_candidate": build,
                "stage": "fold_model_predictions",
                "input_file": str(tensor_path),
                "output_file": str(SCORING_DIR / "predictions.csv"),
                "unit_of_analysis": "subject_session_x_adni_fold",
                "n_rows": int(len(sub_fold)),
                "n_subjects": int(sub_fold["SubjectID"].nunique()),
                "n_cn": int(sub_fold.drop_duplicates(["SubjectID", "session_id", "experiment_id"])["y"].eq(0).sum()),
                "n_ad": int(sub_fold.drop_duplicates(["SubjectID", "session_id", "experiment_id"])["y"].eq(1).sum()),
                "n_sessions": int(sub_fold[["SubjectID", "session_id"]].drop_duplicates().shape[0]),
                "n_runs_or_scans": manifest_runs,
                "notes": "Five rows per subject/session: one score from each ADNI fold-specific VAE/readout pipeline.",
            }
        )
        sub_ens = preds[
            preds["build_candidate"].eq(build)
            & preds["candidate"].eq(CANDIDATE)
            & preds["prediction_level"].eq("ensemble_mean_score_majority_vote")
        ]
        rows.append(
            {
                "build_candidate": build,
                "stage": "ensemble_subject_score",
                "input_file": str(SCORING_DIR / "predictions.csv"),
                "output_file": str(SCORING_DIR / "predictions.csv"),
                "unit_of_analysis": "one_subject_session_ensemble_row",
                "n_rows": int(len(sub_ens)),
                "n_subjects": int(sub_ens["SubjectID"].nunique()),
                "n_cn": int(sub_ens["y"].eq(0).sum()),
                "n_ad": int(sub_ens["y"].eq(1).sum()),
                "n_sessions": int(sub_ens[["SubjectID", "session_id"]].drop_duplicates().shape[0]),
                "n_runs_or_scans": manifest_runs,
                "notes": "Final score is mean over five ADNI fold scores; final y_pred is majority vote over five ADNI-threshold decisions.",
            }
        )
        mrow = metrics[metrics["build_candidate"].eq(build)].iloc[0]
        rows.append(
            {
                "build_candidate": build,
                "stage": "final_metric_computation",
                "input_file": str(SCORING_DIR / "predictions.csv"),
                "output_file": str(OUT / "oasis_metric_recheck.csv"),
                "unit_of_analysis": "one_metric_from_180_subject_session_ensemble_rows",
                "n_rows": 1,
                "n_subjects": int(mrow["n"]),
                "n_cn": int(mrow["n_cn"]),
                "n_ad": int(mrow["n_ad"]),
                "n_sessions": int(sub_ens[["SubjectID", "session_id"]].drop_duplicates().shape[0]),
                "n_runs_or_scans": manifest_runs,
                "notes": "AUC/PR-AUC computed once per build from ensemble subject-level y_score and labels; not mean fold AUC and not run-level AUC.",
            }
        )
    return pd.DataFrame(rows)


def prediction_level_counts(preds: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base_runs = run_count(manifest)
    sub = preds[preds["candidate"].eq(CANDIDATE)].copy()
    for (build, level), g in sub.groupby(["build_candidate", "prediction_level"], dropna=False):
        unique = g.drop_duplicates(["SubjectID", "session_id", "experiment_id"])
        rows.append(
            {
                "build_candidate": build,
                "prediction_file": str(SCORING_DIR / "predictions.csv"),
                "prediction_level": level,
                "n_rows": int(len(g)),
                "n_unique_subjects": int(g["SubjectID"].nunique()),
                "n_cn": int(unique["y"].eq(0).sum()),
                "n_ad": int(unique["y"].eq(1).sum()),
                "n_unique_runs_or_scans_if_available": base_runs,
            }
        )
    return pd.DataFrame(rows).sort_values(["build_candidate", "prediction_level"])


def metric_recheck(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for build in BUILDS:
        g = preds[
            preds["build_candidate"].eq(build)
            & preds["candidate"].eq(CANDIDATE)
            & preds["prediction_level"].eq("ensemble_mean_score_majority_vote")
        ].copy()
        rows.append({"build_candidate": build, **binary_metrics(g["y"], g["y_score"], g["y_pred"])})
    return pd.DataFrame(rows)


def audit_markdown(data_flow: pd.DataFrame, counts: pd.DataFrame, metrics: pd.DataFrame) -> str:
    return f"""# OASIS Run-Handling and Subject-Level Scoring Audit

## Short Answer To Martin

In the final OASIS 90 CN / 90 AD evaluation, the unit of analysis for the reported final AUC/PR-AUC was **one row per OASIS subject/session tensor**, not individual rs-fMRI runs. The final selected model was evaluated on 180 subject/session rows per OASIS build: 90 CN and 90 AD.

When an OASIS subject/session had two usable rs-fMRI runs, those runs were **not counted as independent observations** in the final metrics. The handling depended on the tensor build:

- `concatenated_timeseries`: QC-usable ROI time series were concatenated within subject/session first; then one connectome tensor was computed for that subject/session.
- `runwise_140TR_pilot_parity`: each usable run was truncated to 140 TR, a run-level connectome was computed and normalized, then normalized run connectomes were averaged into one subject/session tensor.
- `runwise164_pilot_parity`: each usable run was used at 164 TR, a run-level connectome was computed and normalized, then normalized run connectomes were averaged into one subject/session tensor.

Thus there was run aggregation **before** ADNI model scoring, at the tensor construction stage. There was also fold-model aggregation **after** scoring: each subject/session tensor was scored by all five ADNI fold-specific pipelines, and the final subject/session score was the mean of the five fold scores.

## Evidence From Files And Code

- Tensor build definitions: `{MEGA_BUILD_SCRIPT}` lines 61-76 define the three final OASIS builds and their construction notes.
- Mega manifest construction: `{MEGA_BUILD_SCRIPT}` lines 305-374 builds `mega_manifest.csv`, preserving `selected_qc_runs`, `selected_run_ids`, and `selected_total_timepoints`.
- Build integrity checks: `{MEGA_BUILD_SCRIPT}` lines 883-888 require exactly 180 rows with 90 CN and 90 AD.
- Tensor merge/order check: `{MEGA_BUILD_SCRIPT}` lines 898-907 save one mega tensor per build and require identical subject/session/experiment order across builds.
- Runwise tensor construction for the new batch: `{RUNWISE_NEW_SCRIPT}` lines 234-272 compute per-run connectomes, normalize each run, average normalized runs, then normalize the average.
- Pilot preflight description: `{PILOT_PREFLIGHT_SCRIPT}` lines 56-64 describes `runwise_connectome_average` versus `concatenated_timeseries`.
- Scoring build selection: `{SCORING_SCRIPT}` lines 73-76 defines the three OASIS tensors used for scoring.
- Final selected ADNI candidate: `{SCORING_SCRIPT}` lines 89-95 defines `promoted_beta3p75_oof_ecdf` and its ADNI run/calibration directories.
- Fold scoring: `{SCORING_SCRIPT}` lines 617-672 loops over ADNI folds, encodes OASIS tensors, reconstructs the ADNI readout, applies OOF-ECDF, and writes fold-model predictions.
- Ensemble aggregation: `{SCORING_SCRIPT}` lines 684-728 groups by subject/session identity, computes mean fold score, and uses positive votes >=3 for the thresholded label.
- Metric computation: `{SCORING_SCRIPT}` lines 733-760 computes metrics from grouped prediction rows using `y`, `y_score`, and `y_pred`.

## Row Counts

### Prediction Levels

{counts.to_markdown(index=False)}

### Final Metric Recheck

{metrics.to_markdown(index=False, floatfmt=".6g")}

## Caveats

The final tensor package fully determines the evaluated unit and scoring aggregation. It records selected run counts and run IDs in `mega_manifest.csv`, but the final `predictions.csv` no longer contains separate run-level score rows because run-level data were collapsed into subject/session tensors before scoring. Therefore this audit can confirm that runs were not independently scored in the final metrics, and can distinguish concatenation versus runwise connectome averaging by build, but it does not re-open raw BOLD files or recompute run-level connectomes.
"""


def aggregation_formula() -> str:
    return """# OASIS Aggregation Formula

Let \\(b\\) denote an OASIS tensor build and \\(s\\) an OASIS subject/session/experiment row.

## Run Aggregation Before Scoring

The OASIS tensor construction first collapses selected rs-fMRI runs into one tensor row per subject/session:

```text
concatenated_timeseries:
  X_s = connectome(concat(TS_{s,r} for r in QC_selected_runs_s))

runwise_140TR_pilot_parity:
  X_s = normalize(mean_r(normalize(connectome(TS_{s,r}[1:140]))))

runwise164_pilot_parity:
  X_s = normalize(mean_r(normalize(connectome(TS_{s,r}[1:164]))))
```

So the scoring input for all builds is one tensor \\(X_s\\) per subject/session. Individual runs are not separate prediction rows.

## Fold-Model Score Aggregation After Scoring

For each ADNI fold \\(f \\in \\{{1,2,3,4,5\\}}\\):

```text
mu_{s,f} = VAE_encoder_f(normalize_f(X_s))
raw_score_{s,f} = logreg_f(mu_{s,f}, Age_s, Sex_s)
score_{s,f} = ECDF_f(raw_score_{s,f})      # ECDF learned from ADNI train/dev only
pred_{s,f} = 1[score_{s,f} >= threshold_f] # threshold_f learned from ADNI only
```

The final ensemble row is:

```text
score_s = mean_f(score_{s,f})
raw_score_s = mean_f(raw_score_{s,f})
threshold_display_s = mean_f(threshold_f)
pred_s = 1[sum_f(pred_{s,f}) >= 3]
```

Final OASIS ROC-AUC and PR-AUC are computed once per build from the 180 subject/session-level pairs:

```text
AUC_b    = ROC_AUC(y_s, score_s for s=1..180)
PR_AUC_b = AveragePrecision(y_s, score_s for s=1..180)
```

They are not mean ± SD over five fold-specific AUCs and not run-level AUCs.
"""


def martin_reply(metrics: pd.DataFrame) -> str:
    best = metrics.loc[metrics["build_candidate"].eq("runwise164_pilot_parity")].iloc[0]
    return (
        "Martín, lo revisé directamente en los scripts, manifests y predicciones guardadas. "
        "En la evaluación OASIS final la unidad del AUC fue sujeto/sesión, no corrida individual. "
        "Si un sujeto tenía 2 runs, no entraron como dos observaciones independientes. "
        "En `concatenated_timeseries` se concatenaron las series temporales QC-usables dentro del sujeto/sesión y se calculó un único conectoma. "
        "En `runwise_140TR` y `runwise164` se calculó un conectoma por run, se normalizó por run y luego se promediaron los conectomas para dejar un único tensor por sujeto/sesión. "
        "Después, ese tensor único se pasó por los 5 modelos/folds ADNI; el score final del sujeto fue el promedio de los 5 scores, y la etiqueta binaria fue voto mayoritario de los 5 thresholds ADNI. "
        "El AUC/PR-AUC final se calculó una sola vez sobre 180 sujetos/sesiones (90 CN, 90 AD) usando esos scores finales, no como promedio de AUCs por fold ni por run. "
        f"Para el build runwise164 del modelo final: AUC={best['auc']:.3f}, PR-AUC={best['pr_auc']:.3f}. "
        "No se usaron labels OASIS para entrenar, escalar, calibrar, elegir threshold, hiperparámetros ni seleccionar modelo; sólo para computar métricas finales."
    )


def manuscript_patch(metrics: pd.DataFrame) -> str:
    rw164 = metrics.loc[metrics["build_candidate"].eq("runwise164_pilot_parity")].iloc[0]
    rw140 = metrics.loc[metrics["build_candidate"].eq("runwise_140TR_pilot_parity")].iloc[0]
    concat = metrics.loc[metrics["build_candidate"].eq("concatenated_timeseries")].iloc[0]
    return rf"""% Replacement text for OASIS external-validation methods/results.

\paragraph{{OASIS external validation.}}
For external validation, OASIS rs-fMRI data were converted to subject/session-level connectivity tensors using the same three channels used by the final ADNI model. When multiple QC-usable rs-fMRI runs were available for an OASIS subject/session, runs were not treated as independent test observations. We evaluated three tensor-construction variants: (i) concatenation of QC-usable ROI time series within subject/session before connectome estimation; (ii) run-wise 140TR connectome estimation with per-run normalization followed by averaging of normalized run connectomes; and (iii) run-wise 164TR connectome estimation with the same run-wise normalization and averaging procedure. Each variant therefore produced one tensor row per OASIS subject/session.

Each OASIS subject/session tensor was then scored by each of the five ADNI-trained fold-specific VAE/readout pipelines. For fold \(f\), the OASIS tensor was normalized with the ADNI fold-specific parameters, encoded with the frozen ADNI VAE encoder, and classified using the ADNI-only logistic readout reconstructed from that fold's ADNI train/development latent cache. The fold score was transformed with the ADNI-derived OOF-ECDF mapping and thresholded using the ADNI-derived operating threshold. The final subject/session score was the arithmetic mean of the five fold scores; the final binary prediction was the majority vote of the five fold threshold decisions. OASIS labels were used only for final external metric computation and were not used for training, feature scaling, calibration, threshold selection, hyperparameter selection, or model selection.

\paragraph{{OASIS results.}}
External OASIS metrics were computed once per tensor-construction variant from one ensemble score per subject/session (90 CN and 90 AD), rather than by averaging fold-wise or run-wise AUCs. For the final selected ADNI model, OASIS performance was build-dependent: concatenated time-series tensors yielded ROC--AUC={concat['auc']:.3f} and PR--AUC={concat['pr_auc']:.3f}; run-wise 140TR tensors yielded ROC--AUC={rw140['auc']:.3f} and PR--AUC={rw140['pr_auc']:.3f}; and run-wise 164TR tensors yielded ROC--AUC={rw164['auc']:.3f} and PR--AUC={rw164['pr_auc']:.3f}. These analyses were treated as external stress tests of transferability and were not used for model selection.
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MEGA_DIR / "mega_manifest.csv")
    tensor_manifest = pd.read_csv(MEGA_DIR / "tensor_manifest.csv")
    preds = pd.read_csv(SCORING_DIR / "predictions.csv")
    preds = preds[preds["candidate"].eq(CANDIDATE)].copy()

    metrics = metric_recheck(preds)
    counts = prediction_level_counts(preds, manifest)
    data_flow = build_data_flow(manifest, tensor_manifest, preds, metrics)

    data_flow.to_csv(OUT / "oasis_data_flow_by_build.csv", index=False)
    counts.to_csv(OUT / "oasis_prediction_level_counts.csv", index=False)
    metrics.to_csv(OUT / "oasis_metric_recheck.csv", index=False)
    (OUT / "oasis_run_handling_audit.md").write_text(audit_markdown(data_flow, counts, metrics), encoding="utf-8")
    (OUT / "oasis_aggregation_formula.md").write_text(aggregation_formula(), encoding="utf-8")
    (OUT / "martin_reply.txt").write_text(martin_reply(metrics) + "\n", encoding="utf-8")
    (OUT / "manuscript_patch_oasis_methods_results.tex").write_text(manuscript_patch(metrics), encoding="utf-8")

    command_log = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "output_dir": str(OUT),
        "inputs": {
            "mega_manifest": str(MEGA_DIR / "mega_manifest.csv"),
            "tensor_manifest": str(MEGA_DIR / "tensor_manifest.csv"),
            "scoring_predictions": str(SCORING_DIR / "predictions.csv"),
            "scoring_primary_metrics": str(SCORING_DIR / "primary_metrics.csv"),
            "scoring_script": str(SCORING_SCRIPT),
            "mega_build_script": str(MEGA_BUILD_SCRIPT),
            "runwise_new_tensor_script": str(RUNWISE_NEW_SCRIPT),
        },
        "guardrails": {
            "did_train_models": False,
            "did_run_oasis_scoring": False,
            "did_modify_tensors": False,
            "did_modify_metadata": False,
            "did_modify_configs": False,
            "did_modify_predictions": False,
            "did_modify_ledgers": False,
            "did_modify_model_artifacts": False,
            "did_modify_manuscript": False,
            "writes_outside_audit_output_folder": False,
        },
        "commands": [
            "/home/diego/anaconda3/envs/vae_ad/bin/python -m py_compile scripts/revision_bspc_2026/audit_oasis_run_handling_subject_level_20260626.py",
            "/home/diego/anaconda3/envs/vae_ad/bin/python scripts/revision_bspc_2026/audit_oasis_run_handling_subject_level_20260626.py",
        ],
    }
    (OUT / "command_log.txt").write_text(json.dumps(command_log, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
