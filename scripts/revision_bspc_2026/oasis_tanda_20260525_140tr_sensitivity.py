#!/usr/bin/env python
"""OASIS 140TR temporal-length sensitivity build and scoring package.

Default mode is dry-run only. Real connectome computation requires
--confirm-build. External scoring requires --confirm-score and an existing
140TR tensor. No input OASIS data are modified, no OASIS threshold fitting is
performed, and OASIS labels are used only for final external metrics.

The ADNI temporal homogenization rule found in the local ADNI build scripts is:
if a ROI time series is longer than 140 TR, use the first 140 TR
(`sigs[:140, :]`). The current OASIS QC-ok ROI runs are 164 TR, so this package
uses the first 140 TR from each QC-usable run, computes run-level connectomes,
and averages run-level connectomes per subject/session.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_oasis_tanda_20260525_connectomes import (  # noqa: E402
    CHANNEL_NAMES,
    BuildContext,
    combine_run_matrices,
    compute_channels,
    load_context,
    load_roi_timeseries,
    selected_run_rows,
    write_csv_md,
)
from score_oasis_tanda_20260525_external_adni import (  # noqa: E402
    PRIMARY_THRESHOLD_STRATEGY,
    binary_metrics,
    resolve_device,
)
from score_oasis_tanda_20260525_secondary_models import (  # noqa: E402
    DEFAULT_CH1_RUN,
    DEFAULT_MFR_RUN,
    DEFAULT_OUTPUT_DIR as DEFAULT_SECONDARY_SCORING_DIR,
    DEFAULT_PRIMARY_ADNI_RUN,
    ModelSpec,
    available_specs,
    make_ensemble_predictions,
    metric_tables,
    model_required_paths,
    score_one_spec_on_tensor,
    threshold_transfer_table,
)


DEFAULT_PREFLIGHT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "oasis_tanda_2026_05_25_connectome_build_preflight"
)
DEFAULT_ROI_MAPPING_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "oasis_tanda_2026_05_25_roi_mapping_audit"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "oasis_tanda_2026_05_25_140TR_sensitivity"
)
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "Tanda_2026_05_25"
BUILD_CANDIDATE = "runwise_140TR_connectome_average"
TENSOR_FILENAME = f"tensor_{BUILD_CANDIDATE}.npz"
TARGET_TR = 140


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--preflight-dir", type=Path, default=DEFAULT_PREFLIGHT_DIR)
    parser.add_argument("--roi-mapping-audit-dir", type=Path, default=DEFAULT_ROI_MAPPING_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reference-secondary-scoring-dir", type=Path, default=DEFAULT_SECONDARY_SCORING_DIR)
    parser.add_argument("--primary-adni-run-dir", type=Path, default=DEFAULT_PRIMARY_ADNI_RUN)
    parser.add_argument("--secondary-ch1-run-dir", type=Path, default=DEFAULT_CH1_RUN)
    parser.add_argument("--manufacturer-conditioned-run-dir", type=Path, default=DEFAULT_MFR_RUN)
    parser.add_argument("--confirm-build", action="store_true")
    parser.add_argument("--confirm-score", action="store_true")
    parser.add_argument("--fail-if-output-exists", action="store_true")
    parser.add_argument("--n-jobs-mi", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--dry-run", action="store_true", help="Explicit dry-run alias; default without confirmation flags.")
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def crop_to_140tr(ts: np.ndarray) -> tuple[np.ndarray, str]:
    if ts.shape[0] < TARGET_TR:
        raise ValueError(f"OASIS 140TR sensitivity requires at least {TARGET_TR} TR, got {ts.shape[0]}")
    if ts.shape[0] == TARGET_TR:
        return ts.astype(np.float32), "already_140TR"
    return ts[:TARGET_TR, :].astype(np.float32), "first_140TR"


def make_build_manifest(context: BuildContext) -> pd.DataFrame:
    selected = selected_run_rows(context.run_selection)
    rows: list[dict[str, Any]] = []
    for _, run in selected.iterrows():
        n_timepoints = int(float(run["n_timepoints"]))
        rows.append(
            {
                "subject_id": run["subject_id"],
                "session_id": run["session_id"],
                "run_id": run["run_id"],
                "experiment_id": run["experiment_id"],
                "diagnosis": run["diagnosis"],
                "roi_txt_path": run["roi_txt_path"],
                "original_timepoints": n_timepoints,
                "selected_timepoints": TARGET_TR if n_timepoints >= TARGET_TR else 0,
                "temporal_selection_rule": "first_140TR_if_run_length_ge_140",
                "start_index_0based_inclusive": 0 if n_timepoints >= TARGET_TR else np.nan,
                "end_index_0based_exclusive": TARGET_TR if n_timepoints >= TARGET_TR else np.nan,
                "usable_for_140TR": bool(n_timepoints >= TARGET_TR),
                "reason": "" if n_timepoints >= TARGET_TR else f"run_has_{n_timepoints}_TR",
            }
        )
    return pd.DataFrame(rows)


def copy_subject_manifest(context: BuildContext, output_dir: Path) -> pd.DataFrame:
    subjects = context.subject_manifest[context.subject_manifest["planned_include_subject_session"].astype(bool)].copy()
    subjects = subjects.sort_values(["subject_id", "session_id"]).reset_index(drop=True)
    subjects["build_candidate"] = BUILD_CANDIDATE
    subjects["temporal_selection_rule"] = "runwise_first_140TR_then_average_connectomes"
    subjects["target_tr_per_run"] = TARGET_TR
    write_csv_md(subjects, output_dir / "subject_manifest.csv", output_dir / "subject_manifest.md", "OASIS 140TR Subject Manifest")
    return subjects


def save_roi_mapping(context: BuildContext, output_dir: Path) -> None:
    write_csv_md(context.roi_mapping, output_dir / "roi_mapping_used.csv", output_dir / "roi_mapping_used.md", "ROI Mapping Used")


def build_140tr_tensor(context: BuildContext, output_dir: Path, n_jobs_mi: int) -> tuple[Path, pd.DataFrame]:
    selected = selected_run_rows(context.run_selection)
    build_manifest = make_build_manifest(context)
    bad = build_manifest[~build_manifest["usable_for_140TR"]]
    if not bad.empty:
        raise ValueError("Some selected OASIS runs have fewer than 140 TR:\n" + bad.to_string(index=False))
    subjects = copy_subject_manifest(context, output_dir)
    selected_by_session = {
        key: group.sort_values("run_id")
        for key, group in selected.groupby(["subject_id", "session_id"], dropna=False)
    }

    tensors: list[np.ndarray] = []
    qc_rows: list[dict[str, Any]] = []
    for _, subj in subjects.iterrows():
        key = (subj["subject_id"], subj["session_id"])
        run_group = selected_by_session.get(key)
        if run_group is None or run_group.empty:
            raise ValueError(f"No selected QC-ok runs for {key}")
        run_matrices: list[np.ndarray] = []
        run_ids: list[str] = []
        original_lengths: list[int] = []
        for _, run in run_group.iterrows():
            sid = f"{run['subject_id']}_{run['session_id']}_{run['run_id']}"
            ts = load_roi_timeseries(str(run["roi_txt_path"]), context.roi_mapping, sid)
            ts_140, rule = crop_to_140tr(ts)
            if rule not in {"first_140TR", "already_140TR"}:
                raise RuntimeError(f"Unexpected temporal rule for {sid}: {rule}")
            run_matrices.append(compute_channels(ts_140, n_jobs_mi=n_jobs_mi))
            run_ids.append(str(run["run_id"]))
            original_lengths.append(int(ts.shape[0]))
        subject_tensor = combine_run_matrices(run_matrices)
        tensors.append(subject_tensor)
        qc_rows.append(
            {
                "subject_id": subj["subject_id"],
                "session_id": subj["session_id"],
                "experiment_id": subj["experiment_id"],
                "build_candidate": BUILD_CANDIDATE,
                "n_runs_used": len(run_matrices),
                "run_ids_used": ";".join(run_ids),
                "original_timepoints_by_run": ";".join(str(x) for x in original_lengths),
                "selected_timepoints_per_run": TARGET_TR,
                "temporal_selection_rule": "first_140TR_per_run",
                "tensor_shape": "3x131x131",
                "tensor_finite_fraction": float(np.isfinite(subject_tensor).mean()),
                "tensor_abs_max": float(np.nanmax(np.abs(subject_tensor))),
            }
        )
    tensor = np.stack(tensors, axis=0).astype(np.float32)
    roi_names = context.roi_mapping["ADNI_final_ROI_name"].astype(str).to_numpy()
    network_labels = np.array(["unknown_external_oasis"] * len(roi_names), dtype=str)
    tensor_path = output_dir / TENSOR_FILENAME
    np.savez_compressed(
        tensor_path,
        global_tensor_data=tensor,
        subject_ids=subjects["subject_id"].astype(str).to_numpy(),
        session_ids=subjects["session_id"].astype(str).to_numpy(),
        experiment_ids=subjects["experiment_id"].astype(str).to_numpy(),
        diagnosis=subjects["diagnosis"].astype(str).to_numpy(),
        channel_names=np.array(CHANNEL_NAMES, dtype=str),
        rois_count=np.array(131, dtype=np.int32),
        roi_order_name=np.array("aal3_manual_yeo17_order", dtype=str),
        roi_names_in_order=roi_names.astype(str),
        network_labels_in_order=network_labels,
        build_candidate=np.array(BUILD_CANDIDATE, dtype=str),
        temporal_selection_rule=np.array("first_140TR_per_run_then_average_connectomes", dtype=str),
        target_len_ts=np.array(TARGET_TR, dtype=np.int32),
        python_bandpass_applied=np.array(False),
        external_validation_only=np.array(True),
    )
    return tensor_path, pd.DataFrame(qc_rows)


def model_specs(args: argparse.Namespace) -> list[ModelSpec]:
    primary = normalize_path(args.primary_adni_run_dir)
    ch1 = normalize_path(args.secondary_ch1_run_dir)
    mfr = normalize_path(args.manufacturer_conditioned_run_dir)
    return [
        ModelSpec("locked_v5_1b_ch1_0_2_horizon4480", primary, primary / "classifier_only_readout", "reference_primary", False),
        ModelSpec("secondary_ch1_only_offdiag_channelmean", ch1, ch1 / "classifier_only_readout", "secondary_simplified_auroc", False),
        ModelSpec(
            "secondary_manufacturer_conditioned_deconfounding",
            mfr,
            mfr / "classifier_only_readout_z_plus_age_sex",
            "secondary_deconfounding_sensitivity",
            True,
        ),
    ]


def validate_scoring_artifacts(specs: Sequence[ModelSpec], tensor_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "artifact_type": "oasis_140tr_tensor",
            "label": BUILD_CANDIDATE,
            "model_label": "",
            "path": str(tensor_path),
            "exists": tensor_path.exists(),
            "optional": False,
        }
    ]
    for spec in specs:
        for path in model_required_paths(spec):
            rows.append(
                {
                    "artifact_type": "adni_model_artifact",
                    "label": path.name,
                    "model_label": spec.model_label,
                    "path": str(path),
                    "exists": path.exists(),
                    "optional": bool(spec.optional),
                }
            )
    return pd.DataFrame(rows)


def normalize_oasis_sex(value: Any) -> str:
    text = str(value).strip().upper()
    if text in {"1", "M", "MALE"}:
        return "M"
    if text in {"2", "F", "FEMALE"}:
        return "F"
    return "UNKNOWN"


def load_140tr_tensor(output_dir: Path) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    tensor_path = output_dir / TENSOR_FILENAME
    if not tensor_path.exists():
        raise FileNotFoundError(f"Missing 140TR tensor: {tensor_path}")
    data = np.load(tensor_path, allow_pickle=True)
    tensor = np.asarray(data["global_tensor_data"], dtype=np.float32)
    channel_names = data["channel_names"].astype(str).tolist()
    subjects = pd.DataFrame(
        {
            "SubjectID": data["subject_ids"].astype(str),
            "session_id": data["session_ids"].astype(str),
            "experiment_id": data["experiment_ids"].astype(str),
            "diagnosis": data["diagnosis"].astype(str),
        }
    )
    manifest = pd.read_csv(output_dir / "subject_manifest.csv").rename(columns={"subject_id": "SubjectID"})
    cols = [
        c
        for c in [
            "SubjectID",
            "session_id",
            "Manufacturer",
            "ScannerModel",
            "age_at_MR",
            "sex",
            "selected_qc_runs",
            "selected_run_ids",
            "selected_total_timepoints",
        ]
        if c in manifest.columns
    ]
    subjects = subjects.merge(manifest[cols].drop_duplicates(["SubjectID", "session_id"]), on=["SubjectID", "session_id"], how="left")
    subjects["ResearchGroup_Mapped"] = subjects["diagnosis"].map({"CN": "CN", "AD_DEMENTIA": "AD"})
    subjects["y_true"] = subjects["ResearchGroup_Mapped"].map({"CN": 0, "AD": 1}).astype(int)
    subjects["Age"] = pd.to_numeric(subjects.get("age_at_MR"), errors="coerce")
    subjects["Sex"] = subjects.get("sex", "UNKNOWN").map(normalize_oasis_sex)
    return tensor, subjects, channel_names


def score_140tr_tensor(args: argparse.Namespace, output_dir: Path, artifact_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    specs = available_specs(model_specs(args), artifact_df)
    device = resolve_device(args.device)
    tensor, subjects, channel_names = load_140tr_tensor(output_dir)
    frames: list[pd.DataFrame] = []
    for spec in specs:
        pred = score_one_spec_on_tensor(
            spec=spec,
            tensor=tensor,
            oasis_subjects=subjects,
            oasis_channel_names=channel_names,
            batch_size=int(args.batch_size),
            device=device,
        )
        pred["build_candidate"] = BUILD_CANDIDATE
        frames.append(pred)
    fold_predictions = pd.concat(frames, ignore_index=True)
    ensemble = make_ensemble_predictions(fold_predictions)
    predictions = pd.concat([fold_predictions, ensemble], ignore_index=True, sort=False)
    metrics, confusion, dist = metric_tables(predictions)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    threshold_summary = threshold_transfer_table(predictions)
    write_csv_md(threshold_summary, output_dir / "threshold_transfer_summary.csv", output_dir / "threshold_transfer_summary.md", "Threshold Transfer Summary")
    return metrics, confusion, dist


def placeholder_tables(output_dir: Path, reason: str) -> None:
    placeholder = pd.DataFrame([{"status": "pending", "reason": reason, "build_candidate": BUILD_CANDIDATE}])
    write_csv_md(placeholder, output_dir / "external_scoring_metrics.csv", output_dir / "external_scoring_metrics.md", "External Scoring Metrics")
    write_csv_md(placeholder, output_dir / "confusion_matrices.csv", output_dir / "confusion_matrices.md", "Confusion Matrices")
    write_csv_md(placeholder, output_dir / "score_distribution_summary.csv", output_dir / "score_distribution_summary.md", "Score Distribution Summary")
    write_csv_md(placeholder, output_dir / "comparison_vs_concatenated_and_runwise164.csv", output_dir / "comparison_vs_concatenated_and_runwise164.md", "Comparison vs Existing OASIS Builds")


def comparison_vs_existing(output_dir: Path, args: argparse.Namespace, metrics_140: pd.DataFrame) -> pd.DataFrame:
    ref_path = normalize_path(args.reference_secondary_scoring_dir) / "primary_metrics.csv"
    if not ref_path.exists():
        return pd.DataFrame(
            [
                {
                    "status": "reference_missing",
                    "reference_path": str(ref_path),
                    "build_candidate": BUILD_CANDIDATE,
                }
            ]
        )
    ref = pd.read_csv(ref_path)
    ref = ref[ref["prediction_level"].eq("ensemble_mean_score_majority_vote")].copy()
    cur = metrics_140[metrics_140["prediction_level"].eq("ensemble_mean_score_majority_vote")].copy()
    rows: list[dict[str, Any]] = []
    for _, cur_row in cur.iterrows():
        model = cur_row["adni_model"]
        matched = ref[ref["adni_model"].eq(model)]
        for _, ref_row in matched.iterrows():
            row: dict[str, Any] = {
                "adni_model": model,
                "reference_build_candidate": ref_row["build_candidate"],
                "current_build_candidate": BUILD_CANDIDATE,
            }
            for metric in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]:
                row[f"reference_{metric}"] = float(ref_row[metric])
                row[f"current_140tr_{metric}"] = float(cur_row[metric])
                row[f"delta_140tr_minus_reference_{metric}"] = float(cur_row[metric]) - float(ref_row[metric])
            rows.append(row)
    return pd.DataFrame(rows)


def write_recommendation(
    output_dir: Path,
    build_confirmed: bool,
    score_confirmed: bool,
    metrics: pd.DataFrame | None,
) -> None:
    lines = [
        "# Final Recommendation",
        "",
        "Decision: `temporal_length_sensitivity_only`",
        "",
        "This package tests whether OASIS external scoring is sensitive to ADNI-like 140TR temporal homogenization.",
        "It must not be used for model selection or OASIS threshold fitting.",
        "",
        "## Guardrails",
        "",
        f"- Build confirmed: `{build_confirmed}`",
        f"- Scoring confirmed: `{score_confirmed}`",
        "- OASIS labels are used only for final metrics.",
        "- ADNI-derived thresholds are used for all primary operating points.",
        "- No model training is performed on OASIS.",
        "",
    ]
    if metrics is None:
        lines += [
            "## Status",
            "",
            "Dry-run/package preparation completed. The 140TR tensor and scoring tables remain pending until `--confirm-build` and `--confirm-score` are explicitly used.",
        ]
    else:
        ens = metrics[metrics["prediction_level"].eq("ensemble_mean_score_majority_vote")].copy()
        lines += [
            "## 140TR Ensemble Metrics",
            "",
            ens.to_markdown(index=False),
            "",
            "These results are a temporal-length sensitivity analysis. They do not supersede the locked ADNI model or the primary OASIS external validation protocol by themselves.",
        ]
    (output_dir / "final_recommendation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(output_dir: Path, build_confirmed: bool, score_confirmed: bool) -> None:
    text = f"""# OASIS 140TR Temporal-Length Sensitivity

Status: `{'scored' if score_confirmed else 'prepared_dry_run' if not build_confirmed else 'built_not_scored'}`

This package defines an ADNI-like OASIS external-validation sensitivity tensor:
`{BUILD_CANDIDATE}`.

## Temporal Rule

The local ADNI build scripts homogenize ROI time series to 140 TR by taking the first
140 rows when a run is longer than 140 TR. OASIS QC-ok runs have 164 TR, so each
run is cropped to rows `[0:140]` before connectome computation.

## Build Candidate

1. Crop each QC-usable run to 140 TR.
2. Compute the same three channels per run:
   - `Pearson_Full_FisherZ_Signed`
   - `Pearson_OMST_GCE_Signed_Weighted`
   - `MI_KNN_Symmetric`
3. Average run-level connectomes per subject/session.
4. Preserve the audited ADNI 131 ROI order.

## Guardrails

- Real build requires `--confirm-build`.
- Real scoring requires `--confirm-score` and an existing 140TR tensor.
- No input data are modified.
- No OASIS threshold fitting or model selection is performed.
- OASIS labels are used only for final metrics.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = normalize_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    context_args = argparse.Namespace(
        input_dir=normalize_path(args.input_dir),
        preflight_dir=normalize_path(args.preflight_dir),
        roi_mapping_audit_dir=normalize_path(args.roi_mapping_audit_dir),
        output_dir=output_dir,
        build_candidate="runwise_connectome_average",
        confirm_build=bool(args.confirm_build),
        confirm_roi_order_aal3_1to170=False,
        fail_if_output_exists=bool(args.fail_if_output_exists),
        dry_run=bool(args.dry_run),
    )
    context = load_context(context_args)
    save_roi_mapping(context, output_dir)
    build_manifest = make_build_manifest(context)
    write_csv_md(build_manifest, output_dir / "build_manifest.csv", output_dir / "build_manifest.md", "OASIS 140TR Build Manifest")
    subjects = copy_subject_manifest(context, output_dir)
    planned_qc = (
        build_manifest.groupby(["subject_id", "session_id", "experiment_id", "diagnosis"], dropna=False)
        .agg(
            n_runs_used=("run_id", "count"),
            min_original_timepoints=("original_timepoints", "min"),
            max_original_timepoints=("original_timepoints", "max"),
            selected_timepoints_per_run=("selected_timepoints", "min"),
            all_runs_usable_for_140TR=("usable_for_140TR", "all"),
        )
        .reset_index()
    )
    planned_qc["build_candidate"] = BUILD_CANDIDATE
    planned_qc["status"] = "planned" if not args.confirm_build else "pending_build"
    write_csv_md(planned_qc, output_dir / "tensor_qc.csv", output_dir / "tensor_qc.md", "Tensor QC")

    tensor_path = output_dir / TENSOR_FILENAME
    build_executed = False
    if args.confirm_build:
        if args.fail_if_output_exists and tensor_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing tensor: {tensor_path}")
        tensor_path, tensor_qc = build_140tr_tensor(context, output_dir, int(args.n_jobs_mi))
        write_csv_md(tensor_qc, output_dir / "tensor_qc.csv", output_dir / "tensor_qc.md", "Tensor QC")
        build_executed = True

    scoring_metrics: pd.DataFrame | None = None
    scoring_executed = False
    artifact_df = validate_scoring_artifacts(model_specs(args), tensor_path)
    write_csv_md(artifact_df, output_dir / "scoring_artifact_validation.csv", output_dir / "scoring_artifact_validation.md", "Scoring Artifact Validation")
    if args.confirm_score:
        if not tensor_path.exists():
            raise FileNotFoundError(f"Cannot score: missing 140TR tensor {tensor_path}")
        missing_required = artifact_df[(~artifact_df["exists"]) & (~artifact_df["optional"])]
        if not missing_required.empty:
            raise FileNotFoundError("Cannot score; missing required artifacts:\n" + missing_required.to_string(index=False))
        scoring_metrics, confusion, dist = score_140tr_tensor(args, output_dir, artifact_df)
        write_csv_md(scoring_metrics, output_dir / "external_scoring_metrics.csv", output_dir / "external_scoring_metrics.md", "External Scoring Metrics")
        write_csv_md(confusion, output_dir / "confusion_matrices.csv", output_dir / "confusion_matrices.md", "Confusion Matrices")
        write_csv_md(dist, output_dir / "score_distribution_summary.csv", output_dir / "score_distribution_summary.md", "Score Distribution Summary")
        comparison = comparison_vs_existing(output_dir, args, scoring_metrics)
        write_csv_md(
            comparison,
            output_dir / "comparison_vs_concatenated_and_runwise164.csv",
            output_dir / "comparison_vs_concatenated_and_runwise164.md",
            "Comparison vs Concatenated and Runwise-164 OASIS Scoring",
        )
        scoring_executed = True
    else:
        reason = "confirm_score_not_provided" if not tensor_path.exists() else "confirm_score_not_provided_tensor_available"
        placeholder_tables(output_dir, reason)

    write_readme(output_dir, build_executed, scoring_executed)
    write_recommendation(output_dir, build_executed, scoring_executed, scoring_metrics)
    command_log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "mode": "build_and_score" if scoring_executed else "build_only" if build_executed else "dry_run_only",
        "build_candidate": BUILD_CANDIDATE,
        "target_tr": TARGET_TR,
        "temporal_selection_rule": "first_140TR_per_run",
        "confirm_build": bool(args.confirm_build),
        "confirm_score": bool(args.confirm_score),
        "computed_connectomes": bool(build_executed),
        "scored_oasis": bool(scoring_executed),
        "trained_on_oasis": False,
        "oasis_threshold_fitting": False,
        "oasis_model_selection": False,
        "modified_input_data": False,
        "n_selected_runs_planned": int(len(build_manifest)),
        "n_subject_sessions_planned": int(len(subjects)),
        "tensor_path": str(tensor_path),
        "threshold_strategy": PRIMARY_THRESHOLD_STRATEGY,
    }
    write_json(output_dir / "command_log.json", command_log)
    print(json.dumps({"output_dir": str(output_dir), **command_log}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
