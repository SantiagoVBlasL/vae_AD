#!/usr/bin/env python3
"""Original-paper-model inference for DPARSF-bandpass ROI signals plus Python bandpass.

This is the preprocessing-control counterpart of
``run_original_model_inference_dparsf_bandpass10.py``. It keeps the same guarded
inventory, metadata recovery, AAL3 170->131 reduction/reordering, original model,
and selected channels, but intentionally applies the historical Python bandpass
filter (0.01-0.08 Hz, TR=3.0) before connectome generation.

No retraining is performed.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision_bspc_2026 import run_original_model_inference_dparsf_bandpass10 as base


RUN_NAME = "original_model_inference_dparsf_bandpass10_pythonbandpass"
DEFAULT_OUTPUT_SYMLINK = PROJECT_ROOT / f"results/revision_bspc_2026/{RUN_NAME}"
DEFAULT_BIG_DISK_TARGET = Path(f"/media/diego/Datos/vae_AD_results/revision_bspc_2026/{RUN_NAME}")
TENSOR_FILENAME = "GLOBAL_TENSOR_DPARSF_original_bandpass10_pythonbandpass_AAL3_131ROIs.npz"
SMOKE_TENSOR_FILENAME = "GLOBAL_TENSOR_SMOKE_ONE_SUBJECT_DPARSF_original_bandpass10_pythonbandpass_AAL3_131ROIs.npz"

PYTHON_BANDPASS_APPLIED = True
FILTER_LOW_HZ = 0.01
FILTER_HIGH_HZ = 0.08
TR_SECONDS = 3.0
TARGET_LEN_TS = 140


def patch_base_output_paths() -> None:
    """Make imported helper searches treat this run as the current output."""
    base.DEFAULT_OUTPUT_SYMLINK = DEFAULT_OUTPUT_SYMLINK
    base.DEFAULT_BIG_DISK_TARGET = DEFAULT_BIG_DISK_TARGET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Guarded original-paper-model inference for DPARSF-bandpass ADNI controls with Python bandpass ON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-root", type=Path, default=base.DEFAULT_INPUT_ROOT)
    parser.add_argument("--training-output-dir", type=Path, default=base.DEFAULT_TRAINING_DIR)
    parser.add_argument("--output-symlink", type=Path, default=DEFAULT_OUTPUT_SYMLINK)
    parser.add_argument("--big-disk-target", type=Path, default=DEFAULT_BIG_DISK_TARGET)
    parser.add_argument("--external-log-dir", type=Path, default=base.DEFAULT_EXTERNAL_LOG_DIR)
    parser.add_argument("--aal3-roi-metadata-path", type=Path, default=base.DEFAULT_AAL3_ROI_METADATA_PATH)
    parser.add_argument("--expected-n", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-one-subject", action="store_true")
    parser.add_argument("--allow-non10", action="store_true")
    parser.add_argument("--allow-missing-metadata", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--classifier-types", nargs="+", default=["logreg", "svm"])
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--ensemble-method", choices=["mean", "median"], default="mean")
    return parser.parse_args()


def python_bandpass_preprocess(raw_170: np.ndarray, subject_id: str, fem: Any) -> np.ndarray:
    """Apply historical AAL3 reduction, Python bandpass, scaling, and length matching."""
    from scipy.interpolate import interp1d
    from sklearn.preprocessing import StandardScaler

    reduced = base.reduce_and_reorder_rois(raw_170, subject_id, fem)
    fs = 1.0 / TR_SECONDS
    filtered = fem._bandpass_filter_signals(
        np.nan_to_num(reduced, nan=0.0, posinf=0.0, neginf=0.0),
        FILTER_LOW_HZ,
        FILTER_HIGH_HZ,
        fs,
        fem.FILTER_ORDER,
        subject_id,
        taper_alpha=fem.TAPER_ALPHA,
    )
    scaled = StandardScaler().fit_transform(np.nan_to_num(filtered, nan=0.0, posinf=0.0, neginf=0.0))
    if scaled.shape[0] > TARGET_LEN_TS:
        scaled = scaled[:TARGET_LEN_TS, :]
    elif scaled.shape[0] < TARGET_LEN_TS:
        old = np.linspace(0, 1, scaled.shape[0])
        new = np.linspace(0, 1, TARGET_LEN_TS)
        out = np.zeros((TARGET_LEN_TS, scaled.shape[1]), dtype=np.float32)
        for idx in range(scaled.shape[1]):
            out[:, idx] = interp1d(old, scaled[:, idx], kind="linear", fill_value="extrapolate")(new)
        scaled = out
    return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def validate_filtering(
    subjects: pd.DataFrame,
    audit_dir: Path,
    fem: Any,
    smoke_one_subject: bool = False,
) -> pd.DataFrame:
    validation_subjects = subjects.sort_values("SubjectID").head(1) if smoke_one_subject else subjects.sort_values("SubjectID")
    rows = []
    for _, row in validation_subjects.iterrows():
        sid = str(row["SubjectID"])
        status = "OK"
        before_shape = ""
        after_shape = ""
        try:
            raw = base.read_roi_signal_txt(Path(row["txt_path"]))
            before_shape = str(tuple(raw.shape))
            processed = python_bandpass_preprocess(raw, sid, fem)
            after_shape = str(tuple(processed.shape))
            if processed.shape != (TARGET_LEN_TS, base.EXPECTED_FINAL_ROIS):
                status = f"FAIL: expected_shape=({TARGET_LEN_TS},{base.EXPECTED_FINAL_ROIS});got={processed.shape}"
        except Exception as exc:
            status = f"FAIL: {exc}"
        rows.append(
            {
                "SubjectID": sid,
                "input_shape_before_filter": before_shape,
                "python_bandpass_applied": PYTHON_BANDPASS_APPLIED,
                "filter_low_hz": FILTER_LOW_HZ,
                "filter_high_hz": FILTER_HIGH_HZ,
                "TR": TR_SECONDS,
                "target_len_ts": TARGET_LEN_TS,
                "output_shape_after_filter_and_trim": after_shape,
                "status": status,
            }
        )
    validation = pd.DataFrame(rows)
    validation.to_csv(audit_dir / "filtering_validation.csv", index=False)
    if validation.empty or (validation["status"] != "OK").any():
        raise RuntimeError("Python bandpass filtering validation failed; see audit/filtering_validation.csv")
    return validation


def generate_tensor_from_roi_txt_pythonbandpass(
    subjects: pd.DataFrame,
    features_dir: Path,
    overwrite: bool,
    run_config: Mapping[str, Any],
    fem: Any,
    tensor_filename: str = TENSOR_FILENAME,
) -> Path:
    if list(fem.CONNECTIVITY_CHANNEL_NAMES) != base.CHANNEL_NAMES_MASTER:
        raise RuntimeError(f"Feature extractor channel order mismatch: {fem.CONNECTIVITY_CHANNEL_NAMES}")
    tensor_path = features_dir / tensor_filename
    if tensor_path.exists() and not overwrite:
        return tensor_path

    individual_dir = features_dir / "individual_subject_tensors"
    individual_dir.mkdir(parents=True, exist_ok=True)
    tensors = []
    subject_ids = []
    qc_rows = []
    for _, row in subjects.sort_values("SubjectID").iterrows():
        sid = str(row["SubjectID"])
        raw = base.read_roi_signal_txt(Path(row["txt_path"]))
        processed = python_bandpass_preprocess(raw, sid, fem)
        conn = fem.calculate_all_connectivity_modalities_for_subject(
            sid,
            processed,
            fem.N_NEIGHBORS_MI,
            fem.DFC_WIN_POINTS,
            fem.DFC_STEP,
            fem.GRANGER_MAX_LAG,
            fem.AAL3_ROI_ORDER_MAPPING,
        )
        tensor = base.normalize_subject_tensor(conn["matrices"], fem)
        out_path = individual_dir / f"tensor_7ch_131rois_pythonbandpass_{sid}.npz"
        np.savez_compressed(
            out_path,
            tensor_data=tensor,
            subject_id=sid,
            channel_names=np.array(base.CHANNEL_NAMES_MASTER, dtype=str),
            rois_count=base.EXPECTED_FINAL_ROIS,
            target_len_ts=TARGET_LEN_TS,
            tr_seconds=TR_SECONDS,
            filter_low_hz=FILTER_LOW_HZ,
            filter_high_hz=FILTER_HIGH_HZ,
            python_bandpass_applied=PYTHON_BANDPASS_APPLIED,
            preprocessing_note="DPARSF-bandpass input plus historical Python bandpass 0.01-0.08 Hz.",
        )
        tensors.append(tensor)
        subject_ids.append(sid)
        qc_rows.append(
            {
                "SubjectID": sid,
                "raw_txt_shape": str(tuple(raw.shape)),
                "processed_shape": str(tuple(processed.shape)),
                "tensor_shape": str(tuple(tensor.shape)),
                "python_bandpass_applied": PYTHON_BANDPASS_APPLIED,
                "filter_low_hz": FILTER_LOW_HZ,
                "filter_high_hz": FILTER_HIGH_HZ,
                "TR": TR_SECONDS,
                "target_len_ts": TARGET_LEN_TS,
                "tensor_path": str(out_path),
                "status": "OK",
            }
        )

    global_tensor = np.stack(tensors, axis=0).astype(np.float32)
    np.savez_compressed(
        tensor_path,
        global_tensor_data=global_tensor,
        subject_ids=np.array(subject_ids, dtype=str),
        channel_names=np.array(base.CHANNEL_NAMES_MASTER, dtype=str),
        roi_names_in_order=np.array(run_config.get("roi_names_in_order", [f"ROI_{i+1:03d}" for i in range(131)]), dtype=str),
        network_labels_in_order=np.array(run_config.get("network_labels_in_order", []), dtype=str),
        target_len_ts=TARGET_LEN_TS,
        tr_seconds=TR_SECONDS,
        filter_low_hz=FILTER_LOW_HZ,
        filter_high_hz=FILTER_HIGH_HZ,
        python_bandpass_applied=PYTHON_BANDPASS_APPLIED,
        source_preprocessing="DPARSF_original_bandpass_plus_Python_bandpass",
        notes="Control tensor: DPARSF-bandpass ROI signals were additionally filtered with the historical Python bandpass.",
    )
    pd.DataFrame(qc_rows).to_csv(features_dir / "tensor_generation_qc.csv", index=False)
    return tensor_path


def write_validation_report_pythonbandpass(
    audit_dir: Path,
    subjects: pd.DataFrame,
    expected_n: int,
    missing_metadata: Sequence[str],
    run_config: Mapping[str, Any],
    input_root: Path,
    output_symlink: Path,
    big_target: Path,
    disk_status: str,
) -> None:
    n_detected = len(subjects)
    blockers = []
    if n_detected != expected_n:
        blockers.append(f"Expected {expected_n} subjects but detected {n_detected}.")
    if missing_metadata:
        blockers.append(f"Missing metadata for {len(missing_metadata)} subjects: {', '.join(missing_metadata)}.")
    bad_subjects = subjects[subjects["issue"] != "OK"] if not subjects.empty else subjects
    if not bad_subjects.empty:
        blockers.append(f"Inventory issues in {len(bad_subjects)} rows; see missing_or_duplicate_subjects.csv.")
    args_cfg = run_config.get("args", {})
    lines = [
        "# DPARSF Bandpass10 + Python Bandpass Input Validation",
        "",
        f"Input root: `{input_root}`",
        f"Detected subjects: `{n_detected}`",
        f"Expected subjects: `{expected_n}`",
        f"Blocking inventory issue: `{'YES' if n_detected != expected_n else 'NO'}`",
        "Inventory explanation: recursive scan found the subject IDs below in ROI signal, check-image, and atlas-preview files. No additional filename matching `[0-9]{3}_S_[0-9]{4}` was found under the input root.",
        f"Detected SubjectIDs: `{', '.join(subjects['SubjectID'].astype(str).tolist()) if not subjects.empty else 'none'}`",
        "",
        "## Original Model",
        f"Training output dir: `{base.DEFAULT_TRAINING_DIR}`",
        f"Recovered original metadata path: `{run_config.get('metadata_path', args_cfg.get('metadata_path', 'not_found'))}`",
        f"Selected channel indices: `{args_cfg.get('channels_to_use', base.SELECTED_CHANNELS)}`",
        f"Selected channel names: `{args_cfg.get('selected_channel_names', base.SELECTED_CHANNEL_NAMES)}`",
        "Python bandpass during tensor adapter: `APPLIED`.",
        f"Python bandpass: `{FILTER_LOW_HZ}-{FILTER_HIGH_HZ} Hz`, TR `{TR_SECONDS} s`, target length `{TARGET_LEN_TS} TRs`.",
        "Reason: controlled compatibility experiment; provenance audit found the historical extractor likely applied this Python bandpass.",
        "",
        "## Output",
        f"Local symlink: `{output_symlink}`",
        f"Big-disk target: `{big_target}`",
        f"Symlink valid: `{output_symlink.is_symlink() and output_symlink.resolve() == big_target.resolve() if output_symlink.exists() or output_symlink.is_symlink() else False}`",
        "",
        "## Metadata Recovery",
        "Per-subject metadata recovery is written to `subject_metadata_recovery_report.csv`. Metadata source search order is written to `metadata_search_order.csv`.",
        "",
        "## Previous Prediction Search",
        "Previous prediction matches are written to `previous_predictions_matched.csv`.",
        "",
        "## Blockers",
    ]
    lines.extend(f"- {item}" for item in blockers) if blockers else lines.append("- None.")
    lines.extend(["", "## Disk Status", "```", disk_status.strip(), "```", ""])
    (audit_dir / "input_validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_command_and_manifest_pythonbandpass(
    target: Path,
    command: Sequence[str],
    args: argparse.Namespace,
    subjects: pd.DataFrame,
    metadata: pd.DataFrame,
    blockers: Sequence[str],
    tensor_path: Path,
    dry_run: bool,
) -> None:
    shell_command = shlex.join(command)
    (target / "command.txt").write_text(shell_command + "\n", encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "input_root": str(base.resolve_path(args.input_root)),
        "expected_n": args.expected_n,
        "detected_n": int(len(subjects)),
        "detected_subjects": subjects["SubjectID"].astype(str).tolist(),
        "blockers": list(blockers),
        "training_output_dir": str(base.resolve_path(args.training_output_dir)),
        "model_type": "original_paper_model",
        "channels_to_use": base.SELECTED_CHANNELS,
        "selected_channel_names": base.SELECTED_CHANNEL_NAMES,
        "smoke_one_subject": bool(args.smoke_one_subject),
        "aal3_roi_metadata_path": str(base.resolve_path(args.aal3_roi_metadata_path)),
        "expected_final_rois": base.EXPECTED_FINAL_ROIS,
        "python_bandpass_applied": PYTHON_BANDPASS_APPLIED,
        "filter_low_hz": FILTER_LOW_HZ,
        "filter_high_hz": FILTER_HIGH_HZ,
        "tr_seconds": TR_SECONDS,
        "target_len_ts": TARGET_LEN_TS,
        "python_bandpass_note": "Applied as missing preprocessing-control experiment requested after provenance audit.",
        "tensor_path": str(tensor_path),
        "metadata_path": str(target / "audit/dparsf_bandpass10_metadata_for_inference.csv"),
        "output_symlink": str(base.resolve_path(args.output_symlink)),
        "output_realpath": str(base.resolve_path(args.output_symlink).resolve())
        if base.resolve_path(args.output_symlink).exists()
        else None,
        "big_disk_target": str(args.big_disk_target),
        "classifier_types": args.classifier_types,
        "command": list(command),
        "command_shell": shell_command,
        "missing_metadata_subjects": metadata.loc[~metadata["metadata_found"], "SubjectID"].astype(str).tolist(),
    }
    (target / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_pending_summary_pythonbandpass(path: Path, subjects: pd.DataFrame, blockers: Sequence[str]) -> None:
    lines = [
        "# Original Model Inference: DPARSF Bandpass10 + Python Bandpass",
        "",
        f"N detected: `{len(subjects)}`",
        "Inference status: `NOT_RUN`",
        "",
        "Python bandpass: `APPLIED`, 0.01-0.08 Hz, TR=3.0, target_len_ts=140.",
        "",
        "## Blockers",
    ]
    lines.extend(f"- {item}" for item in blockers) if blockers else lines.append("- None for dry-run.")
    lines.extend(
        [
            "",
            "No predictions are available yet. Dry-run does not generate tensor features or call the original inference script.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def postprocess_inference_outputs_pythonbandpass(target: Path, metadata: pd.DataFrame) -> None:
    tables_dir = target / "Tables"
    ens_path = tables_dir / "covid_predictions_ensemble.csv"
    if not ens_path.exists():
        return
    ens = pd.read_csv(ens_path)
    merged = ens.merge(metadata, on="SubjectID", how="left")
    merged.to_csv(tables_dir / "dparsf_bandpass10_pythonbandpass_predictions_with_metadata.csv", index=False)
    merged.to_csv(tables_dir / "dparsf_bandpass10_predictions_with_metadata.csv", index=False)

    previous = base.collect_previous_predictions(metadata["SubjectID"].astype(str).tolist())
    comparison = previous.merge(
        merged[
            [
                "SubjectID",
                "classifier",
                "y_score_ensemble",
                "y_pred_ensemble",
                "y_pred_majority_vote",
                "y_score_std",
                "Age",
                "Sex",
                "Manufacturer",
            ]
        ],
        on=["SubjectID", "classifier"],
        how="outer",
    )
    comparison = comparison.rename(
        columns={
            "y_score_ensemble": "dparsf_pythonbandpass_y_score_ensemble",
            "y_pred_ensemble": "dparsf_pythonbandpass_y_pred_ensemble",
            "y_pred_majority_vote": "dparsf_pythonbandpass_y_pred_majority_vote",
        }
    )
    comparison["delta_score"] = pd.to_numeric(
        comparison["dparsf_pythonbandpass_y_score_ensemble"], errors="coerce"
    ) - pd.to_numeric(comparison["previous_y_score_ensemble"], errors="coerce")
    comparison["changed_prediction_0p5"] = (
        pd.to_numeric(comparison["previous_y_pred_ensemble"], errors="coerce")
        != pd.to_numeric(comparison["dparsf_pythonbandpass_y_pred_ensemble"], errors="coerce")
    )
    comparison["changed_majority_vote"] = (
        pd.to_numeric(comparison["previous_y_pred_majority_vote"], errors="coerce")
        != pd.to_numeric(comparison["dparsf_pythonbandpass_y_pred_majority_vote"], errors="coerce")
    )
    comparison.to_csv(tables_dir / "subject_level_comparison_vs_previous_preprocessing.csv", index=False)
    write_prediction_summary_pythonbandpass(
        target / "audit/summary_original_model_dparsf_bandpass10_pythonbandpass.md",
        merged,
        comparison,
        len(metadata),
    )


def write_prediction_summary_pythonbandpass(
    path: Path,
    predictions: pd.DataFrame,
    comparison: pd.DataFrame,
    n_detected: int,
) -> None:
    lines = [
        "# Original Model Inference: DPARSF Bandpass10 + Python Bandpass",
        "",
        f"N detected: `{n_detected}`",
        "Python bandpass: `APPLIED`, 0.01-0.08 Hz, TR=3.0, target_len_ts=140.",
        "",
        "## AD-like Counts",
    ]
    for classifier, sub in predictions.groupby("classifier"):
        lines.append(
            f"- {classifier}: AD-like ensemble={int((sub['y_pred_ensemble'] == 1).sum())}/{len(sub)}, "
            f"majority-vote={int((sub['y_pred_majority_vote'] == 1).sum())}/{len(sub)}"
        )
    lines.extend(["", "## Comparison With Previous Preprocessing"])
    if comparison.empty or comparison["previous_y_score_ensemble"].isna().all():
        lines.append("- No previous original-model predictions found for these subjects, except any rows with non-missing previous fields in the CSV.")
    else:
        for classifier, sub in comparison.dropna(subset=["previous_y_score_ensemble"]).groupby("classifier"):
            delta = pd.to_numeric(sub["delta_score"], errors="coerce")
            lines.append(
                f"- {classifier}: mean delta={delta.mean():.4f}, median delta={delta.median():.4f}, "
                f"changed 0.5 prediction={int(sub['changed_prediction_0p5'].sum())}/{len(sub)}, "
                f"changed majority vote={int(sub['changed_majority_vote'].sum())}/{len(sub)}"
            )
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "- This run tests filtering compatibility; it is not an AUC evaluation.",
            "- If DPARSF+Python scores resemble previous preprocessing more than DPARSF-only scores, Python filtering is likely a material compatibility factor.",
            "- If DPARSF+Python scores remain low, the earlier false positives were likely driven more by other preprocessing/QC/scanner factors.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    patch_base_output_paths()
    args = parse_args()
    input_root = base.resolve_path(args.input_root)
    training_dir = base.resolve_path(args.training_output_dir)
    output_symlink = base.resolve_path(args.output_symlink)
    big_target = args.big_disk_target

    audit_dir, tables_dir, features_dir = base.ensure_output_layout(output_symlink, big_target, args.dry_run)
    if args.overwrite and not args.dry_run:
        base.cleanup_generated_outputs_for_overwrite(big_target)
        tables_dir = big_target / "Tables"
        features_dir = big_target / "features_or_tensor"
    disk_status = base.disk_status_text(input_root, Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026"))

    run_config = base.load_run_config(training_dir)
    base.validate_original_model_config(run_config)
    aal3_roi_metadata_path = base.resolve_path(args.aal3_roi_metadata_path)
    try:
        fem = base.configure_aal3_roi_processing(aal3_roi_metadata_path)
    except Exception as exc:
        base.write_roi_failure_readme(audit_dir, aal3_roi_metadata_path, pd.DataFrame(), str(exc))
        raise

    subjects = base.inventory_input(input_root, audit_dir, args.expected_n)
    metadata, missing_metadata = base.build_metadata(subjects, audit_dir, input_root, training_dir)
    previous_predictions = base.collect_previous_predictions(subjects["SubjectID"].astype(str).tolist())
    previous_predictions.to_csv(audit_dir / "previous_predictions_matched.csv", index=False)
    roi_validation = base.validate_roi_reduction(
        subjects,
        audit_dir,
        aal3_roi_metadata_path,
        fem,
        smoke_one_subject=args.smoke_one_subject,
    )
    filtering_validation = validate_filtering(subjects, audit_dir, fem, smoke_one_subject=args.smoke_one_subject)

    tensor_path = features_dir / TENSOR_FILENAME
    command = base.build_inference_command(
        args.python_executable,
        training_dir,
        tensor_path,
        audit_dir / "dparsf_bandpass10_metadata_for_inference.csv",
        output_symlink,
        args.classifier_types,
        args.ensemble_method,
        args.decision_threshold,
    )
    blockers = base.preprocessing_blocked(
        subjects,
        args.expected_n,
        missing_metadata,
        args.allow_non10,
        args.allow_missing_metadata,
    )

    write_validation_report_pythonbandpass(
        audit_dir,
        subjects,
        args.expected_n,
        missing_metadata,
        run_config,
        input_root,
        output_symlink,
        big_target,
        disk_status,
    )
    write_command_and_manifest_pythonbandpass(big_target, command, args, subjects, metadata, blockers, tensor_path, args.dry_run)
    write_pending_summary_pythonbandpass(audit_dir / "summary_original_model_dparsf_bandpass10_pythonbandpass.md", subjects, blockers)

    print(disk_status)
    print(f"Detected subjects ({len(subjects)}/{args.expected_n} expected):")
    print(subjects[["SubjectID", "txt_count", "mat_count", "txt_shape", "issue"]].to_string(index=False))
    print(f"Original model path: {training_dir}")
    print(f"Selected channels: {base.SELECTED_CHANNELS}")
    print(f"Selected channel names: {', '.join(base.SELECTED_CHANNEL_NAMES)}")
    print(f"Python bandpass: APPLIED ({FILTER_LOW_HZ}-{FILTER_HIGH_HZ} Hz, TR={TR_SECONDS}, target_len_ts={TARGET_LEN_TS})")
    print(f"AAL3 ROI metadata path: {aal3_roi_metadata_path}")
    print(f"ROI reduction/reordering active: {bool((fem.AAL3_ROI_ORDER_MAPPING or {}).get('new_order_indices'))}")
    print("ROI reduction validation:")
    print(roi_validation.to_string(index=False))
    print("Filtering validation:")
    print(filtering_validation.to_string(index=False))
    print(f"Metadata path: {audit_dir / 'dparsf_bandpass10_metadata_for_inference.csv'}")
    print(f"Missing metadata subjects: {missing_metadata}")
    print("Metadata recovery:")
    print(
        metadata[
            [
                "SubjectID",
                "found_metadata",
                "metadata_source_file",
                "ResearchGroup_Mapped",
                "Age",
                "Sex",
                "Manufacturer",
                "Site3",
            ]
        ].to_string(index=False)
    )
    print(f"Previous prediction matches: {len(previous_predictions)} rows")
    if not previous_predictions.empty:
        print(
            previous_predictions[
                [
                    "SubjectID",
                    "classifier",
                    "previous_predictions_source_file",
                    "previous_y_score_ensemble",
                    "previous_y_pred_ensemble",
                    "previous_y_pred_majority_vote",
                    "previous_y_score",
                    "previous_y_pred",
                ]
            ].head(30).to_string(index=False)
        )
    print(f"Output symlink path: {output_symlink}")
    if output_symlink.exists() or output_symlink.is_symlink():
        print(f"Output realpath: {output_symlink.resolve()}")
    else:
        print("Output realpath: symlink missing")
    print(f"Big-disk target: {big_target}")
    print("\nCommand:")
    print(shlex.join(command))

    if args.dry_run:
        print("\nDry-run requested. Inference was not launched and tensor generation was not run.")
        if blockers:
            print("Blocking issues for real run:")
            for item in blockers:
                print(f"- {item}")
        return 0

    if base.output_has_existing_outputs(big_target) and not args.overwrite:
        raise RuntimeError(f"Refusing real run because target has existing outputs; pass --overwrite: {big_target}")
    if blockers:
        raise RuntimeError("Refusing real run due to blocking issues:\n" + "\n".join(f"- {item}" for item in blockers))
    if not output_symlink.is_symlink() or output_symlink.resolve() != big_target.resolve():
        raise RuntimeError(f"Refusing real run because output symlink is invalid: {output_symlink}")

    if args.smoke_one_subject:
        smoke_features_dir = features_dir / "smoke_one_subject"
        smoke_features_dir.mkdir(parents=True, exist_ok=True)
        smoke_subjects = subjects.sort_values("SubjectID").head(1)
        smoke_tensor_path = generate_tensor_from_roi_txt_pythonbandpass(
            smoke_subjects,
            smoke_features_dir,
            args.overwrite,
            run_config,
            fem,
            tensor_filename=SMOKE_TENSOR_FILENAME,
        )
        smoke_tensor = np.load(smoke_tensor_path, allow_pickle=False)["global_tensor_data"]
        write_command_and_manifest_pythonbandpass(big_target, command, args, smoke_subjects, metadata, [], smoke_tensor_path, False)
        print("\nSmoke-one-subject requested. Classifier inference was not launched.")
        print(f"Smoke subject: {smoke_subjects['SubjectID'].iloc[0]}")
        print(f"Smoke tensor path: {smoke_tensor_path}")
        print(f"Smoke tensor shape: {tuple(smoke_tensor.shape)}")
        return 0

    tensor_path = generate_tensor_from_roi_txt_pythonbandpass(subjects, features_dir, args.overwrite, run_config, fem)
    command = base.build_inference_command(
        args.python_executable,
        training_dir,
        tensor_path,
        audit_dir / "dparsf_bandpass10_metadata_for_inference.csv",
        output_symlink,
        args.classifier_types,
        args.ensemble_method,
        args.decision_threshold,
    )
    write_command_and_manifest_pythonbandpass(big_target, command, args, subjects, metadata, [], tensor_path, False)

    args.external_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.external_log_dir / f"run_original_model_inference_dparsf_bandpass10_pythonbandpass_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    with log_path.open("w", encoding="utf-8") as log_f:
        completed = subprocess.run(command, cwd=str(PROJECT_ROOT), text=True, stdout=log_f, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Inference command failed with exit code {completed.returncode}; see {log_path}")
    postprocess_inference_outputs_pythonbandpass(big_target, metadata)
    print(f"Real inference completed. Log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
