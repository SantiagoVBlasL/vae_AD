#!/usr/bin/env python3
"""Build the exploratory ADNI all-available-timepoints [1,0,2] tensor branch.

Default mode is a dry-run/preflight. Real tensor computation requires
``--confirm-build`` and writes only to the branch-local data directory under
``/media/diego/Datos/vae_AD_data/revision_bspc_2026``. Locked v5.1b tensors,
metadata, configs, ledgers, and model outputs are never modified.

The branch is explicitly exploratory because original n_TR is already known to
be associated with diagnosis, Manufacturer, and SiteCode.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_v5_dparsf10000_no_pybandpass_manifest_and_extract import (  # noqa: E402
    CHANNEL_NAMES as LOCKED_CHANNEL_NAMES,
    OUTPUT_ROIS,
    build_roi_reduction_and_order,
    load_signal,
    mi_knn,
    orient_reduce_reorder,
    pearson_full,
    pearson_omst,
    robustscale_offdiag,
    standardize_timeseries,
)


BRANCH_NAME = "adni_all_available_timepoints_ch1_0_2_exploratory"
RESULTS_DIR = PROJECT_ROOT / "results/revision_bspc_2026" / f"{BRANCH_NAME}_tensor_build"
DATA_BRANCH_DIR = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026"
    "/adni_all_available_timepoints_ch1_0_2_exploratory"
)
SUBJECT_TIMEPOINT_TABLE = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/adni_timepoint_availability_confounding_audit"
    / "subject_timepoint_table.csv"
)
LOCKED_TENSOR = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026"
    "/adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors"
    "/GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
)
LOCKED_METADATA = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026"
    "/adni_expanded_v5_1_batch20260514b_no_pybandpass"
    "/training_ready_metadata_v5_1_batch20260514b_no_pybandpass.csv"
)

TENSOR_FILENAME = "GLOBAL_TENSOR_ADNI_all_available_timepoints_ch1_0_2_exploratory.npz"
METADATA_FILENAME = "training_ready_metadata_adni_all_available_timepoints_ch1_0_2_exploratory.csv"

# Preserve the locked tensor's first three channel-axis indices so downstream
# configs can keep channels_to_use=[1,0,2] with the same channel semantics.
BRANCH_CHANNEL_INDICES = [0, 1, 2]
INTENDED_CHANNELS_TO_USE = [1, 0, 2]
SELECTED_CHANNEL_NAMES_IN_INTENDED_ORDER = [
    "Pearson_Full_FisherZ_Signed",
    "Pearson_OMST_GCE_Signed_Weighted",
    "MI_KNN_Symmetric",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or dry-run the exploratory ADNI all-timepoints [1,0,2] tensor branch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true", help="Explicit dry-run alias; default without --confirm-build.")
    parser.add_argument("--confirm-build", action="store_true", help="Actually compute and write the branch tensor.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing branch tensor/metadata outputs.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Subject-level parallel workers for real build.")
    parser.add_argument("--pairwise-n-jobs", type=int, default=1, help="Within-subject pairwise workers for MI.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--data-branch-dir", type=Path, default=DATA_BRANCH_DIR)
    parser.add_argument("--subject-timepoint-table", type=Path, default=SUBJECT_TIMEPOINT_TABLE)
    parser.add_argument("--locked-tensor", type=Path, default=LOCKED_TENSOR)
    parser.add_argument("--locked-metadata", type=Path, default=LOCKED_METADATA)
    return parser.parse_args()


def write_csv_md(df: pd.DataFrame, csv_path: Path, title: str) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    md_path = csv_path.with_suffix(".md")
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        if df.empty:
            f.write("_No rows._\n")
        else:
            f.write(df.to_markdown(index=False))
            f.write("\n")


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return math.nan
    return out if math.isfinite(out) else math.nan


def require_inputs(args: argparse.Namespace) -> None:
    required = [args.subject_timepoint_table, args.locked_tensor, args.locked_metadata]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input(s): " + ", ".join(missing))


def load_subjects(args: argparse.Namespace) -> pd.DataFrame:
    subjects = pd.read_csv(args.subject_timepoint_table)
    subjects = subjects.sort_values("tensor_index").reset_index(drop=True)
    subjects["SubjectID"] = subjects["SubjectID"].astype(str)
    subjects["roi_signal_path"] = subjects["roi_signal_path"].astype(str)
    subjects["roi_signal_path_exists_now"] = subjects["roi_signal_path"].map(lambda p: Path(p).exists())
    subjects["branch_n_timepoints_used"] = pd.to_numeric(
        subjects["original_n_timepoints"], errors="coerce"
    )
    subjects["branch_timepoint_policy"] = "all_available_variable_length"
    subjects["branch_include"] = True
    if subjects["SubjectID"].duplicated().any():
        dup = sorted(subjects.loc[subjects["SubjectID"].duplicated(), "SubjectID"].unique())
        raise RuntimeError(f"Duplicate SubjectID rows in subject table: {dup}")
    missing_paths = subjects[~subjects["roi_signal_path_exists_now"]]
    if not missing_paths.empty:
        raise FileNotFoundError(
            "Missing ROI-signal path(s): "
            + ", ".join(missing_paths["SubjectID"].head(10).tolist())
        )
    return subjects


def load_locked_reference(args: argparse.Namespace, subjects: pd.DataFrame) -> dict[str, Any]:
    npz = np.load(args.locked_tensor, allow_pickle=False)
    tensor = npz["global_tensor_data"].astype(np.float32)
    subject_ids = npz["subject_ids"].astype(str)
    channel_names = npz["channel_names"].astype(str)
    if list(channel_names[:3]) != list(np.asarray(LOCKED_CHANNEL_NAMES[:3], dtype=str)):
        raise RuntimeError(
            "Locked tensor first three channel names differ from build helper constants: "
            f"{list(channel_names[:3])} vs {LOCKED_CHANNEL_NAMES[:3]}"
        )
    locked_by_subject = {sid: idx for idx, sid in enumerate(subject_ids)}
    missing_locked = [sid for sid in subjects["SubjectID"].tolist() if sid not in locked_by_subject]
    if missing_locked:
        raise RuntimeError(f"Subjects missing from locked tensor: {missing_locked[:10]}")
    locked_indices = np.asarray([locked_by_subject[sid] for sid in subjects["SubjectID"].tolist()], dtype=int)
    return {
        "tensor": tensor,
        "subject_ids": subject_ids,
        "channel_names": channel_names,
        "locked_indices_for_subject_table": locked_indices,
        "roi_names": npz["roi_names_in_order"].astype(str),
        "network_labels": npz["network_labels_in_order"].astype(str),
        "target_len_ts": int(npz["target_len_ts"]),
        "tr_seconds": float(npz["tr_seconds"]),
        "roi_order_name": str(npz["roi_order_name"]),
        "python_bandpass_applied": bool(npz["python_bandpass_applied"]),
    }


def _preprocess_all_available(raw: np.ndarray, roi_info: dict[str, Any]) -> tuple[np.ndarray | None, dict[str, Any], str]:
    reduced = orient_reduce_reorder(raw, roi_info)
    if reduced is None:
        return None, {}, "roi_reduce_or_reorder_failed"
    if not np.isfinite(reduced).any():
        return None, {}, "no_finite_values_after_roi_reduction"
    standardized = standardize_timeseries(reduced).astype(np.float32)
    qc = {
        "raw_reduced_shape": str(tuple(int(x) for x in reduced.shape)),
        "raw_reduced_finite_fraction": float(np.isfinite(reduced).mean()),
        "raw_reduced_mean": float(np.nanmean(reduced)),
        "raw_reduced_std": float(np.nanstd(reduced)),
        "all_available_processed_shape": str(tuple(int(x) for x in standardized.shape)),
        "all_available_processed_nan_count": int(np.isnan(standardized).sum()),
        "all_available_processed_min": float(np.nanmin(standardized)),
        "all_available_processed_max": float(np.nanmax(standardized)),
    }
    return standardized, qc, "ok"


def _compute_selected_channels(ts: np.ndarray, pairwise_n_jobs: int) -> tuple[np.ndarray, dict[str, str], list[dict[str, Any]]]:
    raw_by_index: dict[int, np.ndarray] = {}
    status_by_index: dict[int, str] = {}

    omst, omst_status = pearson_omst(ts)
    raw_by_index[0] = omst
    status_by_index[0] = omst_status
    raw_by_index[1] = pearson_full(ts)
    status_by_index[1] = "ok"
    raw_by_index[2] = mi_knn(ts, n_jobs=pairwise_n_jobs)
    status_by_index[2] = "ok"

    matrices: list[np.ndarray] = []
    qc_rows: list[dict[str, Any]] = []
    for channel_idx in BRANCH_CHANNEL_INDICES:
        channel_name = LOCKED_CHANNEL_NAMES[channel_idx]
        raw = raw_by_index[channel_idx]
        scaled = robustscale_offdiag(raw).astype(np.float32)
        matrices.append(scaled)
        qc_rows.append(
            {
                "channel_index": channel_idx,
                "channel_name": channel_name,
                "calc_status": status_by_index[channel_idx],
                "raw_min": float(np.nanmin(raw)),
                "raw_max": float(np.nanmax(raw)),
                "raw_nan_count": int(np.isnan(raw).sum()),
                "scaled_min": float(np.nanmin(scaled)),
                "scaled_max": float(np.nanmax(scaled)),
                "scaled_nan_count": int(np.isnan(scaled).sum()),
            }
        )
    return np.stack(matrices, axis=0).astype(np.float32), {
        LOCKED_CHANNEL_NAMES[i]: status_by_index[i] for i in BRANCH_CHANNEL_INDICES
    }, qc_rows


def process_subject(task: dict[str, Any]) -> tuple[np.ndarray | None, dict[str, Any], list[dict[str, Any]]]:
    started = time.time()
    sid = str(task["SubjectID"])
    path = Path(str(task["roi_signal_path"]))
    pairwise_n_jobs = int(task["pairwise_n_jobs"])
    roi_info = task["roi_info"]
    row: dict[str, Any] = {
        "SubjectID": sid,
        "roi_signal_path": str(path),
        "status": "pending",
        "load_status": "",
        "raw_shape": "",
        "original_n_timepoints": task.get("original_n_timepoints", np.nan),
        "locked_n_timepoints_used": task.get("locked_n_timepoints_used", np.nan),
        "elapsed_sec": np.nan,
    }
    channel_rows: list[dict[str, Any]] = []
    try:
        raw, _var, raw_shape, load_status = load_signal(path)
        row["raw_shape"] = raw_shape
        row["load_status"] = load_status
        if raw is None:
            row["status"] = "load_failed"
            return None, row, channel_rows
        ts, pre_qc, pre_status = _preprocess_all_available(raw, roi_info)
        row.update(pre_qc)
        row["preprocess_status"] = pre_status
        if ts is None:
            row["status"] = "preprocess_failed"
            return None, row, channel_rows
        if ts.shape[1] != OUTPUT_ROIS:
            row["status"] = f"bad_roi_count:{ts.shape}"
            return None, row, channel_rows
        tensor, statuses, qc_rows = _compute_selected_channels(ts, pairwise_n_jobs=pairwise_n_jobs)
        if tensor.shape != (len(BRANCH_CHANNEL_INDICES), OUTPUT_ROIS, OUTPUT_ROIS):
            row["status"] = f"bad_tensor_shape:{tensor.shape}"
            return None, row, channel_rows
        row["channel_statuses"] = json.dumps(statuses, sort_keys=True)
        row["tensor_shape"] = str(tuple(int(x) for x in tensor.shape))
        row["tensor_finite_fraction"] = float(np.isfinite(tensor).mean())
        row["tensor_nan_count"] = int(np.isnan(tensor).sum())
        row["tensor_min"] = float(np.nanmin(tensor))
        row["tensor_max"] = float(np.nanmax(tensor))
        row["status"] = "ok"
        channel_rows = [{"SubjectID": sid, **qc} for qc in qc_rows]
        return tensor, row, channel_rows
    except Exception as exc:
        row["status"] = f"failed:{exc}"
        return None, row, channel_rows
    finally:
        row["elapsed_sec"] = time.time() - started


def build_tensor(subjects: pd.DataFrame, args: argparse.Namespace) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    roi_info = build_roi_reduction_and_order()
    tasks: list[dict[str, Any]] = []
    for _, subject in subjects.iterrows():
        payload = subject.to_dict()
        payload["roi_info"] = roi_info
        payload["pairwise_n_jobs"] = args.pairwise_n_jobs
        tasks.append(payload)

    tensors_by_subject: dict[str, np.ndarray] = {}
    qc_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []

    if args.n_jobs <= 1:
        for task in tasks:
            tensor, qc, channel_qc = process_subject(task)
            qc_rows.append(qc)
            channel_rows.extend(channel_qc)
            if tensor is not None and qc["status"] == "ok":
                tensors_by_subject[qc["SubjectID"]] = tensor
    else:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as pool:
            futures = {pool.submit(process_subject, task): str(task["SubjectID"]) for task in tasks}
            for future in as_completed(futures):
                tensor, qc, channel_qc = future.result()
                qc_rows.append(qc)
                channel_rows.extend(channel_qc)
                if tensor is not None and qc["status"] == "ok":
                    tensors_by_subject[qc["SubjectID"]] = tensor

    qc_df = pd.DataFrame(qc_rows).sort_values("SubjectID").reset_index(drop=True)
    failed = qc_df[~qc_df["status"].eq("ok")]
    if not failed.empty:
        failed_preview = failed[["SubjectID", "status"]].head(20).to_dict(orient="records")
        raise RuntimeError(f"Subject build failures: {failed_preview}")

    ordered: list[np.ndarray] = []
    missing: list[str] = []
    for sid in subjects["SubjectID"].astype(str).tolist():
        tensor = tensors_by_subject.get(sid)
        if tensor is None:
            missing.append(sid)
        else:
            ordered.append(tensor)
    if missing:
        raise RuntimeError(f"Built tensor missing subjects: {missing[:20]}")
    channel_qc = pd.DataFrame(channel_rows).sort_values(["SubjectID", "channel_index"]).reset_index(drop=True)
    return np.stack(ordered, axis=0).astype(np.float32), qc_df, channel_qc


def ensure_real_build_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    tensor_dir = args.data_branch_dir / "subject_tensors"
    tensor_path = tensor_dir / TENSOR_FILENAME
    metadata_path = args.data_branch_dir / METADATA_FILENAME
    existing = [path for path in [tensor_path, metadata_path] if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing branch output(s) without --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    tensor_dir.mkdir(parents=True, exist_ok=True)
    args.data_branch_dir.mkdir(parents=True, exist_ok=True)
    return tensor_path, metadata_path


def save_branch_outputs(
    tensor: np.ndarray,
    subjects: pd.DataFrame,
    locked: dict[str, Any],
    tensor_path: Path,
    metadata_path: Path,
) -> None:
    metadata = subjects.copy()
    metadata["original_n_TR"] = metadata["original_n_timepoints"]
    metadata["all_available_timepoints_branch"] = BRANCH_NAME
    metadata["timepoint_policy"] = "all_available_variable_length_exploratory"
    metadata["exploratory_confounding_stress_test"] = True
    metadata.to_csv(metadata_path, index=False)
    np.savez_compressed(
        tensor_path,
        global_tensor_data=tensor.astype(np.float32),
        subject_ids=subjects["SubjectID"].astype(str).to_numpy(),
        channel_names=np.asarray([LOCKED_CHANNEL_NAMES[i] for i in BRANCH_CHANNEL_INDICES], dtype=str),
        original_channel_indices=np.asarray(BRANCH_CHANNEL_INDICES, dtype=np.int32),
        intended_channels_to_use=np.asarray(INTENDED_CHANNELS_TO_USE, dtype=np.int32),
        intended_channel_names=np.asarray(SELECTED_CHANNEL_NAMES_IN_INTENDED_ORDER, dtype=str),
        rois_count=np.asarray(OUTPUT_ROIS, dtype=np.int32),
        target_len_ts=np.asarray("all_available_variable_length", dtype=str),
        original_n_timepoints=pd.to_numeric(subjects["original_n_timepoints"], errors="coerce").to_numpy(),
        locked_n_timepoints_used=pd.to_numeric(subjects["locked_n_timepoints_used"], errors="coerce").to_numpy(),
        tr_seconds=np.asarray(locked["tr_seconds"], dtype=np.float32),
        python_bandpass_applied=np.asarray(False),
        preprocessing_source=np.asarray("ADNI_ROISignals_AAL3_all_available_no_pybandpass", dtype=str),
        dataset_name=np.asarray(BRANCH_NAME, dtype=str),
        source_locked_tensor=np.asarray(str(LOCKED_TENSOR), dtype=str),
        roi_order_name=np.asarray(locked["roi_order_name"], dtype=str),
        roi_names_in_order=locked["roi_names"].astype(str),
        network_labels_in_order=locked["network_labels"].astype(str),
        exploratory_confounding_stress_test=np.asarray(True),
    )


def offdiag_values(matrix: np.ndarray) -> np.ndarray:
    mask = ~np.eye(matrix.shape[-1], dtype=bool)
    if matrix.ndim == 2:
        return matrix[mask]
    return matrix[:, mask]


def channel_stats(tensor: np.ndarray, channel_axis_index: int, label: str, channel_name: str) -> dict[str, Any]:
    values = tensor[:, channel_axis_index]
    offdiag = offdiag_values(values)
    diag = np.diagonal(values, axis1=1, axis2=2)
    sym_diff = np.abs(values - np.transpose(values, (0, 2, 1)))
    return {
        "source": label,
        "channel_axis_index": channel_axis_index,
        "channel_name": channel_name,
        "n_subjects": int(values.shape[0]),
        "finite_fraction": float(np.isfinite(values).mean()),
        "mean_all": float(np.nanmean(values)),
        "std_all": float(np.nanstd(values)),
        "min_all": float(np.nanmin(values)),
        "max_all": float(np.nanmax(values)),
        "mean_offdiag": float(np.nanmean(offdiag)),
        "std_offdiag": float(np.nanstd(offdiag)),
        "diag_mean_abs": float(np.nanmean(np.abs(diag))),
        "diag_max_abs": float(np.nanmax(np.abs(diag))),
        "max_symmetry_abs_diff": float(np.nanmax(sym_diff)),
    }


def tensor_qc_table(
    subjects: pd.DataFrame,
    locked: dict[str, Any],
    alltr_tensor: np.ndarray | None,
    tensor_path: Path | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    locked_tensor = locked["tensor"][locked["locked_indices_for_subject_table"]][:, BRANCH_CHANNEL_INDICES]
    rows.append(
        {
            "qc_item": "locked_subject_pool_reference",
            "status": "ok",
            "tensor_path": str(LOCKED_TENSOR),
            "shape": str(tuple(int(x) for x in locked_tensor.shape)),
            "n_subjects": int(locked_tensor.shape[0]),
            "n_channels": int(locked_tensor.shape[1]),
            "n_rois": int(locked_tensor.shape[-1]),
            "finite_fraction": float(np.isfinite(locked_tensor).mean()),
            "max_symmetry_abs_diff": float(np.nanmax(np.abs(locked_tensor - np.transpose(locked_tensor, (0, 1, 3, 2))))),
            "diag_max_abs": float(np.nanmax(np.abs(np.diagonal(locked_tensor, axis1=2, axis2=3)))),
            "notes": "Locked 140TR tensor restricted to the 646-row training-ready metadata subject pool.",
        }
    )
    if alltr_tensor is None:
        rows.append(
            {
                "qc_item": "allTR_branch_tensor",
                "status": "pending_not_built_dry_run",
                "tensor_path": str(tensor_path) if tensor_path else str(DATA_BRANCH_DIR / "subject_tensors" / TENSOR_FILENAME),
                "shape": f"planned ({len(subjects)}, 3, 131, 131)",
                "n_subjects": int(len(subjects)),
                "n_channels": 3,
                "n_rois": 131,
                "finite_fraction": np.nan,
                "max_symmetry_abs_diff": np.nan,
                "diag_max_abs": np.nan,
                "notes": "Real tensor QC will run only after --confirm-build.",
            }
        )
    else:
        rows.append(
            {
                "qc_item": "allTR_branch_tensor",
                "status": "ok",
                "tensor_path": str(tensor_path),
                "shape": str(tuple(int(x) for x in alltr_tensor.shape)),
                "n_subjects": int(alltr_tensor.shape[0]),
                "n_channels": int(alltr_tensor.shape[1]),
                "n_rois": int(alltr_tensor.shape[-1]),
                "finite_fraction": float(np.isfinite(alltr_tensor).mean()),
                "max_symmetry_abs_diff": float(np.nanmax(np.abs(alltr_tensor - np.transpose(alltr_tensor, (0, 1, 3, 2))))),
                "diag_max_abs": float(np.nanmax(np.abs(np.diagonal(alltr_tensor, axis1=2, axis2=3)))),
                "notes": "Branch tensor built from all available subject-specific timepoints.",
            }
        )
    return pd.DataFrame(rows)


def channel_distribution_comparison(
    subjects: pd.DataFrame,
    locked: dict[str, Any],
    alltr_tensor: np.ndarray | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    locked_tensor = locked["tensor"][locked["locked_indices_for_subject_table"]][:, BRANCH_CHANNEL_INDICES]
    for axis_idx, original_idx in enumerate(BRANCH_CHANNEL_INDICES):
        channel_name = LOCKED_CHANNEL_NAMES[original_idx]
        rows.append(channel_stats(locked_tensor, axis_idx, "locked_140TR", channel_name))
        if alltr_tensor is not None:
            rows.append(channel_stats(alltr_tensor, axis_idx, "all_available_timepoints", channel_name))
        else:
            rows.append(
                {
                    "source": "all_available_timepoints",
                    "channel_axis_index": axis_idx,
                    "channel_name": channel_name,
                    "n_subjects": int(len(subjects)),
                    "finite_fraction": np.nan,
                    "mean_all": np.nan,
                    "std_all": np.nan,
                    "min_all": np.nan,
                    "max_all": np.nan,
                    "mean_offdiag": np.nan,
                    "std_offdiag": np.nan,
                    "diag_mean_abs": np.nan,
                    "diag_max_abs": np.nan,
                    "max_symmetry_abs_diff": np.nan,
                    "status": "pending_not_built_dry_run",
                }
            )
    return pd.DataFrame(rows)


def matrix_distance_vs_locked(
    subjects: pd.DataFrame,
    locked: dict[str, Any],
    alltr_tensor: np.ndarray | None,
) -> pd.DataFrame:
    columns = [
        "SubjectID",
        "channel_axis_index",
        "channel_name",
        "frobenius_offdiag",
        "mean_abs_diff_offdiag",
        "pearson_corr_offdiag",
        "locked_norm_offdiag",
        "alltr_norm_offdiag",
        "original_n_timepoints",
        "locked_n_timepoints_used",
        "ResearchGroup_Mapped",
        "Manufacturer",
        "SiteCode",
        "classifier_outer_fold",
    ]
    if alltr_tensor is None:
        return pd.DataFrame(
            [
                {
                    "SubjectID": "pending_not_built_dry_run",
                    "channel_axis_index": "",
                    "channel_name": "",
                    "frobenius_offdiag": np.nan,
                    "mean_abs_diff_offdiag": np.nan,
                    "pearson_corr_offdiag": np.nan,
                    "locked_norm_offdiag": np.nan,
                    "alltr_norm_offdiag": np.nan,
                    "original_n_timepoints": np.nan,
                    "locked_n_timepoints_used": 140,
                    "ResearchGroup_Mapped": "",
                    "Manufacturer": "",
                    "SiteCode": "",
                    "classifier_outer_fold": "",
                }
            ],
            columns=columns,
        )
    locked_tensor = locked["tensor"][locked["locked_indices_for_subject_table"]][:, BRANCH_CHANNEL_INDICES]
    mask = ~np.eye(OUTPUT_ROIS, dtype=bool)
    rows: list[dict[str, Any]] = []
    for subj_idx, subject in subjects.reset_index(drop=True).iterrows():
        for axis_idx, original_idx in enumerate(BRANCH_CHANNEL_INDICES):
            locked_vals = locked_tensor[subj_idx, axis_idx][mask].astype(np.float64)
            alltr_vals = alltr_tensor[subj_idx, axis_idx][mask].astype(np.float64)
            diff = alltr_vals - locked_vals
            if np.std(locked_vals) <= 1e-12 or np.std(alltr_vals) <= 1e-12:
                corr = np.nan
            else:
                corr = float(np.corrcoef(locked_vals, alltr_vals)[0, 1])
            rows.append(
                {
                    "SubjectID": subject["SubjectID"],
                    "channel_axis_index": axis_idx,
                    "channel_name": LOCKED_CHANNEL_NAMES[original_idx],
                    "frobenius_offdiag": float(np.linalg.norm(diff)),
                    "mean_abs_diff_offdiag": float(np.mean(np.abs(diff))),
                    "pearson_corr_offdiag": corr,
                    "locked_norm_offdiag": float(np.linalg.norm(locked_vals)),
                    "alltr_norm_offdiag": float(np.linalg.norm(alltr_vals)),
                    "original_n_timepoints": finite_float(subject["original_n_timepoints"]),
                    "locked_n_timepoints_used": finite_float(subject["locked_n_timepoints_used"]),
                    "ResearchGroup_Mapped": subject.get("ResearchGroup_Mapped", ""),
                    "Manufacturer": subject.get("Manufacturer", ""),
                    "SiteCode": str(subject.get("SiteCode", "")).zfill(3),
                    "classifier_outer_fold": subject.get("classifier_outer_fold", ""),
                }
            )
    return pd.DataFrame(rows)


def _welch_by_binary(df: pd.DataFrame, outcome: str, group_col: str, a: str, b: str) -> dict[str, Any]:
    a_vals = pd.to_numeric(df.loc[df[group_col].astype(str).eq(a), outcome], errors="coerce").dropna()
    b_vals = pd.to_numeric(df.loc[df[group_col].astype(str).eq(b), outcome], errors="coerce").dropna()
    if len(a_vals) < 2 or len(b_vals) < 2:
        return {"status": "insufficient_counts", "statistic": np.nan, "p_value": np.nan, "effect": np.nan}
    stat, p_value = stats.ttest_ind(a_vals, b_vals, equal_var=False, nan_policy="omit")
    return {
        "status": "ok",
        "statistic": float(stat),
        "p_value": float(p_value),
        "effect": float(b_vals.mean() - a_vals.mean()),
    }


def _kruskal_by_group(df: pd.DataFrame, outcome: str, group_col: str) -> dict[str, Any]:
    groups = []
    labels = []
    for label, group in df.groupby(group_col, dropna=True):
        vals = pd.to_numeric(group[outcome], errors="coerce").dropna()
        if len(vals) >= 2:
            groups.append(vals)
            labels.append(str(label))
    if len(groups) < 2:
        return {"status": "insufficient_groups", "statistic": np.nan, "p_value": np.nan, "effect": np.nan, "details": ""}
    stat, p_value = stats.kruskal(*groups)
    return {
        "status": "ok",
        "statistic": float(stat),
        "p_value": float(p_value),
        "effect": np.nan,
        "details": "groups=" + ",".join(labels),
    }


def confounding_guardrail_after_build(distance_df: pd.DataFrame, built: bool) -> pd.DataFrame:
    outcomes = ["frobenius_offdiag", "alltr_norm_offdiag"]
    rows: list[dict[str, Any]] = []
    if not built:
        for outcome in outcomes:
            for test_name in ["spearman_vs_n_TR", "welch_by_diagnosis", "kruskal_by_manufacturer", "kruskal_by_sitecode"]:
                rows.append(
                    {
                        "outcome": outcome,
                        "test_name": test_name,
                        "channel_name": "all_selected_channels",
                        "status": "pending_not_built_dry_run",
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "effect": np.nan,
                        "details": "Run with --confirm-build to compute allTR tensor and guardrail associations.",
                    }
                )
        return pd.DataFrame(rows)

    for channel_name, channel_df in distance_df.groupby("channel_name", dropna=False):
        for outcome in outcomes:
            x = pd.to_numeric(channel_df["original_n_timepoints"], errors="coerce")
            y = pd.to_numeric(channel_df[outcome], errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum() >= 3:
                stat, p_value = stats.spearmanr(x[mask], y[mask])
                rows.append(
                    {
                        "outcome": outcome,
                        "test_name": "spearman_vs_n_TR",
                        "channel_name": channel_name,
                        "status": "ok",
                        "statistic": float(stat),
                        "p_value": float(p_value),
                        "effect": float(stat),
                        "details": f"n={int(mask.sum())}",
                    }
                )
            diagnosis = _welch_by_binary(channel_df, outcome, "ResearchGroup_Mapped", "CN", "AD")
            rows.append(
                {
                    "outcome": outcome,
                    "test_name": "welch_by_diagnosis_CN_vs_AD",
                    "channel_name": channel_name,
                    "status": diagnosis["status"],
                    "statistic": diagnosis["statistic"],
                    "p_value": diagnosis["p_value"],
                    "effect": diagnosis["effect"],
                    "details": "effect=AD-CN mean difference",
                }
            )
            for group_col, test_name in [
                ("Manufacturer", "kruskal_by_manufacturer"),
                ("SiteCode", "kruskal_by_sitecode"),
            ]:
                result = _kruskal_by_group(channel_df, outcome, group_col)
                rows.append(
                    {
                        "outcome": outcome,
                        "test_name": test_name,
                        "channel_name": channel_name,
                        "status": result["status"],
                        "statistic": result["statistic"],
                        "p_value": result["p_value"],
                        "effect": result["effect"],
                        "details": result.get("details", ""),
                    }
                )
    return pd.DataFrame(rows)


def ntr_distribution_by_fold(subjects: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold, group in subjects.groupby("classifier_outer_fold", dropna=False):
        for dx, dx_group in group.groupby("ResearchGroup_Mapped", dropna=False):
            ntr = pd.to_numeric(dx_group["original_n_timepoints"], errors="coerce")
            rows.append(
                {
                    "classifier_outer_fold": fold,
                    "ResearchGroup_Mapped": dx,
                    "n": int(len(dx_group)),
                    "mean_n_TR": float(ntr.mean()),
                    "sd_n_TR": float(ntr.std(ddof=1)) if len(dx_group) > 1 else np.nan,
                    "median_n_TR": float(ntr.median()),
                    "min_n_TR": float(ntr.min()),
                    "max_n_TR": float(ntr.max()),
                }
            )
    return pd.DataFrame(rows)


def build_manifest(args: argparse.Namespace, subjects: pd.DataFrame, will_build: bool, tensor_path: Path | None) -> pd.DataFrame:
    cn_ad = subjects[subjects["ResearchGroup_Mapped"].isin(["CN", "AD"])]
    rows = [
        {
            "stage": "dry_run_preflight",
            "status": "complete",
            "real_build": False,
            "requires_confirm_build": False,
            "output": str(args.results_dir),
            "notes": "Preflight reports only.",
        },
        {
            "stage": "tensor_build",
            "status": "complete" if will_build else "blocked_not_launched",
            "real_build": bool(will_build),
            "requires_confirm_build": True,
            "output": str(tensor_path or (args.data_branch_dir / "subject_tensors" / TENSOR_FILENAME)),
            "notes": "Builds all-available-timepoints branch tensor only; locked tensor untouched.",
        },
        {
            "stage": "metadata_write",
            "status": "complete" if will_build else "blocked_not_launched",
            "real_build": bool(will_build),
            "requires_confirm_build": True,
            "output": str(args.data_branch_dir / METADATA_FILENAME),
            "notes": "Stores original n_TR per subject in branch metadata.",
        },
        {
            "stage": "tensor_qc",
            "status": "complete" if will_build else "pending_after_confirm_build",
            "real_build": bool(will_build),
            "requires_confirm_build": False,
            "output": str(args.results_dir / "tensor_qc.csv"),
            "notes": "Includes finite fraction, symmetry, diagonal, distribution, and distance checks.",
        },
        {
            "stage": "model_training",
            "status": "not_launched",
            "real_build": False,
            "requires_confirm_build": False,
            "output": "",
            "notes": "No VAE or classifier training is performed by this package.",
        },
    ]
    for row in rows:
        row.update(
            {
                "branch_name": BRANCH_NAME,
                "planned_subject_pool_n": int(len(subjects)),
                "planned_classifier_pool_n": int(len(cn_ad)),
                "planned_cn": int((cn_ad["ResearchGroup_Mapped"] == "CN").sum()),
                "planned_ad": int((cn_ad["ResearchGroup_Mapped"] == "AD").sum()),
                "planned_mci": int((subjects["ResearchGroup_Mapped"] == "MCI").sum()),
                "tensor_channel_axis": "0=Pearson_OMST, 1=Pearson_Full, 2=MI_KNN",
                "intended_channels_to_use": "[1,0,2]",
                "exploratory_label": "exploratory_upper_bound_confounding_stress_test",
            }
        )
    return pd.DataFrame(rows)


def write_readme(args: argparse.Namespace, will_build: bool) -> None:
    mode = "confirmed_build" if will_build else "dry_run_no_build"
    text = f"""# ADNI All-Available-Timepoints [1,0,2] Tensor Build

Mode: `{mode}`

This package prepares the explicitly exploratory branch
`{BRANCH_NAME}`. The branch is an upper-bound/confounding-stress-test
analysis because original timepoint count is associated with diagnosis,
Manufacturer, and SiteCode.

## Safety

- Locked v5.1b 140TR tensors are not modified.
- Locked metadata, ledgers, configs, and model outputs are not modified.
- Real tensor computation requires `--confirm-build`.
- No VAE or classifier training is launched by this package.

## Branch Data Output

- Data branch: `{args.data_branch_dir}`
- Tensor path: `{args.data_branch_dir / 'subject_tensors' / TENSOR_FILENAME}`
- Metadata path: `{args.data_branch_dir / METADATA_FILENAME}`

## Channel Axis

The branch tensor stores only the first three locked channel-axis entries:

- axis 0: `Pearson_OMST_GCE_Signed_Weighted`
- axis 1: `Pearson_Full_FisherZ_Signed`
- axis 2: `MI_KNN_Symmetric`

This preserves the locked model semantics for future `channels_to_use=[1,0,2]`.

## Interpretation Guardrail

This branch must not be promoted based on AUC alone. Any apparent internal
improvement must also pass n_TR/site/manufacturer leakage checks and external
OASIS validation.
"""
    (args.results_dir / "README.md").write_text(text, encoding="utf-8")


def write_final_recommendation(args: argparse.Namespace, will_build: bool) -> None:
    if will_build:
        decision = (
            "Tensor branch built. Proceed only to read-only QC review before any training. "
            "Any future FULL run remains exploratory and confounding-risk aware."
        )
    else:
        decision = (
            "Dry-run PASS. The package is ready, but no tensor was built. "
            "Run with --confirm-build only if the exploratory/confounding-stress-test branch is explicitly approved."
        )
    text = f"""# Final Recommendation

Decision: `{ 'branch_tensor_built_review_qc_before_training' if will_build else 'dry_run_ready_no_build' }`

{decision}

The all-available-timepoints branch remains exploratory because n_TR is
diagnosis-, Manufacturer-, and SiteCode-associated. It should not replace the
locked 140TR model unless it improves internal metrics, does not increase
nuisance leakage, and improves or preserves OASIS external ranking.
"""
    (args.results_dir / "final_recommendation.md").write_text(text, encoding="utf-8")


def write_command_log(
    args: argparse.Namespace,
    will_build: bool,
    tensor_path: Path | None,
    metadata_path: Path | None,
    started: str,
) -> None:
    command_log = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "started": started,
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "argv": sys.argv,
        "mode": "confirmed_build" if will_build else "dry_run",
        "confirm_build": bool(args.confirm_build),
        "overwrite": bool(args.overwrite),
        "n_jobs": int(args.n_jobs),
        "pairwise_n_jobs": int(args.pairwise_n_jobs),
        "inputs": {
            "subject_timepoint_table": str(args.subject_timepoint_table),
            "locked_tensor": str(args.locked_tensor),
            "locked_metadata": str(args.locked_metadata),
        },
        "outputs": {
            "results_dir": str(args.results_dir),
            "data_branch_dir": str(args.data_branch_dir),
            "tensor_path": str(tensor_path) if tensor_path else str(args.data_branch_dir / "subject_tensors" / TENSOR_FILENAME),
            "metadata_path": str(metadata_path) if metadata_path else str(args.data_branch_dir / METADATA_FILENAME),
        },
        "safety": {
            "training_launched": False,
            "locked_tensor_modified": False,
            "locked_metadata_modified": False,
            "ledger_modified": False,
            "locked_model_outputs_modified": False,
            "real_tensor_build_launched": bool(will_build),
        },
        "exploratory_label": "exploratory_upper_bound_confounding_stress_test",
    }
    (args.results_dir / "command_log.json").write_text(
        json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    started = datetime.now().isoformat(timespec="seconds")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    require_inputs(args)
    subjects = load_subjects(args)
    locked = load_locked_reference(args, subjects)

    will_build = bool(args.confirm_build)
    tensor_path: Path | None = None
    metadata_path: Path | None = None
    alltr_tensor: np.ndarray | None = None

    if will_build:
        tensor_path, metadata_path = ensure_real_build_paths(args)
        alltr_tensor, subject_qc, channel_qc = build_tensor(subjects, args)
        save_branch_outputs(alltr_tensor, subjects, locked, tensor_path, metadata_path)
        write_csv_md(subject_qc, args.results_dir / "subject_build_qc.csv", "Subject Build QC")
        write_csv_md(channel_qc, args.results_dir / "channel_build_qc.csv", "Channel Build QC")

    subject_cols = [
        "SubjectID",
        "tensor_index",
        "ResearchGroup_Mapped",
        "Diagnosis",
        "classifier_pool_role",
        "classifier_outer_fold",
        "in_vae_pool_any_fold",
        "SiteCode",
        "Manufacturer",
        "Age",
        "Sex",
        "roi_signal_path",
        "raw_shape",
        "original_n_timepoints",
        "locked_n_timepoints_used",
        "branch_n_timepoints_used",
        "branch_timepoint_policy",
        "roi_signal_path_exists_now",
    ]
    write_csv_md(
        subjects[[c for c in subject_cols if c in subjects.columns]],
        args.results_dir / "subject_timepoint_used.csv",
        "Subject Timepoints Used",
    )

    manifest = build_manifest(args, subjects, will_build, tensor_path)
    write_csv_md(manifest, args.results_dir / "build_manifest.csv", "Build Manifest")
    tensor_qc = tensor_qc_table(subjects, locked, alltr_tensor, tensor_path)
    write_csv_md(tensor_qc, args.results_dir / "tensor_qc.csv", "Tensor QC")
    distribution = channel_distribution_comparison(subjects, locked, alltr_tensor)
    write_csv_md(
        distribution,
        args.results_dir / "channel_distribution_comparison.csv",
        "Channel Distribution Comparison",
    )
    distances = matrix_distance_vs_locked(subjects, locked, alltr_tensor)
    write_csv_md(
        distances,
        args.results_dir / "matrix_distance_vs_140TR.csv",
        "Matrix Distance vs 140TR",
    )
    guardrail = confounding_guardrail_after_build(distances, built=will_build)
    write_csv_md(
        guardrail,
        args.results_dir / "confounding_guardrail_after_build.csv",
        "Confounding Guardrail After Build",
    )
    ntr_fold = ntr_distribution_by_fold(subjects)
    write_csv_md(
        ntr_fold,
        args.results_dir / "ntr_distribution_by_fold.csv",
        "n_TR Distribution by Fold",
    )
    write_readme(args, will_build)
    write_final_recommendation(args, will_build)
    write_command_log(args, will_build, tensor_path, metadata_path, started)

    print(
        json.dumps(
            {
                "mode": "confirmed_build" if will_build else "dry_run",
                "results_dir": str(args.results_dir),
                "data_branch_dir": str(args.data_branch_dir),
                "tensor_built": bool(will_build),
                "training_launched": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
