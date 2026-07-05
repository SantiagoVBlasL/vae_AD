#!/usr/bin/env python3
"""Read-only inventory/QC audit for Martin's OASIS batch.

This script only reads the OASIS input folder and writes lightweight audit
tables under results/. It does not concatenate runs, compute connectomes, or
modify any input data.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import scipy.io as scipy_io
except Exception:  # pragma: no cover - reported in command log if unavailable.
    scipy_io = None


RUN_RE = re.compile(
    r"sub-(OAS\d+)_ses-(d\d+)_task-([A-Za-z0-9]+)_run-(\d+)_bold",
    re.IGNORECASE,
)
EXPERIMENT_RE = re.compile(r"(OAS\d+)_MR_(d\d+)", re.IGNORECASE)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def normalized_ext(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    if path.suffix:
        return path.suffix.lower()
    return ""


def extract_ids(text: str) -> dict[str, str | None]:
    m = RUN_RE.search(text)
    if m:
        subject = f"sub-{m.group(1).upper()}"
        session = f"ses-{m.group(2).lower()}"
        run = f"run-{int(m.group(4)):02d}"
        return {
            "subject_id": subject,
            "session_id": session,
            "run_id": run,
            "task": m.group(3).lower(),
            "experiment_id": f"{subject.replace('sub-', '')}_MR_{session.replace('ses-', '')}",
        }

    m = EXPERIMENT_RE.search(text)
    if m:
        subject = f"sub-{m.group(1).upper()}"
        session = f"ses-{m.group(2).lower()}"
        return {
            "subject_id": subject,
            "session_id": session,
            "run_id": None,
            "task": None,
            "experiment_id": f"{subject.replace('sub-', '')}_MR_{session.replace('ses-', '')}",
        }

    m = re.search(r"sub-(OAS\d+)", text, re.IGNORECASE)
    subject = f"sub-{m.group(1).upper()}" if m else None
    m = re.search(r"ses-(d\d+)", text, re.IGNORECASE)
    session = f"ses-{m.group(1).lower()}" if m else None
    experiment = None
    if subject and session:
        experiment = f"{subject.replace('sub-', '')}_MR_{session.replace('ses-', '')}"
    return {
        "subject_id": subject,
        "session_id": session,
        "run_id": None,
        "task": None,
        "experiment_id": experiment,
    }


def classify_file(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    rel_lower = str(rel).lower()
    name_lower = path.name.lower()
    ext = normalized_ext(path)

    if "resultsaal3" in rel_lower or name_lower.startswith("roisignals_"):
        if ext in {".mat", ".txt", ".csv", ".tsv"}:
            return "ROI time series"
    if "_bold" in name_lower and ext in {".nii", ".nii.gz"}:
        return "BOLD NIfTI"
    if "t1w" in name_lower and ext in {".nii", ".nii.gz"}:
        return "T1w NIfTI"
    if ext == ".json":
        return "JSON metadata"
    if ext in {".png", ".gif"} and ("check" in rel_lower or "_bold" in name_lower):
        return "QC image"
    if (
        name_lower.startswith("fd_")
        or name_lower.startswith("rp_")
        or "headmotion" in name_lower
        or "exclude" in name_lower
    ):
        return "motion/QC metadata"
    if ext in {".csv", ".tsv"} or name_lower.startswith("readme"):
        return "metadata"
    if ext == ".txt" and path.parent == root:
        return "metadata"
    if ext == ".mat" and "realignparameter" in rel_lower:
        return "realignment transform"
    return "unknown"


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        x = float(value)
        if math.isnan(x):
            return None
        return x
    except Exception:
        return None


def to_sub_id(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("sub-"):
        return text
    m = re.search(r"(OAS\d+)", text, re.IGNORECASE)
    if m:
        return f"sub-{m.group(1).upper()}"
    return text


def to_session_id(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    m = re.search(r"(d\d+)", text, re.IGNORECASE)
    if m:
        return f"ses-{m.group(1).lower()}"
    return text


def metadata_rows(metadata_path: Path) -> pd.DataFrame:
    if not metadata_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(metadata_path)
    out_rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        experiment = row.get("experiment_id")
        subject = to_sub_id(row.get("subject_id"))
        session = to_session_id(row.get("session_id"))
        if pd.notna(experiment):
            ids = extract_ids(str(experiment))
            subject = subject or ids["subject_id"]
            session = session or ids["session_id"]
        if not experiment and subject and session:
            experiment = f"{subject.replace('sub-', '')}_MR_{session.replace('ses-', '')}"
        out_rows.append(
            {
                "subject_id": subject,
                "session_id": session,
                "experiment_id": experiment,
                "diagnosis": row.get("diagnosis"),
                "diagnosis_confidence": row.get("diagnosis_confidence"),
                "TR_seconds": row.get("TR_seconds"),
                "ScannerModel": row.get("ScannerModel"),
                "Manufacturer": row.get("Manufacturer"),
                "age_at_MR": row.get("age_at_MR"),
                "sex": row.get("sex"),
                "CDRTOT": row.get("CDRTOT"),
                "CDRSUM": row.get("CDRSUM"),
                "clinical_delta_days": row.get("clinical_delta_days"),
                "metadata_source_file": str(metadata_path),
            }
        )
    return pd.DataFrame(out_rows)


def split_path_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    parts = re.split(r"[;|]\s*|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def expected_bold_rows(download_qc_path: Path) -> pd.DataFrame:
    if not download_qc_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(download_qc_path)
    bold_path_col = None
    for col in df.columns:
        c = col.lower()
        if "bold" in c and "path" in c:
            bold_path_col = col
            break
    suspicious_col = None
    for col in df.columns:
        c = col.lower()
        if "suspicious" in c or "tiny" in c:
            if "path" in c or "file" in c:
                suspicious_col = col
                break
    rows = []
    for _, row in df.iterrows():
        suspicious_paths = set()
        if suspicious_col:
            suspicious_paths = {Path(p).name for p in split_path_list(row.get(suspicious_col))}
        for raw_path in split_path_list(row.get(bold_path_col)):
            ids = extract_ids(raw_path)
            if not ids["subject_id"] or not ids["session_id"] or not ids["run_id"]:
                continue
            rows.append(
                {
                    **ids,
                    "raw_bold_path_from_download_qc": raw_path,
                    "raw_bold_expected": True,
                    "raw_bold_flagged_tiny_or_suspicious": Path(raw_path).name in suspicious_paths,
                }
            )
    return pd.DataFrame(rows)


def find_numeric_matrix(mat_dict: dict[str, Any]) -> tuple[str | None, np.ndarray | None]:
    preferred = ["signals", "ROISignals", "roi_signals", "Signal", "data"]
    for key in preferred:
        arr = mat_dict.get(key)
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
            return key, arr
    for key, arr in mat_dict.items():
        if key.startswith("__"):
            continue
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
            return key, arr
    return None, None


def load_text_matrix(path: Path) -> tuple[tuple[int, int] | None, str | None]:
    for delimiter_name, delimiter in [("comma", ","), ("whitespace", None), ("tab", "\t")]:
        try:
            arr = np.loadtxt(path, delimiter=delimiter)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return tuple(arr.shape), delimiter_name
        except Exception:
            continue
    return None, None


def roi_signal_qc(root: Path, inventory_df: pd.DataFrame) -> pd.DataFrame:
    roi_files = inventory_df[
        (inventory_df["modality_type"] == "ROI time series")
        & (inventory_df["extension"].isin([".mat", ".txt"]))
    ].copy()
    mat_rows = roi_files[roi_files["extension"] == ".mat"]
    rows = []
    for _, inv in mat_rows.iterrows():
        path = Path(inv["absolute_path"])
        ids = extract_ids(str(path))
        row: dict[str, Any] = {
            **ids,
            "roi_mat_path": str(path),
            "roi_txt_path": "",
            "mat_key": "",
            "mat_loadable": False,
            "txt_pair_exists": False,
            "txt_loadable": False,
            "txt_delimiter": "",
            "txt_shape": "",
            "n_timepoints": np.nan,
            "n_rois": np.nan,
            "matrix_orientation": "",
            "finite_fraction": np.nan,
            "nan_count": np.nan,
            "n_all_nan_roi_columns": np.nan,
            "all_nan_roi_column_indices_1based": "",
            "n_partial_nan_roi_columns": np.nan,
            "min_value": np.nan,
            "max_value": np.nan,
            "mean_value": np.nan,
            "std_value": np.nan,
            "aal3_170_shape_match": False,
            "aal3_131_order_match": "unknown_requires_roi_mapping",
            "roi_order_detected": False,
            "qc_ok_for_connectome_input_after_order_confirmation": False,
            "qc_notes": "",
        }
        arr = None
        if scipy_io is None:
            row["qc_notes"] = "scipy_unavailable_for_mat_loading"
        else:
            try:
                mat = scipy_io.loadmat(path)
                key, arr = find_numeric_matrix(mat)
                row["mat_key"] = key or ""
                row["mat_loadable"] = arr is not None
            except Exception as exc:
                row["qc_notes"] = f"mat_load_error: {exc}"
        if arr is not None:
            arr = np.asarray(arr, dtype=float)
            row["n_timepoints"] = int(arr.shape[0])
            row["n_rois"] = int(arr.shape[1])
            row["matrix_orientation"] = "timepoints_by_rois"
            finite = np.isfinite(arr)
            row["finite_fraction"] = float(finite.mean())
            row["nan_count"] = int((~finite).sum())
            col_finite_fraction = finite.mean(axis=0)
            all_nan_cols = np.where(col_finite_fraction == 0)[0] + 1
            partial_nan_cols = np.where((col_finite_fraction > 0) & (col_finite_fraction < 1))[0] + 1
            row["n_all_nan_roi_columns"] = int(len(all_nan_cols))
            row["all_nan_roi_column_indices_1based"] = ",".join(str(int(x)) for x in all_nan_cols)
            row["n_partial_nan_roi_columns"] = int(len(partial_nan_cols))
            if finite.any():
                finite_values = arr[finite]
                row["min_value"] = float(np.min(finite_values))
                row["max_value"] = float(np.max(finite_values))
                row["mean_value"] = float(np.mean(finite_values))
                row["std_value"] = float(np.std(finite_values))
            row["aal3_170_shape_match"] = int(arr.shape[1]) == 170
            row["qc_ok_for_connectome_input_after_order_confirmation"] = (
                int(arr.shape[1]) == 170
                and int(arr.shape[0]) >= 100
                and float(finite.mean()) >= 0.95
                and int(len(partial_nan_cols)) == 0
            )
            if not row["qc_notes"]:
                if row["aal3_170_shape_match"] and len(all_nan_cols):
                    row["qc_notes"] = (
                        "shape_matches_AAL3_170_with_all_nan_roi_columns; "
                        "confirm whether ADNI 131-ROI mapping excludes these columns"
                    )
                elif row["aal3_170_shape_match"]:
                    row["qc_notes"] = "shape_matches_AAL3_170_but_explicit_ROI_order_file_not_found"
                else:
                    row["qc_notes"] = "unexpected_roi_count"
        txt_path = path.with_suffix(".txt")
        if txt_path.exists():
            row["txt_pair_exists"] = True
            row["roi_txt_path"] = str(txt_path)
            shape, delimiter = load_text_matrix(txt_path)
            row["txt_loadable"] = shape is not None
            row["txt_shape"] = str(shape) if shape else ""
            row["txt_delimiter"] = delimiter or ""
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["subject_id", "session_id", "run_id"], na_position="last"
    )


def build_inventory(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        stat = path.stat()
        ids = extract_ids(str(path))
        rows.append(
            {
                "file_path": str(path.relative_to(root)),
                "absolute_path": str(path),
                "extension": normalized_ext(path),
                "size_bytes": int(stat.st_size),
                "size_mb": round(stat.st_size / (1024 * 1024), 6),
                "subject_id": ids["subject_id"],
                "session_id": ids["session_id"],
                "run_id": ids["run_id"],
                "task": ids["task"],
                "experiment_id": ids["experiment_id"],
                "modality_type": classify_file(path, root),
                "is_suspicious_tiny_file": stat.st_size < 1024 and normalized_ext(path) not in {".txt", ".tsv", ".csv"},
            }
        )
    return pd.DataFrame(rows)


def init_run_row(subject_id: str | None, session_id: str | None, run_id: str | None) -> dict[str, Any]:
    experiment = None
    if subject_id and session_id:
        experiment = f"{subject_id.replace('sub-', '')}_MR_{session_id.replace('ses-', '')}"
    return {
        "subject_id": subject_id,
        "session_id": session_id,
        "run_id": run_id,
        "experiment_id": experiment,
        "task": "rest",
        "raw_bold_expected": False,
        "raw_bold_flagged_tiny_or_suspicious": False,
        "has_roi_mat": False,
        "has_roi_txt": False,
        "roi_qc_ok": False,
        "n_timepoints": np.nan,
        "n_rois": np.nan,
        "finite_fraction": np.nan,
        "has_check_png": False,
        "has_check_gif": False,
        "has_fd_power": False,
        "has_fd_vandijk": False,
        "has_fd_jenkinson": False,
        "has_realign_params": False,
        "has_mean_nifti": False,
        "has_wmean_nifti": False,
        "n_associated_files": 0,
        "roi_mat_path": "",
        "roi_txt_path": "",
        "diagnosis": "",
        "diagnosis_confidence": "",
        "TR_seconds": np.nan,
        "Manufacturer": "",
        "ScannerModel": "",
        "age_at_MR": np.nan,
        "sex": "",
        "CDRTOT": np.nan,
        "CDRSUM": np.nan,
    }


def build_run_manifest(
    inventory_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    expected_bold_df: pd.DataFrame,
    roi_qc_df: pd.DataFrame,
) -> pd.DataFrame:
    runs: dict[tuple[str | None, str | None, str | None], dict[str, Any]] = {}

    def get_row(subject: str | None, session: str | None, run: str | None) -> dict[str, Any]:
        key = (subject, session, run)
        if key not in runs:
            runs[key] = init_run_row(subject, session, run)
        return runs[key]

    for _, row in expected_bold_df.iterrows():
        rr = get_row(row["subject_id"], row["session_id"], row["run_id"])
        rr["raw_bold_expected"] = True
        rr["raw_bold_flagged_tiny_or_suspicious"] = bool(
            rr["raw_bold_flagged_tiny_or_suspicious"]
            or row.get("raw_bold_flagged_tiny_or_suspicious", False)
        )

    for _, row in inventory_df.iterrows():
        if not row.get("subject_id") or not row.get("session_id") or not row.get("run_id"):
            continue
        rr = get_row(row["subject_id"], row["session_id"], row["run_id"])
        rr["n_associated_files"] += 1
        file_path = str(row["file_path"])
        name = Path(file_path).name.lower()
        modality = row["modality_type"]
        ext = row["extension"]
        if modality == "ROI time series" and ext == ".mat":
            rr["has_roi_mat"] = True
            rr["roi_mat_path"] = str(row["absolute_path"])
        if modality == "ROI time series" and ext == ".txt":
            rr["has_roi_txt"] = True
            rr["roi_txt_path"] = str(row["absolute_path"])
        if modality == "QC image" and ext == ".png":
            rr["has_check_png"] = True
        if modality == "QC image" and ext == ".gif":
            rr["has_check_gif"] = True
        if name.startswith("fd_power"):
            rr["has_fd_power"] = True
        if name.startswith("fd_vandijk"):
            rr["has_fd_vandijk"] = True
        if name.startswith("fd_jenkinson"):
            rr["has_fd_jenkinson"] = True
        if name.startswith("rp_"):
            rr["has_realign_params"] = True
        if name.startswith("meana") and ext == ".nii":
            rr["has_mean_nifti"] = True
        if name.startswith("wmeana") and ext == ".nii":
            rr["has_wmean_nifti"] = True

    for _, row in roi_qc_df.iterrows():
        rr = get_row(row["subject_id"], row["session_id"], row["run_id"])
        rr["has_roi_mat"] = True
        rr["has_roi_txt"] = bool(row.get("txt_pair_exists", False))
        rr["roi_qc_ok"] = bool(row.get("qc_ok_for_connectome_input_after_order_confirmation", False))
        rr["n_timepoints"] = row.get("n_timepoints")
        rr["n_rois"] = row.get("n_rois")
        rr["finite_fraction"] = row.get("finite_fraction")
        rr["roi_mat_path"] = row.get("roi_mat_path", "")
        rr["roi_txt_path"] = row.get("roi_txt_path", "")

    meta_by_exp = {
        row["experiment_id"]: row
        for _, row in metadata_df.iterrows()
        if pd.notna(row.get("experiment_id"))
    }
    for rr in runs.values():
        meta = meta_by_exp.get(rr["experiment_id"])
        if meta is not None:
            for col in [
                "diagnosis",
                "diagnosis_confidence",
                "TR_seconds",
                "Manufacturer",
                "ScannerModel",
                "age_at_MR",
                "sex",
                "CDRTOT",
                "CDRSUM",
            ]:
                rr[col] = meta.get(col, rr.get(col))

        if rr["roi_qc_ok"]:
            rr["run_availability_status"] = "roi_signal_qc_ok_after_roi_order_confirmation"
        elif rr["has_roi_mat"]:
            rr["run_availability_status"] = "roi_signal_present_but_qc_flagged"
        elif rr["raw_bold_expected"]:
            rr["run_availability_status"] = "raw_bold_expected_but_no_processed_roi_signal"
        else:
            rr["run_availability_status"] = "associated_non_roi_file_only"

    out = pd.DataFrame(runs.values())
    if out.empty:
        return out
    return out.sort_values(["subject_id", "session_id", "run_id"], na_position="last")


def build_subject_session_manifest(run_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = run_df.groupby(["subject_id", "session_id", "experiment_id"], dropna=False)
    run_summary: dict[tuple[Any, Any, Any], pd.DataFrame] = {
        key: g for key, g in grouped
    }

    seen_keys = set()
    for _, meta in metadata_df.iterrows():
        key = (meta.get("subject_id"), meta.get("session_id"), meta.get("experiment_id"))
        seen_keys.add(key)
        g = run_summary.get(key, pd.DataFrame())
        expected = int(g["raw_bold_expected"].sum()) if not g.empty else 0
        roi_runs = int(g["has_roi_mat"].sum()) if not g.empty else 0
        roi_ok = int(g["roi_qc_ok"].sum()) if not g.empty else 0
        if roi_ok >= 2:
            recommendation = "usable_after_roi_order_resolution_two_or_more_qc_ok_runs"
        elif roi_ok == 1:
            recommendation = "usable_after_roi_order_resolution_single_qc_ok_run"
        else:
            recommendation = "not_usable_until_processed_roi_signal_available"
        rows.append(
            {
                "subject_id": meta.get("subject_id"),
                "session_id": meta.get("session_id"),
                "experiment_id": meta.get("experiment_id"),
                "diagnosis": meta.get("diagnosis"),
                "diagnosis_confidence": meta.get("diagnosis_confidence"),
                "TR_seconds": meta.get("TR_seconds"),
                "Manufacturer": meta.get("Manufacturer"),
                "ScannerModel": meta.get("ScannerModel"),
                "age_at_MR": meta.get("age_at_MR"),
                "sex": meta.get("sex"),
                "expected_raw_bold_runs": expected,
                "processed_roi_runs": roi_runs,
                "roi_qc_ok_runs": roi_ok,
                "has_multiple_qc_ok_runs": roi_ok >= 2,
                "subject_session_recommendation": recommendation,
                "run_combination_note": (
                    "if multiple QC-ok runs are retained, pre-specify concatenate time series or average per-run Fisher-z connectomes"
                    if roi_ok >= 2
                    else "single QC-ok run or no ROI run available"
                ),
            }
        )

    for key, g in run_summary.items():
        if key in seen_keys:
            continue
        subject_id, session_id, experiment_id = key
        rows.append(
            {
                "subject_id": subject_id,
                "session_id": session_id,
                "experiment_id": experiment_id,
                "diagnosis": "",
                "diagnosis_confidence": "",
                "TR_seconds": np.nan,
                "Manufacturer": "",
                "ScannerModel": "",
                "age_at_MR": np.nan,
                "sex": "",
                "expected_raw_bold_runs": int(g["raw_bold_expected"].sum()),
                "processed_roi_runs": int(g["has_roi_mat"].sum()),
                "roi_qc_ok_runs": int(g["roi_qc_ok"].sum()),
                "has_multiple_qc_ok_runs": int(g["roi_qc_ok"].sum()) >= 2,
                "subject_session_recommendation": "metadata_resolution_needed",
                "run_combination_note": "run files exist but subject/session metadata was not found",
            }
        )

    return pd.DataFrame(rows).sort_values(["subject_id", "session_id"], na_position="last")


def build_diagnosis_counts(subject_df: pd.DataFrame, run_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for diagnosis, g in subject_df.groupby("diagnosis", dropna=False):
        rows.append(
            {
                "level": "subject_session",
                "diagnosis": diagnosis,
                "n": int(len(g)),
                "n_with_at_least_one_roi_qc_ok_run": int((g["roi_qc_ok_runs"] >= 1).sum()),
                "n_with_two_or_more_roi_qc_ok_runs": int((g["roi_qc_ok_runs"] >= 2).sum()),
            }
        )
    for diagnosis, g in run_df.groupby("diagnosis", dropna=False):
        rows.append(
            {
                "level": "run",
                "diagnosis": diagnosis,
                "n": int(len(g)),
                "n_with_at_least_one_roi_qc_ok_run": int(g["roi_qc_ok"].sum()),
                "n_with_two_or_more_roi_qc_ok_runs": np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_missing_ambiguous(
    root: Path,
    subject_df: pd.DataFrame,
    run_df: pd.DataFrame,
    roi_qc_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    roi_order_candidates = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and ("label" in p.name.lower() or "aal3" in p.name.lower())
        and not p.name.lower().startswith("roisignals_")
    ]
    if not roi_order_candidates:
        rows.append(
            {
                "issue_type": "roi_order_metadata_missing",
                "subject_id": "GLOBAL",
                "session_id": "",
                "run_id": "",
                "severity": "blocking_for_final_connectome_build",
                "details": "ROI signals have AAL3-shaped 170 columns, but no explicit ROI label/order file was found in this batch.",
                "recommended_action": "confirm DPABI/AAL3 ordering and ADNI 170-to-131 mapping before building external-validation tensors",
            }
        )

    if not roi_qc_df.empty and "n_all_nan_roi_columns" in roi_qc_df.columns:
        n_with_all_nan = int((roi_qc_df["n_all_nan_roi_columns"].fillna(0) > 0).sum())
        if n_with_all_nan:
            most_common_nan_sets = (
                roi_qc_df["all_nan_roi_column_indices_1based"]
                .fillna("")
                .value_counts()
                .head(5)
                .to_dict()
            )
            rows.append(
                {
                    "issue_type": "all_nan_roi_columns_present",
                    "subject_id": "GLOBAL",
                    "session_id": "",
                    "run_id": "",
                    "severity": "blocking_for_final_connectome_build",
                    "details": (
                        f"{n_with_all_nan} ROI-signal runs contain complete all-NaN ROI columns; "
                        f"most_common_nan_column_sets={most_common_nan_sets}"
                    ),
                    "recommended_action": (
                        "confirm whether the ADNI AAL3 170-to-131 mapping drops these columns; "
                        "otherwise define a pre-specified ROI exclusion/imputation rule before connectome construction"
                    ),
                }
            )

    for _, row in run_df.iterrows():
        if bool(row.get("raw_bold_expected")) and not bool(row.get("has_roi_mat")):
            rows.append(
                {
                    "issue_type": "expected_bold_without_processed_roi_signal",
                    "subject_id": row.get("subject_id"),
                    "session_id": row.get("session_id"),
                    "run_id": row.get("run_id"),
                    "severity": "run_level",
                    "details": "Raw/download-QC BOLD run is expected, but no ROISignals file was found in ResultsAAL3.",
                    "recommended_action": "treat as run-level unavailable; do not discard the session if another QC-ok run exists",
                }
            )
        if bool(row.get("has_roi_mat")) and not str(row.get("diagnosis", "")).strip():
            rows.append(
                {
                    "issue_type": "roi_run_without_diagnosis_metadata",
                    "subject_id": row.get("subject_id"),
                    "session_id": row.get("session_id"),
                    "run_id": row.get("run_id"),
                    "severity": "subject_level",
                    "details": "ROISignals exist but diagnosis metadata was not found in the pilot metadata table.",
                    "recommended_action": "resolve metadata before use",
                }
            )

    for _, row in subject_df.iterrows():
        if not str(row.get("diagnosis", "")).strip() or str(row.get("diagnosis", "")).lower() == "nan":
            rows.append(
                {
                    "issue_type": "missing_diagnosis",
                    "subject_id": row.get("subject_id"),
                    "session_id": row.get("session_id"),
                    "run_id": "",
                    "severity": "subject_level",
                    "details": "Subject/session has no diagnosis label.",
                    "recommended_action": "resolve before external validation",
                }
            )
        if int(row.get("roi_qc_ok_runs", 0)) == 0:
            rows.append(
                {
                    "issue_type": "no_qc_ok_roi_signal_run",
                    "subject_id": row.get("subject_id"),
                    "session_id": row.get("session_id"),
                    "run_id": "",
                    "severity": "subject_level",
                    "details": "No processed ROI signal run passed basic shape/finite QC.",
                    "recommended_action": "exclude subject/session unless ROI signals can be recovered",
                }
            )

    for _, row in roi_qc_df.iterrows():
        if not bool(row.get("aal3_170_shape_match")):
            rows.append(
                {
                    "issue_type": "unexpected_roi_count",
                    "subject_id": row.get("subject_id"),
                    "session_id": row.get("session_id"),
                    "run_id": row.get("run_id"),
                    "severity": "run_level",
                    "details": f"ROI matrix has n_rois={row.get('n_rois')}, expected 170 for DPABI AAL3.",
                    "recommended_action": "exclude or resolve ROI format before connectome construction",
                }
            )
        if safe_float(row.get("finite_fraction")) is not None and safe_float(row.get("finite_fraction")) < 0.999:
            rows.append(
                {
                    "issue_type": "nonfinite_roi_values",
                    "subject_id": row.get("subject_id"),
                    "session_id": row.get("session_id"),
                    "run_id": row.get("run_id"),
                    "severity": "run_level",
                    "details": f"finite_fraction={row.get('finite_fraction')}",
                    "recommended_action": "exclude run or inspect source preprocessing",
                }
            )

    return pd.DataFrame(rows)


def markdown_escape(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value)
    text = text.replace("|", "\\|")
    text = text.replace("\n", " ")
    return text


def write_markdown_table(df: pd.DataFrame, path: Path, title: str, max_rows: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"Rows: {len(df)}\n\n")
        if df.empty:
            f.write("No rows.\n")
            return
        display = df.head(max_rows)
        if len(df) > max_rows:
            f.write(f"Showing first {max_rows} rows.\n\n")
        cols = list(display.columns)
        f.write("| " + " | ".join(markdown_escape(c) for c in cols) + " |\n")
        f.write("| " + " | ".join("---" for _ in cols) + " |\n")
        for _, row in display.iterrows():
            f.write("| " + " | ".join(markdown_escape(row.get(c)) for c in cols) + " |\n")


def write_csv_md(df: pd.DataFrame, out_csv: Path, title: str, max_md_rows: int = 200) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    write_markdown_table(df, out_csv.with_suffix(".md"), title, max_rows=max_md_rows)


def make_final_recommendation(
    output_path: Path,
    inventory_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    expected_bold_df: pd.DataFrame,
    run_df: pd.DataFrame,
    subject_df: pd.DataFrame,
    roi_qc_df: pd.DataFrame,
    missing_df: pd.DataFrame,
) -> str:
    diagnosis_counts = subject_df["diagnosis"].value_counts(dropna=False).to_dict()
    n_subject_sessions = len(subject_df)
    n_roi_runs = int(roi_qc_df.shape[0])
    n_roi_ok = int(roi_qc_df["qc_ok_for_connectome_input_after_order_confirmation"].sum()) if not roi_qc_df.empty else 0
    n_subjects_roi_ok = int((subject_df["roi_qc_ok_runs"] >= 1).sum()) if not subject_df.empty else 0
    n_subjects_multi = int((subject_df["roi_qc_ok_runs"] >= 2).sum()) if not subject_df.empty else 0
    n_raw_expected = int(expected_bold_df.shape[0])
    n_roi_runs_with_all_nan_cols = int((roi_qc_df["n_all_nan_roi_columns"].fillna(0) > 0).sum()) if not roi_qc_df.empty and "n_all_nan_roi_columns" in roi_qc_df.columns else 0
    common_nan_col_sets = (
        roi_qc_df["all_nan_roi_column_indices_1based"].fillna("").value_counts().head(3).to_dict()
        if not roi_qc_df.empty and "all_nan_roi_column_indices_1based" in roi_qc_df.columns
        else {}
    )
    n_missing_raw_roi = int(
        (
            (run_df["raw_bold_expected"].fillna(False))
            & (~run_df["has_roi_mat"].fillna(False))
        ).sum()
    ) if not run_df.empty else 0
    n_files = int(len(inventory_df))
    ext_counts = Counter(inventory_df["extension"].fillna(""))
    modality_counts = Counter(inventory_df["modality_type"].fillna(""))
    roi_order_blocker = (
        not missing_df.empty
        and (missing_df["issue_type"] == "roi_order_metadata_missing").any()
    )
    if metadata_df.empty:
        recommendation = "needs_metadata_resolution"
    elif roi_order_blocker:
        recommendation = "needs_roi_order_resolution"
    elif n_subjects_roi_ok < n_subject_sessions:
        recommendation = "needs_metadata_resolution"
    else:
        recommendation = "ready_for_connectome_build"

    lines = [
        "# Final Recommendation",
        "",
        f"Recommendation: `{recommendation}`",
        "",
        "## What Is Available",
        "",
        f"- Input folder: `{inventory_df['absolute_path'].iloc[0].split('/data/Tanda_2026_05_25/')[0] + '/data/Tanda_2026_05_25' if not inventory_df.empty else 'data/Tanda_2026_05_25'}`",
        f"- Total files inventoried: {n_files}",
        f"- File extensions: {dict(ext_counts)}",
        f"- Modality/type counts: {dict(modality_counts)}",
        f"- Metadata subject/sessions: {n_subject_sessions}",
        f"- Diagnosis counts: {diagnosis_counts}",
        f"- Raw/download-QC BOLD runs expected: {n_raw_expected}",
        f"- Processed AAL3 ROI-signal runs found: {n_roi_runs}",
        f"- ROI-signal runs passing basic shape/finite QC: {n_roi_ok}",
        f"- ROI-signal runs with complete all-NaN ROI columns: {n_roi_runs_with_all_nan_cols}",
        f"- Most common all-NaN ROI column sets: {common_nan_col_sets}",
        f"- Subject/sessions with at least one QC-ok ROI run: {n_subjects_roi_ok}",
        f"- Subject/sessions with two or more QC-ok ROI runs: {n_subjects_multi}",
        f"- Expected BOLD runs without processed ROI signals: {n_missing_raw_roi}",
        "",
        "## Interpretation",
        "",
        (
            "The batch contains diagnosis metadata and processed AAL3 ROI time-series for external validation, "
            "with multiple runs available for many subject/sessions. Basic ROI-signal QC indicates AAL3-shaped "
            "170-column matrices. The matrices also contain a consistent set of complete all-NaN ROI columns, "
            "which must be reconciled against the ADNI ROI filtering/mapping. This audit did not find an explicit ROI label/order file in the batch. "
            "Because the ADNI model uses an AAL3-derived 131-ROI tensor after project-specific ROI filtering, "
            "the OASIS ROI order and the 170-to-131 mapping must be confirmed before connectome construction."
        ),
        "",
        "## Safe External-Validation Plan",
        "",
        "1. Keep OASIS strictly as an external validation/stress-test cohort; do not merge it with ADNI training or threshold selection.",
        "2. Confirm that the DPABI AAL3 ROI order in these `ROISignals` files matches the ADNI preprocessing order and document the 170-to-131 ROI mapping.",
        "3. Apply run-level QC. If one run is bad, exclude only that run; retain the subject/session if at least one QC-ok run remains.",
        "4. Pre-specify the multi-run strategy before computing connectomes: either concatenate QC-ok run time series within subject/session, or compute per-run connectomes and average Fisher-z/static channels.",
        "5. Build exactly one OASIS connectome tensor per subject/session after QC and ROI-order confirmation.",
        "6. Apply the locked ADNI preprocessing/readout externally. Do not tune model hyperparameters, channels, thresholds, or calibration on OASIS.",
        "7. Report TR/scanner/domain shift explicitly. OASIS TR is approximately 2.2 s, while the ADNI model was trained around TR 3 s.",
        "",
        "## Current Blockers",
        "",
    ]
    if recommendation == "needs_roi_order_resolution":
        lines.append("- Blocking: ROI order/mapping to the ADNI AAL3-131 tensor is not explicitly documented in this batch.")
    if n_roi_runs_with_all_nan_cols:
        lines.append("- Blocking: all-NaN ROI columns must be confirmed as outside the ADNI 131-ROI retained set or handled by a pre-specified rule.")
    if n_missing_raw_roi:
        lines.append("- Non-blocking run-level issue: some raw/download-QC BOLD runs lack processed ROI signals; these can be handled by run-level exclusion.")
    if not lines[-1].startswith("-"):
        lines.append("- No major blockers beyond preserving the external-validation-only design.")
    lines.extend(
        [
            "",
            "## Explicit Non-Actions In This Audit",
            "",
            "- No OASIS data were merged with ADNI.",
            "- No time series were concatenated.",
            "- No connectomes were computed.",
            "- No input files were modified.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return recommendation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/Tanda_2026_05_25")
    parser.add_argument(
        "--output-dir",
        default="results/revision_bspc_2026/oasis_tanda_2026_05_25_audit",
    )
    args = parser.parse_args()

    root = Path(args.input_dir).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    command_log: dict[str, Any] = {
        "script": str(Path(__file__).resolve()),
        "argv": sys.argv,
        "timestamp_start": now_iso(),
        "input_dir": str(root),
        "output_dir": str(out),
        "read_only_input": True,
        "connectomes_computed": False,
        "runs_concatenated": False,
        "scipy_available": scipy_io is not None,
    }
    if not root.exists():
        raise FileNotFoundError(f"Input folder not found: {root}")

    metadata_path = root / "oasis3_pilot_30cn_30ad_subjects.csv"
    download_qc_path = root / "download_qc.csv"

    inventory_df = build_inventory(root)
    metadata_df = metadata_rows(metadata_path)
    expected_bold_df = expected_bold_rows(download_qc_path)
    roi_qc_df = roi_signal_qc(root, inventory_df)
    run_df = build_run_manifest(inventory_df, metadata_df, expected_bold_df, roi_qc_df)
    subject_df = build_subject_session_manifest(run_df, metadata_df)
    diagnosis_df = build_diagnosis_counts(subject_df, run_df)
    missing_df = build_missing_ambiguous(root, subject_df, run_df, roi_qc_df)

    write_csv_md(inventory_df, out / "file_inventory.csv", "OASIS Tanda 2026-05-25 File Inventory", max_md_rows=250)
    write_csv_md(run_df, out / "run_manifest.csv", "OASIS Run-Level Manifest", max_md_rows=250)
    write_csv_md(subject_df, out / "subject_session_manifest.csv", "OASIS Subject/Session Manifest", max_md_rows=200)
    write_csv_md(diagnosis_df, out / "diagnosis_counts.csv", "OASIS Diagnosis Counts", max_md_rows=200)
    write_csv_md(roi_qc_df, out / "roi_signal_qc.csv", "OASIS ROI Signal QC", max_md_rows=250)
    write_csv_md(missing_df, out / "missing_or_ambiguous_subjects.csv", "Missing or Ambiguous OASIS Items", max_md_rows=250)

    recommendation = make_final_recommendation(
        out / "final_recommendation.md",
        inventory_df,
        metadata_df,
        expected_bold_df,
        run_df,
        subject_df,
        roi_qc_df,
        missing_df,
    )

    command_log.update(
        {
            "timestamp_end": now_iso(),
            "status": "completed",
            "recommendation": recommendation,
            "n_files": int(len(inventory_df)),
            "n_metadata_subject_sessions": int(len(metadata_df)),
            "n_expected_raw_bold_runs": int(len(expected_bold_df)),
            "n_run_manifest_rows": int(len(run_df)),
            "n_roi_signal_runs": int(len(roi_qc_df)),
            "n_roi_signal_qc_ok_runs_after_order_confirmation": int(
                roi_qc_df["qc_ok_for_connectome_input_after_order_confirmation"].sum()
            )
            if not roi_qc_df.empty
            else 0,
            "n_subject_sessions_with_roi_qc_ok": int((subject_df["roi_qc_ok_runs"] >= 1).sum())
            if not subject_df.empty
            else 0,
            "outputs": [
                "file_inventory.csv",
                "file_inventory.md",
                "run_manifest.csv",
                "run_manifest.md",
                "subject_session_manifest.csv",
                "subject_session_manifest.md",
                "diagnosis_counts.csv",
                "diagnosis_counts.md",
                "roi_signal_qc.csv",
                "roi_signal_qc.md",
                "missing_or_ambiguous_subjects.csv",
                "missing_or_ambiguous_subjects.md",
                "final_recommendation.md",
                "command_log.json",
            ],
        }
    )
    (out / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
    print(json.dumps({k: command_log[k] for k in ["status", "recommendation", "n_files", "n_roi_signal_runs", "n_subject_sessions_with_roi_qc_ok"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
