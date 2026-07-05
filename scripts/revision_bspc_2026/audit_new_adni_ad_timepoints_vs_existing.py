#!/usr/bin/env python3
"""Metadata-only audit of newly downloaded ADNI AD fMRI timepoints.

This script compares a new ADNI AD fMRI image list against the original paper
metadata and currently discovered expanded metadata. It does not train, does not
preprocess images, and does not load tensors/checkpoints/joblibs.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NEW_LIST = PROJECT_ROOT / "data/AD_fMRI_4_28_2026.csv"
DEFAULT_ORIGINAL_METADATA = PROJECT_ROOT / "data/SubjectsData_AAL3_procesado2.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/new_adni_ad_timepoint_overlap_audit"

SUBJECT_COLS = ["Subject", "SubjectID", "PTID", "subject_id", "RID"]
IMAGE_COLS = ["Image Data ID", "ImageID", "image_id", "IMAGEUID", "Image ID", "Image Data ID"]
VISIT_COLS = ["Visit", "VISCODE", "VisitCode", "VISCODE2"]
DATE_COLS = ["Acq Date", "AcquisitionDate", "ScanDate", "EXAMDATE", "StudyDate", "ArchiveDate"]
DX_COLS = ["Group", "ResearchGroup_Mapped", "ResearchGroup", "DX", "Diagnosis"]
AGE_COLS = ["Age", "AGE"]
SEX_COLS = ["Sex", "PTGENDER", "Gender"]
MANUFACTURER_COLS = ["Manufacturer"]
DESCRIPTION_COLS = ["Description", "Series Description"]
MODALITY_COLS = ["Modality"]
TYPE_COLS = ["Type"]
FORMAT_COLS = ["Format"]
TR_COLS = ["TR", "RepetitionTime", "Repetition Time"]

SMALL_NPZ_KEYS = {
    "subject_ids",
    "SubjectID",
    "SubjectIDs",
    "subject_id",
    "subjects",
    "ptids",
    "PTID",
    "image_ids",
    "ImageID",
    "ImageIDs",
    "image_id",
    "IMAGEUID",
    "visits",
    "Visit",
    "VISCODE",
    "acq_dates",
    "AcqDate",
    "StudyDate",
    "ScanDate",
}
LARGE_NPZ_KEY_PATTERNS = ("tensor", "global_tensor", "data", "arr_", "X", "features")


@dataclass
class SourceLoadResult:
    records: pd.DataFrame
    source_row: Dict[str, Any]
    warnings: List[str]
    npz_small_keys_read: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit newly downloaded ADNI AD fMRI rows against existing subject/timepoint metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--new-list", type=Path, default=DEFAULT_NEW_LIST)
    parser.add_argument("--original-metadata", type=Path, default=DEFAULT_ORIGINAL_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-csv-mb", type=float, default=150.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-npz", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def find_new_list(path: Path) -> Path:
    path = resolve_path(path)
    if path.exists():
        return path
    hits = sorted((PROJECT_ROOT / "data").rglob("AD_fMRI_4_28_2026.csv"))
    if not hits:
        raise FileNotFoundError(
            "New ADNI AD fMRI list not found. Expected "
            f"{path} or data/**/AD_fMRI_4_28_2026.csv"
        )
    return hits[0]


def first_existing_col(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower_map = {str(col).strip().lower(): col for col in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        key = candidate.strip().lower()
        if key in lower_map:
            return lower_map[key]
    return None


def csv_header(path: Path) -> List[str]:
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return []


def normalize_subject(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    text = re.sub(r"\s+", "", text)
    match = re.search(r"\d{3}_S_\d+", text)
    return match.group(0) if match else text


def normalize_image_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    text = re.sub(r"\s+", "", text)
    if re.fullmatch(r"I?\d+", text):
        text = text.lstrip("I")
    return text


def normalize_visit(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", "", str(value).strip().lower())


def normalize_date(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return ""
    return parsed.strftime("%Y-%m-%d")


def normalize_scalar(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def join_unique(values: Iterable[Any]) -> str:
    out = []
    seen = set()
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text or text.lower() == "nan":
            continue
        if text not in seen:
            seen.add(text)
            out.append(text)
    return "|".join(out)


def split_set(text: str) -> set[str]:
    if not text:
        return set()
    return {part for part in str(text).split("|") if part}


def safe_numeric_image(image_id: str) -> float:
    if image_id and str(image_id).isdigit():
        return float(image_id)
    return float("inf")


def visit_priority(visit_norm: str) -> int:
    visit = visit_norm.lower().strip()
    if visit in {"sc", "screening"} or "screen" in visit:
        return 0
    if "init" in visit or "initial" in visit:
        return 1
    if visit in {"bl", "baseline"} or "baseline" in visit:
        return 2
    if visit == "v02":
        return 3
    return 10


def recommended_scan_for_subject(sub: pd.DataFrame) -> pd.Series:
    ranked = sub.copy()
    ranked["_visit_rank"] = ranked["Visit_normalized"].map(visit_priority)
    ranked["_date_sort"] = ranked["AcqDate"].replace("", "9999-99-99")
    ranked["_image_sort"] = ranked["ImageID_normalized"].map(safe_numeric_image)
    ranked = ranked.sort_values(["_visit_rank", "_date_sort", "_image_sort", "new_row_index"])
    return ranked.iloc[0]


def read_new_list(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    subject_col = first_existing_col(df.columns, SUBJECT_COLS)
    image_col = first_existing_col(df.columns, IMAGE_COLS)
    visit_col = first_existing_col(df.columns, VISIT_COLS)
    date_col = first_existing_col(df.columns, DATE_COLS)
    if subject_col is None:
        raise ValueError(f"Could not detect subject column in new list. Columns: {list(df.columns)}")
    rows = []
    for idx, row in df.iterrows():
        subject = normalize_subject(row.get(subject_col, ""))
        image_id = normalize_image_id(row.get(image_col, "")) if image_col else ""
        visit_raw = normalize_scalar(row.get(visit_col, "")) if visit_col else ""
        date_norm = normalize_date(row.get(date_col, "")) if date_col else ""
        rows.append(
            {
                "new_row_index": idx,
                "Subject": subject,
                "ImageID_normalized": image_id,
                "Visit": visit_raw,
                "Visit_normalized": normalize_visit(visit_raw),
                "AcqDate": date_norm,
                "Group": normalize_scalar(row.get(first_existing_col(df.columns, DX_COLS) or "", "")),
                "Age": normalize_scalar(row.get(first_existing_col(df.columns, AGE_COLS) or "", "")),
                "Sex": normalize_scalar(row.get(first_existing_col(df.columns, SEX_COLS) or "", "")),
                "Description": normalize_scalar(row.get(first_existing_col(df.columns, DESCRIPTION_COLS) or "", "")),
                "Modality": normalize_scalar(row.get(first_existing_col(df.columns, MODALITY_COLS) or "", "")),
                "Type": normalize_scalar(row.get(first_existing_col(df.columns, TYPE_COLS) or "", "")),
                "Format": normalize_scalar(row.get(first_existing_col(df.columns, FORMAT_COLS) or "", "")),
                "Downloaded": normalize_scalar(row.get("Downloaded", "")),
                "TR_if_available": normalize_scalar(row.get(first_existing_col(df.columns, TR_COLS) or "", "")),
            }
        )
    return pd.DataFrame(rows)


def load_csv_source(path: Path, source_kind: str, max_csv_mb: float) -> SourceLoadResult:
    warnings: List[str] = []
    columns = csv_header(path)
    if not columns:
        return SourceLoadResult(pd.DataFrame(), {}, [f"could_not_read_header: {path}"], [])

    size_mb = path.stat().st_size / (1024 * 1024)
    subject_col = first_existing_col(columns, SUBJECT_COLS)
    image_col = first_existing_col(columns, IMAGE_COLS)
    visit_col = first_existing_col(columns, VISIT_COLS)
    date_col = first_existing_col(columns, DATE_COLS)
    dx_col = first_existing_col(columns, DX_COLS)
    age_col = first_existing_col(columns, AGE_COLS)
    sex_col = first_existing_col(columns, SEX_COLS)
    manufacturer_col = first_existing_col(columns, MANUFACTURER_COLS)

    source_row = {
        "path": str(path),
        "source_kind": source_kind,
        "n_rows": 0,
        "detected_subject_col": subject_col or "",
        "detected_image_col": image_col or "",
        "detected_visit_col": visit_col or "",
        "detected_date_col": date_col or "",
        "detected_dx_col": dx_col or "",
        "notes": "",
    }

    if subject_col is None:
        source_row["notes"] = "skipped_no_subject_column"
        return SourceLoadResult(pd.DataFrame(), source_row, [], [])
    if size_mb > max_csv_mb:
        source_row["notes"] = f"skipped_large_csv_{size_mb:.1f}MB"
        warnings.append(f"Skipped large CSV ({size_mb:.1f} MB): {path}")
        return SourceLoadResult(pd.DataFrame(), source_row, warnings, [])

    usecols = [col for col in [subject_col, image_col, visit_col, date_col, dx_col, age_col, sex_col, manufacturer_col] if col]
    usecols = list(dict.fromkeys(usecols))
    try:
        df = pd.read_csv(path, dtype=str, usecols=usecols, low_memory=False)
    except Exception as exc:
        source_row["notes"] = f"failed_read_csv: {exc}"
        warnings.append(f"Failed to read CSV {path}: {exc}")
        return SourceLoadResult(pd.DataFrame(), source_row, warnings, [])

    records = pd.DataFrame(
        {
            "source_path": str(path),
            "source_kind": source_kind,
            "Subject": df[subject_col].map(normalize_subject),
            "ImageID_normalized": df[image_col].map(normalize_image_id) if image_col else "",
            "Visit_normalized": df[visit_col].map(normalize_visit) if visit_col else "",
            "Visit_raw": df[visit_col].map(normalize_scalar) if visit_col else "",
            "AcqDate": df[date_col].map(normalize_date) if date_col else "",
            "Diagnosis": df[dx_col].map(normalize_scalar) if dx_col else "",
            "Age": df[age_col].map(normalize_scalar) if age_col else "",
            "Sex": df[sex_col].map(normalize_scalar) if sex_col else "",
            "Manufacturer": df[manufacturer_col].map(normalize_scalar) if manufacturer_col else "",
        }
    )
    records = records[records["Subject"] != ""].copy()
    source_row["n_rows"] = int(len(records))
    source_row["notes"] = "used_csv"
    return SourceLoadResult(records, source_row, warnings, [])


def as_flat_str_list(value: Any) -> List[str]:
    arr = np.asarray(value)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    out = []
    for item in arr.tolist():
        if isinstance(item, bytes):
            item = item.decode("utf-8", errors="replace")
        out.append(str(item))
    return out


def load_npz_source(path: Path) -> SourceLoadResult:
    warnings: List[str] = []
    small_keys_read: List[str] = []
    source_row = {
        "path": str(path),
        "source_kind": "npz_metadata",
        "n_rows": 0,
        "detected_subject_col": "",
        "detected_image_col": "",
        "detected_visit_col": "",
        "detected_date_col": "",
        "detected_dx_col": "",
        "notes": "",
    }
    try:
        with np.load(path, allow_pickle=True) as npz:
            keys = list(npz.files)
            subject_key = next((key for key in keys if key in SMALL_NPZ_KEYS and "subject" in key.lower()), None)
            if subject_key is None:
                subject_key = next((key for key in keys if key.lower() in {"ptid", "ptids"}), None)
            if subject_key is None:
                source_row["notes"] = f"skipped_npz_no_subject_metadata_keys; keys={','.join(keys[:20])}"
                return SourceLoadResult(pd.DataFrame(), source_row, [], [])
            if any(pattern == subject_key or pattern in subject_key.lower() for pattern in LARGE_NPZ_KEY_PATTERNS):
                source_row["notes"] = f"skipped_npz_subject_key_looked_large: {subject_key}"
                return SourceLoadResult(pd.DataFrame(), source_row, [], [])
            subjects = as_flat_str_list(npz[subject_key])
            small_keys_read.append(f"{path}:{subject_key}")

            def read_optional_key(candidates: Sequence[str]) -> List[str]:
                key = next((candidate for candidate in candidates if candidate in keys), None)
                if key is None:
                    return []
                small_keys_read.append(f"{path}:{key}")
                return as_flat_str_list(npz[key])

            images = read_optional_key(["image_ids", "ImageID", "ImageIDs", "image_id", "IMAGEUID"])
            visits = read_optional_key(["visits", "Visit", "VISCODE"])
            dates = read_optional_key(["acq_dates", "AcqDate", "StudyDate", "ScanDate"])
    except Exception as exc:
        source_row["notes"] = f"failed_npz_metadata_read: {exc}"
        warnings.append(f"Failed NPZ metadata read {path}: {exc}")
        return SourceLoadResult(pd.DataFrame(), source_row, warnings, small_keys_read)

    n = len(subjects)
    records = pd.DataFrame(
        {
            "source_path": str(path),
            "source_kind": "npz_metadata",
            "Subject": [normalize_subject(value) for value in subjects],
            "ImageID_normalized": [normalize_image_id(images[i]) if i < len(images) else "" for i in range(n)],
            "Visit_normalized": [normalize_visit(visits[i]) if i < len(visits) else "" for i in range(n)],
            "Visit_raw": [normalize_scalar(visits[i]) if i < len(visits) else "" for i in range(n)],
            "AcqDate": [normalize_date(dates[i]) if i < len(dates) else "" for i in range(n)],
            "Diagnosis": "",
            "Age": "",
            "Sex": "",
            "Manufacturer": "",
        }
    )
    records = records[records["Subject"] != ""].copy()
    source_row.update(
        {
            "n_rows": int(len(records)),
            "detected_subject_col": subject_key,
            "detected_image_col": "image_ids_if_present",
            "detected_visit_col": "visits_if_present",
            "detected_date_col": "dates_if_present",
            "notes": "used_npz_small_metadata_keys_only",
        }
    )
    return SourceLoadResult(records, source_row, warnings, small_keys_read)


def unique_paths(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    out = []
    for path in paths:
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def discover_csv_sources(original_metadata: Path, new_list: Path) -> List[Tuple[Path, str]]:
    paths: List[Tuple[Path, str]] = []
    if original_metadata.exists():
        paths.append((original_metadata, "original_paper_metadata"))
    for path in sorted((PROJECT_ROOT / "data/revision_bspc_2026").glob("**/subject_metadata*.csv")):
        paths.append((path, "expanded_metadata"))
    for pattern in ["**/pooled_test_predictions*.csv", "**/subject_level_predictions*.csv"]:
        for path in sorted((PROJECT_ROOT / "results/revision_bspc_2026").glob(pattern)):
            paths.append((path, "prediction_metadata"))
    for base_path, pattern, kind in [
        (Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026"), "**/subject_metadata*.csv", "expanded_or_inference_metadata"),
        (Path("/media/diego/Datos/adni_expansion"), "**/subject_metadata*.csv", "adni_expansion_metadata"),
        (Path("/media/diego/Datos/adni_expansion"), "**/metadata*.csv", "adni_expansion_metadata"),
    ]:
        if base_path.exists():
            for path in sorted(base_path.glob(pattern)):
                paths.append((path, kind))
    deduped = []
    seen = set()
    new_resolved = str(new_list.resolve())
    for path, kind in paths:
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        if key == new_resolved or key in seen:
            continue
        seen.add(key)
        deduped.append((path, kind))
    return deduped


def discover_npz_sources() -> List[Path]:
    candidates: List[Path] = []
    for base_path, patterns in [
        (PROJECT_ROOT / "data/revision_bspc_2026", ["**/GLOBAL_TENSOR*.npz", "**/*global*tensor*.npz"]),
        (PROJECT_ROOT / "data", ["AAL3_dynamicROIs*/GLOBAL_TENSOR*.npz"]),
        (Path("/media/diego/Datos/adni_expansion"), ["**/GLOBAL_TENSOR*.npz", "**/*global*tensor*.npz"]),
        (Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026"), ["**/GLOBAL_TENSOR*.npz", "**/*global*tensor*.npz"]),
    ]:
        if not base_path.exists():
            continue
        for pattern in patterns:
            candidates.extend(sorted(base_path.glob(pattern)))
    return unique_paths(candidates)


def aggregate_subject_records(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame(
            columns=[
                "Subject",
                "rows_count",
                "diagnosis_values",
                "image_ids_if_available",
                "visits_if_available",
                "acq_dates_if_available",
                "sources_where_subject_found",
                "n_sources_found",
            ]
        )
    grouped = []
    for subject, sub in records.groupby("Subject", dropna=False):
        sources = sorted(set(str(value) for value in sub["source_path"].dropna() if str(value)))
        grouped.append(
            {
                "Subject": subject,
                "rows_count": int(len(sub)),
                "diagnosis_values": join_unique(sub["Diagnosis"]),
                "image_ids_if_available": join_unique(sub["ImageID_normalized"]),
                "visits_if_available": join_unique(sub["Visit_normalized"]),
                "acq_dates_if_available": join_unique(sub["AcqDate"]),
                "sources_where_subject_found": "|".join(sources),
                "n_sources_found": len(sources),
            }
        )
    return pd.DataFrame(grouped)


def exact_timepoint_status(new_sub: pd.DataFrame, existing_sub: pd.DataFrame, no_match_status: str) -> str:
    if existing_sub.empty:
        return no_match_status
    new_images = set(value for value in new_sub["ImageID_normalized"] if value)
    old_images = set(value for value in existing_sub["ImageID_normalized"] if value)
    if new_images and old_images and new_images.intersection(old_images):
        return "SAME_IMAGE_ID"
    new_visits = set(value for value in new_sub["Visit_normalized"] if value)
    old_visits = set(value for value in existing_sub["Visit_normalized"] if value)
    if new_visits and old_visits and new_visits.intersection(old_visits):
        return "SAME_VISIT"
    new_dates = set(value for value in new_sub["AcqDate"] if value)
    old_dates = set(value for value in existing_sub["AcqDate"] if value)
    if new_dates and old_dates and new_dates.intersection(old_dates):
        return "SAME_ACQ_DATE"
    if not (old_images or old_visits or old_dates):
        return "AMBIGUOUS_OLD_METADATA_LACKS_TIMEPOINT_FIELDS"
    return "NEW_TIMEPOINT"


def build_new_subject_summary(new_inventory: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject, sub in new_inventory.groupby("Subject"):
        rec = recommended_scan_for_subject(sub)
        age_num = pd.to_numeric(sub["Age"], errors="coerce")
        rows.append(
            {
                "Subject": subject,
                "n_new_images": int(len(sub)),
                "visits_new": join_unique(sub["Visit"]),
                "image_ids_new": join_unique(sub["ImageID_normalized"]),
                "acq_dates_new": join_unique(sub["AcqDate"]),
                "age_min": age_num.min(),
                "age_max": age_num.max(),
                "sex": join_unique(sub["Sex"]),
                "group": join_unique(sub["Group"]),
                "has_multiple_timepoints_in_new_list": bool(len(sub) > 1),
                "recommended_single_scan_rule": "prefer sc, init, bl, v02; then earliest acquisition date; then lowest ImageID",
                "recommended_image_id_to_process": rec["ImageID_normalized"],
                "recommended_visit": rec["Visit"],
                "recommended_acq_date": rec["AcqDate"],
                "reason_for_recommendation": f"visit_rank={visit_priority(rec['Visit_normalized'])}; deterministic one-scan-per-subject rule",
            }
        )
    return pd.DataFrame(rows).sort_values("Subject")


def build_overlap_table(
    new_inventory: pd.DataFrame,
    existing_records: pd.DataFrame,
    prefix: str,
    no_match_status: str,
) -> pd.DataFrame:
    existing_summary = aggregate_subject_records(existing_records)
    rows = []
    for subject, new_sub in new_inventory.groupby("Subject"):
        existing_sub = existing_records[existing_records["Subject"] == subject] if not existing_records.empty else pd.DataFrame()
        summary_row = existing_summary[existing_summary["Subject"] == subject]
        found = not existing_sub.empty
        status = exact_timepoint_status(new_sub, existing_sub, no_match_status)
        rows.append(
            {
                "Subject": subject,
                f"in_{prefix}_by_subject": bool(found),
                f"{prefix}_rows_count": int(len(existing_sub)),
                f"{prefix}_diagnosis_values": summary_row["diagnosis_values"].iloc[0] if not summary_row.empty else "",
                f"{prefix}_image_ids_if_available": summary_row["image_ids_if_available"].iloc[0] if not summary_row.empty else "",
                f"{prefix}_visits_if_available": summary_row["visits_if_available"].iloc[0] if not summary_row.empty else "",
                f"{prefix}_acq_dates_if_available": summary_row["acq_dates_if_available"].iloc[0] if not summary_row.empty else "",
                f"{prefix}_sources_where_subject_found": summary_row["sources_where_subject_found"].iloc[0] if not summary_row.empty else "",
                "status_subject_level": "EXISTING_SUBJECT_IN_ORIGINAL" if found and prefix == "original" else (
                    "NEW_SUBJECT_RELATIVE_TO_ORIGINAL" if prefix == "original" else ""
                ),
                f"exact_timepoint_status_vs_{prefix}": status,
            }
        )
    return pd.DataFrame(rows).sort_values("Subject")


def build_global_overlap(
    new_inventory: pd.DataFrame,
    all_records: pd.DataFrame,
    original_subjects: set[str],
    expanded_subjects: set[str],
) -> pd.DataFrame:
    summary = aggregate_subject_records(all_records)
    rows = []
    for subject, new_sub in new_inventory.groupby("Subject"):
        existing_sub = all_records[all_records["Subject"] == subject] if not all_records.empty else pd.DataFrame()
        summary_row = summary[summary["Subject"] == subject]
        exact_status = exact_timepoint_status(new_sub, existing_sub, "NO_EXISTING_SUBJECT_MATCH")
        if existing_sub.empty:
            status_global = "COMPLETELY_NEW_SUBJECT"
        elif exact_status == "NEW_TIMEPOINT":
            status_global = "EXISTING_SUBJECT_NEW_TIMEPOINT"
        else:
            status_global = "EXISTING_SUBJECT_SAME_OR_AMBIGUOUS_TIMEPOINT"
        rows.append(
            {
                "Subject": subject,
                "in_all_existing_by_subject": bool(not existing_sub.empty),
                "already_in_original": subject in original_subjects,
                "already_in_expanded": subject in expanded_subjects,
                "all_existing_rows_count": int(len(existing_sub)),
                "all_existing_diagnosis_values": summary_row["diagnosis_values"].iloc[0] if not summary_row.empty else "",
                "all_existing_image_ids_if_available": summary_row["image_ids_if_available"].iloc[0] if not summary_row.empty else "",
                "all_existing_visits_if_available": summary_row["visits_if_available"].iloc[0] if not summary_row.empty else "",
                "all_existing_acq_dates_if_available": summary_row["acq_dates_if_available"].iloc[0] if not summary_row.empty else "",
                "sources_where_subject_found": summary_row["sources_where_subject_found"].iloc[0] if not summary_row.empty else "",
                "n_sources_found": int(summary_row["n_sources_found"].iloc[0]) if not summary_row.empty else 0,
                "exact_timepoint_status_vs_all_existing": exact_status,
                "status_global": status_global,
            }
        )
    return pd.DataFrame(rows).sort_values("Subject")


def subject_rows_with_overlap(
    new_inventory: pd.DataFrame,
    global_overlap: pd.DataFrame,
    statuses: Sequence[str],
) -> pd.DataFrame:
    subjects = set(global_overlap.loc[global_overlap["status_global"].isin(statuses), "Subject"])
    merged = new_inventory[new_inventory["Subject"].isin(subjects)].merge(global_overlap, on="Subject", how="left")
    return merged.sort_values(["Subject", "AcqDate", "ImageID_normalized"])


def build_longitudinal_report(
    new_summary: pd.DataFrame,
    global_overlap: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    overlap_by_subject = global_overlap.set_index("Subject").to_dict(orient="index")
    for _, row in new_summary.iterrows():
        if not bool(row["has_multiple_timepoints_in_new_list"]):
            continue
        subject = row["Subject"]
        overlap = overlap_by_subject.get(subject, {})
        rows.append(
            {
                "Subject": subject,
                "n_new_images": row["n_new_images"],
                "already_in_original": bool(overlap.get("already_in_original", False)),
                "already_in_expanded": bool(overlap.get("already_in_expanded", False)),
                "leakage_risk_if_treated_as_independent": "HIGH",
                "recommended_handling": "select one scan per subject or use GroupKFold/LeaveOneGroupOut by Subject if longitudinal modelling is intended",
            }
        )
    return pd.DataFrame(rows)


def expanded_dataset_counts(new_subjects: set[str], records: pd.DataFrame) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if records.empty:
        return counts
    for version in ["v1", "v2", "v3", "v4"]:
        sub = records[records["source_path"].str.contains(f"expanded_{version}|adni_expanded_{version}", case=False, regex=True, na=False)]
        counts[version] = len(new_subjects.intersection(set(sub["Subject"])))
    counts["any_expanded"] = len(new_subjects.intersection(set(records[records["source_kind"].str.contains("expanded", na=False)]["Subject"])))
    return counts


def write_readme(
    path: Path,
    new_inventory: pd.DataFrame,
    new_summary: pd.DataFrame,
    original_overlap: pd.DataFrame,
    global_overlap: pd.DataFrame,
    expanded_counts: Dict[str, int],
    warnings: Sequence[str],
) -> None:
    n_rows = len(new_inventory)
    n_subjects = new_inventory["Subject"].nunique()
    n_multi = int(new_summary["has_multiple_timepoints_in_new_list"].sum())
    n_original = int(original_overlap["in_original_by_subject"].sum())
    n_new = int((global_overlap["status_global"] == "COMPLETELY_NEW_SUBJECT").sum())
    n_existing_new_tp = int((global_overlap["status_global"] == "EXISTING_SUBJECT_NEW_TIMEPOINT").sum())
    exact_counts = global_overlap["exact_timepoint_status_vs_all_existing"].value_counts(dropna=False).to_dict()
    n_same_image = int(exact_counts.get("SAME_IMAGE_ID", 0))
    n_same_visit = int(exact_counts.get("SAME_VISIT", 0))
    n_same_date = int(exact_counts.get("SAME_ACQ_DATE", 0))
    n_ambiguous = int(
        sum(int(count) for status, count in exact_counts.items() if str(status).startswith("AMBIGUOUS"))
    )
    n_same_or_ambiguous = int((global_overlap["status_global"] == "EXISTING_SUBJECT_SAME_OR_AMBIGUOUS_TIMEPOINT").sum())
    additional_rows = int(n_rows - n_subjects)
    tr_values = join_unique(new_inventory["TR_if_available"])
    tr_note = (
        f"TR values present in the new list: `{tr_values}`."
        if tr_values
        else "The new CSV does not include TR or imaging protocol columns, so TR=3 cannot be independently verified from this file."
    )

    lines = [
        "# New ADNI AD fMRI Timepoint Overlap Audit",
        "",
        "Metadata-only audit. No training, preprocessing, checkpoints, joblibs, or tensor arrays were loaded.",
        "",
        "## Main Counts",
        f"- Rows/images in new AD list: `{n_rows}`",
        f"- Unique subjects in new AD list: `{n_subjects}`",
        f"- Subjects with multiple timepoints/images in new list: `{n_multi}`",
        f"- Additional image rows beyond one per subject: `{additional_rows}`",
        f"- New-list subjects already in original paper metadata by subject: `{n_original}`",
        f"- New-list subjects already in expanded metadata v1: `{expanded_counts.get('v1', 0)}`",
        f"- New-list subjects already in expanded metadata v2: `{expanded_counts.get('v2', 0)}`",
        f"- New-list subjects already in expanded metadata v3: `{expanded_counts.get('v3', 0)}`",
        f"- New-list subjects already in expanded metadata v4: `{expanded_counts.get('v4', 0)}`",
        f"- New-list subjects already in any expanded metadata: `{expanded_counts.get('any_expanded', 0)}`",
        f"- Completely new subjects across discovered sources: `{n_new}`",
        f"- Existing subjects with evidence of new timepoints: `{n_existing_new_tp}`",
        f"- Existing subjects matching a previous ImageID: `{n_same_image}`",
        f"- Existing subjects matching a previous Visit only: `{n_same_visit}`",
        f"- Existing subjects matching a previous Acq Date only: `{n_same_date}`",
        f"- Existing subjects with ambiguous exact timepoint status: `{n_ambiguous}`",
        f"- Existing subjects with same or ambiguous timepoint status: `{n_same_or_ambiguous}`",
        "",
        "## TR Metadata",
        f"- {tr_note}",
        "- Martin's TR=3 statement is therefore treated as external acquisition knowledge unless a separate ADNI protocol export is provided.",
        "",
        "## Interpretation",
        "- Subject-level overlap uses normalized ADNI PTID strings such as `002_S_1261`.",
        "- Image-level overlap normalizes `I123456` and `123456` as the same ImageID.",
        "- Visit/date matching is used only when ImageID is unavailable or non-overlapping.",
        "- A subject already present in the original or expanded datasets must not be counted as an additional independent patient just because a new scan/timepoint was downloaded.",
        "- Adding multiple timepoints requires subject-grouped splitting, e.g. GroupKFold or LeaveOneGroupOut by Subject/PTID, to avoid longitudinal leakage.",
        "",
        "## One-Scan Recommendation Rule",
        "- For one scan per subject, prefer visits in this order: `sc`, `init`, `bl`, `v02`; then earliest acquisition date; then lowest normalized ImageID as deterministic tie-break.",
        "- Recommended ImageIDs are listed in `new_ad_subject_summary.csv`.",
        "",
        "## Caveats",
        "- Some expanded metadata files preserve subject-level metadata only and lack ImageID/Visit/Acq Date, so exact timepoint status can be ambiguous.",
        "- `EXISTING_SUBJECT_NEW_TIMEPOINT` means the subject existed before and available ImageID/Visit/Date evidence did not match discovered old records.",
        "- `COMPLETELY_NEW_SUBJECT` means absent from original paper metadata and all discovered expanded/inference/adni_expansion metadata plus small NPZ subject metadata.",
    ]
    if warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in warnings)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def git_hash() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            text=True,
            capture_output=True,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else ""
    except Exception:
        return ""


def main() -> int:
    args = parse_args()
    new_list = find_new_list(args.new_list)
    original_metadata = resolve_path(args.original_metadata)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    warnings: List[str] = []
    small_npz_keys_read: List[str] = []

    new_inventory = read_new_list(new_list)
    new_subjects = set(new_inventory["Subject"])

    source_rows: List[Dict[str, Any]] = []
    csv_records: List[pd.DataFrame] = []
    original_records = pd.DataFrame()

    for path, kind in discover_csv_sources(original_metadata, new_list):
        result = load_csv_source(path, kind, args.max_csv_mb)
        warnings.extend(result.warnings)
        if result.source_row:
            source_rows.append(result.source_row)
        if not result.records.empty:
            csv_records.append(result.records)
            if kind == "original_paper_metadata" and path.resolve() == original_metadata.resolve():
                original_records = result.records.copy()

    npz_records: List[pd.DataFrame] = []
    if not args.skip_npz:
        for path in discover_npz_sources():
            result = load_npz_source(path)
            warnings.extend(result.warnings)
            small_npz_keys_read.extend(result.npz_small_keys_read)
            if result.source_row and result.source_row.get("notes", "").startswith("used"):
                source_rows.append(result.source_row)
            if not result.records.empty:
                npz_records.append(result.records)

    all_records = pd.concat(csv_records + npz_records, ignore_index=True) if (csv_records or npz_records) else pd.DataFrame()
    if all_records.empty:
        all_records = pd.DataFrame(
            columns=[
                "source_path",
                "source_kind",
                "Subject",
                "ImageID_normalized",
                "Visit_normalized",
                "Visit_raw",
                "AcqDate",
                "Diagnosis",
                "Age",
                "Sex",
                "Manufacturer",
            ]
        )

    expanded_records = all_records[all_records["source_kind"].str.contains("expanded", na=False)].copy()
    original_subjects = set(original_records["Subject"]) if not original_records.empty else set()
    expanded_subjects = set(expanded_records["Subject"]) if not expanded_records.empty else set()

    new_summary = build_new_subject_summary(new_inventory)
    original_overlap = build_overlap_table(new_inventory, original_records, "original", "NO_ORIGINAL_SUBJECT_MATCH")
    global_overlap = build_global_overlap(new_inventory, all_records, original_subjects, expanded_subjects)
    expanded_counts = expanded_dataset_counts(new_subjects, all_records)

    completely_new = subject_rows_with_overlap(new_inventory, global_overlap, ["COMPLETELY_NEW_SUBJECT"])
    existing_new_tp = subject_rows_with_overlap(new_inventory, global_overlap, ["EXISTING_SUBJECT_NEW_TIMEPOINT"])
    existing_ambig = subject_rows_with_overlap(new_inventory, global_overlap, ["EXISTING_SUBJECT_SAME_OR_AMBIGUOUS_TIMEPOINT"])
    leakage_report = build_longitudinal_report(new_summary, global_overlap)

    new_inventory.drop(columns=["Visit_normalized", "TR_if_available"], errors="ignore").to_csv(
        output_dir / "new_ad_list_inventory.csv", index=False
    )
    new_summary.to_csv(output_dir / "new_ad_subject_summary.csv", index=False)
    original_overlap.to_csv(output_dir / "overlap_vs_original_subject_level.csv", index=False)
    global_overlap.to_csv(output_dir / "overlap_vs_all_existing_subject_level.csv", index=False)
    completely_new.to_csv(output_dir / "completely_new_subjects.csv", index=False)
    existing_new_tp.to_csv(output_dir / "existing_subjects_new_timepoints.csv", index=False)
    existing_ambig.to_csv(output_dir / "already_existing_or_ambiguous_subjects.csv", index=False)
    leakage_report.to_csv(output_dir / "longitudinal_leakage_risk_report.csv", index=False)
    pd.DataFrame(source_rows).to_csv(output_dir / "metadata_sources_used.csv", index=False)

    write_readme(
        output_dir / "README.md",
        new_inventory,
        new_summary,
        original_overlap,
        global_overlap,
        expanded_counts,
        warnings,
    )

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": git_hash(),
        "input_paths": {
            "new_ad_list": str(new_list),
            "original_metadata": str(original_metadata),
        },
        "output_path": str(output_dir),
        "matching_rules": {
            "subject": "normalized ADNI PTID/SubjectID, e.g. ###_S_####",
            "image": "strip leading I and compare numeric string",
            "visit": "lowercase and strip whitespace",
            "date": "parse to yyyy-mm-dd where possible",
            "timepoint_priority": "ImageID, then Visit, then Acq Date",
        },
        "number_of_discovered_metadata_sources_used": int(len(source_rows)),
        "small_npz_metadata_keys_read": small_npz_keys_read,
        "no_training": True,
        "no_preprocessing": True,
        "no_checkpoint_loading": True,
        "no_joblib_loading": True,
        "no_tensor_loading": True,
        "no_tensor_loading_note": "Only small NPZ metadata keys such as subject_ids/image_ids were read when present; large tensor arrays were skipped.",
        "warnings": warnings,
    }
    (output_dir / "audit_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print((output_dir / "README.md").read_text(encoding="utf-8"))
    print(f"Outputs written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
