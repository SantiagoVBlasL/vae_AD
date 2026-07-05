#!/usr/bin/env python3
"""Metadata-first OASIS-3 feasibility audit for ADNI fMRI external validation.

The script intentionally avoids training, image preprocessing, image download,
and large-array loading. It discovers local CSV/TSV/JSON metadata, inventories
candidate resting-state BOLD sessions, maps provisional cognitive labels when
clear, and selects a small smoke-test candidate set if enough metadata exists.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/oasis3_feasibility_audit"

DEFAULT_SEARCH_ROOTS = [
    PROJECT_ROOT / "data/OASIS3",
    PROJECT_ROOT / "data/oasis3",
    PROJECT_ROOT / "media/diego/Datos/OASIS3",
    PROJECT_ROOT / "media/diego/Datos/oasis3",
    Path("/media/diego/Datos/OASIS3"),
    Path("/media/diego/Datos/oasis3"),
]

SUBJECT_COLS = [
    "subject_id",
    "subject",
    "participant_id",
    "participants_id",
    "Subject",
    "SubjectID",
    "participant",
    "OASISID",
    "OASIS_ID",
    "ID",
]
SESSION_COLS = ["session_id", "session", "ses", "Session", "Visit", "visit", "MR ID", "MR_ID", "label"]
DX_COLS = [
    "diagnosis",
    "dx",
    "clinical_diagnosis",
    "cognitive_status",
    "dementia_status",
    "label",
    "Group",
    "cdr",
    "CDR",
]
CDR_COLS = ["CDR", "cdr", "cdr_global", "CDRGlobal", "cdrglob", "ClinicalDementiaRating"]
AGE_COLS = ["age", "Age", "AGE", "age_at_scan", "AgeAtScan", "age_at_visit", "AgeAtVisit"]
SEX_COLS = ["sex", "Sex", "gender", "Gender", "M/F", "PTGENDER"]
MODALITY_COLS = ["modality", "Modality", "scan_modality", "image_modality", "type", "Type"]
TASK_COLS = ["task", "TaskName", "task_name", "BIDS task", "BIDSTask"]
DESCRIPTION_COLS = ["description", "Description", "series_description", "SeriesDescription", "scan_description"]
PATH_COLS = ["path", "Path", "filepath", "FilePath", "file", "File", "filename", "Filename", "URI"]
TR_COLS = ["RepetitionTime", "TR", "tr", "repetition_time", "Repetition Time"]
NVOL_COLS = ["n_volumes", "NumVolumes", "NumberOfVolumes", "volumes", "nvols", "dim4"]
MANUFACTURER_COLS = ["Manufacturer", "manufacturer", "ScannerManufacturer", "scanner_manufacturer"]
MODEL_COLS = ["ManufacturersModelName", "ScannerModel", "scanner_model", "ModelName", "model"]
DATE_COLS = ["acq_date", "Acq Date", "AcquisitionDate", "ScanDate", "scan_date", "date", "Date"]
DAYS_COLS = ["days_from_entry", "days_from_baseline", "DaysFromEntry", "days"]

OUTPUT_COLUMNS_SOURCES = [
    "path",
    "file_type",
    "n_rows",
    "detected_columns",
    "notes",
]

OUTPUT_COLUMNS_SESSION = [
    "subject_id",
    "session_id",
    "diagnosis_raw",
    "cdr",
    "dementia_status",
    "clinical_label_raw",
    "provisional_label_CN_AD_MCI_unknown",
    "label_rule_used",
    "age",
    "sex",
    "scan_modality",
    "bids_task",
    "has_task_rest_bold",
    "bold_file_path",
    "json_path",
    "TR_RepetitionTime",
    "n_volumes",
    "scanner_manufacturer",
    "scanner_model",
    "acquisition_date",
    "days_from_entry",
    "metadata_source",
    "session_rank_per_subject",
    "eligibility_status",
    "exclusion_reason",
]

OUTPUT_COLUMNS_SUBJECT = [
    "subject_id",
    "n_sessions",
    "n_task_rest_sessions",
    "candidate_baseline_session",
    "candidate_baseline_reason",
    "clinical_label_available",
    "provisional_label_CN_AD_MCI_unknown",
    "age_baseline",
    "sex",
    "scanner_manufacturer_summary",
    "scanner_model_summary",
    "eligible_for_smoke_test",
]

OUTPUT_COLUMNS_SMOKE = [
    "subject_id",
    "session_id",
    "provisional_label_CN_AD_MCI_unknown",
    "age",
    "sex",
    "TR_RepetitionTime",
    "scanner_manufacturer",
    "scanner_model",
    "json_path",
    "bold_file_path",
    "selection_reason",
]


@dataclass
class TableMetadata:
    path: Path
    file_type: str
    n_rows: int
    detected_columns: Dict[str, str]
    notes: str
    frame: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Metadata-first OASIS-3 feasibility audit for ADNI fMRI external validation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--oasis-root", type=Path, default=None)
    parser.add_argument("--clinical-csv", type=Path, default=None)
    parser.add_argument("--sessions-csv", type=Path, default=None)
    parser.add_argument("--bids-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    return path if path.is_absolute() else PROJECT_ROOT / path


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


def first_col(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    exact = {str(col): str(col) for col in columns}
    lower = {str(col).strip().lower(): str(col) for col in columns}
    for candidate in candidates:
        if candidate in exact:
            return exact[candidate]
        key = candidate.strip().lower()
        if key in lower:
            return lower[key]
    return None


def safe_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_subject(value: Any) -> str:
    text = safe_text(value)
    if not text:
        return ""
    text = text.replace("_", "-") if text.startswith("OAS") and "_" in text else text
    text = re.sub(r"\s+", "", text)
    if text.startswith("sub-"):
        return text
    if re.match(r"^OAS\d+", text, flags=re.I):
        return "sub-" + text.upper()
    return text


def normalize_session(value: Any) -> str:
    text = safe_text(value)
    if not text:
        return ""
    text = re.sub(r"\s+", "", text)
    if text.startswith("ses-"):
        return text
    if re.match(r"^\d+$", text):
        return f"ses-{int(text):04d}"
    return text


def parse_float(value: Any) -> Optional[float]:
    text = safe_text(value)
    if not text:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def normalize_date(value: Any) -> str:
    text = safe_text(value)
    if not text:
        return ""
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return ""
    return parsed.strftime("%Y-%m-%d")


def join_unique(values: Iterable[Any]) -> str:
    seen = set()
    out = []
    for value in values:
        text = safe_text(value)
        if not text:
            continue
        if text not in seen:
            seen.add(text)
            out.append(text)
    return "|".join(out)


def classify_metadata_file(path: Path) -> str:
    name = path.name.lower()
    text = str(path).lower()
    if name == "participants.tsv":
        return "bids_participants_tsv"
    if name.endswith("_sessions.tsv") or name == "sessions.tsv":
        return "bids_sessions_tsv"
    if name.endswith(".json"):
        if "task-rest" in text and "_bold" in text:
            return "bids_task_rest_bold_json"
        return "json_metadata"
    if any(token in name for token in ["clinical", "cogn", "cdr", "diagnosis", "demograph", "demo"]):
        return "clinical_or_cognitive_table"
    if any(token in name for token in ["session", "scan", "xnat", "manifest", "mr"]):
        return "session_or_xnat_table"
    if name.endswith(".tsv"):
        return "tsv_metadata"
    return "csv_metadata"


def iter_metadata_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    if root.is_file() and root.suffix.lower() in {".csv", ".tsv", ".json"}:
        return [root]
    files: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in {".git", "__pycache__", ".datalad", ".ipynb_checkpoints"}
        ]
        for filename in filenames:
            suffix = Path(filename).suffix.lower()
            if suffix in {".csv", ".tsv", ".json"}:
                files.append(Path(dirpath) / filename)
    return files


def discover_metadata_files(args: argparse.Namespace) -> Tuple[List[Path], List[Path]]:
    explicit = [resolve_path(args.clinical_csv), resolve_path(args.sessions_csv)]
    explicit = [path for path in explicit if path is not None]
    roots = []
    if args.oasis_root is not None:
        roots.append(resolve_path(args.oasis_root))
    if args.bids_root is not None:
        roots.append(resolve_path(args.bids_root))
    roots.extend(DEFAULT_SEARCH_ROOTS)
    roots = [path for path in roots if path is not None]

    files = []
    for path in explicit:
        if path.exists():
            files.append(path)
    for root in roots:
        if root.exists():
            files.extend(iter_metadata_files(root))
    return unique_paths(files), unique_paths([path for path in roots if path.exists()])


def detect_columns(columns: Sequence[str]) -> Dict[str, str]:
    return {
        "subject": first_col(columns, SUBJECT_COLS) or "",
        "session": first_col(columns, SESSION_COLS) or "",
        "diagnosis": first_col(columns, DX_COLS) or "",
        "cdr": first_col(columns, CDR_COLS) or "",
        "age": first_col(columns, AGE_COLS) or "",
        "sex": first_col(columns, SEX_COLS) or "",
        "modality": first_col(columns, MODALITY_COLS) or "",
        "task": first_col(columns, TASK_COLS) or "",
        "description": first_col(columns, DESCRIPTION_COLS) or "",
        "path": first_col(columns, PATH_COLS) or "",
        "tr": first_col(columns, TR_COLS) or "",
        "n_volumes": first_col(columns, NVOL_COLS) or "",
        "manufacturer": first_col(columns, MANUFACTURER_COLS) or "",
        "model": first_col(columns, MODEL_COLS) or "",
        "date": first_col(columns, DATE_COLS) or "",
        "days": first_col(columns, DAYS_COLS) or "",
    }


def read_table(path: Path) -> TableMetadata:
    suffix = path.suffix.lower()
    sep = "\t" if suffix == ".tsv" else ","
    file_type = classify_metadata_file(path)
    try:
        df = pd.read_csv(path, sep=sep, dtype=str, low_memory=False)
    except Exception as exc:
        return TableMetadata(
            path=path,
            file_type=file_type,
            n_rows=0,
            detected_columns={},
            notes=f"failed_read: {exc}",
            frame=pd.DataFrame(),
        )
    detected = detect_columns(df.columns)
    notes = "used_table"
    if not detected.get("subject"):
        notes = "used_table_no_subject_column"
    return TableMetadata(path=path, file_type=file_type, n_rows=len(df), detected_columns=detected, notes=notes, frame=df)


def parse_bids_ids_from_path(path: Path) -> Tuple[str, str, str]:
    subject = ""
    session = ""
    task = ""
    for part in path.parts:
        if part.startswith("sub-"):
            subject = part
        elif part.startswith("ses-"):
            session = part
    match = re.search(r"task-([A-Za-z0-9]+)", path.name)
    if match:
        task = match.group(1)
    return subject, session, task


def read_json_sidecar(path: Path) -> Tuple[Dict[str, Any], str]:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace")), "used_json"
    except Exception as exc:
        return {}, f"failed_json_read: {exc}"


def is_rest_like_text(*values: Any) -> bool:
    text = " ".join(safe_text(value).lower() for value in values if safe_text(value))
    return bool(re.search(r"\brest\b|task-rest|rsfmri|resting|bold", text))


def provisional_label(row: Dict[str, Any]) -> Tuple[str, str]:
    cdr_val = parse_float(row.get("cdr", ""))
    text_parts = [
        row.get("diagnosis_raw", ""),
        row.get("dementia_status", ""),
        row.get("clinical_label_raw", ""),
    ]
    text = " ".join(safe_text(value).lower() for value in text_parts if safe_text(value))

    if cdr_val is not None:
        if cdr_val == 0:
            return "CN", "CDR==0"
        if cdr_val == 0.5:
            return "MCI", "CDR==0.5"
        if cdr_val >= 1:
            return "AD", "CDR>=1 interpreted as dementia/AD-compatible"

    if re.search(r"\bmci\b|mild cognitive impairment", text):
        return "MCI", "explicit MCI text"
    if re.search(r"alzheimer|ad dementia|dementia due to ad|\bad\b", text):
        return "AD", "explicit AD/dementia text"
    if re.search(r"\bdementia\b", text) and not re.search(r"no dementia|non[- ]?demented|without dementia", text):
        return "AD", "explicit dementia text"
    if re.search(r"cognitively normal|normal control|\bnormal\b|no dementia|non[- ]?demented|cognitively intact|healthy", text):
        return "CN", "explicit normal/no dementia text"
    return "unknown", "no clear CN/AD/MCI rule matched"


def table_rows_to_sessions(table: TableMetadata) -> pd.DataFrame:
    df = table.frame
    detected = table.detected_columns
    if df.empty or not detected.get("subject"):
        return pd.DataFrame(columns=OUTPUT_COLUMNS_SESSION)

    rows = []
    for _, rec in df.iterrows():
        subject = normalize_subject(rec.get(detected["subject"], ""))
        if not subject:
            continue
        session = normalize_session(rec.get(detected.get("session", ""), "")) if detected.get("session") else ""
        diagnosis = safe_text(rec.get(detected.get("diagnosis", ""), "")) if detected.get("diagnosis") else ""
        cdr = safe_text(rec.get(detected.get("cdr", ""), "")) if detected.get("cdr") else ""
        age = safe_text(rec.get(detected.get("age", ""), "")) if detected.get("age") else ""
        sex = safe_text(rec.get(detected.get("sex", ""), "")) if detected.get("sex") else ""
        modality = safe_text(rec.get(detected.get("modality", ""), "")) if detected.get("modality") else ""
        task = safe_text(rec.get(detected.get("task", ""), "")) if detected.get("task") else ""
        description = safe_text(rec.get(detected.get("description", ""), "")) if detected.get("description") else ""
        path_value = safe_text(rec.get(detected.get("path", ""), "")) if detected.get("path") else ""
        tr = safe_text(rec.get(detected.get("tr", ""), "")) if detected.get("tr") else ""
        n_volumes = safe_text(rec.get(detected.get("n_volumes", ""), "")) if detected.get("n_volumes") else ""
        manufacturer = safe_text(rec.get(detected.get("manufacturer", ""), "")) if detected.get("manufacturer") else ""
        model = safe_text(rec.get(detected.get("model", ""), "")) if detected.get("model") else ""
        acq_date = normalize_date(rec.get(detected.get("date", ""), "")) if detected.get("date") else ""
        days = safe_text(rec.get(detected.get("days", ""), "")) if detected.get("days") else ""
        rest_like = is_rest_like_text(task, description, path_value, modality)
        has_session_evidence = bool(session or rest_like or modality or path_value or table.file_type in {"bids_sessions_tsv", "session_or_xnat_table"})
        if not has_session_evidence:
            continue
        row = {
            "subject_id": subject,
            "session_id": session,
            "diagnosis_raw": diagnosis,
            "cdr": cdr,
            "dementia_status": diagnosis,
            "clinical_label_raw": diagnosis,
            "age": age,
            "sex": sex,
            "scan_modality": modality,
            "bids_task": task,
            "has_task_rest_bold": bool(rest_like),
            "bold_file_path": path_value if rest_like and path_value else "",
            "json_path": path_value if path_value.endswith(".json") else "",
            "TR_RepetitionTime": tr,
            "n_volumes": n_volumes,
            "scanner_manufacturer": manufacturer,
            "scanner_model": model,
            "acquisition_date": acq_date,
            "days_from_entry": days,
            "metadata_source": str(table.path),
        }
        label, rule = provisional_label(row)
        row["provisional_label_CN_AD_MCI_unknown"] = label
        row["label_rule_used"] = rule
        rows.append(row)
    return pd.DataFrame(rows, columns=[col for col in OUTPUT_COLUMNS_SESSION if col not in {"session_rank_per_subject", "eligibility_status", "exclusion_reason"}])


def json_to_session_row(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    metadata, note = read_json_sidecar(path)
    subject, session, task_from_path = parse_bids_ids_from_path(path)
    task = safe_text(metadata.get("TaskName", "")) or task_from_path
    is_bold_json = path.name.endswith("_bold.json") or "_bold" in path.name
    is_rest = is_rest_like_text(task, path.name, path)
    nii_gz = path.with_suffix("")
    if nii_gz.name.endswith(".nii"):
        bold_path = str(nii_gz)
    else:
        bold_path = str(path).replace(".json", ".nii.gz") if is_bold_json else ""
    row = {
        "subject_id": normalize_subject(subject),
        "session_id": normalize_session(session),
        "diagnosis_raw": "",
        "cdr": "",
        "dementia_status": "",
        "clinical_label_raw": "",
        "age": "",
        "sex": "",
        "scan_modality": "BOLD" if is_bold_json else "",
        "bids_task": task,
        "has_task_rest_bold": bool(is_bold_json and is_rest),
        "bold_file_path": bold_path if is_bold_json else "",
        "json_path": str(path),
        "TR_RepetitionTime": safe_text(metadata.get("RepetitionTime", "")),
        "n_volumes": safe_text(metadata.get("NumberOfVolumes", metadata.get("NumVolumes", ""))),
        "scanner_manufacturer": safe_text(metadata.get("Manufacturer", "")),
        "scanner_model": safe_text(metadata.get("ManufacturersModelName", "")),
        "acquisition_date": normalize_date(metadata.get("AcquisitionDate", "")),
        "days_from_entry": "",
        "metadata_source": str(path),
    }
    label, rule = provisional_label(row)
    row["provisional_label_CN_AD_MCI_unknown"] = label
    row["label_rule_used"] = rule
    source = {
        "path": str(path),
        "file_type": classify_metadata_file(path),
        "n_rows": 1,
        "detected_columns": "|".join(sorted(metadata.keys())[:40]),
        "notes": note,
    }
    return row, source


def subject_info_from_tables(tables: Sequence[TableMetadata]) -> pd.DataFrame:
    rows = []
    for table in tables:
        df = table.frame
        det = table.detected_columns
        if df.empty or not det.get("subject"):
            continue
        for _, rec in df.iterrows():
            subject = normalize_subject(rec.get(det["subject"], ""))
            if not subject:
                continue
            row = {
                "subject_id": subject,
                "age": safe_text(rec.get(det.get("age", ""), "")) if det.get("age") else "",
                "sex": safe_text(rec.get(det.get("sex", ""), "")) if det.get("sex") else "",
                "diagnosis_raw": safe_text(rec.get(det.get("diagnosis", ""), "")) if det.get("diagnosis") else "",
                "cdr": safe_text(rec.get(det.get("cdr", ""), "")) if det.get("cdr") else "",
                "dementia_status": safe_text(rec.get(det.get("diagnosis", ""), "")) if det.get("diagnosis") else "",
                "clinical_label_raw": safe_text(rec.get(det.get("diagnosis", ""), "")) if det.get("diagnosis") else "",
                "source": str(table.path),
            }
            label, rule = provisional_label(row)
            row["provisional_label_CN_AD_MCI_unknown"] = label
            row["label_rule_used"] = rule
            rows.append(row)
    return pd.DataFrame(rows)


def enrich_sessions_with_subject_info(sessions: pd.DataFrame, subject_info: pd.DataFrame) -> pd.DataFrame:
    if sessions.empty:
        return sessions
    if subject_info.empty:
        return sessions
    best_rows = []
    for subject, sub in subject_info.groupby("subject_id"):
        ranked = sub.copy()
        ranked["_label_rank"] = ranked["provisional_label_CN_AD_MCI_unknown"].map(lambda x: 0 if x != "unknown" else 1)
        ranked["_clinical_rank"] = ranked["cdr"].eq("").astype(int) + ranked["diagnosis_raw"].eq("").astype(int)
        best_rows.append(ranked.sort_values(["_label_rank", "_clinical_rank"]).iloc[0].drop(labels=["_label_rank", "_clinical_rank"]))
    best = pd.DataFrame(best_rows).set_index("subject_id").to_dict(orient="index")
    out = sessions.copy()
    for idx, row in out.iterrows():
        info = best.get(row["subject_id"], {})
        for col in ["age", "sex", "diagnosis_raw", "cdr", "dementia_status", "clinical_label_raw"]:
            if not safe_text(row.get(col, "")) and safe_text(info.get(col, "")):
                out.at[idx, col] = info[col]
        label, rule = provisional_label(out.loc[idx].to_dict())
        out.at[idx, "provisional_label_CN_AD_MCI_unknown"] = label
        out.at[idx, "label_rule_used"] = rule
    return out


def session_sort_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_has_rest_rank"] = (~out["has_task_rest_bold"].astype(bool)).astype(int)
    out["_date_sort"] = out["acquisition_date"].replace("", "9999-99-99")
    out["_days_sort"] = pd.to_numeric(out["days_from_entry"], errors="coerce").fillna(10**9)
    out["_session_sort"] = out["session_id"].replace("", "zzzz")
    return out


def add_session_eligibility(sessions: pd.DataFrame) -> pd.DataFrame:
    if sessions.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS_SESSION)
    rows = []
    for _, row in sessions.iterrows():
        reasons = []
        if not bool(row.get("has_task_rest_bold", False)):
            reasons.append("no_task_rest_bold_metadata")
        if not safe_text(row.get("TR_RepetitionTime", "")):
            reasons.append("missing_TR")
        if safe_text(row.get("provisional_label_CN_AD_MCI_unknown", "unknown")) == "unknown":
            reasons.append("unknown_clinical_label")
        if not safe_text(row.get("age", "")):
            reasons.append("missing_age")
        if not safe_text(row.get("sex", "")):
            reasons.append("missing_sex")
        status = "ELIGIBLE_FOR_SMOKE_TEST" if not reasons else "NEEDS_REVIEW"
        row_out = row.to_dict()
        row_out["eligibility_status"] = status
        row_out["exclusion_reason"] = "|".join(reasons)
        rows.append(row_out)
    out = pd.DataFrame(rows)
    ranked = []
    for subject, sub in out.groupby("subject_id", dropna=False):
        sub2 = session_sort_key(sub)
        sub2 = sub2.sort_values(["_has_rest_rank", "_date_sort", "_days_sort", "_session_sort", "metadata_source"])
        sub2["session_rank_per_subject"] = range(1, len(sub2) + 1)
        ranked.append(sub2.drop(columns=["_has_rest_rank", "_date_sort", "_days_sort", "_session_sort"]))
    return pd.concat(ranked, ignore_index=True)[OUTPUT_COLUMNS_SESSION]


def build_subject_summary(sessions: pd.DataFrame, subject_info: pd.DataFrame) -> pd.DataFrame:
    if sessions.empty and subject_info.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS_SUBJECT)
    subjects = sorted(set(sessions.get("subject_id", pd.Series(dtype=str))).union(set(subject_info.get("subject_id", pd.Series(dtype=str)))))
    rows = []
    info_by_subject = {}
    if not subject_info.empty:
        for subject, sub in subject_info.groupby("subject_id"):
            info_by_subject[subject] = sub.iloc[0].to_dict()
    for subject in subjects:
        sub = sessions[sessions["subject_id"] == subject] if not sessions.empty else pd.DataFrame()
        rest = sub[sub["has_task_rest_bold"].astype(bool)] if not sub.empty else pd.DataFrame()
        baseline = pd.Series(dtype=object)
        reason = ""
        if not rest.empty:
            baseline = session_sort_key(rest).sort_values(["_date_sort", "_days_sort", "_session_sort", "metadata_source"]).iloc[0]
            reason = "first ranked task-rest session by date/days/session"
        elif not sub.empty:
            baseline = session_sort_key(sub).sort_values(["_date_sort", "_days_sort", "_session_sort", "metadata_source"]).iloc[0]
            reason = "no task-rest metadata; first available session row"
        info = info_by_subject.get(subject, {})
        label = safe_text(baseline.get("provisional_label_CN_AD_MCI_unknown", "")) or safe_text(info.get("provisional_label_CN_AD_MCI_unknown", "unknown")) or "unknown"
        age = safe_text(baseline.get("age", "")) or safe_text(info.get("age", ""))
        sex = safe_text(baseline.get("sex", "")) or safe_text(info.get("sex", ""))
        clinical_available = label != "unknown" or bool(safe_text(info.get("cdr", "")) or safe_text(info.get("diagnosis_raw", "")))
        rows.append(
            {
                "subject_id": subject,
                "n_sessions": int(len(sub)),
                "n_task_rest_sessions": int(len(rest)),
                "candidate_baseline_session": safe_text(baseline.get("session_id", "")),
                "candidate_baseline_reason": reason,
                "clinical_label_available": bool(clinical_available),
                "provisional_label_CN_AD_MCI_unknown": label,
                "age_baseline": age,
                "sex": sex,
                "scanner_manufacturer_summary": join_unique(sub.get("scanner_manufacturer", pd.Series(dtype=str))),
                "scanner_model_summary": join_unique(sub.get("scanner_model", pd.Series(dtype=str))),
                "eligible_for_smoke_test": bool(
                    not rest.empty
                    and label in {"CN", "AD"}
                    and safe_text(baseline.get("TR_RepetitionTime", ""))
                    and age
                    and sex
                ),
            }
        )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS_SUBJECT)


def select_smoke_candidates(sessions: pd.DataFrame) -> pd.DataFrame:
    if sessions.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS_SMOKE)
    eligible = sessions[
        (sessions["eligibility_status"] == "ELIGIBLE_FOR_SMOKE_TEST")
        & (sessions["provisional_label_CN_AD_MCI_unknown"].isin(["CN", "AD"]))
    ].copy()
    if eligible.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS_SMOKE)
    selected_rows = []
    used_subjects = set()
    for label in ["CN", "AD"]:
        sub = eligible[eligible["provisional_label_CN_AD_MCI_unknown"] == label].copy()
        sub = session_sort_key(sub).sort_values(["_date_sort", "_days_sort", "_session_sort", "subject_id"])
        for _, row in sub.iterrows():
            if row["subject_id"] in used_subjects:
                continue
            used_subjects.add(row["subject_id"])
            selected_rows.append(
                {
                    "subject_id": row["subject_id"],
                    "session_id": row["session_id"],
                    "provisional_label_CN_AD_MCI_unknown": row["provisional_label_CN_AD_MCI_unknown"],
                    "age": row["age"],
                    "sex": row["sex"],
                    "TR_RepetitionTime": row["TR_RepetitionTime"],
                    "scanner_manufacturer": row["scanner_manufacturer"],
                    "scanner_model": row["scanner_model"],
                    "json_path": row["json_path"],
                    "bold_file_path": row["bold_file_path"],
                    "selection_reason": f"baseline eligible task-rest {label}; one session per subject",
                }
            )
            if sum(1 for x in selected_rows if x["provisional_label_CN_AD_MCI_unknown"] == label) >= 5:
                break
    return pd.DataFrame(selected_rows, columns=OUTPUT_COLUMNS_SMOKE)


def feasibility_verdict(
    metadata_files: Sequence[Path],
    sessions: pd.DataFrame,
    subjects: pd.DataFrame,
    smoke: pd.DataFrame,
) -> str:
    if not metadata_files:
        return "NEED_DOWNLOAD_METADATA"
    if sessions.empty:
        return "NEED_BIDS_JSON"
    if int(sessions["has_task_rest_bold"].sum()) == 0:
        return "NEED_BIDS_JSON"
    if subjects.empty or not subjects["clinical_label_available"].any():
        return "NEED_CLINICAL_MAPPING"
    if smoke.empty:
        if sessions["TR_RepetitionTime"].replace("", pd.NA).isna().all():
            return "NEED_BIDS_JSON"
        return "NEED_CLINICAL_MAPPING"
    cn = int((smoke["provisional_label_CN_AD_MCI_unknown"] == "CN").sum())
    ad = int((smoke["provisional_label_CN_AD_MCI_unknown"] == "AD").sum())
    if cn >= 1 and ad >= 1:
        return "READY_FOR_SMOKE_TEST"
    return "NEED_CLINICAL_MAPPING"


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


def write_label_mapping_report(path: Path, subject_summary: pd.DataFrame, session_inventory: pd.DataFrame) -> None:
    label_counts = {}
    if not subject_summary.empty:
        label_counts = subject_summary["provisional_label_CN_AD_MCI_unknown"].value_counts(dropna=False).to_dict()
    lines = [
        "# OASIS-3 Provisional Label Mapping",
        "",
        "Labels are provisional and must not be treated as ADNI-compatible without clinical review.",
        "",
        "Rules used:",
        "- `CN`: CDR==0, or explicit cognitively normal / normal control / no dementia text.",
        "- `MCI`: CDR==0.5, or explicit MCI / mild cognitive impairment text.",
        "- `AD`: CDR>=1, or explicit AD dementia / Alzheimer / dementia text.",
        "- `unknown`: no clear rule matched.",
        "",
        "Subject-level provisional label counts:",
    ]
    if label_counts:
        lines.extend(f"- `{label}`: `{count}`" for label, count in sorted(label_counts.items()))
    else:
        lines.append("- No subject-level labels available from local metadata.")
    if not session_inventory.empty and "label_rule_used" in session_inventory.columns:
        lines.extend(["", "Session label-rule counts:"])
        for rule, count in session_inventory["label_rule_used"].value_counts(dropna=False).items():
            lines.append(f"- `{rule}`: `{count}`")
    lines.extend(
        [
            "",
            "Uncertainty:",
            "- OASIS-3 clinical labels may be visit-specific and longitudinal.",
            "- One session per subject should be selected unless an explicitly longitudinal design is used.",
            "- External validation should report the exact clinical mapping rule used.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(
    path: Path,
    roots_found: Sequence[Path],
    metadata_files: Sequence[Path],
    session_inventory: pd.DataFrame,
    subject_summary: pd.DataFrame,
    smoke: pd.DataFrame,
    verdict: str,
) -> None:
    n_subjects = len(subject_summary)
    n_sessions = len(session_inventory)
    n_rest = int(session_inventory["has_task_rest_bold"].sum()) if not session_inventory.empty else 0
    label_counts = subject_summary["provisional_label_CN_AD_MCI_unknown"].value_counts(dropna=False).to_dict() if not subject_summary.empty else {}
    tr_available = bool(not session_inventory.empty and session_inventory["TR_RepetitionTime"].replace("", pd.NA).notna().any())
    scanner_available = bool(
        not session_inventory.empty
        and (
            session_inventory["scanner_manufacturer"].replace("", pd.NA).notna().any()
            or session_inventory["scanner_model"].replace("", pd.NA).notna().any()
        )
    )
    metadata_present = bool(metadata_files)
    next_step = {
        "READY_FOR_SMOKE_TEST": "Review `oasis3_smoke_test_candidates.csv`, then manually download/prepare only those selected sessions.",
        "NEED_DOWNLOAD_METADATA": "Download OASIS-3 clinical/session exports and BIDS participants/sessions/task-rest JSON metadata from NITRC/XNAT; do not download imaging yet.",
        "NEED_CLINICAL_MAPPING": "Provide or review OASIS-3 clinical diagnosis/CDR metadata before selecting AD/CN smoke-test subjects.",
        "NEED_BIDS_JSON": "Provide BIDS task-rest sidecar JSON metadata or an XNAT session export with task/TR/scanner fields.",
        "NOT_READY": "Resolve missing metadata and label/session mapping before imaging work.",
    }.get(verdict, "Review missing metadata fields.")
    lines = [
        "# OASIS-3 External Validation Feasibility Audit",
        "",
        "Metadata-first audit only. No training, no full image preprocessing, no large download, and no large arrays loaded.",
        "",
        "## Local Metadata Status",
        f"- Metadata present locally: `{metadata_present}`",
        f"- Search roots found: `{len(roots_found)}`",
        f"- Metadata files discovered: `{len(metadata_files)}`",
        "",
        "## Inventory Counts",
        f"- Subjects with any discovered metadata: `{n_subjects}`",
        f"- Candidate imaging/session rows: `{n_sessions}`",
        f"- Candidate task-rest BOLD sessions: `{n_rest}`",
        f"- Provisional CN subjects: `{int(label_counts.get('CN', 0))}`",
        f"- Provisional AD/dementia subjects: `{int(label_counts.get('AD', 0))}`",
        f"- Provisional MCI subjects: `{int(label_counts.get('MCI', 0))}`",
        f"- Unknown-label subjects: `{int(label_counts.get('unknown', 0))}`",
        f"- TR/RepetitionTime available: `{tr_available}`",
        f"- Scanner/manufacturer metadata available: `{scanner_available}`",
        f"- Smoke-test candidates selected: `{len(smoke)}`",
        "",
        "## Feasibility Verdict",
        f"`{verdict}`",
        "",
        "## Recommended Next Step",
        next_step,
        "",
        "## Methodological Notes",
        "- Use one baseline/rest session per subject unless explicitly modelling longitudinal data.",
        "- If longitudinal OASIS sessions are used, split by subject using GroupKFold/LeaveOneGroupOut.",
        "- Do not assume OASIS clinical labels are ADNI-compatible without documenting CDR/diagnosis mapping.",
        "- OASIS must be preprocessing-audited before using the ADNI beta-VAE/classifier pipeline because the ADNI pipeline is filter-sensitive.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_empty_outputs(output_dir: Path) -> None:
    pd.DataFrame(columns=OUTPUT_COLUMNS_SOURCES).to_csv(output_dir / "metadata_sources_used.csv", index=False)
    pd.DataFrame(columns=OUTPUT_COLUMNS_SESSION).to_csv(output_dir / "oasis3_session_inventory.csv", index=False)
    pd.DataFrame(columns=OUTPUT_COLUMNS_SUBJECT).to_csv(output_dir / "oasis3_subject_summary.csv", index=False)
    pd.DataFrame(columns=OUTPUT_COLUMNS_SMOKE).to_csv(output_dir / "oasis3_smoke_test_candidates.csv", index=False)


def main() -> int:
    args = parse_args()
    output_dir = resolve_path(args.output_dir) or DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_files, roots_found = discover_metadata_files(args)
    tables: List[TableMetadata] = []
    metadata_rows: List[Dict[str, Any]] = []
    json_session_rows: List[Dict[str, Any]] = []
    json_source_rows: List[Dict[str, Any]] = []

    for path in metadata_files:
        if path.suffix.lower() == ".json":
            row, source = json_to_session_row(path)
            json_source_rows.append(source)
            if row["subject_id"] or row["has_task_rest_bold"]:
                json_session_rows.append(row)
        else:
            table = read_table(path)
            tables.append(table)
            metadata_rows.append(
                {
                    "path": str(path),
                    "file_type": table.file_type,
                    "n_rows": table.n_rows,
                    "detected_columns": json.dumps(table.detected_columns, sort_keys=True),
                    "notes": table.notes,
                }
            )
    metadata_rows.extend(json_source_rows)
    metadata_sources = pd.DataFrame(metadata_rows, columns=OUTPUT_COLUMNS_SOURCES)

    table_sessions = [table_rows_to_sessions(table) for table in tables]
    session_frames = [df for df in table_sessions if not df.empty]
    if json_session_rows:
        session_frames.append(pd.DataFrame(json_session_rows))
    sessions = pd.concat(session_frames, ignore_index=True) if session_frames else pd.DataFrame(columns=OUTPUT_COLUMNS_SESSION)
    subject_info = subject_info_from_tables(tables)
    sessions = enrich_sessions_with_subject_info(sessions, subject_info)
    sessions = add_session_eligibility(sessions)
    subject_summary = build_subject_summary(sessions, subject_info)
    smoke = select_smoke_candidates(sessions)
    verdict = feasibility_verdict(metadata_files, sessions, subject_summary, smoke)

    if metadata_sources.empty and sessions.empty and subject_summary.empty and smoke.empty:
        write_empty_outputs(output_dir)
    else:
        metadata_sources.to_csv(output_dir / "metadata_sources_used.csv", index=False)
        sessions.to_csv(output_dir / "oasis3_session_inventory.csv", index=False)
        subject_summary.to_csv(output_dir / "oasis3_subject_summary.csv", index=False)
        smoke.to_csv(output_dir / "oasis3_smoke_test_candidates.csv", index=False)

    write_label_mapping_report(output_dir / "label_mapping_report.md", subject_summary, sessions)
    write_readme(output_dir / "README.md", roots_found, metadata_files, sessions, subject_summary, smoke, verdict)

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": git_hash(),
        "inputs": {
            "oasis_root": str(resolve_path(args.oasis_root)) if args.oasis_root else "",
            "clinical_csv": str(resolve_path(args.clinical_csv)) if args.clinical_csv else "",
            "sessions_csv": str(resolve_path(args.sessions_csv)) if args.sessions_csv else "",
            "bids_root": str(resolve_path(args.bids_root)) if args.bids_root else "",
            "default_search_roots_found": [str(path) for path in roots_found],
        },
        "output_dir": str(output_dir),
        "metadata_files_discovered": [str(path) for path in metadata_files],
        "no_training": True,
        "no_full_image_preprocessing": True,
        "no_large_download": True,
        "no_large_arrays_loaded": True,
        "feasibility_verdict": verdict,
    }
    (output_dir / "audit_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print((output_dir / "README.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
