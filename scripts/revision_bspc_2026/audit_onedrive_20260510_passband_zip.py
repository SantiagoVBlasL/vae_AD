#!/usr/bin/env python3
"""Safe import/audit for Martin's 2026-05-10 ADNI passband zip.

This script is intentionally conservative:
- it lists zip contents before any extraction;
- extraction is opt-in via --extract;
- extraction to /home is blocked by default;
- no connectivity computation, tensor generation, inference, or training occurs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUBJECT_RE = re.compile(r"(?<!\d)(\d{3}_S_\d{4})(?!\d)", re.IGNORECASE)
IMAGE_ID_RE = re.compile(r"(?<![A-Z0-9])I?(\d{5,})(?![A-Z0-9])", re.IGNORECASE)

DEFAULT_ZIP = PROJECT_ROOT / "data" / "OneDrive_1_10-5-2026.zip"
DEFAULT_EXTRACT_ROOT = Path("/media/diego/My_Book_Diego/vae_AD_data/adni_passband_20260510")
DEFAULT_V5_OUTPUT_ROOT = Path(
    "/media/diego/My_Book_Diego/vae_AD_data/revision_bspc_2026/adni_expanded_v5_passband_dparsf_only"
)
DEFAULT_V4_METADATA = (
    PROJECT_ROOT
    / "data"
    / "revision_bspc_2026"
    / "adni_expanded_v4_all_available"
    / "subject_metadata_adni_expanded_v4_all_available.csv"
)
DEFAULT_ORIGINAL_METADATA = PROJECT_ROOT / "data" / "SubjectsData_AAL3_procesado2.csv"
DEFAULT_ADNI_DOWNLOAD_NOW = PROJECT_ROOT / "data" / "adni_download_now.csv"
DEFAULT_EXTRA_METADATA = [
    PROJECT_ROOT / "data" / "idaSearch_4_03_2026.csv",
    PROJECT_ROOT / "data" / "AD_fMRI_4_28_2026.csv",
    PROJECT_ROOT / "data" / "RevisionPaperfMRI_2026_04_4_06_2026.csv",
    PROJECT_ROOT / "data" / "RevisionPaperfMRI_2026_04_4_06_2026_SecondBatch.csv",
]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_passband_20260510_audit"
)

SUBJECT_COLS = ["SubjectID", "Subject", "Subject ID", "PTID", "subject_id", "RID"]
DX_COLS = ["ResearchGroup_Mapped", "ResearchGroup", "Research Group", "Group", "DX", "Diagnosis"]
AGE_COLS = ["Age", "AGE"]
SEX_COLS = ["Sex", "PTGENDER", "Gender"]
MANUFACTURER_COLS = ["Manufacturer", "Mfr", "MFG"]
SITE_COLS = ["Site3", "Site", "SITEID", "RID_SITE"]
IMAGE_COLS = ["ImageID", "Image ID", "Image Data ID", "IMAGEUID", "Image Data ID"]
VISIT_COLS = ["Visit", "VISCODE", "VISCODE2", "VisitCode", "Visit Code"]
DATE_COLS = [
    "StudyDate",
    "Study Date",
    "Acq Date",
    "AcquisitionDate",
    "Acquisition Date",
    "ScanDate",
    "Scan Date",
    "EXAMDATE",
    "ArchiveDate",
    "Archive Date",
]
TR_COLS = ["TR", "RepetitionTime", "Repetition Time"]
TE_COLS = ["TE", "EchoTime", "Echo Time"]
FIELD_STRENGTH_COLS = ["FieldStrength", "Field Strength", "MagneticFieldStrength"]
DESCRIPTION_COLS = ["Description", "Series Description", "ProtocolName"]
REASON_COLS = ["reason", "Reason", "selection_reason"]
PHASE_COLS = ["Phase", "Study", "COLPROT"]
PROTOCOL_COLS = ["ImagingProtocol", "Imaging Protocol", "Protocol", "Sequence"]

METADATA_OUTPUT_COLUMNS = [
    "SubjectID",
    "diagnosis",
    "ResearchGroup_Mapped",
    "Age",
    "Sex",
    "Manufacturer",
    "Site",
    "Site3",
    "ImageID",
    "Visit",
    "timepoint",
    "StudyDate",
    "source_metadata_file",
    "all_metadata_sources",
    "already_in_v4",
    "already_in_original",
    "completely_new_subject",
    "candidate_for_v5",
]

CHANNEL_NAMES_MASTER = [
    "Pearson_OMST_GCE_Signed_Weighted",
    "Pearson_Full_FisherZ_Signed",
    "MI_KNN_Symmetric",
    "dFC_AbsDiffMean",
    "dFC_StdDev",
    "DistanceCorr",
    "Granger_F_lag1",
]


@dataclass(frozen=True)
class MetadataSource:
    path_label: str
    source_kind: str
    priority: int
    dataframe: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Martin's ADNI passband OneDrive zip without computing features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP, dest="zip_path")
    parser.add_argument("--extract-root", type=Path, default=DEFAULT_EXTRACT_ROOT)
    parser.add_argument("--local-symlink", type=Path, default=None)
    parser.add_argument("--v4-metadata", type=Path, default=DEFAULT_V4_METADATA)
    parser.add_argument("--original-metadata", type=Path, default=DEFAULT_ORIGINAL_METADATA)
    parser.add_argument("--adni-download-now", type=Path, default=DEFAULT_ADNI_DOWNLOAD_NOW)
    parser.add_argument(
        "--extra-metadata",
        type=Path,
        nargs="*",
        default=DEFAULT_EXTRA_METADATA,
        help="Optional auxiliary local ADNI metadata CSVs used only for recovery when primary files are incomplete.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--extract", action="store_true", help="Extract the zip after listing/inventory checks.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite/recreate audit output files.")
    parser.add_argument(
        "--overwrite-extracted-files",
        action="store_true",
        help="Allow replacing already extracted files whose size differs from the zip entry.",
    )
    parser.add_argument(
        "--allow-home-extract",
        action="store_true",
        help="Override the guard that blocks extraction under /home.",
    )
    parser.add_argument("--max-csv-mb", type=float, default=100.0)
    parser.add_argument("--max-zip-shape-mb", type=float, default=8.0)
    parser.add_argument("--skip-zip-shape-inspection", action="store_true")
    parser.add_argument(
        "--final-subject-list-confirmed",
        action="store_true",
        help="Clear the final v5 rebuild gate after the complete final training subject list has been confirmed.",
    )
    return parser.parse_args()


def resolve_path(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    return path if path.is_absolute() else PROJECT_ROOT / path


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(f"Output directory exists and is not empty; pass --overwrite: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def normalize_subject_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    text = re.sub(r"\s+", "", text)
    match = SUBJECT_RE.search(text)
    return match.group(1).upper() if match else text


def normalize_diagnosis(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    text = text.replace(" ", "")
    if text in {"", "NAN", "NONE"}:
        return ""
    if text in {"CN", "NL", "NORMAL", "CONTROL", "CONTROLS"}:
        return "CN"
    if text in {"AD", "DEMENTIA", "ALZHEIMERS", "ALZHEIMER", "ALZHEIMER'S"}:
        return "AD"
    if text in {"MCI", "EMCI", "LMCI", "SMC"}:
        return "MCI"
    if "MCI" in text:
        return "MCI"
    if text.startswith("AD"):
        return "AD"
    return text


def normalize_image_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    text = text.lstrip("I")
    return text if text and text != "NAN" else ""


def normalize_scalar(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def normalize_date(value: Any) -> str:
    text = normalize_scalar(value)
    if not text:
        return ""
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return ""
    return parsed.strftime("%Y-%m-%d")


def normalize_visit(value: Any) -> str:
    text = normalize_scalar(value)
    return re.sub(r"\s+", " ", text)


def normalize_site(value: Any, subject_id: str = "") -> str:
    text = normalize_scalar(value)
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if text:
        return text
    if subject_id and "_" in subject_id:
        return subject_id.split("_", 1)[0].lstrip("0") or subject_id.split("_", 1)[0]
    return ""


def parse_manufacturer_from_protocol(value: Any) -> str:
    text = normalize_scalar(value)
    if not text:
        return ""
    match = re.search(r"Manufacturer=([^;]+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def normalize_manufacturer(value: Any) -> str:
    text = normalize_scalar(value)
    upper = text.upper()
    if "SIEMENS" in upper:
        return "SIEMENS"
    if "PHILIPS" in upper:
        return "Philips"
    if "GE" in upper:
        return "GE MEDICAL SYSTEMS"
    return text


def first_existing_col(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower = {str(col).strip().lower(): str(col) for col in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        hit = lower.get(candidate.strip().lower())
        if hit is not None:
            return hit
    return None


def safe_numeric(value: Any, default: float = float("inf")) -> float:
    text = normalize_image_id(value)
    if text.isdigit():
        return float(text)
    return default


def join_unique(values: Iterable[Any], sep: str = "|") -> str:
    out: List[str] = []
    seen = set()
    for value in values:
        text = normalize_scalar(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return sep.join(out)


def split_joined(text: Any) -> List[str]:
    scalar = normalize_scalar(text)
    if not scalar:
        return []
    return [item for item in scalar.split("|") if item]


def find_subjects(text: str) -> List[str]:
    return sorted({match.group(1).upper() for match in SUBJECT_RE.finditer(text or "")})


def infer_role(path_text: str) -> str:
    path = Path(path_text)
    name_low = path.name.lower()
    full_low = path_text.lower()
    suffix = path.suffix.lower()
    if name_low.startswith("roisignals_") and suffix in {".txt", ".mat"}:
        return "roi_signal"
    if "roisignals" in full_low and suffix in {".txt", ".mat"}:
        return "roi_signal_candidate"
    if "meanfmriatlas" in name_low:
        return "atlas_preview"
    if suffix in {".png", ".jpg", ".jpeg"} and "check" in full_low:
        return "qc_check_image"
    if suffix == ".gif":
        return "preview_gif"
    if suffix == ".csv":
        return "csv"
    if suffix in {".nii", ".gz"}:
        return "nifti_or_archive"
    return "other"


def provenance_tokens(path_text: str) -> str:
    terms = [
        "pasabandas",
        "passband",
        "bandpass",
        "ROISignals",
        "ResultsAAL3",
        "FunImg",
        "ARWSDCFN",
        "ARWSDCF",
        "ARWSDC",
        "DPARSF",
        "Filtered",
        "Detrend",
        "Normalize",
    ]
    low = path_text.lower()
    hits = [term for term in terms if term.lower() in low]
    if "arwsdcfn" in low:
        hits.extend(["DPARSF_prefix_ARWSDCFN", "likely_filtered_F_component"])
    elif "arwsdcf" in low:
        hits.extend(["DPARSF_prefix_ARWSDCF", "likely_filtered_F_component"])
    return join_unique(hits)


def list_zip_contents(zip_path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            date_time = ""
            try:
                date_time = datetime(*info.date_time).isoformat()
            except Exception:
                pass
            subjects = find_subjects(info.filename)
            rows.append(
                {
                    "path": info.filename,
                    "basename": Path(info.filename).name,
                    "parent": str(Path(info.filename).parent),
                    "is_dir": info.is_dir(),
                    "suffix": Path(info.filename).suffix.lower(),
                    "file_size_bytes": int(info.file_size),
                    "compress_size_bytes": int(info.compress_size),
                    "crc32": f"{info.CRC:08x}",
                    "date_time": date_time,
                    "subject_ids": "|".join(subjects),
                    "subject_id_count": len(subjects),
                    "inferred_file_role": infer_role(info.filename),
                    "provenance_tokens": provenance_tokens(info.filename),
                }
            )
    return pd.DataFrame(rows)


def path_is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except (OSError, ValueError):
        return False


def assert_safe_zip_member(root: Path, member_name: str) -> Path:
    target = root / member_name
    try:
        resolved = target.resolve()
        resolved.relative_to(root.resolve())
    except (OSError, ValueError):
        raise RuntimeError(f"Unsafe zip member path outside extract root: {member_name}")
    return target


def is_home_path(path: Path) -> bool:
    try:
        path.resolve().relative_to(Path.home().resolve())
        return True
    except (OSError, ValueError):
        return False


def extract_zip_safely(
    zip_path: Path,
    extract_root: Path,
    overwrite_extracted_files: bool,
    allow_home_extract: bool,
) -> pd.DataFrame:
    if is_home_path(extract_root) and not allow_home_extract:
        raise RuntimeError(
            "Refusing extraction under /home. Use a big-disk --extract-root such as "
            "/media/diego/My_Book_Diego/vae_AD_data/adni_passband_20260510."
        )
    extract_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            target = assert_safe_zip_member(extract_root, info.filename)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                rows.append({"path": info.filename, "target": str(target), "status": "directory"})
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                existing_size = target.stat().st_size
                if existing_size == info.file_size:
                    rows.append(
                        {
                            "path": info.filename,
                            "target": str(target),
                            "status": "exists_same_size_skipped",
                            "existing_size_bytes": int(existing_size),
                            "zip_size_bytes": int(info.file_size),
                        }
                    )
                    continue
                if not overwrite_extracted_files:
                    raise RuntimeError(
                        f"Extract target exists with different size: {target}. "
                        "Pass --overwrite-extracted-files to replace extracted payload files."
                    )
            with zf.open(info, "r") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            rows.append(
                {
                    "path": info.filename,
                    "target": str(target),
                    "status": "extracted",
                    "existing_size_bytes": "",
                    "zip_size_bytes": int(info.file_size),
                }
            )
    return pd.DataFrame(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_extracted_files(extract_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if not extract_root.exists():
        return pd.DataFrame(
            columns=[
                "path",
                "abs_path",
                "basename",
                "parent",
                "suffix",
                "file_size_bytes",
                "mtime",
                "sha256",
                "subject_ids",
                "subject_id_count",
                "inferred_file_role",
                "provenance_tokens",
            ]
        )
    for path in sorted(p for p in extract_root.rglob("*") if p.is_file()):
        rel = str(path.relative_to(extract_root))
        subjects = find_subjects(rel)
        rows.append(
            {
                "path": rel,
                "abs_path": str(path),
                "basename": path.name,
                "parent": str(path.parent.relative_to(extract_root)),
                "suffix": path.suffix.lower(),
                "file_size_bytes": int(path.stat().st_size),
                "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                "sha256": sha256_file(path),
                "subject_ids": "|".join(subjects),
                "subject_id_count": len(subjects),
                "inferred_file_role": infer_role(rel),
                "provenance_tokens": provenance_tokens(rel),
            }
        )
    return pd.DataFrame(rows)


def read_txt_shape_from_bytes(payload: bytes) -> Dict[str, Any]:
    for delimiter, label in [(b",", "comma"), (None, "whitespace")]:
        try:
            data = np.loadtxt(io.BytesIO(payload), delimiter=delimiter.decode("ascii") if delimiter else None)
            if data.ndim == 1:
                shape = (int(data.shape[0]),)
                rows, cols = int(data.shape[0]), 1
            else:
                shape = tuple(int(x) for x in data.shape)
                rows, cols = int(data.shape[0]), int(data.shape[1])
            return {
                "shape": str(shape),
                "n_rows": rows,
                "n_cols": cols,
                "shape_status": f"ok_{label}_delimited",
                "value_min": float(np.nanmin(data)) if data.size else np.nan,
                "value_max": float(np.nanmax(data)) if data.size else np.nan,
                "nan_count": int(np.isnan(data).sum()) if np.issubdtype(data.dtype, np.floating) else 0,
            }
        except Exception as exc:
            last_exc = exc
    return {
        "shape": "",
        "n_rows": np.nan,
        "n_cols": np.nan,
        "shape_status": f"txt_read_failed: {last_exc}",
        "value_min": np.nan,
        "value_max": np.nan,
        "nan_count": np.nan,
    }


def read_txt_shape_from_path(path: Path) -> Dict[str, Any]:
    try:
        return read_txt_shape_from_bytes(path.read_bytes())
    except Exception as exc:
        return {
            "shape": "",
            "n_rows": np.nan,
            "n_cols": np.nan,
            "shape_status": f"txt_open_failed: {exc}",
            "value_min": np.nan,
            "value_max": np.nan,
            "nan_count": np.nan,
        }


def read_mat_shape_from_bytes(payload: bytes) -> Dict[str, Any]:
    try:
        import scipy.io  # type: ignore

        entries = scipy.io.whosmat(io.BytesIO(payload))
        if not entries:
            return {"shape": "", "n_rows": np.nan, "n_cols": np.nan, "shape_status": "mat_no_variables"}
        preferred = sorted(entries, key=lambda item: (0 if "roi" in item[0].lower() or "signal" in item[0].lower() else 1, item[0]))
        name, shape, klass = preferred[0]
        rows = int(shape[0]) if len(shape) >= 1 else 1
        cols = int(shape[1]) if len(shape) >= 2 else 1
        return {
            "shape": str(tuple(int(x) for x in shape)),
            "n_rows": rows,
            "n_cols": cols,
            "shape_status": f"ok_whosmat:{name}:{klass}",
        }
    except Exception as exc:
        return {"shape": "", "n_rows": np.nan, "n_cols": np.nan, "shape_status": f"mat_read_failed: {exc}"}


def read_mat_shape_from_path(path: Path) -> Dict[str, Any]:
    try:
        return read_mat_shape_from_bytes(path.read_bytes())
    except Exception as exc:
        return {"shape": "", "n_rows": np.nan, "n_cols": np.nan, "shape_status": f"mat_open_failed: {exc}"}


def signal_rows_from_zip(
    zip_path: Path,
    zip_inventory: pd.DataFrame,
    inspect_shapes: bool,
    max_zip_shape_mb: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    max_bytes = int(max_zip_shape_mb * 1024 * 1024)
    signal_like = zip_inventory[
        zip_inventory["inferred_file_role"].isin(["roi_signal", "roi_signal_candidate"])
    ].copy()
    with zipfile.ZipFile(zip_path) as zf:
        for _, row in signal_like.iterrows():
            path_text = str(row["path"])
            subjects = split_joined(row["subject_ids"])
            subject_id = subjects[0] if subjects else ""
            suffix = str(row["suffix"]).lower()
            shape_info = {
                "shape": "",
                "n_rows": np.nan,
                "n_cols": np.nan,
                "shape_status": "not_read",
                "value_min": np.nan,
                "value_max": np.nan,
                "nan_count": np.nan,
            }
            if inspect_shapes and int(row["file_size_bytes"]) <= max_bytes:
                payload = zf.read(path_text)
                if suffix == ".txt":
                    shape_info = read_txt_shape_from_bytes(payload)
                elif suffix == ".mat":
                    shape_info.update(read_mat_shape_from_bytes(payload))
            elif int(row["file_size_bytes"]) > max_bytes:
                shape_info["shape_status"] = f"not_read_zip_member_gt_{max_zip_shape_mb:g}mb"
            rows.append(
                {
                    "source_kind": "zip",
                    "SubjectID": subject_id,
                    "path": path_text,
                    "abs_path": "",
                    "basename": row["basename"],
                    "suffix": suffix,
                    "file_size_bytes": row["file_size_bytes"],
                    "sha256_or_crc32": row["crc32"],
                    "inferred_file_role": row["inferred_file_role"],
                    "provenance_tokens": row["provenance_tokens"],
                    "shape_source": "zip_stream" if shape_info["shape_status"] != "not_read" else "",
                    **shape_info,
                }
            )
    return rows


def signal_rows_from_extracted(extracted_inventory: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if extracted_inventory.empty:
        return rows
    signal_like = extracted_inventory[
        extracted_inventory["inferred_file_role"].isin(["roi_signal", "roi_signal_candidate"])
    ].copy()
    for _, row in signal_like.iterrows():
        subjects = split_joined(row["subject_ids"])
        subject_id = subjects[0] if subjects else ""
        suffix = str(row["suffix"]).lower()
        path = Path(str(row["abs_path"]))
        shape_info = {
            "shape": "",
            "n_rows": np.nan,
            "n_cols": np.nan,
            "shape_status": "not_read",
            "value_min": np.nan,
            "value_max": np.nan,
            "nan_count": np.nan,
        }
        if suffix == ".txt":
            shape_info = read_txt_shape_from_path(path)
        elif suffix == ".mat":
            shape_info.update(read_mat_shape_from_path(path))
        rows.append(
            {
                "source_kind": "extracted",
                "SubjectID": subject_id,
                "path": row["path"],
                "abs_path": row["abs_path"],
                "basename": row["basename"],
                "suffix": suffix,
                "file_size_bytes": row["file_size_bytes"],
                "sha256_or_crc32": row["sha256"],
                "inferred_file_role": row["inferred_file_role"],
                "provenance_tokens": row["provenance_tokens"],
                "shape_source": "extracted_file" if shape_info["shape_status"] != "not_read" else "",
                **shape_info,
            }
        )
    return rows


def build_detected_signal_files(
    zip_path: Path,
    zip_inventory: pd.DataFrame,
    extracted_inventory: pd.DataFrame,
    inspect_zip_shapes: bool,
    max_zip_shape_mb: float,
) -> pd.DataFrame:
    rows = signal_rows_from_zip(zip_path, zip_inventory, inspect_zip_shapes, max_zip_shape_mb)
    rows.extend(signal_rows_from_extracted(extracted_inventory))
    if not rows:
        return pd.DataFrame(
            columns=[
                "source_kind",
                "SubjectID",
                "path",
                "abs_path",
                "basename",
                "suffix",
                "file_size_bytes",
                "sha256_or_crc32",
                "shape",
                "n_rows",
                "n_cols",
                "shape_status",
            ]
        )
    return pd.DataFrame(rows).sort_values(["source_kind", "SubjectID", "suffix", "path"])


def all_subject_ids_from_inventories(zip_inventory: pd.DataFrame, extracted_inventory: pd.DataFrame) -> List[str]:
    subjects = set()
    for df in [zip_inventory, extracted_inventory]:
        if df.empty or "subject_ids" not in df.columns:
            continue
        for item in df["subject_ids"].fillna("").astype(str):
            subjects.update(split_joined(item))
    return sorted(subjects)


def build_detected_subjects(
    zip_inventory: pd.DataFrame,
    extracted_inventory: pd.DataFrame,
    signal_files: pd.DataFrame,
) -> pd.DataFrame:
    subject_ids = all_subject_ids_from_inventories(zip_inventory, extracted_inventory)
    rows: List[Dict[str, Any]] = []
    primary_source = "extracted" if (not extracted_inventory.empty and not signal_files[signal_files["source_kind"] == "extracted"].empty) else "zip"
    for sid in subject_ids:
        zip_rows = zip_inventory[zip_inventory["subject_ids"].fillna("").astype(str).str.contains(sid, regex=False)]
        ext_rows = (
            extracted_inventory[extracted_inventory["subject_ids"].fillna("").astype(str).str.contains(sid, regex=False)]
            if not extracted_inventory.empty
            else pd.DataFrame()
        )
        sig = signal_files[signal_files["SubjectID"] == sid].copy() if not signal_files.empty else pd.DataFrame()
        primary_sig = sig[sig["source_kind"] == primary_source].copy() if not sig.empty else pd.DataFrame()
        if primary_sig.empty:
            primary_sig = sig
        txt = primary_sig[primary_sig["suffix"] == ".txt"].copy() if not primary_sig.empty else pd.DataFrame()
        mat = primary_sig[primary_sig["suffix"] == ".mat"].copy() if not primary_sig.empty else pd.DataFrame()
        shape_values = join_unique(txt["shape"].tolist() if not txt.empty else [])
        issue_parts = []
        if txt.empty and mat.empty:
            issue_parts.append("no_roi_signal_txt_or_mat")
        if len(txt) > 1:
            issue_parts.append(f"multiple_txt_signals={len(txt)}")
        if len(mat) > 1:
            issue_parts.append(f"multiple_mat_signals={len(mat)}")
        if not txt.empty:
            cols = pd.to_numeric(txt["n_cols"], errors="coerce").dropna().astype(int).unique().tolist()
            if cols and not all(col in {131, 166, 170} for col in cols):
                issue_parts.append(f"unexpected_txt_cols={cols}")
        rows.append(
            {
                "SubjectID": sid,
                "primary_source": primary_source,
                "zip_file_count": int(len(zip_rows)),
                "extracted_file_count": int(len(ext_rows)),
                "signal_file_count": int(len(primary_sig)),
                "txt_signal_count": int(len(txt)),
                "mat_signal_count": int(len(mat)),
                "has_txt_signal": bool(len(txt) > 0),
                "has_mat_signal": bool(len(mat) > 0),
                "txt_shapes": shape_values,
                "txt_paths": join_unique(txt["path"].tolist() if not txt.empty else []),
                "mat_paths": join_unique(mat["path"].tolist() if not mat.empty else []),
                "other_zip_paths": join_unique(zip_rows["path"].head(12).tolist() if not zip_rows.empty else []),
                "issue": ";".join(issue_parts) if issue_parts else "OK",
            }
        )
    return pd.DataFrame(rows).sort_values("SubjectID")


def read_csv_if_exists(path: Path, max_csv_mb: float) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_csv_mb:
        raise RuntimeError(f"Metadata CSV is larger than --max-csv-mb ({size_mb:.1f} MB): {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def csv_header_from_bytes(payload: bytes) -> List[str]:
    try:
        return list(pd.read_csv(io.BytesIO(payload), nrows=0).columns)
    except Exception:
        return []


def looks_like_metadata_columns(columns: Sequence[str]) -> bool:
    return first_existing_col(columns, SUBJECT_COLS) is not None and (
        first_existing_col(columns, DX_COLS) is not None
        or first_existing_col(columns, AGE_COLS) is not None
        or first_existing_col(columns, IMAGE_COLS) is not None
    )


def discover_extracted_metadata_csvs(extract_root: Path, max_csv_mb: float) -> List[MetadataSource]:
    sources: List[MetadataSource] = []
    if not extract_root.exists():
        return sources
    for path in sorted(extract_root.rglob("*.csv")):
        try:
            columns = list(pd.read_csv(path, nrows=0).columns)
        except Exception:
            continue
        if not looks_like_metadata_columns(columns):
            continue
        df = read_csv_if_exists(path, max_csv_mb)
        if df is None:
            continue
        sources.append(
            MetadataSource(
                path_label=str(path),
                source_kind="extracted_metadata_csv",
                priority=0,
                dataframe=df,
            )
        )
    return sources


def discover_zip_metadata_csvs(zip_path: Path, max_csv_mb: float) -> List[MetadataSource]:
    sources: List[MetadataSource] = []
    max_bytes = int(max_csv_mb * 1024 * 1024)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".csv") or info.file_size > max_bytes:
                continue
            payload = zf.read(info.filename)
            if not looks_like_metadata_columns(csv_header_from_bytes(payload)):
                continue
            df = pd.read_csv(io.BytesIO(payload), dtype=str, keep_default_na=False)
            sources.append(
                MetadataSource(
                    path_label=f"zip://{zip_path}!/{info.filename}",
                    source_kind="zip_metadata_csv",
                    priority=1,
                    dataframe=df,
                )
            )
    return sources


def normalize_metadata_dataframe(source: MetadataSource) -> pd.DataFrame:
    df = source.dataframe.copy()
    sid_col = first_existing_col(df.columns, SUBJECT_COLS)
    if sid_col is None:
        return pd.DataFrame()

    dx_col = first_existing_col(df.columns, DX_COLS)
    age_col = first_existing_col(df.columns, AGE_COLS)
    sex_col = first_existing_col(df.columns, SEX_COLS)
    manufacturer_col = first_existing_col(df.columns, MANUFACTURER_COLS)
    site_col = first_existing_col(df.columns, SITE_COLS)
    image_col = first_existing_col(df.columns, IMAGE_COLS)
    visit_col = first_existing_col(df.columns, VISIT_COLS)
    date_col = first_existing_col(df.columns, DATE_COLS)
    tr_col = first_existing_col(df.columns, TR_COLS)
    te_col = first_existing_col(df.columns, TE_COLS)
    field_col = first_existing_col(df.columns, FIELD_STRENGTH_COLS)
    desc_col = first_existing_col(df.columns, DESCRIPTION_COLS)
    reason_col = first_existing_col(df.columns, REASON_COLS)
    phase_col = first_existing_col(df.columns, PHASE_COLS)
    protocol_col = first_existing_col(df.columns, PROTOCOL_COLS)

    rows: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        sid = normalize_subject_id(row.get(sid_col, ""))
        if not SUBJECT_RE.fullmatch(sid):
            continue
        manufacturer = normalize_manufacturer(row.get(manufacturer_col, "")) if manufacturer_col else ""
        if not manufacturer and protocol_col:
            manufacturer = normalize_manufacturer(parse_manufacturer_from_protocol(row.get(protocol_col, "")))
        visit = normalize_visit(row.get(visit_col, "")) if visit_col else ""
        date = normalize_date(row.get(date_col, "")) if date_col else ""
        rows.append(
            {
                "SubjectID": sid,
                "source_metadata_file": source.path_label,
                "source_kind": source.source_kind,
                "source_priority": source.priority,
                "source_row_index": int(idx),
                "diagnosis": normalize_diagnosis(row.get(dx_col, "")) if dx_col else "",
                "ResearchGroup_Mapped": normalize_diagnosis(row.get(dx_col, "")) if dx_col else "",
                "ResearchGroup_Raw": normalize_scalar(row.get(dx_col, "")) if dx_col else "",
                "Age": normalize_scalar(row.get(age_col, "")) if age_col else "",
                "Sex": normalize_scalar(row.get(sex_col, "")) if sex_col else "",
                "Manufacturer": manufacturer,
                "Site": normalize_site(row.get(site_col, ""), sid) if site_col else normalize_site("", sid),
                "Site3": normalize_site(row.get(site_col, ""), sid) if site_col else normalize_site("", sid),
                "ImageID": normalize_image_id(row.get(image_col, "")) if image_col else "",
                "Visit": visit,
                "timepoint": visit,
                "StudyDate": date,
                "TR": normalize_scalar(row.get(tr_col, "")) if tr_col else "",
                "TE": normalize_scalar(row.get(te_col, "")) if te_col else "",
                "FieldStrength": normalize_scalar(row.get(field_col, "")) if field_col else "",
                "Description": normalize_scalar(row.get(desc_col, "")) if desc_col else "",
                "reason": normalize_scalar(row.get(reason_col, "")) if reason_col else "",
                "Phase": normalize_scalar(row.get(phase_col, "")) if phase_col else "",
            }
        )
    return pd.DataFrame(rows)


def metadata_quality(row: Mapping[str, Any]) -> int:
    score = 0
    for col in ["diagnosis", "Age", "Sex", "Manufacturer", "Site3"]:
        if normalize_scalar(row.get(col, "")):
            score += 2
    for col in ["ImageID", "Visit", "StudyDate", "TR", "Description"]:
        if normalize_scalar(row.get(col, "")):
            score += 1
    reason = normalize_scalar(row.get("reason", "")).lower()
    if "preferred" in reason:
        score += 4
    return score


def visit_rank(visit: Any) -> int:
    text = normalize_scalar(visit).lower()
    if "initial" in text or "init" in text:
        return 0
    if text in {"bl", "baseline"} or "baseline" in text:
        return 1
    if "screen" in text or text == "sc":
        return 2
    if text.startswith("v") and text[1:].isdigit():
        return 10 + int(text[1:])
    if "month" in text:
        match = re.search(r"(\d+)", text)
        if match:
            return 20 + int(match.group(1))
    return 50


def metadata_sort_key(row: Mapping[str, Any]) -> Tuple[int, int, int, str, float, int]:
    reason = normalize_scalar(row.get("reason", "")).lower()
    is_preferred = 0 if "preferred" in reason else 1
    source_priority = int(row.get("source_priority", 99))
    visit_priority = visit_rank(row.get("Visit", ""))
    date = normalize_scalar(row.get("StudyDate", "")) or "9999-99-99"
    image = safe_numeric(row.get("ImageID", ""))
    return (is_preferred, source_priority, visit_priority, date, image, -metadata_quality(row))


def choose_best_metadata(sub: pd.DataFrame) -> Dict[str, Any]:
    if sub.empty:
        return {}
    records = sub.to_dict(orient="records")
    records.sort(key=metadata_sort_key)
    # Break ties in favor of richer rows.
    best_rank = metadata_sort_key(records[0])[:4]
    tied = [row for row in records if metadata_sort_key(row)[:4] == best_rank]
    tied.sort(key=lambda row: -metadata_quality(row))
    return tied[0]


def build_metadata_sources(
    original_metadata: Path,
    v4_metadata: Path,
    adni_download_now: Path,
    extra_metadata: Sequence[Path],
    extract_root: Path,
    zip_path: Path,
    max_csv_mb: float,
) -> Tuple[List[MetadataSource], pd.DataFrame]:
    sources: List[MetadataSource] = []
    source_rows: List[Dict[str, Any]] = []
    static_sources = [
        (adni_download_now, "adni_download_now", 2),
        (original_metadata, "original_metadata", 3),
        (v4_metadata, "v4_metadata", 4),
    ]
    for path, kind, priority in static_sources:
        df = read_csv_if_exists(path, max_csv_mb)
        if df is None:
            source_rows.append({"source_metadata_file": str(path), "source_kind": kind, "status": "missing"})
            continue
        sources.append(MetadataSource(str(path), kind, priority, df))
        source_rows.append(
            {
                "source_metadata_file": str(path),
                "source_kind": kind,
                "status": "loaded",
                "rows": len(df),
                "columns": "|".join(map(str, df.columns)),
            }
        )
    for path in extra_metadata:
        df = read_csv_if_exists(path, max_csv_mb)
        if df is None:
            source_rows.append({"source_metadata_file": str(path), "source_kind": "extra_metadata", "status": "missing"})
            continue
        sources.append(MetadataSource(str(path), "extra_metadata", 5, df))
        source_rows.append(
            {
                "source_metadata_file": str(path),
                "source_kind": "extra_metadata",
                "status": "loaded",
                "rows": len(df),
                "columns": "|".join(map(str, df.columns)),
            }
        )
    for source in discover_extracted_metadata_csvs(extract_root, max_csv_mb):
        sources.append(source)
        source_rows.append(
            {
                "source_metadata_file": source.path_label,
                "source_kind": source.source_kind,
                "status": "loaded",
                "rows": len(source.dataframe),
                "columns": "|".join(map(str, source.dataframe.columns)),
            }
        )
    for source in discover_zip_metadata_csvs(zip_path, max_csv_mb):
        sources.append(source)
        source_rows.append(
            {
                "source_metadata_file": source.path_label,
                "source_kind": source.source_kind,
                "status": "loaded",
                "rows": len(source.dataframe),
                "columns": "|".join(map(str, source.dataframe.columns)),
            }
        )
    return sources, pd.DataFrame(source_rows)


def combine_metadata_hits(sources: Sequence[MetadataSource]) -> pd.DataFrame:
    frames = []
    for source in sources:
        norm = normalize_metadata_dataframe(source)
        if not norm.empty:
            frames.append(norm)
    if not frames:
        return pd.DataFrame()
    hits = pd.concat(frames, ignore_index=True, sort=False)
    hits["metadata_quality"] = hits.apply(metadata_quality, axis=1)
    return hits.sort_values(["SubjectID", "source_priority", "source_row_index"])


def subject_set_from_source(sources: Sequence[MetadataSource], source_kind: str) -> set[str]:
    out: set[str] = set()
    for source in sources:
        if source.source_kind != source_kind:
            continue
        norm = normalize_metadata_dataframe(source)
        if not norm.empty:
            out.update(norm["SubjectID"].astype(str).tolist())
    return out


def build_metadata_recovery_report(
    detected_subjects: pd.DataFrame,
    signal_files: pd.DataFrame,
    metadata_hits: pd.DataFrame,
    v4_subjects: set[str],
    original_subjects: set[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    signal_subjects = (
        sorted(signal_files.loc[signal_files["suffix"] == ".txt", "SubjectID"].dropna().astype(str).unique())
        if not signal_files.empty
        else []
    )
    subject_ids = sorted(set(detected_subjects["SubjectID"].astype(str).tolist()) | set(signal_subjects))
    sig_primary_source = "extracted" if (not signal_files.empty and (signal_files["source_kind"] == "extracted").any()) else "zip"
    for sid in subject_ids:
        sub_hits = metadata_hits[metadata_hits["SubjectID"] == sid].copy() if not metadata_hits.empty else pd.DataFrame()
        best = choose_best_metadata(sub_hits)
        sig = signal_files[(signal_files["SubjectID"] == sid) & (signal_files["source_kind"] == sig_primary_source)]
        if sig.empty and not signal_files.empty:
            sig = signal_files[signal_files["SubjectID"] == sid]
        txt = sig[sig["suffix"] == ".txt"] if not sig.empty else pd.DataFrame()
        valid_txt = False
        if not txt.empty:
            cols = pd.to_numeric(txt["n_cols"], errors="coerce").dropna().astype(int).tolist()
            valid_txt = bool(cols) and all(col in {131, 166, 170} for col in cols)
        already_v4 = sid in v4_subjects
        already_original = sid in original_subjects
        completely_new = not already_v4 and not already_original
        has_required_metadata = bool(
            normalize_scalar(best.get("ResearchGroup_Mapped", ""))
            and normalize_scalar(best.get("Age", ""))
            and normalize_scalar(best.get("Sex", ""))
        )
        candidate = bool(valid_txt and has_required_metadata)
        row = {col: "" for col in METADATA_OUTPUT_COLUMNS}
        row.update(
            {
                "SubjectID": sid,
                "diagnosis": best.get("diagnosis", ""),
                "ResearchGroup_Mapped": best.get("ResearchGroup_Mapped", ""),
                "Age": best.get("Age", ""),
                "Sex": best.get("Sex", ""),
                "Manufacturer": best.get("Manufacturer", ""),
                "Site": best.get("Site", ""),
                "Site3": best.get("Site3", ""),
                "ImageID": best.get("ImageID", ""),
                "Visit": best.get("Visit", ""),
                "timepoint": best.get("timepoint", ""),
                "StudyDate": best.get("StudyDate", ""),
                "source_metadata_file": best.get("source_metadata_file", "MISSING"),
                "all_metadata_sources": join_unique(sub_hits["source_metadata_file"].tolist() if not sub_hits.empty else []),
                "already_in_v4": already_v4,
                "already_in_original": already_original,
                "completely_new_subject": completely_new,
                "candidate_for_v5": candidate,
                "metadata_hit_count": int(len(sub_hits)),
                "has_valid_txt_signal": valid_txt,
                "txt_signal_count": int(len(txt)),
                "txt_shapes": join_unique(txt["shape"].tolist() if not txt.empty else []),
                "reason": best.get("reason", ""),
                "TR": best.get("TR", ""),
                "Description": best.get("Description", ""),
                "metadata_quality": best.get("metadata_quality", 0),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("SubjectID")


def build_new_vs_existing_subjects(metadata_report: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "SubjectID",
        "diagnosis",
        "ResearchGroup_Mapped",
        "Age",
        "Sex",
        "Manufacturer",
        "Site3",
        "ImageID",
        "Visit",
        "StudyDate",
        "already_in_v4",
        "already_in_original",
        "completely_new_subject",
        "candidate_for_v5",
        "source_metadata_file",
        "has_valid_txt_signal",
        "txt_signal_count",
        "txt_shapes",
        "reason",
    ]
    return metadata_report[[col for col in keep if col in metadata_report.columns]].copy()


def build_duplicate_subjects_or_timepoints(
    detected_subjects: pd.DataFrame,
    metadata_hits: pd.DataFrame,
    metadata_report: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    subject_ids = sorted(set(detected_subjects["SubjectID"].astype(str).tolist()))
    for sid in subject_ids:
        det = detected_subjects[detected_subjects["SubjectID"] == sid].iloc[0].to_dict()
        hits = metadata_hits[metadata_hits["SubjectID"] == sid].copy() if not metadata_hits.empty else pd.DataFrame()
        images = sorted({normalize_image_id(x) for x in hits.get("ImageID", pd.Series(dtype=str)).tolist() if normalize_image_id(x)})
        visits = sorted({normalize_scalar(x) for x in hits.get("Visit", pd.Series(dtype=str)).tolist() if normalize_scalar(x)})
        dates = sorted({normalize_scalar(x) for x in hits.get("StudyDate", pd.Series(dtype=str)).tolist() if normalize_scalar(x)})
        signal_problem = str(det.get("issue", "")) != "OK"
        metadata_problem = len(images) > 1 or len(dates) > 1
        if not signal_problem and not metadata_problem:
            continue
        best = choose_best_metadata(hits)
        report_row = (
            metadata_report[metadata_report["SubjectID"] == sid].iloc[0].to_dict()
            if not metadata_report[metadata_report["SubjectID"] == sid].empty
            else {}
        )
        rows.append(
            {
                "SubjectID": sid,
                "diagnosis": report_row.get("diagnosis", best.get("diagnosis", "")),
                "signal_issue": det.get("issue", ""),
                "txt_signal_count": det.get("txt_signal_count", 0),
                "mat_signal_count": det.get("mat_signal_count", 0),
                "metadata_hit_count": int(len(hits)),
                "distinct_image_ids": "|".join(images),
                "distinct_visits": "|".join(visits),
                "distinct_study_dates": "|".join(dates),
                "recommended_keep_image_id": best.get("ImageID", ""),
                "recommended_keep_visit": best.get("Visit", ""),
                "recommended_keep_study_date": best.get("StudyDate", ""),
                "recommended_keep_source": best.get("source_metadata_file", ""),
                "recommended_rule": "prefer metadata row marked preferred; otherwise initial/baseline, earliest StudyDate, then lowest ImageID",
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "SubjectID",
                "diagnosis",
                "signal_issue",
                "txt_signal_count",
                "mat_signal_count",
                "metadata_hit_count",
                "distinct_image_ids",
                "distinct_visits",
                "distinct_study_dates",
                "recommended_keep_image_id",
                "recommended_keep_visit",
                "recommended_keep_study_date",
                "recommended_keep_source",
                "recommended_rule",
            ]
        )
    return pd.DataFrame(rows).sort_values("SubjectID")


def build_preprocessing_provenance_report(
    zip_inventory: pd.DataFrame,
    extracted_inventory: pd.DataFrame,
    signal_files: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    frames = [("zip", zip_inventory), ("extracted", extracted_inventory)]
    for source_kind, df in frames:
        if df.empty:
            continue
        for _, row in df.iterrows():
            tokens = split_joined(row.get("provenance_tokens", ""))
            if not tokens:
                continue
            rows.append(
                {
                    "source_kind": source_kind,
                    "path": row.get("path", ""),
                    "evidence_type": "path_token",
                    "matched_terms": "|".join(tokens),
                    "interpretation": provenance_interpretation(tokens),
                }
            )
    signal_tokens = join_unique(signal_files.get("provenance_tokens", pd.Series(dtype=str)).tolist() if not signal_files.empty else [])
    n_txt = int((signal_files["suffix"] == ".txt").sum()) if not signal_files.empty else 0
    n_mat = int((signal_files["suffix"] == ".mat").sum()) if not signal_files.empty else 0
    rows.append(
        {
            "source_kind": "summary",
            "path": "",
            "evidence_type": "signal_inventory",
            "matched_terms": signal_tokens,
            "interpretation": (
                f"Detected {n_txt} ROI signal txt files and {n_mat} mat files. "
                "Paths containing ROISignals_AAL3_FunImgARWSDCFN are consistent with DPARSF-preprocessed AAL3 ROI signals."
            ),
        }
    )
    rows.append(
        {
            "source_kind": "summary",
            "path": "",
            "evidence_type": "python_bandpass_policy",
            "matched_terms": "FunImgARWSDCFN|DPARSF_prefix_ARWSDCFN|likely_filtered_F_component",
            "interpretation": (
                "Python bandpass should be OFF for this dataset. Martin described these as passband/pasabandas data, "
                "and prior controls showed DPARSF-only vs DPARSF+Python-bandpass can strongly change predictions."
            ),
        }
    )
    return pd.DataFrame(rows)


def provenance_interpretation(tokens: Sequence[str]) -> str:
    token_set = {token.lower() for token in tokens}
    parts = []
    if "roisignals" in token_set:
        parts.append("AAL3 ROI signal artifact")
    if "funimg" in token_set:
        parts.append("DPARSF functional-image derivative naming")
    if "arwsdcfn" in token_set or "dparsf_prefix_arwsdcfn" in token_set:
        parts.append("DPARSF ARWSDCFN prefix; F is compatible with filtering/bandpass")
    if "bandpass" in token_set or "passband" in token_set or "pasabandas" in token_set:
        parts.append("explicit passband/bandpass naming")
    return "; ".join(parts) if parts else "path-level provenance token"


def duplicate_file_report(zip_inventory: pd.DataFrame, extracted_inventory: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if not zip_inventory.empty:
        for (size, crc), sub in zip_inventory[~zip_inventory["is_dir"]].groupby(["file_size_bytes", "crc32"], dropna=False):
            if len(sub) > 1:
                rows.append(
                    {
                        "source_kind": "zip",
                        "duplicate_key": f"size={size};crc32={crc}",
                        "n_files": len(sub),
                        "paths": join_unique(sub["path"].tolist()),
                    }
                )
    if not extracted_inventory.empty:
        for sha, sub in extracted_inventory.groupby("sha256", dropna=False):
            if len(sub) > 1:
                rows.append(
                    {
                        "source_kind": "extracted",
                        "duplicate_key": f"sha256={sha}",
                        "n_files": len(sub),
                        "paths": join_unique(sub["path"].tolist()),
                    }
                )
    return pd.DataFrame(rows)


def create_local_symlink(local_symlink: Optional[Path], extract_root: Path) -> Dict[str, Any]:
    if local_symlink is None:
        return {"requested": False, "status": "not_requested", "path": "", "target": str(extract_root)}
    local_symlink = resolve_path(local_symlink) or local_symlink
    if not extract_root.exists():
        return {
            "requested": True,
            "status": "not_created_target_missing",
            "path": str(local_symlink),
            "target": str(extract_root),
        }
    if local_symlink.exists() or local_symlink.is_symlink():
        if local_symlink.is_symlink() and local_symlink.resolve() == extract_root.resolve():
            return {"requested": True, "status": "already_correct", "path": str(local_symlink), "target": str(extract_root)}
        raise RuntimeError(f"Local symlink path exists but does not point to extract root: {local_symlink}")
    local_symlink.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(extract_root, local_symlink)
    return {"requested": True, "status": "created", "path": str(local_symlink), "target": str(extract_root)}


def count_by_diagnosis(metadata_report: pd.DataFrame, candidate_only: bool = False) -> Dict[str, int]:
    df = metadata_report.copy()
    if candidate_only and "candidate_for_v5" in df.columns:
        df = df[df["candidate_for_v5"].astype(bool)]
    counts = {key: 0 for key in ["AD", "CN", "MCI", "UNKNOWN"]}
    for diag in df.get("diagnosis", pd.Series(dtype=str)).map(normalize_diagnosis):
        if diag in {"AD", "CN", "MCI"}:
            counts[diag] += 1
        else:
            counts["UNKNOWN"] += 1
    return counts


def count_metadata_subjects(path: Path) -> int:
    df = read_csv_if_exists(path, max_csv_mb=500.0)
    if df is None:
        return 0
    sid_col = first_existing_col(df.columns, SUBJECT_COLS)
    if sid_col is None:
        return int(len(df))
    return int(df[sid_col].map(normalize_subject_id).replace("", np.nan).dropna().nunique())


def expected_new_ad_subjects(metadata_hits: pd.DataFrame, v4_subjects: set[str]) -> List[str]:
    if metadata_hits.empty:
        return []
    adni = metadata_hits[metadata_hits["source_kind"] == "adni_download_now"].copy()
    if adni.empty:
        return []
    adni["diagnosis_norm"] = adni["diagnosis"].map(normalize_diagnosis)
    adni = adni[(adni["diagnosis_norm"] == "AD") & (~adni["SubjectID"].isin(v4_subjects))]
    preferred = adni[adni["reason"].str.contains("preferred", case=False, na=False)].copy()
    source = preferred if not preferred.empty else adni
    return sorted(source["SubjectID"].dropna().astype(str).unique().tolist())


def format_count_dict(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{key}={counts.get(key, 0)}" for key in ["AD", "CN", "MCI", "UNKNOWN"])


def compute_safety(metadata_report: pd.DataFrame, detected_subjects: pd.DataFrame, provenance: pd.DataFrame) -> Tuple[bool, List[str]]:
    blockers: List[str] = []
    candidate = metadata_report[metadata_report["candidate_for_v5"].astype(bool)] if not metadata_report.empty else pd.DataFrame()
    if candidate.empty:
        blockers.append("No candidate_for_v5 subjects with valid ROI signal txt and required metadata.")
    bad_inventory = detected_subjects[detected_subjects["issue"] != "OK"] if not detected_subjects.empty else pd.DataFrame()
    blocking_bad = bad_inventory[
        bad_inventory["has_txt_signal"].astype(bool)
        & bad_inventory["issue"].str.contains("multiple_txt|unexpected_txt_cols|no_roi", na=False)
    ] if not bad_inventory.empty else pd.DataFrame()
    if not blocking_bad.empty:
        blockers.append(f"{len(blocking_bad)} detected subjects have blocking ROI signal inventory issues.")
    missing_required = metadata_report[
        metadata_report["has_valid_txt_signal"].astype(bool)
        & (
            metadata_report["diagnosis"].astype(str).eq("")
            | metadata_report["Age"].astype(str).eq("")
            | metadata_report["Sex"].astype(str).eq("")
        )
    ] if not metadata_report.empty else pd.DataFrame()
    if not missing_required.empty:
        blockers.append(f"{len(missing_required)} subjects with valid txt signals are missing diagnosis/Age/Sex metadata.")
    prov_text = " ".join(provenance.get("matched_terms", pd.Series(dtype=str)).astype(str).tolist()).lower()
    if "arwsdcf" not in prov_text and "bandpass" not in prov_text and "passband" not in prov_text:
        blockers.append("No strong DPARSF/passband provenance token was detected.")
    return not blockers, blockers


def proposed_feature_command(extract_root: Path, output_root: Path, local_tensor_symlink: Path) -> str:
    return " ".join(
        [
            sh_quote(sys.executable),
            "scripts/revision_bspc_2026/build_adni_expanded_v5_passband_dparsf_only.py",
            "--input-root",
            sh_quote(str(extract_root)),
            "--metadata",
            "results/revision_bspc_2026/adni_passband_20260510_audit/metadata_recovery_report.csv",
            "--dataset-name",
            "adni_expanded_v5_passband_dparsf_only",
            "--output-root",
            sh_quote(str(output_root)),
            "--local-symlink",
            sh_quote(str(local_tensor_symlink)),
            "--aal3-roi-metadata",
            "data/ROI_MNI_V7_vol.txt",
            "--aal3-manual-network-order",
            "data/aal3_131_manual_network_order.csv",
            "--tr",
            "3.0",
            "--target-len",
            "140",
            "--channels",
            "all7",
            "--python-bandpass",
            "off",
            "--dry-run",
        ]
    )


def audit_extract_command(args: argparse.Namespace) -> str:
    parts = [
        sh_quote(sys.executable),
        "scripts/revision_bspc_2026/audit_onedrive_20260510_passband_zip.py",
        "--zip",
        sh_quote(str(args.zip_path)),
        "--extract-root",
        sh_quote(str(args.extract_root)),
    ]
    if args.local_symlink:
        parts.extend(["--local-symlink", sh_quote(str(args.local_symlink))])
    parts.extend(
        [
            "--v4-metadata",
            sh_quote(str(args.v4_metadata)),
            "--original-metadata",
            sh_quote(str(args.original_metadata)),
            "--adni-download-now",
            sh_quote(str(args.adni_download_now)),
            "--output-dir",
            sh_quote(str(args.output_dir)),
            "--overwrite",
            "--extract",
        ]
    )
    return " ".join(parts)


def sh_quote(text: str) -> str:
    if not text:
        return "''"
    if re.fullmatch(r"[A-Za-z0-9_./:=+-]+", text):
        return text
    return "'" + text.replace("'", "'\"'\"'") + "'"


def write_feature_extraction_plan(
    path: Path,
    extract_root: Path,
    metadata_report: pd.DataFrame,
    safe_to_compute: bool,
    blockers: Sequence[str],
    v4_subject_count: int,
    final_subject_list_confirmed: bool,
) -> None:
    output_root = DEFAULT_V5_OUTPUT_ROOT
    local_symlink = Path("data/revision_bspc_2026/adni_expanded_v5_passband_dparsf_only")
    command = proposed_feature_command(extract_root, output_root, local_symlink)
    counts = count_by_diagnosis(metadata_report, candidate_only=True)
    lines = [
        "# Feature Extraction Plan: adni_expanded_v5_passband_dparsf_only",
        "",
        "Prepared only. Do not run connectivity computation or training from this audit step.",
        "",
        "## Dataset",
        "",
        "- Dataset name: `adni_expanded_v5_passband_dparsf_only`",
        f"- Input ROI root: `{extract_root}`",
        f"- Output tensor root: `{output_root}`",
        f"- Local symlink: `{local_symlink}`",
        f"- Candidate subjects in audit: `{int(metadata_report['candidate_for_v5'].astype(bool).sum()) if not metadata_report.empty else 0}`",
        f"- Candidate diagnosis counts: `{format_count_dict(counts)}`",
        f"- v4 subject count for scope comparison: `{v4_subject_count}`",
        "- Scope status: this zip is a 60-subject passband batch/source package, not the full final training dataset.",
        "",
        "## Preprocessing Contract",
        "",
        "- Python bandpass: `OFF`",
        "- Rationale: input is Martin passband/DPARSF-derived ROI signals; applying the historical Python 0.01-0.08 Hz bandpass would mix filtering regimes.",
        "- Rebuild rule: recompute connectivity matrices for every subject in the confirmed final training dataset using this same DPARSF-bandpass-only recipe.",
        "- No-mix rule: do not combine old tensors/matrices generated with Python bandpass ON with new DPARSF-only matrices.",
        "- TR: `3.0`",
        "- target_len: `140`",
        "- ROI reduction/reorder: `170 -> 131` using `data/ROI_MNI_V7_vol.txt` and `data/aal3_131_manual_network_order.csv`",
        "- Channels: compute all 7 existing channels in historical order:",
    ]
    lines.extend(f"  - `{idx}`: `{name}`" for idx, name in enumerate(CHANNEL_NAMES_MASTER))
    lines.extend(
        [
            "- Downstream model selection can still select `[4, 1, 0]`; the v5 tensor should retain all channels.",
            "- No model training in this step.",
            "",
            "## Validation Before Running",
            "",
            "- Confirm `metadata_recovery_report.csv` has no missing diagnosis/Age/Sex for candidate subjects.",
            "- Confirm `detected_signal_files.csv` txt shapes are compatible with 170 ROI input.",
            "- Confirm duplicate/timepoint policy in `duplicate_subjects_or_timepoints.csv`.",
            "- Confirm the complete final training subject list. This must cover all retained v4 subjects plus confirmed additions/removals, not just the 60 subjects in this zip.",
            "- Confirm DPARSF-bandpass ROI signals are available for every subject in that final list.",
            "- Run the builder first with `--dry-run`; only then run the real feature extraction.",
            "",
            "## Safety Status",
            "",
            f"- Import audit passed: `{'YES' if safe_to_compute else 'NO'}`",
            f"- Final subject list confirmed: `{'YES' if final_subject_list_confirmed else 'NO'}`",
            f"- Safe to compute connectivity from this audit state: `{'YES' if safe_to_compute and final_subject_list_confirmed else 'NO'}`",
        ]
    )
    if blockers:
        lines.extend(f"- Blocker: {item}" for item in blockers)
    lines.extend(["", "## Exact Next Command", "", "```bash", command, "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_readme(
    path: Path,
    args: argparse.Namespace,
    zip_inventory: pd.DataFrame,
    extracted_inventory: pd.DataFrame,
    detected_subjects: pd.DataFrame,
    signal_files: pd.DataFrame,
    duplicates: pd.DataFrame,
    metadata_report: pd.DataFrame,
    metadata_hits: pd.DataFrame,
    provenance: pd.DataFrame,
    symlink_status: Mapping[str, Any],
    safe_to_compute: bool,
    blockers: Sequence[str],
    v4_subject_count: int,
    final_subject_list_confirmed: bool,
    extraction_error: str = "",
) -> None:
    signal_subjects = sorted(
        signal_files.loc[signal_files["suffix"] == ".txt", "SubjectID"].dropna().astype(str).unique().tolist()
    ) if not signal_files.empty else []
    detected_n = len(signal_subjects)
    counts = count_by_diagnosis(metadata_report, candidate_only=False)
    candidate_counts = count_by_diagnosis(metadata_report, candidate_only=True)
    detected_ad_df = metadata_report[
        metadata_report["SubjectID"].isin(signal_subjects)
        & (metadata_report["diagnosis"].map(normalize_diagnosis) == "AD")
    ] if not metadata_report.empty else pd.DataFrame()
    detected_ad_subjects = sorted(detected_ad_df["SubjectID"].astype(str).unique().tolist()) if not detected_ad_df.empty else []
    new_df = metadata_report[metadata_report["completely_new_subject"].astype(bool)] if not metadata_report.empty else pd.DataFrame()
    new_count = int(len(new_df))
    new_ad_df = new_df[new_df["diagnosis"].map(normalize_diagnosis) == "AD"] if not new_df.empty else pd.DataFrame()
    new_ad_count = int(len(new_ad_df))
    v4_subjects = set(
        metadata_hits.loc[metadata_hits["source_kind"] == "v4_metadata", "SubjectID"].dropna().astype(str).tolist()
    ) if not metadata_hits.empty else set()
    expected_ad = expected_new_ad_subjects(metadata_hits, v4_subjects)
    present_expected_ad = sorted(set(expected_ad).intersection(signal_subjects))
    missing_expected_ad = sorted(set(expected_ad) - set(signal_subjects))
    if extraction_error:
        extracted_state = f"FAILED: {extraction_error}"
    else:
        extracted_state = "YES" if args.extract and not extracted_inventory.empty else ("ALREADY_PRESENT" if not extracted_inventory.empty else "NO")
    provenance_text = " ".join(provenance.get("interpretation", pd.Series(dtype=str)).astype(str).tolist())
    dparsf_compatible = "YES" if "DPARSF" in provenance_text or "ARWSDCF" in provenance_text else "UNCLEAR"
    if extraction_error or (not args.extract and extracted_inventory.empty):
        next_command = audit_extract_command(args)
    elif safe_to_compute and final_subject_list_confirmed:
        next_command = proposed_feature_command(
            args.extract_root,
            DEFAULT_V5_OUTPUT_ROOT,
            Path("data/revision_bspc_2026/adni_expanded_v5_passband_dparsf_only"),
        )
    else:
        next_command = "Confirm the full final subject list and DPARSF-bandpass ROI availability for every retained subject; then rerun this audit with --final-subject-list-confirmed."

    lines = [
        "# ADNI Passband 2026-05-10 Audit",
        "",
        "No connectivity computation, tensor generation, model inference, or training was performed.",
        "",
        "## Explicit Answers",
        "",
        f"- Zip entries listed: `{len(zip_inventory)}`",
        f"- Extraction requested: `{'YES' if args.extract else 'NO'}`",
        f"- Extracted payload inventoried: `{extracted_state}`",
        f"- Subjects detected from ROI `.txt` signals: `{detected_n}`",
        f"- AD/CN/MCI/UNKNOWN counts across recovered subject metadata: `{format_count_dict(counts)}`",
        f"- Candidate-for-v5 AD/CN/MCI/UNKNOWN counts: `{format_count_dict(candidate_counts)}`",
        f"- Is this the full final dataset or only a batch? `ONLY_A_BATCH`: v4 has `{v4_subject_count}` subjects, while this zip has `{detected_n}` ROI-signal subjects.",
        f"- Detected AD subjects after resolving metadata/timepoints: `{len(detected_ad_subjects)}` (`{', '.join(detected_ad_subjects) if detected_ad_subjects else 'none'}`)",
        f"- Completely new subjects relative to v4 and original metadata: `{new_count}`",
        f"- Completely new AD subjects after metadata reconciliation: `{new_ad_count}`",
        f"- Expected new AD subjects from `adni_download_now.csv`: `{', '.join(expected_ad) if expected_ad else 'none_detected_in_metadata'}`",
        f"- Expected new AD subjects present in ROI signals: `{', '.join(present_expected_ad) if present_expected_ad else 'none'}`",
        f"- Expected new AD subjects missing from ROI signals: `{', '.join(missing_expected_ad) if missing_expected_ad else 'none'}`",
        "- Is `301_S_6592` intended as AD? `YES`: it is AD in original/v4 metadata and is already in v4/original, so it is not a new AD addition.",
        "- Is `114_S_6039` missing? `YES`: it is expected from `adni_download_now.csv` but has no ROI signal file in this zip.",
        f"- Is it 3 or 4 new AD subjects after resolving timepoints? `{len(detected_ad_subjects)}` detected AD subject(s) total; `{new_ad_count}` are completely new relative to v4/original. See `duplicate_subjects_or_timepoints.csv` for timepoint resolution.",
        f"- Which timepoint should be kept if there are duplicates? `preferred` rows from metadata first; otherwise initial/baseline, earliest StudyDate, then lowest ImageID. Concrete recommendations are in `duplicate_subjects_or_timepoints.csv`.",
        f"- Compatible with DPARSF-bandpass-only feature extraction? `{dparsf_compatible}`",
        "- Should Python bandpass be disabled? `YES`",
        f"- Import audit passed? `{'YES' if safe_to_compute else 'NO'}`",
        f"- Final subject list confirmed? `{'YES' if final_subject_list_confirmed else 'NO'}`",
        f"- Is it safe to compute connectivity now? `{'YES' if safe_to_compute and final_subject_list_confirmed else 'NO'}`",
    ]
    if blockers:
        lines.extend(f"  Blocker: {item}" for item in blockers)
    lines.extend(["", "## Exact Next Command", "", "```bash", next_command, "```", ""])
    lines.extend(
        [
            "## Key Inputs",
            "",
            f"- Zip: `{args.zip_path}`",
            f"- Extract root: `{args.extract_root}`",
            f"- Local symlink status: `{json.dumps(dict(symlink_status), sort_keys=True)}`",
            f"- v4 metadata: `{args.v4_metadata}`",
            f"- original metadata: `{args.original_metadata}`",
            f"- ADNI download metadata: `{args.adni_download_now}`",
            f"- Auxiliary metadata: `{', '.join(str(p) for p in args.extra_metadata) if getattr(args, 'extra_metadata', None) else 'none'}`",
            "",
            "## Output Tables",
            "",
            "- `zip_inventory.csv`: zip listing, sizes, CRCs, subject IDs, and path-level provenance tokens.",
            "- `extracted_file_inventory.csv`: recursive inventory of extracted files when available.",
            "- `detected_subjects.csv`: subject-level file and ROI signal counts.",
            "- `detected_signal_files.csv`: ROI signal files with shape detection where possible.",
            "- `duplicate_subjects_or_timepoints.csv`: duplicate signal/timepoint audit and keep recommendation.",
            "- `preprocessing_provenance_report.csv`: DPARSF/passband evidence and Python-bandpass policy.",
            "- `metadata_recovery_report.csv`: reconciled per-subject metadata and v5 candidacy flags.",
            "- `new_vs_existing_subjects.csv`: compact new/existing subject comparison.",
            "- `feature_extraction_plan.md`: prepared v5 extraction plan; not executed.",
            "",
            "## Interpretation",
            "",
            "The relevant provenance signal is the `ResultsAAL3/ROISignals_AAL3_FunImgARWSDCFN` path pattern in Martin's zip. "
            "Together with Martin's passband/pasabandas note and the prior DPARSF-only versus DPARSF+Python-bandpass control, "
            "this audit treats the v5 dataset as DPARSF-bandpass-only input and explicitly keeps Python bandpass OFF.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    output_dir: Path,
    tables: Mapping[str, pd.DataFrame],
    json_payloads: Mapping[str, Mapping[str, Any]],
) -> None:
    for name, df in tables.items():
        df.to_csv(output_dir / name, index=False)
    for name, payload in json_payloads.items():
        (output_dir / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.zip_path = resolve_path(args.zip_path)
    args.extract_root = resolve_path(args.extract_root)
    args.local_symlink = resolve_path(args.local_symlink)
    args.v4_metadata = resolve_path(args.v4_metadata)
    args.original_metadata = resolve_path(args.original_metadata)
    args.adni_download_now = resolve_path(args.adni_download_now)
    args.extra_metadata = [resolve_path(path) for path in args.extra_metadata]
    args.output_dir = resolve_path(args.output_dir)

    assert args.zip_path is not None
    assert args.extract_root is not None
    assert args.output_dir is not None
    assert args.v4_metadata is not None
    assert args.original_metadata is not None
    assert args.adni_download_now is not None

    if not args.zip_path.exists():
        raise FileNotFoundError(f"Input zip not found: {args.zip_path}")

    prepare_output_dir(args.output_dir, args.overwrite)
    zip_inventory = list_zip_contents(args.zip_path)
    extraction_manifest = pd.DataFrame()
    extraction_error = ""
    if args.extract:
        try:
            extraction_manifest = extract_zip_safely(
                args.zip_path,
                args.extract_root,
                args.overwrite_extracted_files,
                args.allow_home_extract,
            )
        except Exception as exc:
            extraction_error = str(exc)
            extraction_manifest = pd.DataFrame(
                [
                    {
                        "path": "",
                        "target": str(args.extract_root),
                        "status": "failed",
                        "error": extraction_error,
                    }
                ]
            )

    extracted_inventory = inventory_extracted_files(args.extract_root)
    symlink_status = create_local_symlink(args.local_symlink, args.extract_root)

    signal_files = build_detected_signal_files(
        args.zip_path,
        zip_inventory,
        extracted_inventory,
        inspect_zip_shapes=not args.skip_zip_shape_inspection,
        max_zip_shape_mb=args.max_zip_shape_mb,
    )
    detected_subjects = build_detected_subjects(zip_inventory, extracted_inventory, signal_files)

    sources, metadata_sources_report = build_metadata_sources(
        args.original_metadata,
        args.v4_metadata,
        args.adni_download_now,
        [path for path in args.extra_metadata if path is not None],
        args.extract_root,
        args.zip_path,
        args.max_csv_mb,
    )
    metadata_hits = combine_metadata_hits(sources)
    v4_subjects = subject_set_from_source(sources, "v4_metadata")
    original_subjects = subject_set_from_source(sources, "original_metadata")
    metadata_report = build_metadata_recovery_report(
        detected_subjects,
        signal_files,
        metadata_hits,
        v4_subjects,
        original_subjects,
    )
    new_vs_existing = build_new_vs_existing_subjects(metadata_report)
    duplicates = build_duplicate_subjects_or_timepoints(detected_subjects, metadata_hits, metadata_report)
    provenance = build_preprocessing_provenance_report(zip_inventory, extracted_inventory, signal_files)
    duplicate_files = duplicate_file_report(zip_inventory, extracted_inventory)
    safe_to_compute, blockers = compute_safety(metadata_report, detected_subjects, provenance)
    if extraction_error:
        blockers = [f"Extraction failed: {extraction_error}"] + list(blockers)
        safe_to_compute = False
    elif args.extract and extracted_inventory.empty:
        blockers = ["Extraction was requested but no extracted files were inventoried."] + list(blockers)
        safe_to_compute = False
    v4_subject_count = count_metadata_subjects(args.v4_metadata)
    connectivity_safe = bool(safe_to_compute and args.final_subject_list_confirmed)
    reported_blockers = list(blockers)
    if not args.final_subject_list_confirmed:
        reported_blockers.append(
            "Final subject list is not confirmed; do not compute connectivity for v5 yet."
        )

    write_outputs(
        args.output_dir,
        {
            "zip_inventory.csv": zip_inventory,
            "extracted_file_inventory.csv": extracted_inventory,
            "detected_subjects.csv": detected_subjects,
            "detected_signal_files.csv": signal_files,
            "duplicate_subjects_or_timepoints.csv": duplicates,
            "preprocessing_provenance_report.csv": provenance,
            "metadata_recovery_report.csv": metadata_report,
            "new_vs_existing_subjects.csv": new_vs_existing,
            "metadata_raw_hits.csv": metadata_hits,
            "metadata_source_inventory.csv": metadata_sources_report,
            "duplicate_file_report.csv": duplicate_files,
            "extraction_manifest.csv": extraction_manifest,
        },
        {
            "audit_manifest.json": {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "zip": str(args.zip_path),
                "extract": bool(args.extract),
                "extract_root": str(args.extract_root),
                "local_symlink": str(args.local_symlink) if args.local_symlink else "",
                "symlink_status": dict(symlink_status),
                "output_dir": str(args.output_dir),
                "import_audit_passed": bool(safe_to_compute),
                "final_subject_list_confirmed": bool(args.final_subject_list_confirmed),
                "safe_to_compute_connectivity": connectivity_safe,
                "blockers": reported_blockers,
                "extraction_error": extraction_error,
                "python_bandpass": "OFF",
                "training_run": False,
                "connectivity_computation_run": False,
            }
        },
    )
    write_feature_extraction_plan(
        args.output_dir / "feature_extraction_plan.md",
        args.extract_root,
        metadata_report,
        safe_to_compute,
        reported_blockers,
        v4_subject_count,
        args.final_subject_list_confirmed,
    )
    write_readme(
        args.output_dir / "README.md",
        args,
        zip_inventory,
        extracted_inventory,
        detected_subjects,
        signal_files,
        duplicates,
        metadata_report,
        metadata_hits,
        provenance,
        symlink_status,
        safe_to_compute,
        reported_blockers,
        v4_subject_count,
        args.final_subject_list_confirmed,
        extraction_error,
    )

    signal_subjects = sorted(
        signal_files.loc[signal_files["suffix"] == ".txt", "SubjectID"].dropna().astype(str).unique().tolist()
    ) if not signal_files.empty else []
    counts = count_by_diagnosis(metadata_report, candidate_only=False)
    print(f"Wrote audit outputs to {args.output_dir}")
    print(f"Zip entries: {len(zip_inventory)}")
    print(f"ROI txt subjects detected: {len(signal_subjects)}")
    print(f"Recovered diagnosis counts: {format_count_dict(counts)}")
    print("Python bandpass: OFF")
    print(f"Import audit passed: {'YES' if safe_to_compute else 'NO'}")
    print(f"Final subject list confirmed: {'YES' if args.final_subject_list_confirmed else 'NO'}")
    print(f"Safe to compute connectivity now: {'YES' if connectivity_safe else 'NO'}")
    if reported_blockers:
        print("Blockers:")
        for item in reported_blockers:
            print(f"- {item}")
    return 1 if extraction_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
