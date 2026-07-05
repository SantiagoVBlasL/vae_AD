#!/usr/bin/env python3
"""Build OASIS-3 TR~2.2 external-validation candidate manifests.

This is a metadata/header audit only. It does not preprocess, calculate
connectivity, copy large files, train, or modify source data.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import struct
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "oasis3_external_validation_tr2_candidate_manifest"
)
DEFAULT_METADATA_RAW = Path("/media/diego/Datos/vae_AD_data/OASIS3/metadata_raw")
DEFAULT_MR_JSON = DEFAULT_METADATA_RAW / "imported" / "OASIS3_MR_json.csv"
DEFAULT_ELIGIBILITY = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "oasis3_external_validation_cohort_sizing"
    / "oasis3_external_validation_eligibility_subjects.csv"
)
DEFAULT_REST_CLINICAL = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "oasis3_clinical_mapping_audit"
    / "oasis3_rest_bold_clinical_candidates.csv"
)
DEFAULT_SEARCH_ROOTS = [
    PROJECT_ROOT / "data" / "oasis3",
    Path("/media/diego/Datos"),
    Path("/media/diego/My_Book_Diego"),
]
DEFAULT_DATA_SEARCH_ROOT = PROJECT_ROOT / "data"

TR2_MIN = 2.15
TR2_MAX = 2.25
TR_NEAR_MIN = 2.0
TR_NEAR_MAX = 2.5
TR3_MIN = 2.95
TR3_MAX = 3.05

MAIN_COLUMNS = [
    "subject_id",
    "session_id",
    "bids_session",
    "acquisition",
    "run",
    "task",
    "TR_seconds",
    "Manufacturer",
    "ScannerModel",
    "SeriesDescription",
    "n_volumes",
    "diagnosis",
    "diagnosis_confidence",
    "CDRTOT",
    "CDRSUM",
    "clinical_delta_days",
    "age_at_MR",
    "sex",
    "session_rank_per_subject",
    "is_first_mr_visit",
    "is_first_usable_tr2_visit",
    "bold_identifier",
    "local_bold_path_available",
    "local_json_path_available",
    "selected_local_or_metadata_path",
    "experiment_id",
    "label_source",
    "selection_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build OASIS-3 TR~2.2 AD/CN external-validation candidate manifests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mr-json", type=Path, default=DEFAULT_MR_JSON)
    parser.add_argument("--eligibility", type=Path, default=DEFAULT_ELIGIBILITY)
    parser.add_argument("--rest-clinical", type=Path, default=DEFAULT_REST_CLINICAL)
    parser.add_argument("--metadata-raw", type=Path, default=DEFAULT_METADATA_RAW)
    parser.add_argument("--data-search-root", type=Path, default=DEFAULT_DATA_SEARCH_ROOT)
    parser.add_argument("--search-root", type=Path, action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null", "."} else text


def parse_float(value: Any) -> Optional[float]:
    text = clean(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def tr_seconds(value: Any) -> Optional[float]:
    num = parse_float(value)
    if num is None:
        return None
    if 100 <= num <= 10000:
        return num / 1000.0
    return num


def normalize_subject(value: Any) -> str:
    text = clean(value)
    if not text:
        return ""
    if text.startswith("sub-"):
        return text
    match = re.search(r"(OAS3\d{4,})", text)
    if match:
        return f"sub-{match.group(1)}"
    return text


def raw_subject(subject_id: str) -> str:
    return subject_id.replace("sub-", "", 1)


def parse_session_from_filename(filename: str, subject_id: str) -> Tuple[str, str]:
    match = re.search(r"_(ses|sess)-d(\d+)", filename)
    if not match:
        return "", ""
    bids_session = f"{match.group(1)}-d{match.group(2)}"
    return f"{raw_subject(subject_id)}_MR_d{match.group(2)}", bids_session


def parse_task(filename: str, scan_category: Any, series_description: Any) -> str:
    match = re.search(r"_task-([^_]+)", filename)
    if match:
        return match.group(1)
    category = clean(scan_category)
    if "rest" in category.lower():
        return "rest"
    desc = clean(series_description).lower()
    if "rest" in desc or "connect" in desc:
        return "rest"
    return category


def parse_run(filename: str, acquisition_number: Any) -> str:
    match = re.search(r"_run-([0-9]+)", filename)
    if match:
        return match.group(1)
    return clean(acquisition_number)


def parse_acquisition(filename: str) -> str:
    match = re.search(r"_acq-([^_]+)", filename)
    return match.group(1) if match else ""


def bool_from_any(value: Any) -> bool:
    return clean(value).lower() in {"true", "1", "yes", "y"}


def is_rest_task(task: Any, filename: Any, scan_category: Any, series_description: Any) -> bool:
    text = " ".join(clean(v).lower() for v in [task, filename, scan_category, series_description])
    return "task-rest" in text or "bold-rest" in text or "rest" == clean(task).lower() or "connect" in text


def normalize_diagnosis(value: Any) -> str:
    text = clean(value).upper()
    if text in {"CN", "AD_DEMENTIA", "MCI", "UNKNOWN"}:
        return text
    if text == "AD":
        return "AD_DEMENTIA"
    if "DEMENT" in text or "ALZ" in text:
        return "AD_DEMENTIA"
    if "CONTROL" in text or "NORMAL" in text:
        return "CN"
    if "MCI" in text:
        return "MCI"
    return "UNKNOWN"


def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {path}. Pass --overwrite.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)


def find_matching_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    patterns = [
        "*participants.tsv",
        "*participants.csv",
        "*sessions.tsv",
        "*sessions.csv",
        "*scans.tsv",
        "*scans.csv",
        "*_bold.json",
        "*_bold.nii",
        "*_bold.nii.gz",
        "*oasis*.csv",
        "*oasis*.tsv",
        "*oasis*.json",
        "*OASIS*.csv",
        "*OASIS*.tsv",
        "*OASIS*.json",
    ]
    cmd = ["find", str(root), "-type", "f", "("]
    for idx, pattern in enumerate(patterns):
        if idx:
            cmd.append("-o")
        cmd.extend(["-iname", pattern])
    cmd.append(")")
    completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if completed.returncode not in {0, 1}:
        return []
    return [Path(line) for line in completed.stdout.splitlines() if line.strip()]


def classify_file(path: Path) -> str:
    name = path.name.lower()
    if name.endswith("_bold.nii.gz") or name.endswith("_bold.nii"):
        return "bold_nifti"
    if name.endswith("_bold.json"):
        return "bold_json_sidecar"
    if "participants" in name and name.endswith((".tsv", ".csv")):
        return "participants_table"
    if "sessions" in name and name.endswith((".tsv", ".csv")):
        return "sessions_table"
    if "scans" in name and name.endswith((".tsv", ".csv")):
        return "scans_table"
    if any(token in name for token in ["cdr", "diagnos", "cognitive", "psychometric", "demographic", "uds"]):
        return "clinical_cognitive_metadata"
    if "mr_json" in name:
        return "mr_json_metadata"
    if "oasis" in name:
        return "oasis_metadata"
    return "other"


def inventory_files(search_roots: Sequence[Path], data_search_root: Path) -> pd.DataFrame:
    roots: List[Path] = []
    for root in list(search_roots) + [data_search_root]:
        try:
            resolved = root.resolve()
        except Exception:
            resolved = root
        if resolved not in roots:
            roots.append(resolved)
    records: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for root in roots:
        for path in find_matching_files(root):
            key = str(path.resolve()) if path.exists() else str(path)
            if key in seen:
                continue
            seen.add(key)
            try:
                size = path.stat().st_size
            except OSError:
                size = ""
            records.append(
                {
                    "path": str(path),
                    "root": str(root),
                    "file_name": path.name,
                    "file_kind": classify_file(path),
                    "size_bytes": size,
                }
            )
    return pd.DataFrame(records)


def lookup_by_basename(files: pd.DataFrame, kind: Optional[str] = None) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    if files.empty:
        return out
    sub = files if kind is None else files[files["file_kind"].eq(kind)]
    for _, row in sub.iterrows():
        out.setdefault(clean(row["file_name"]), []).append(clean(row["path"]))
    return out


def nifti_volume_count(path: str) -> str:
    if not path:
        return ""
    try:
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rb") as f:
            header = f.read(348)
        if len(header) < 348:
            return ""
        sizeof_le = struct.unpack("<i", header[:4])[0]
        endian = "<" if sizeof_le == 348 else ">"
        sizeof_hdr = struct.unpack(f"{endian}i", header[:4])[0]
        if sizeof_hdr != 348:
            return ""
        dims = struct.unpack(f"{endian}8h", header[40:56])
        ndim = int(dims[0])
        if ndim >= 4:
            return str(int(dims[4]))
        return "1"
    except Exception:
        return ""


def first_existing_path(paths: Iterable[str]) -> str:
    for path in paths:
        if path and Path(path).exists():
            return path
    return ""


def build_mr_bold_inventory(mr_json: pd.DataFrame) -> pd.DataFrame:
    if mr_json.empty:
        return pd.DataFrame()
    filename_col = "filename"
    category_col = "scan category"
    bold = mr_json[
        mr_json[filename_col].str.contains("_bold\\.json", case=False, na=False)
        | mr_json[category_col].str.contains("bold", case=False, na=False)
    ].copy()
    rows: List[Dict[str, Any]] = []
    for _, row in bold.iterrows():
        subject = normalize_subject(row.get("subject_id", ""))
        filename = clean(row.get(filename_col, ""))
        session_id, bids_session = parse_session_from_filename(filename, subject)
        task = parse_task(filename, row.get(category_col, ""), row.get("SeriesDescription", ""))
        rows.append(
            {
                "subject_id": subject,
                "session_id": session_id,
                "bids_session": bids_session,
                "filename": filename,
                "json_path": filename,
                "bold_identifier": filename,
                "scan_category": clean(row.get(category_col, "")),
                "acquisition": parse_acquisition(filename),
                "run": parse_run(filename, row.get("AcquisitionNumber", "")),
                "task": task,
                "TR": clean(row.get("RepetitionTime", "")),
                "TR_seconds": tr_seconds(row.get("RepetitionTime", "")),
                "Manufacturer": clean(row.get("Manufacturer", "")),
                "ScannerModel": clean(row.get("ManufacturersModelName", "")),
                "SeriesDescription": clean(row.get("SeriesDescription", "")),
                "SequenceName": clean(row.get("SequenceName", "")),
                "PulseSequenceDetails": clean(row.get("PulseSequenceDetails", "")),
                "MagneticFieldStrength": clean(row.get("MagneticFieldStrength", "")),
                "ImageType": clean(row.get("ImageType", "")),
                "is_resting_state": is_rest_task(task, filename, row.get(category_col, ""), row.get("SeriesDescription", "")),
            }
        )
    return pd.DataFrame(rows)


def clinical_by_filename(rest_clinical: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    if rest_clinical.empty:
        return {}
    rows: Dict[str, Dict[str, Any]] = {}
    for _, row in rest_clinical.iterrows():
        key = Path(clean(row.get("bold_file_path", row.get("json_path", "")))).name
        if key and key not in rows:
            rows[key] = row.to_dict()
    return rows


def eligibility_by_session(eligibility: pd.DataFrame) -> Dict[Tuple[str, str], Dict[str, Any]]:
    if eligibility.empty:
        return {}
    rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for _, row in eligibility.iterrows():
        sid = normalize_subject(row.get("subject_id", ""))
        session_id = clean(row.get("session_id", ""))
        key = (sid, session_id)
        if sid and session_id and key not in rows:
            rows[key] = row.to_dict()
    return rows


def enrich_inventory(
    inventory: pd.DataFrame,
    rest_clinical: pd.DataFrame,
    eligibility: pd.DataFrame,
    files: pd.DataFrame,
) -> pd.DataFrame:
    clinical = clinical_by_filename(rest_clinical)
    elig = eligibility_by_session(eligibility)
    json_lookup = lookup_by_basename(files, "bold_json_sidecar")
    nifti_lookup = lookup_by_basename(files, "bold_nifti")
    rows: List[Dict[str, Any]] = []
    for _, row in inventory.iterrows():
        filename = clean(row.get("filename", ""))
        c = clinical.get(filename, {})
        e = elig.get((clean(row.get("subject_id", "")), clean(row.get("session_id", ""))), {})
        diagnosis = normalize_diagnosis(c.get("provisional_label", e.get("provisional_label", "")))
        confidence = clean(c.get("label_confidence", e.get("label_confidence", "")))
        if not confidence and diagnosis == "UNKNOWN":
            confidence = "low"
        json_paths = json_lookup.get(filename, [])
        stem = filename[:-5] if filename.endswith(".json") else filename
        nifti_paths = []
        for candidate_name in [f"{stem}.nii.gz", f"{stem}.nii"]:
            nifti_paths.extend(nifti_lookup.get(candidate_name, []))
        local_bold_path = first_existing_path(nifti_paths)
        local_json_path = first_existing_path(json_paths)
        n_volumes = clean(e.get("n_volumes", ""))
        if not n_volumes and local_bold_path:
            n_volumes = nifti_volume_count(local_bold_path)
        bold_identifier = clean(row.get("bold_identifier", "")) or clean(c.get("bold_file_path", e.get("bold_file_path", "")))
        selected_path = local_bold_path or local_json_path or bold_identifier
        out = row.to_dict()
        out.update(
            {
                "diagnosis": diagnosis,
                "diagnosis_confidence": confidence,
                "CDRTOT": clean(e.get("CDRTOT", "")),
                "CDRSUM": clean(e.get("CDRSUM", "")),
                "clinical_delta_days": clean(e.get("abs_delta_clinical_to_MR_days", c.get("clinical_day_delta", ""))),
                "age_at_MR": clean(e.get("age_at_MR", "")),
                "sex": clean(e.get("sex", "")),
                "session_rank_per_subject": clean(e.get("session_rank_per_subject", "")),
                "experiment_id": clean(e.get("experiment_id", e.get("MR_ID", ""))),
                "label_source": "clinical_mapping_audit" if c else ("eligibility_subjects" if e else "MR_json_only"),
                "bold_identifier": bold_identifier,
                "local_bold_path": local_bold_path,
                "local_json_path": local_json_path,
                "local_bold_path_available": "yes" if local_bold_path else "no",
                "local_json_path_available": "yes" if local_json_path else "no",
                "bold_identifier_available": "yes" if bold_identifier else "no",
                "selected_local_or_metadata_path": selected_path,
                "n_volumes": n_volumes,
            }
        )
        rows.append(out)
    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df["TR_distance_from_2p2"] = out_df["TR_seconds"].map(
            lambda x: "" if pd.isna(x) else abs(float(x) - 2.2)
        )
        out_df["TR_class"] = out_df["TR_seconds"].map(classify_tr)
    return out_df


def classify_tr(value: Any) -> str:
    tr = parse_float(value)
    if tr is None:
        return "missing"
    if TR2_MIN <= tr <= TR2_MAX:
        return "target_2p15_2p25"
    if TR_NEAR_MIN <= tr <= TR_NEAR_MAX:
        return "near_tr2_outside_target"
    if TR3_MIN <= tr <= TR3_MAX:
        return "tr3_reference"
    return "far_from_tr2"


def rank_inventory(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_session_day"] = out["session_id"].map(lambda x: parse_float(re.search(r"_d(\d+)", clean(x)).group(1)) if re.search(r"_d(\d+)", clean(x)) else None)
    out["_rank_from_existing"] = pd.to_numeric(out["session_rank_per_subject"], errors="coerce")
    out["_run_num"] = pd.to_numeric(out["run"], errors="coerce").fillna(999)
    out = out.sort_values(
        ["subject_id", "_rank_from_existing", "_session_day", "_run_num", "filename"],
        na_position="last",
    ).reset_index(drop=True)
    out["computed_run_rank_per_subject"] = out.groupby("subject_id").cumcount() + 1
    first_session = out.groupby("subject_id")["_session_day"].transform("min")
    out["is_first_mr_visit"] = (out["_session_day"].eq(first_session)).map(lambda x: "yes" if x else "no")
    return out


def has_identifier(row: Mapping[str, Any]) -> bool:
    return clean(row.get("bold_identifier_available", "")).lower() == "yes"


def tr_in_target(row: Mapping[str, Any]) -> bool:
    tr = parse_float(row.get("TR_seconds", ""))
    return tr is not None and TR2_MIN <= tr <= TR2_MAX


def tr_near(row: Mapping[str, Any]) -> bool:
    tr = parse_float(row.get("TR_seconds", ""))
    return tr is not None and TR_NEAR_MIN <= tr <= TR_NEAR_MAX and not (TR2_MIN <= tr <= TR2_MAX)


def tr3(row: Mapping[str, Any]) -> bool:
    tr = parse_float(row.get("TR_seconds", ""))
    return tr is not None and TR3_MIN <= tr <= TR3_MAX


def base_main_mask(df: pd.DataFrame) -> pd.Series:
    return (
        df["is_resting_state"].astype(bool)
        & df.apply(tr_in_target, axis=1)
        & df["diagnosis"].isin(["CN", "AD_DEMENTIA"])
        & df["diagnosis_confidence"].str.lower().eq("high")
        & df.apply(has_identifier, axis=1)
    )


def exclusion_reasons(row: Mapping[str, Any]) -> str:
    reasons: List[str] = []
    if not bool(row.get("is_resting_state", False)):
        reasons.append("no_resting_state")
    if not clean(row.get("bold_identifier", "")):
        reasons.append("path_identifier_missing")
    if clean(row.get("TR_class", "")) == "missing":
        reasons.append("TR_missing")
    elif clean(row.get("TR_class", "")) == "far_from_tr2":
        reasons.append("TR_far_from_2p2")
    if clean(row.get("diagnosis", "")) == "UNKNOWN":
        reasons.append("UNKNOWN_or_low_confidence")
    elif clean(row.get("diagnosis_confidence", "")).lower() == "low":
        reasons.append("UNKNOWN_or_low_confidence")
    elif clean(row.get("diagnosis", "")) in {"CN", "AD_DEMENTIA"} and clean(row.get("diagnosis_confidence", "")).lower() != "high":
        reasons.append("diagnosis_not_high_confidence")
    if clean(row.get("TR_class", "")) == "tr3_reference":
        reasons.append("TR3_reference_not_TR2_request")
    if not reasons:
        reasons.append("not_primary_candidate")
    return "|".join(reasons)


def classify_candidates(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ranked = rank_inventory(df)
    main_pool = ranked[base_main_mask(ranked)].copy()
    main_pool = main_pool.sort_values(
        ["subject_id", "_rank_from_existing", "_session_day", "_run_num", "TR_distance_from_2p2"],
        na_position="last",
    )
    main_indices = set(main_pool.drop_duplicates("subject_id", keep="first").index.tolist())
    ranked["candidate_class"] = "excluded"
    ranked["selection_reason"] = ""
    ranked.loc[list(main_indices), "candidate_class"] = "main"
    ranked.loc[list(main_indices), "selection_reason"] = "first_usable_rest_high_confidence_CN_or_AD_TR2p2"

    for idx, row in ranked.iterrows():
        if idx in main_indices:
            continue
        reason = ""
        if bool(base_main_mask(pd.DataFrame([row])).iloc[0]):
            reason = "longitudinal_extra_visit_or_nonprimary_run"
        elif row["is_resting_state"] and has_identifier(row) and clean(row["diagnosis"]) == "MCI" and (tr_in_target(row) or tr_near(row)):
            reason = "MCI_secondary_candidate"
        elif (
            row["is_resting_state"]
            and has_identifier(row)
            and clean(row["diagnosis"]) in {"CN", "AD_DEMENTIA"}
            and clean(row["diagnosis_confidence"]).lower() in {"medium", "uncertain"}
            and (tr_in_target(row) or tr_near(row))
        ):
            reason = "diagnosis_uncertain_secondary_candidate"
        elif (
            row["is_resting_state"]
            and has_identifier(row)
            and clean(row["diagnosis"]) in {"CN", "AD_DEMENTIA"}
            and clean(row["diagnosis_confidence"]).lower() == "high"
            and tr_near(row)
        ):
            reason = "TR_near_but_outside_2p15_2p25"
        elif (
            row["is_resting_state"]
            and has_identifier(row)
            and clean(row["diagnosis"]) in {"CN", "AD_DEMENTIA"}
            and clean(row["diagnosis_confidence"]).lower() == "high"
            and tr3(row)
        ):
            reason = "TR3_reference_only_not_TR2_request"
        if reason:
            ranked.at[idx, "candidate_class"] = "secondary"
            ranked.at[idx, "selection_reason"] = reason
        else:
            ranked.at[idx, "selection_reason"] = exclusion_reasons(row)

    first_usable_subjects = set(ranked.loc[ranked["candidate_class"].eq("main"), "subject_id"])
    ranked["is_first_usable_tr2_visit"] = ranked.apply(
        lambda r: "yes" if r.name in main_indices else ("no" if r["subject_id"] in first_usable_subjects else ""),
        axis=1,
    )
    main = ranked[ranked["candidate_class"].eq("main")].copy()
    secondary = ranked[ranked["candidate_class"].eq("secondary")].copy()
    excluded = ranked[ranked["candidate_class"].eq("excluded")].copy()
    return ranked, main, secondary, excluded


def format_output(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = ""
    return out[list(columns)].reset_index(drop=True)


def diagnosis_counts(main: pd.DataFrame, secondary: pd.DataFrame, excluded: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for scope, df in [
        ("inventory_all_bold_runs", inventory),
        ("main_candidates", main),
        ("secondary_candidates", secondary),
        ("excluded", excluded),
    ]:
        if df.empty:
            continue
        grouped = (
            df.groupby(["diagnosis", "diagnosis_confidence"], dropna=False)
            .agg(n_runs=("subject_id", "size"), n_subjects=("subject_id", "nunique"))
            .reset_index()
        )
        for _, row in grouped.iterrows():
            rows.append(
                {
                    "scope": scope,
                    "diagnosis": row["diagnosis"],
                    "diagnosis_confidence": row["diagnosis_confidence"],
                    "n_runs": int(row["n_runs"]),
                    "n_subjects": int(row["n_subjects"]),
                }
            )
    return pd.DataFrame(rows)


def scanner_counts(main: pd.DataFrame, secondary: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for scope, df in [("main_candidates", main), ("secondary_candidates", secondary), ("inventory_all_bold_runs", inventory)]:
        if df.empty:
            continue
        table = (
            df.groupby(["Manufacturer", "ScannerModel", "TR_class"], dropna=False)
            .agg(n_runs=("subject_id", "size"), n_subjects=("subject_id", "nunique"))
            .reset_index()
        )
        table.insert(0, "scope", scope)
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def write_readme(
    output_dir: Path,
    main: pd.DataFrame,
    secondary: pd.DataFrame,
    excluded: pd.DataFrame,
    inventory: pd.DataFrame,
    files: pd.DataFrame,
    command_log: Mapping[str, Any],
) -> None:
    main_cn = int((main["diagnosis"].eq("CN")).sum()) if not main.empty else 0
    main_ad = int((main["diagnosis"].eq("AD_DEMENTIA")).sum()) if not main.empty else 0
    main_local_bold = int(main["local_bold_path_available"].eq("yes").sum()) if not main.empty else 0
    main_identifier = int(main["bold_identifier_available"].eq("yes").sum()) if not main.empty else 0
    main_first_mr = int(main["is_first_mr_visit"].eq("yes").sum()) if not main.empty else 0
    tr3_pool = inventory[
        inventory["is_resting_state"].astype(bool)
        & inventory["diagnosis"].isin(["CN", "AD_DEMENTIA"])
        & inventory["diagnosis_confidence"].str.lower().eq("high")
        & inventory.apply(tr3, axis=1)
    ]
    scanner_text = "none"
    if not main.empty:
        scanner = (
            main.groupby(["Manufacturer", "ScannerModel"], dropna=False)
            .agg(n_subjects=("subject_id", "nunique"))
            .reset_index()
            .sort_values(["Manufacturer", "ScannerModel"])
        )
        scanner_text = "; ".join(
            f"{clean(r.Manufacturer) or 'UNKNOWN'} {clean(r.ScannerModel) or 'UNKNOWN'}: {int(r.n_subjects)}"
            for _, r in scanner.iterrows()
        )
    lines = [
        "# OASIS-3 TR~2.2 External Validation Candidate Manifest",
        "",
        f"Generated: `{command_log['generated']}`",
        "",
        "Metadata/header-only audit. No preprocessing, connectivity calculation, training, or source-data modification was performed.",
        "",
        "## Answers",
        "",
        f"- Main TR~2.2 high-confidence CN subjects/runs: `{main_cn}`.",
        f"- Main TR~2.2 high-confidence AD_DEMENTIA subjects/runs: `{main_ad}`.",
        f"- Main candidates with BIDS/path identifier available: `{main_identifier}`.",
        f"- Main candidates with local BOLD NIfTI path available: `{main_local_bold}`.",
        f"- Main candidates that are also first MR visit by session rank/day: `{main_first_mr}`.",
        f"- Main candidates are one first usable TR~2.2 rest run per subject: `{len(main)}`.",
        f"- Main candidate scanners: `{scanner_text}`.",
        f"- High-confidence CN/AD resting-state TR~3 rows found: `{len(tr3_pool)}` runs, `{tr3_pool['subject_id'].nunique() if not tr3_pool.empty else 0}` subjects.",
        "- Exact CSV to send Martin: `oasis3_tr2_main_candidates_for_martin.csv`.",
        "- Limitation: OASIS-3 TR~2.2 is not the same acquisition regime as ADNI TR~3. It is suitable only as external validation/stress-test, especially because the ADNI model uses static channels `[1,0,2]` (Pearson Full, OMST Pearson, MI).",
        "",
        "## Inventory",
        "",
        f"- BOLD rows inventoried from OASIS metadata: `{len(inventory)}`.",
        f"- Secondary candidates: `{len(secondary)}`.",
        f"- Excluded rows: `{len(excluded)}`.",
        f"- Local matching OASIS/BOLD/metadata files discovered: `{len(files)}`.",
        f"- Local BOLD NIfTI files discovered: `{int(files['file_kind'].eq('bold_nifti').sum()) if not files.empty else 0}`.",
        f"- Local BOLD JSON sidecars discovered: `{int(files['file_kind'].eq('bold_json_sidecar').sum()) if not files.empty else 0}`.",
        "",
        "## Outputs",
        "",
        "- `oasis3_tr2_main_candidates_for_martin.csv`",
        "- `oasis3_tr2_secondary_candidates.csv`",
        "- `oasis3_tr2_excluded_with_reasons.csv`",
        "- `oasis3_tr2_diagnosis_counts.csv`",
        "- `oasis3_tr2_scanner_counts.csv`",
        "- `oasis3_tr2_subject_session_inventory.csv`",
        "- `command_log.json`",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = prepare_output_dir(args.output_dir, args.overwrite)
    search_roots = args.search_root if args.search_root else DEFAULT_SEARCH_ROOTS

    files = inventory_files(search_roots, args.data_search_root)
    mr_json = read_csv(args.mr_json)
    rest_clinical = read_csv(args.rest_clinical, required=False)
    eligibility = read_csv(args.eligibility, required=False)

    mr_bold = build_mr_bold_inventory(mr_json)
    enriched = enrich_inventory(mr_bold, rest_clinical, eligibility, files)
    ranked, main, secondary, excluded = classify_candidates(enriched)

    inventory_cols = [
        "subject_id",
        "session_id",
        "bids_session",
        "acquisition",
        "run",
        "task",
        "scan_category",
        "TR",
        "TR_seconds",
        "TR_class",
        "is_resting_state",
        "Manufacturer",
        "ScannerModel",
        "SeriesDescription",
        "SequenceName",
        "PulseSequenceDetails",
        "n_volumes",
        "diagnosis",
        "diagnosis_confidence",
        "CDRTOT",
        "CDRSUM",
        "clinical_delta_days",
        "age_at_MR",
        "sex",
        "session_rank_per_subject",
        "is_first_mr_visit",
        "is_first_usable_tr2_visit",
        "bold_identifier",
        "local_bold_path_available",
        "local_json_path_available",
        "selected_local_or_metadata_path",
        "experiment_id",
        "label_source",
        "candidate_class",
        "selection_reason",
    ]
    main_out = format_output(main, MAIN_COLUMNS)
    secondary_out = format_output(secondary, MAIN_COLUMNS + ["candidate_class"])
    excluded_out = format_output(excluded, inventory_cols)
    inventory_out = format_output(ranked, inventory_cols)
    diag = diagnosis_counts(main, secondary, excluded, ranked)
    scanner = scanner_counts(main, secondary, ranked)

    main_out.to_csv(output_dir / "oasis3_tr2_main_candidates_for_martin.csv", index=False)
    secondary_out.to_csv(output_dir / "oasis3_tr2_secondary_candidates.csv", index=False)
    excluded_out.to_csv(output_dir / "oasis3_tr2_excluded_with_reasons.csv", index=False)
    diag.to_csv(output_dir / "oasis3_tr2_diagnosis_counts.csv", index=False)
    scanner.to_csv(output_dir / "oasis3_tr2_scanner_counts.csv", index=False)
    inventory_out.to_csv(output_dir / "oasis3_tr2_subject_session_inventory.csv", index=False)

    command_log = {
        "script": str(Path(__file__).resolve()),
        "generated": datetime.now().isoformat(timespec="seconds"),
        "mr_json": str(args.mr_json),
        "eligibility": str(args.eligibility),
        "rest_clinical": str(args.rest_clinical),
        "search_roots": [str(p) for p in search_roots],
        "data_search_root": str(args.data_search_root),
        "output_dir": str(output_dir),
        "file_inventory_counts": files["file_kind"].value_counts().to_dict() if not files.empty else {},
        "n_bold_inventory_rows": int(len(ranked)),
        "n_main_candidates": int(len(main)),
        "n_secondary_candidates": int(len(secondary)),
        "n_excluded_rows": int(len(excluded)),
        "main_CN": int(main["diagnosis"].eq("CN").sum()) if not main.empty else 0,
        "main_AD_DEMENTIA": int(main["diagnosis"].eq("AD_DEMENTIA").sum()) if not main.empty else 0,
        "python_bandpass_applied": False,
        "preprocessing_run": False,
        "connectivity_calculation_run": False,
        "training_run": False,
        "source_data_modified": False,
    }
    (output_dir / "command_log.json").write_text(
        json.dumps(command_log, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_readme(output_dir, main, secondary, excluded, ranked, files, command_log)

    print(f"output_dir={output_dir}")
    print(f"bold_inventory_rows={len(ranked)}")
    print(f"main_candidates={len(main)}")
    print(f"main_CN={command_log['main_CN']}")
    print(f"main_AD_DEMENTIA={command_log['main_AD_DEMENTIA']}")
    print(f"secondary_candidates={len(secondary)}")
    print(f"excluded_rows={len(excluded)}")
    print(f"local_bold_nifti_files={int(files['file_kind'].eq('bold_nifti').sum()) if not files.empty else 0}")
    print("preprocessing_run=False")
    print("connectivity_calculation_run=False")
    print("training_run=False")
    print("source_data_modified=False")


if __name__ == "__main__":
    main()
