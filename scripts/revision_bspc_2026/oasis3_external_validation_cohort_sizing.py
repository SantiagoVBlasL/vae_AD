#!/usr/bin/env python3
"""Metadata-only OASIS-3 external-validation cohort sizing audit.

This script estimates OASIS-3 CN/AD_DEMENTIA rs-fMRI eligibility using only
metadata CSV/TSV/JSON files and previous audit outputs. It does not download
images, preprocess, train, copy large files, or load arrays.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision_bspc_2026 import oasis3_clinical_mapping_audit as clinical  # noqa: E402


DEFAULT_METADATA_RAW = Path("/media/diego/Datos/vae_AD_data/OASIS3/metadata_raw")
DEFAULT_FEASIBILITY_DIR = PROJECT_ROOT / "results/revision_bspc_2026/oasis3_feasibility_audit"
DEFAULT_CLINICAL_DIR = PROJECT_ROOT / "results/revision_bspc_2026/oasis3_clinical_mapping_audit"
DEFAULT_SMOKE_REVIEW_DIR = PROJECT_ROOT / "results/revision_bspc_2026/oasis3_smoke_test_pre_download_review"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/oasis3_external_validation_cohort_sizing"
DEFAULT_DOWNLOAD_ROOT = Path("/media/diego/Datos/vae_AD_data/OASIS3/external_validation_raw")

REST_CANDIDATES = "oasis3_rest_bold_clinical_candidates.csv"
SUBJECT_MAPPING = "subject_level_clinical_mapping_draft.csv"
FEASIBILITY_INVENTORY = "oasis3_session_inventory.csv"

CLINICAL_FILE_BASENAMES = [
    "OASIS3_UDSb4_cdr.csv",
    "OASIS3_UDSd1_diagnoses.csv",
    "OASIS3_UDSa1_participant_demo.csv",
    "OASIS3_demographics.csv",
    "OASIS3_unchanged_CDR_cognitively_healthy.csv",
]

OUTPUT_COLUMNS = [
    "subject_id",
    "session_id",
    "MR_ID",
    "experiment_id",
    "provisional_label",
    "label_confidence",
    "CDRTOT",
    "CDRSUM",
    "DEMENTED",
    "PROBAD",
    "POSSAD",
    "dx_fields",
    "clinical_date_or_day",
    "MR_session_day",
    "abs_delta_clinical_to_MR_days",
    "has_task_rest_bold",
    "bold_scan_id",
    "bold_file_path",
    "json_path",
    "RepetitionTime",
    "TR",
    "n_volumes",
    "manufacturer",
    "scanner_model",
    "sequence_description",
    "age_at_MR",
    "sex",
    "session_rank_per_subject",
    "exclusion_reason",
    "eligible_strict",
    "eligible_relaxed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Metadata-only OASIS-3 external-validation cohort sizing audit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metadata-raw", type=Path, default=DEFAULT_METADATA_RAW)
    parser.add_argument("--feasibility-dir", type=Path, default=DEFAULT_FEASIBILITY_DIR)
    parser.add_argument("--clinical-audit-dir", type=Path, default=DEFAULT_CLINICAL_DIR)
    parser.add_argument("--smoke-review-dir", type=Path, default=DEFAULT_SMOKE_REVIEW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--download-root", type=Path, default=DEFAULT_DOWNLOAD_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


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


def safe_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def parse_float(value: Any) -> Optional[float]:
    text = safe_text(value)
    if not text or text == ".":
        return None
    try:
        return float(text)
    except ValueError:
        return clinical.parse_float(text)


def bool_from_any(value: Any) -> bool:
    text = safe_text(value).lower()
    return text in {"true", "1", "yes", "y"}


def join_unique(values: Iterable[Any], max_items: int = 20) -> str:
    seen = set()
    out: List[str] = []
    for value in values:
        text = safe_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= max_items:
            out.append("...")
            break
    return "|".join(out)


def require_file(path: Path, label: str) -> Path:
    path = resolve(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    path = resolve(path)
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {path}. Pass --overwrite.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def prefer_primary(paths: Sequence[Path]) -> List[Path]:
    def key(path: Path) -> Tuple[int, int, str]:
        text = str(path)
        imported = 1 if "/imported/" in text else 0
        return imported, len(text), text

    return sorted(paths, key=key)


def find_first_file(root: Path, basename: str) -> Optional[Path]:
    paths = [p for p in root.rglob(basename) if p.is_file()]
    return prefer_primary(paths)[0] if paths else None


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, low_memory=False)


def load_key_clinical_rows(metadata_raw: Path) -> Tuple[pd.DataFrame, List[str], List[str]]:
    loaded: Dict[str, pd.DataFrame] = {}
    used_files: List[str] = []
    warnings: List[str] = []
    for basename in CLINICAL_FILE_BASENAMES:
        path = find_first_file(metadata_raw, basename)
        if path is None:
            warnings.append(f"missing_clinical_metadata_file:{basename}")
            continue
        try:
            loaded[str(path)] = clinical.load_table(path)
            used_files.append(str(path))
        except Exception as exc:
            warnings.append(f"failed_load_clinical_metadata:{path}:{exc}")
    rows, row_warnings = clinical.clinical_rows_from_loaded(loaded)
    warnings.extend(row_warnings)
    return rows, used_files, warnings


def load_mr_json(metadata_raw: Path) -> Tuple[pd.DataFrame, str, List[str]]:
    path = find_first_file(metadata_raw, "OASIS3_MR_json.csv")
    warnings: List[str] = []
    if path is None:
        return pd.DataFrame(), "", ["missing_OASIS3_MR_json.csv"]
    try:
        mr = load_csv(path)
    except Exception as exc:
        return pd.DataFrame(), str(path), [f"failed_load_mr_json:{path}:{exc}"]
    return mr, str(path), warnings


def source_type(path: str) -> str:
    low = path.lower()
    if "udsb4" in low:
        return "cdr"
    if "udsd1" in low:
        return "dx"
    if "udsa1" in low:
        return "visit_demo"
    if "demographics" in low:
        return "demographics"
    if "cognitively_normal" in low or "cognorm" in low:
        return "cognorm"
    return "other"


def build_clinical_index(clinical_rows: pd.DataFrame) -> Dict[Tuple[str, str], pd.DataFrame]:
    index: Dict[Tuple[str, str], pd.DataFrame] = {}
    if clinical_rows.empty:
        return index
    temp = clinical_rows.copy()
    temp["_source_type"] = temp["clinical_source_file"].map(source_type)
    temp["_clinical_day_num"] = pd.to_numeric(temp["clinical_day"], errors="coerce")
    temp["_confidence_rank"] = temp["row_label_confidence"].map(clinical.confidence_rank)
    temp["_label_rank"] = temp["row_label"].map(clinical.label_priority)
    for key, group in temp.groupby(["subject_id", "_source_type"], dropna=False):
        index[key] = group.copy()
    return index


def nearest_row(
    clinical_index: Dict[Tuple[str, str], pd.DataFrame],
    subject_id: str,
    row_type: str,
    mr_day: Optional[float],
) -> Optional[pd.Series]:
    group = clinical_index.get((subject_id, row_type))
    if group is None or group.empty:
        return None
    sub = group.copy()
    if mr_day is not None and sub["_clinical_day_num"].notna().any():
        sub["_abs_delta"] = (sub["_clinical_day_num"] - mr_day).abs()
        sub = sub.sort_values(["_abs_delta", "_confidence_rank", "_label_rank"], ascending=[True, False, False])
        return sub.iloc[0]
    sub = sub.sort_values(["_confidence_rank", "_label_rank"], ascending=[False, False])
    return sub.iloc[0]


def get_any(row: Optional[pd.Series], names: Sequence[str]) -> str:
    if row is None:
        return ""
    lower = {str(c).lower(): c for c in row.index}
    for name in names:
        col = lower.get(name.lower())
        if col is not None:
            text = safe_text(row.get(col, ""))
            if text:
                return text
    return ""


def dx_summary(cdr_row: Optional[pd.Series], dx_row: Optional[pd.Series]) -> str:
    parts = []
    fields = ["dx1", "dx1_code", "dx2", "dx2_code", "dx3", "dx3_code", "dx4", "dx4_code", "dx5", "dx5_code", "dxmethod"]
    for field in fields:
        value = get_any(dx_row, [field]) or get_any(cdr_row, [field])
        if value:
            parts.append(f"{field}={value}")
    return "; ".join(parts)


def clinical_day_summary(rows: Sequence[Optional[pd.Series]]) -> str:
    return join_unique([get_any(row, ["clinical_day", "days_to_visit"]) for row in rows if row is not None], max_items=10)


def age_for_session(rows: Sequence[Optional[pd.Series]]) -> str:
    for row in rows:
        value = get_any(row, ["age at visit", "age_at_visit", "age", "Age", "AgeatEntry"])
        if value:
            return value
    return ""


def sex_for_subject(rows: Sequence[Optional[pd.Series]], subject_map_row: Optional[pd.Series]) -> str:
    for row in rows:
        value = get_any(row, ["GENDER", "INSEX", "sex", "Sex"])
        if value:
            return value
    if subject_map_row is not None:
        return safe_text(subject_map_row.get("sex", ""))
    return ""


def build_mr_lookup(mr_json: pd.DataFrame) -> pd.DataFrame:
    if mr_json.empty:
        return pd.DataFrame()
    mr = mr_json.copy()
    if "filename" in mr.columns:
        mr = mr[mr["filename"].astype(str).str.contains("bold|task-rest", case=False, na=False)].copy()
    out = pd.DataFrame(
        {
            "subject_id": "sub-" + mr.get("subject_id", pd.Series("", index=mr.index)).astype(str),
            "session_id": mr.get("label", pd.Series("", index=mr.index)).astype(str),
            "json_path": mr.get("filename", pd.Series("", index=mr.index)).astype(str),
            "MR_ID": mr.get("label", pd.Series("", index=mr.index)).astype(str),
            "experiment_id": mr.get("label", pd.Series("", index=mr.index)).astype(str),
            "bold_scan_id": mr.get("acccession", pd.Series("", index=mr.index)).astype(str),
            "RepetitionTime_mrjson": mr.get("RepetitionTime", pd.Series("", index=mr.index)).astype(str),
            "manufacturer_mrjson": mr.get("Manufacturer", pd.Series("", index=mr.index)).astype(str),
            "scanner_model_mrjson": mr.get("ManufacturersModelName", pd.Series("", index=mr.index)).astype(str),
            "sequence_description": mr.get("SeriesDescription", pd.Series("", index=mr.index)).astype(str),
            "scan_category": mr.get("scan category", pd.Series("", index=mr.index)).astype(str),
        }
    )
    return out.drop_duplicates(["subject_id", "session_id", "json_path"], keep="first")


def merge_mr_metadata(rest: pd.DataFrame, mr_lookup: pd.DataFrame) -> pd.DataFrame:
    if mr_lookup.empty:
        out = rest.copy()
        for col in ["MR_ID", "experiment_id", "bold_scan_id", "RepetitionTime_mrjson", "manufacturer_mrjson", "scanner_model_mrjson", "sequence_description", "scan_category"]:
            out[col] = ""
        return out
    return rest.merge(mr_lookup, on=["subject_id", "session_id", "json_path"], how="left")


def merge_feasibility(rest: pd.DataFrame, feasibility_path: Path) -> pd.DataFrame:
    if not feasibility_path.exists():
        rest = rest.copy()
        rest["n_volumes"] = ""
        return rest
    feasible = load_csv(feasibility_path)
    keep = [c for c in ["subject_id", "session_id", "json_path", "n_volumes"] if c in feasible.columns]
    if {"subject_id", "session_id", "json_path"}.issubset(keep):
        feasible = feasible[keep].drop_duplicates(["subject_id", "session_id", "json_path"], keep="first")
        return rest.merge(feasible, on=["subject_id", "session_id", "json_path"], how="left")
    rest = rest.copy()
    rest["n_volumes"] = ""
    return rest


def strict_failures(row: Dict[str, Any]) -> List[str]:
    failures = []
    if row["provisional_label"] not in {"CN", "AD_DEMENTIA"}:
        failures.append("label_not_CN_or_AD_DEMENTIA")
    if row["label_confidence"] != "high":
        failures.append("label_confidence_not_high")
    if not bool_from_any(row["has_task_rest_bold"]):
        failures.append("missing_task_rest_bold")
    if not safe_text(row["RepetitionTime"]):
        failures.append("missing_TR")
    if not safe_text(row["manufacturer"]):
        failures.append("missing_manufacturer")
    if not safe_text(row["scanner_model"]):
        failures.append("missing_scanner_model")
    delta = parse_float(row["abs_delta_clinical_to_MR_days"])
    if delta is not None and delta > 90:
        failures.append("clinical_to_MR_delta_gt_90_days")
    return failures


def relaxed_failures(row: Dict[str, Any]) -> List[str]:
    failures = []
    if row["provisional_label"] not in {"CN", "AD_DEMENTIA"}:
        failures.append("label_not_CN_or_AD_DEMENTIA")
    if row["label_confidence"] not in {"high", "medium"}:
        failures.append("label_confidence_not_high_or_medium")
    if not bool_from_any(row["has_task_rest_bold"]):
        failures.append("missing_task_rest_bold")
    if not safe_text(row["RepetitionTime"]):
        failures.append("missing_TR")
    if not safe_text(row["manufacturer"]):
        failures.append("missing_manufacturer")
    if not safe_text(row["scanner_model"]):
        failures.append("missing_scanner_model")
    delta = parse_float(row["abs_delta_clinical_to_MR_days"])
    if delta is not None and delta > 365:
        failures.append("clinical_to_MR_delta_gt_365_days")
    return failures


def enrich_candidates(
    rest_candidates: pd.DataFrame,
    feasibility_path: Path,
    mr_lookup: pd.DataFrame,
    clinical_index: Dict[Tuple[str, str], pd.DataFrame],
    subject_map: pd.DataFrame,
) -> pd.DataFrame:
    rest = rest_candidates.copy()
    rest = merge_mr_metadata(rest, mr_lookup)
    rest = merge_feasibility(rest, feasibility_path)
    subject_map_idx = subject_map.set_index("subject_id").to_dict(orient="index") if not subject_map.empty else {}
    rows: List[Dict[str, Any]] = []
    for _, src in rest.iterrows():
        subject = safe_text(src.get("subject_id", ""))
        session = safe_text(src.get("session_id", ""))
        mr_day = parse_float(src.get("session_day", "")) or clinical.parse_session_day(session)
        cdr_row = nearest_row(clinical_index, subject, "cdr", mr_day)
        dx_row = nearest_row(clinical_index, subject, "dx", mr_day)
        visit_demo_row = nearest_row(clinical_index, subject, "visit_demo", mr_day)
        demographics_row = nearest_row(clinical_index, subject, "demographics", mr_day)
        subj = subject_map_idx.get(subject)
        rep_time = safe_text(src.get("RepetitionTime_mrjson", "")) or safe_text(src.get("RepetitionTime", ""))
        manufacturer = safe_text(src.get("manufacturer_mrjson", "")) or safe_text(src.get("scanner_manufacturer", ""))
        scanner_model = safe_text(src.get("scanner_model_mrjson", "")) or safe_text(src.get("scanner_model", ""))
        out = {
            "subject_id": subject,
            "session_id": session,
            "MR_ID": safe_text(src.get("MR_ID", "")) or session,
            "experiment_id": safe_text(src.get("experiment_id", "")) or session,
            "provisional_label": safe_text(src.get("provisional_label", "")),
            "label_confidence": safe_text(src.get("label_confidence", "")),
            "CDRTOT": get_any(cdr_row, ["CDRTOT", "CDR"]),
            "CDRSUM": get_any(cdr_row, ["CDRSUM"]),
            "DEMENTED": get_any(dx_row, ["DEMENTED"]),
            "PROBAD": get_any(dx_row, ["PROBAD"]),
            "POSSAD": get_any(dx_row, ["POSSAD"]),
            "dx_fields": dx_summary(cdr_row, dx_row),
            "clinical_date_or_day": clinical_day_summary([cdr_row, dx_row]),
            "MR_session_day": "" if mr_day is None else f"{mr_day:g}",
            "abs_delta_clinical_to_MR_days": safe_text(src.get("clinical_day_delta", "")),
            "has_task_rest_bold": safe_text(src.get("has_task_rest_bold", "")),
            "bold_scan_id": safe_text(src.get("bold_scan_id", "")),
            "RepetitionTime": rep_time,
            "TR": rep_time,
            "n_volumes": safe_text(src.get("n_volumes", "")),
            "manufacturer": manufacturer,
            "scanner_model": scanner_model,
            "sequence_description": safe_text(src.get("sequence_description", "")) or safe_text(src.get("scan_category", "")),
            "age_at_MR": age_for_session([cdr_row, dx_row, visit_demo_row, demographics_row]),
            "sex": sex_for_subject([demographics_row, visit_demo_row, cdr_row, dx_row], pd.Series(subj) if subj else None),
            "json_path": safe_text(src.get("json_path", "")),
            "bold_file_path": safe_text(src.get("bold_file_path", "")),
            "label_warning": safe_text(src.get("label_warning", "")),
            "provisional_label_rule": safe_text(src.get("provisional_label_rule", "")),
        }
        strict = strict_failures(out)
        relaxed = relaxed_failures(out)
        out["exclusion_reason"] = ";".join(strict)
        out["eligible_strict"] = not strict
        out["eligible_relaxed"] = not relaxed
        rows.append(out)
    enriched = pd.DataFrame(rows)
    if enriched.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    enriched["_sequence_preference"] = np.where(
        enriched["sequence_description"].astype(str).str.contains("connect", case=False, na=False),
        0,
        np.where(enriched["sequence_description"].astype(str).str.contains("test", case=False, na=False), 2, 1),
    )
    enriched["_tr_missing"] = enriched["TR"].astype(str).str.strip().eq("").astype(int)
    enriched["_json_sort"] = enriched["json_path"].astype(str)
    enriched = (
        enriched.sort_values(
            ["subject_id", "session_id", "_sequence_preference", "_tr_missing", "_json_sort"],
            na_position="last",
        )
        .drop_duplicates(["subject_id", "session_id"], keep="first")
        .copy()
    )
    enriched["_session_day_num"] = pd.to_numeric(enriched["MR_session_day"], errors="coerce")
    session_rank = (
        enriched[["subject_id", "session_id", "_session_day_num"]]
        .drop_duplicates(["subject_id", "session_id"])
        .sort_values(["subject_id", "_session_day_num", "session_id"], na_position="last")
    )
    session_rank["session_rank_per_subject"] = session_rank.groupby("subject_id").cumcount() + 1
    enriched = enriched.merge(session_rank[["subject_id", "session_id", "session_rank_per_subject"]], on=["subject_id", "session_id"], how="left")
    return enriched[OUTPUT_COLUMNS + ["label_warning", "provisional_label_rule"]].sort_values(
        ["subject_id", "session_rank_per_subject", "json_path"], na_position="last"
    )


def selection_sort_frame(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()
    temp["_conf_rank"] = temp["label_confidence"].map(clinical.confidence_rank).fillna(0)
    temp["_delta"] = pd.to_numeric(temp["abs_delta_clinical_to_MR_days"], errors="coerce").fillna(10**9)
    temp["_session_day"] = pd.to_numeric(temp["MR_session_day"], errors="coerce").fillna(10**9)
    temp["_tr_missing"] = temp["TR"].astype(str).str.strip().eq("").astype(int)
    temp["_json"] = temp.get("json_path", pd.Series("", index=temp.index)).astype(str)
    return temp.sort_values(["subject_id", "_conf_rank", "_delta", "_session_day", "_tr_missing", "_json"], ascending=[True, False, True, True, True, True])


def select_one_per_subject(candidates: pd.DataFrame, eligibility_col: str) -> pd.DataFrame:
    eligible = candidates[candidates[eligibility_col].astype(bool)].copy()
    if eligible.empty:
        return eligible[OUTPUT_COLUMNS].copy()
    sorted_df = selection_sort_frame(eligible)
    selected = sorted_df.drop_duplicates("subject_id", keep="first").copy()
    return selected[OUTPUT_COLUMNS].sort_values(["provisional_label", "subject_id", "session_id"]).reset_index(drop=True)


def age_bin(value: Any) -> str:
    age = parse_float(value)
    if age is None:
        return "unknown"
    if age < 65:
        return "<65"
    if age < 70:
        return "65-70"
    if age < 75:
        return "70-75"
    if age < 80:
        return "75-80"
    if age < 85:
        return "80-85"
    return ">=85"


def delta_bin(value: Any) -> str:
    delta = parse_float(value)
    if delta is None:
        return "unknown"
    if delta <= 30:
        return "<=30"
    if delta <= 90:
        return "31-90"
    if delta <= 180:
        return "91-180"
    if delta <= 365:
        return "181-365"
    return ">365"


def session_day_bin(value: Any) -> str:
    day = parse_float(value)
    if day is None:
        return "unknown"
    if day == 0:
        return "d0000"
    if day <= 365:
        return "d0001_0365"
    if day <= 1095:
        return "d0366_1095"
    return "d1096_plus"


def with_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["age_bin"] = out["age_at_MR"].map(age_bin)
    out["clinical_to_MR_delta_bin"] = out["abs_delta_clinical_to_MR_days"].map(delta_bin)
    out["session_day_bin"] = out["MR_session_day"].map(session_day_bin)
    out["sex"] = out["sex"].replace("", "unknown").fillna("unknown")
    out["manufacturer"] = out["manufacturer"].replace("", "unknown").fillna("unknown")
    out["scanner_model"] = out["scanner_model"].replace("", "unknown").fillna("unknown")
    out["TR"] = out["TR"].replace("", "unknown").fillna("unknown")
    return out


def build_counts(strict: pd.DataFrame, relaxed: pd.DataFrame, all_candidates: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    sets = {"strict": with_bins(strict), "relaxed": with_bins(relaxed)}
    candidate_sets = {
        "strict": all_candidates[all_candidates["eligible_strict"].astype(bool)].copy(),
        "relaxed": all_candidates[all_candidates["eligible_relaxed"].astype(bool)].copy(),
    }
    dimensions = {
        "overall": None,
        "label": "provisional_label",
        "manufacturer": "manufacturer",
        "scanner_model": "scanner_model",
        "TR": "TR",
        "sex": "sex",
        "age_bin": "age_bin",
        "clinical_to_MR_delta_bin": "clinical_to_MR_delta_bin",
        "session_day_bin": "session_day_bin",
    }
    for set_name, df in sets.items():
        cand = with_bins(candidate_sets[set_name]) if not candidate_sets[set_name].empty else pd.DataFrame()
        for dim, col in dimensions.items():
            if col is None:
                rows.append(
                    {
                        "eligibility_set": set_name,
                        "dimension": dim,
                        "level": "all",
                        "n_subjects": int(df["subject_id"].nunique()) if not df.empty else 0,
                        "n_selected_rows": int(len(df)),
                        "n_candidate_rows": int(len(cand)) if not cand.empty else 0,
                    }
                )
                continue
            if df.empty:
                rows.append({"eligibility_set": set_name, "dimension": dim, "level": "none", "n_subjects": 0, "n_selected_rows": 0, "n_candidate_rows": 0})
                continue
            selected_counts = df.groupby(col, dropna=False).agg(n_subjects=("subject_id", "nunique"), n_selected_rows=("subject_id", "size")).reset_index()
            cand_counts = (
                cand.groupby(col, dropna=False).agg(n_candidate_rows=("subject_id", "size")).reset_index()
                if not cand.empty and col in cand.columns
                else pd.DataFrame(columns=[col, "n_candidate_rows"])
            )
            merged = selected_counts.merge(cand_counts, on=col, how="left")
            for _, row in merged.iterrows():
                rows.append(
                    {
                        "eligibility_set": set_name,
                        "dimension": dim,
                        "level": safe_text(row[col]),
                        "n_subjects": int(row["n_subjects"]),
                        "n_selected_rows": int(row["n_selected_rows"]),
                        "n_candidate_rows": int(row["n_candidate_rows"]) if pd.notna(row.get("n_candidate_rows")) else 0,
                    }
                )
    return pd.DataFrame(rows)


def build_balance_table(strict: pd.DataFrame, relaxed: pd.DataFrame) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for set_name, df in {"strict": with_bins(strict), "relaxed": with_bins(relaxed)}.items():
        for stratum, col in [
            ("label_x_manufacturer", "manufacturer"),
            ("label_x_scanner_model", "scanner_model"),
            ("label_x_TR", "TR"),
            ("label_x_sex", "sex"),
            ("label_x_age_bin", "age_bin"),
        ]:
            if df.empty:
                rows.append(pd.DataFrame([{"eligibility_set": set_name, "stratum": stratum, "level": "none", "CN": 0, "AD_DEMENTIA": 0, "total": 0}]))
                continue
            tab = pd.crosstab(df[col], df["provisional_label"])
            for label in ["CN", "AD_DEMENTIA"]:
                if label not in tab.columns:
                    tab[label] = 0
            tab = tab[["CN", "AD_DEMENTIA"]]
            tab["total"] = tab.sum(axis=1)
            tab = tab.reset_index().rename(columns={col: "level"})
            tab.insert(0, "stratum", stratum)
            tab.insert(0, "eligibility_set", set_name)
            rows.append(tab)
    return pd.concat(rows, ignore_index=True)


def class_balance_verdict(df: pd.DataFrame) -> str:
    counts = df["provisional_label"].value_counts().to_dict() if not df.empty else {}
    cn = counts.get("CN", 0)
    ad = counts.get("AD_DEMENTIA", 0)
    if min(cn, ad) < 10:
        return "too_small_for_external_validation"
    ratio = min(cn, ad) / max(cn, ad) if max(cn, ad) else 0
    if ratio >= 0.5:
        return "acceptable"
    if ratio >= 0.25:
        return "usable_but_imbalanced"
    return "strongly_imbalanced"


def scanner_balance_verdict(df: pd.DataFrame) -> str:
    if df.empty:
        return "unknown"
    manufacturers = df["manufacturer"].replace("", "unknown").fillna("unknown").nunique()
    models = df["scanner_model"].replace("", "unknown").fillna("unknown").nunique()
    if manufacturers == 1 and models == 1:
        return "single_scanner_domain"
    if manufacturers == 1:
        return "single_manufacturer_multi_model"
    return "multi_manufacturer"


def write_design_recommendation(path: Path, strict: pd.DataFrame, relaxed: pd.DataFrame, candidates: pd.DataFrame, warnings: Sequence[str]) -> None:
    strict_counts = strict["provisional_label"].value_counts().to_dict() if not strict.empty else {}
    relaxed_counts = relaxed["provisional_label"].value_counts().to_dict() if not relaxed.empty else {}
    strict_n = int(strict["subject_id"].nunique()) if not strict.empty else 0
    relaxed_n = int(relaxed["subject_id"].nunique()) if not relaxed.empty else 0
    strict_class = class_balance_verdict(strict)
    relaxed_class = class_balance_verdict(relaxed)
    scanner_verdict = scanner_balance_verdict(strict)
    n_vol_known = int(candidates["n_volumes"].astype(str).str.strip().replace("nan", "").ne("").sum()) if "n_volumes" in candidates else 0
    if strict_n >= 40 and min(strict_counts.get("CN", 0), strict_counts.get("AD_DEMENTIA", 0)) >= 10:
        use_verdict = "OASIS-3 is metadata-compatible as an external validation candidate after smoke-test download, preprocessing QC, and ROI/time-series validation."
    elif relaxed_n >= 20:
        use_verdict = "OASIS-3 is better treated first as a stress-test or pilot external-validation cohort until strict criteria are improved."
    else:
        use_verdict = "OASIS-3 is not ready for external validation from current metadata criteria."
    warning_block = "\n".join(f"- {w}" for w in warnings) if warnings else "- None."
    lines = [
        "# OASIS-3 External Validation Design Recommendation",
        "",
        "This is a metadata-only cohort sizing audit. It did not download images, preprocess, train, copy large files, or load arrays.",
        "",
        "## Estimated Cohort Size",
        "",
        f"- Strict selected subjects: {strict_n} ({json.dumps(strict_counts, sort_keys=True)})",
        f"- Relaxed selected subjects: {relaxed_n} ({json.dumps(relaxed_counts, sort_keys=True)})",
        f"- Strict class balance verdict: `{strict_class}`",
        f"- Relaxed class balance verdict: `{relaxed_class}`",
        "",
        "## Scanner And TR Balance",
        "",
        f"- Strict scanner/manufacturer verdict: `{scanner_verdict}`",
        "- Scanner balance is part of the external-domain question, not a reason to mix OASIS into ADNI training.",
        "- If the eligible cohort is dominated by one OASIS scanner/model/TR, it is still useful as an external stress-test but not as a broad scanner-invariance benchmark.",
        "",
        "## Feasibility Verdict",
        "",
        use_verdict,
        "",
        "OASIS should not be mixed into ADNI training unless the experiment is explicitly redesigned as a separate domain-harmonized multi-cohort study. Mixing OASIS into the ADNI training set would blur the external-validation claim and could introduce dataset-specific clinical/preprocessing artifacts.",
        "",
        "## Unknowns Before Image Download",
        "",
        f"- `n_volumes` available in metadata for candidate rows: {n_vol_known}/{len(candidates)}.",
        "- BOLD image existence, usable duration, motion, ROI extraction quality, AAL3 coverage, and preprocessing compatibility still require smoke-test download and QC.",
        "- Clinical labels are provisional OASIS mappings, not automatically ADNI-equivalent diagnoses.",
        "",
        "## Warnings",
        "",
        warning_block,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(path: Path, strict: pd.DataFrame, relaxed: pd.DataFrame, balance: pd.DataFrame, warnings: Sequence[str]) -> None:
    strict_counts = strict["provisional_label"].value_counts().to_dict() if not strict.empty else {}
    relaxed_counts = relaxed["provisional_label"].value_counts().to_dict() if not relaxed.empty else {}
    strict_scanner = (
        strict.groupby(["manufacturer", "scanner_model", "TR"], dropna=False).size().reset_index(name="n").sort_values("n", ascending=False).head(10)
        if not strict.empty
        else pd.DataFrame()
    )
    scanner_lines = []
    for _, row in strict_scanner.iterrows():
        scanner_lines.append(f"- {row['manufacturer']} / {row['scanner_model']} / TR={row['TR']}: {int(row['n'])}")
    warning_block = "\n".join(f"- {w}" for w in warnings) if warnings else "- None."
    recommendation = "proceed_to_smoke_test_then_qc" if len(strict) >= 10 and set(strict["provisional_label"]) >= {"CN", "AD_DEMENTIA"} else "revise_selection_before_download"
    lines = [
        "# OASIS-3 External Validation Cohort Sizing",
        "",
        "Metadata-only audit. No images were downloaded, no preprocessing or training was run, and no arrays/checkpoints/joblibs were loaded.",
        "",
        "## Summary",
        "",
        f"- Strict compatible CN/AD_DEMENTIA subjects: {len(strict)} ({json.dumps(strict_counts, sort_keys=True)})",
        f"- Relaxed compatible CN/AD_DEMENTIA subjects: {len(relaxed)} ({json.dumps(relaxed_counts, sort_keys=True)})",
        f"- Strict class balance: `{class_balance_verdict(strict)}`",
        f"- Strict scanner domain: `{scanner_balance_verdict(strict)}`",
        f"- Recommendation: `{recommendation}`",
        "",
        "## Strict Scanner/TR Distribution",
        "",
        "\n".join(scanner_lines) if scanner_lines else "- None.",
        "",
        "## Main Limitations",
        "",
        "- Metadata cannot confirm image usability, motion, frame count, ROI signal quality, or preprocessing compatibility.",
        "- OASIS clinical labels are provisional mappings and should be reviewed before external-validation claims.",
        "- External validation should remain separate from ADNI training unless a domain-harmonized multi-cohort design is explicitly defined.",
        "",
        "## Warnings",
        "",
        warning_block,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(path: Path, strict: pd.DataFrame, download_root: Path) -> None:
    manifest = pd.DataFrame(
        [
            {
                "experiment_id": row["experiment_id"],
                "subject_id": row["subject_id"],
                "session_id": row["session_id"],
                "target_label": row["provisional_label"],
                "scan_type_requested": "bold",
                "output_root": str(download_root),
                "include_in_download": True,
                "reason": "strict_metadata_eligible; download only after manual review and smoke-test plan approval",
            }
            for _, row in strict.iterrows()
        ]
    )
    manifest.to_csv(path, index=False)


def main() -> int:
    args = parse_args()
    metadata_raw = resolve(args.metadata_raw)
    clinical_dir = resolve(args.clinical_audit_dir)
    feasibility_dir = resolve(args.feasibility_dir)
    smoke_review_dir = resolve(args.smoke_review_dir)
    output_dir = prepare_output_dir(args.output_dir, args.overwrite)
    if not metadata_raw.exists():
        raise FileNotFoundError(f"Missing metadata raw root: {metadata_raw}")

    rest_path = require_file(clinical_dir / REST_CANDIDATES, "rest BOLD clinical candidates")
    subject_map_path = require_file(clinical_dir / SUBJECT_MAPPING, "subject-level clinical mapping")
    feasibility_path = feasibility_dir / FEASIBILITY_INVENTORY
    rest = load_csv(rest_path)
    subject_map = load_csv(subject_map_path)
    clinical_rows, clinical_files, clinical_warnings = load_key_clinical_rows(metadata_raw)
    mr_json, mr_json_source, mr_warnings = load_mr_json(metadata_raw)
    mr_lookup = build_mr_lookup(mr_json)
    clinical_index = build_clinical_index(clinical_rows)
    candidates = enrich_candidates(rest, feasibility_path, mr_lookup, clinical_index, subject_map)
    strict = select_one_per_subject(candidates, "eligible_strict")
    relaxed = select_one_per_subject(candidates, "eligible_relaxed")
    counts = build_counts(strict, relaxed, candidates)
    balance = build_balance_table(strict, relaxed)
    warnings = clinical_warnings + mr_warnings
    if not smoke_review_dir.exists():
        warnings.append(f"smoke_review_dir_missing:{smoke_review_dir}")
    if candidates["n_volumes"].astype(str).str.strip().replace("nan", "").eq("").all():
        warnings.append("n_volumes_unavailable_in_metadata_for_all_candidates")

    candidates[OUTPUT_COLUMNS].to_csv(output_dir / "oasis3_external_validation_eligibility_subjects.csv", index=False)
    strict.to_csv(output_dir / "oasis3_external_validation_selected_subjects_strict.csv", index=False)
    relaxed.to_csv(output_dir / "oasis3_external_validation_selected_subjects_relaxed.csv", index=False)
    counts.to_csv(output_dir / "oasis3_external_validation_counts.csv", index=False)
    balance.to_csv(output_dir / "oasis3_external_validation_balance_table.csv", index=False)
    write_manifest(output_dir / "oasis3_external_validation_download_manifest_strict.csv", strict, args.download_root)
    write_design_recommendation(output_dir / "oasis3_external_validation_design_recommendation.md", strict, relaxed, candidates, warnings)
    write_readme(output_dir / "README.md", strict, relaxed, balance, warnings)

    audit_manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_hash": git_hash(),
        "project_root": str(PROJECT_ROOT),
        "metadata_raw": str(metadata_raw),
        "clinical_audit_dir": str(clinical_dir),
        "feasibility_dir": str(feasibility_dir),
        "smoke_review_dir": str(smoke_review_dir),
        "output_dir": str(output_dir),
        "inputs": {
            "rest_bold_clinical_candidates": str(rest_path),
            "subject_level_clinical_mapping": str(subject_map_path),
            "feasibility_session_inventory": str(feasibility_path),
            "mr_json_metadata": mr_json_source,
            "clinical_files": clinical_files,
        },
        "outputs": {
            "eligibility_subjects": str(output_dir / "oasis3_external_validation_eligibility_subjects.csv"),
            "selected_strict": str(output_dir / "oasis3_external_validation_selected_subjects_strict.csv"),
            "selected_relaxed": str(output_dir / "oasis3_external_validation_selected_subjects_relaxed.csv"),
            "counts": str(output_dir / "oasis3_external_validation_counts.csv"),
            "balance_table": str(output_dir / "oasis3_external_validation_balance_table.csv"),
            "design_recommendation": str(output_dir / "oasis3_external_validation_design_recommendation.md"),
            "strict_download_manifest": str(output_dir / "oasis3_external_validation_download_manifest_strict.csv"),
            "readme": str(output_dir / "README.md"),
        },
        "strict_rule": "CN/AD_DEMENTIA; high confidence; task-rest BOLD; TR and scanner present; clinical-to-MR delta <=90 days when computable; one selected session per subject.",
        "relaxed_rule": "CN/AD_DEMENTIA; high or medium confidence; task-rest BOLD; TR and scanner present; clinical-to-MR delta <=365 days when computable; one selected session per subject.",
        "selection_rule": "Sort by subject, confidence rank descending, clinical-to-MR delta ascending, MR session day ascending, TR present, json filename; keep first per subject.",
        "no_image_download": True,
        "no_preprocessing": True,
        "no_training": True,
        "no_large_copy": True,
        "no_large_array_loading": True,
        "warnings": warnings,
    }
    (output_dir / "audit_manifest.json").write_text(json.dumps(audit_manifest, indent=2), encoding="utf-8")

    print(f"Wrote outputs to {output_dir}")
    print(f"Candidate BOLD rows: {len(candidates)}")
    print(f"Strict selected subjects: {len(strict)}")
    print(f"Relaxed selected subjects: {len(relaxed)}")
    print(f"Warnings: {len(warnings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
