#!/usr/bin/env python3
"""Read-only OASIS-3 clinical-label mapping audit.

This script reads only OASIS-3 metadata files and the previous feasibility
session inventory. It does not download images, preprocess, train, copy large
files, or load arrays/checkpoints/joblibs.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA_RAW = Path("/media/diego/Datos/vae_AD_data/OASIS3/metadata_raw")
DEFAULT_SYMLINK_METADATA_RAW = PROJECT_ROOT / "data/oasis3/metadata_raw"
DEFAULT_FEASIBILITY_INVENTORY = PROJECT_ROOT / "results/revision_bspc_2026/oasis3_feasibility_audit/oasis3_session_inventory.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/oasis3_clinical_mapping_audit"

SUBJECT_CANDIDATES = ["OASISID", "OASIS3_id", "subject_id", "participant_id", "Subject", "SubjectID", "OASIS_ID"]
SESSION_CANDIDATES = ["OASIS_session_label", "session_id", "session", "Session", "Visit", "visit"]
DATE_CANDIDATES = ["date", "Date", "scan_date", "ScanDate", "acq_date", "Acq Date", "AcquisitionDate"]
DAY_CANDIDATES = ["days_to_visit", "days_from_entry", "days_from_baseline", "days", "DaysFromEntry"]
AGE_CANDIDATES = ["age at visit", "AgeatEntry", "age", "Age", "AGE", "age_at_visit", "age_at_scan"]
SEX_CANDIDATES = ["GENDER", "INSEX", "sex", "Sex", "gender", "Gender"]

CLINICAL_NAME_PATTERNS = [
    "demo",
    "udsa1",
    "udsa2",
    "udsb4",
    "udsd1",
    "udsd2",
    "cognorm",
    "psychometrics",
    "pychometrics",
    "cdr",
    "diagnos",
    "clinical",
    "unchanged_cdr",
]

LABEL_COLUMN_REGEX = re.compile(
    r"(cdr|cdrsum|cdrtot|naccudsd|naccalzd|naccadc|demented|diagnos|diagnosis|dx|cognitive|cognit|normal|mci|prob(ad)?|poss(ad)?|alz|ad$|dementia|normcog|alzdis)",
    re.IGNORECASE,
)
CDR_COLUMN_REGEX = re.compile(r"(^cdr$|cdrtot|cdrsum|cdr_global|cdrglob|clinicaldementiarating|min of cdrtot|max of cdrtot)", re.IGNORECASE)
DX_COLUMN_REGEX = re.compile(r"(^dx\d*$|diagnos|diagnosis|demented|normcog|mci|probad|possad|alzdis|naccudsd|naccalzd|naccadc)", re.IGNORECASE)

AD_FLAG_COLUMNS = ["PROBAD", "POSSAD", "alzdis", "NACCALZD", "NACCADC"]
MCI_FLAG_PREFIXES = ["MCI", "MCIN", "MCIAMEM", "MCIAPLUS", "MCINON"]
DX_TEXT_COLUMNS = ["dx1", "dx2", "dx3", "dx4", "dx5", "diagnosis", "Diagnosis", "clinical_diagnosis", "cognitive_status"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only OASIS-3 clinical-label mapping audit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metadata-raw", type=Path, default=DEFAULT_METADATA_RAW)
    parser.add_argument("--symlink-metadata-raw", type=Path, default=DEFAULT_SYMLINK_METADATA_RAW)
    parser.add_argument("--feasibility-session-inventory", type=Path, default=DEFAULT_FEASIBILITY_INVENTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-dictionary-bytes", type=int, default=25_000_000)
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


def normalize_subject(value: Any) -> str:
    text = safe_text(value)
    if not text:
        return ""
    text = re.sub(r"\s+", "", text)
    if text.startswith("sub-"):
        return text
    match = re.search(r"OAS\d{5}", text, flags=re.IGNORECASE)
    if match:
        return "sub-" + match.group(0).upper()
    return text


def normalize_session(value: Any) -> str:
    text = safe_text(value)
    if not text:
        return ""
    text = re.sub(r"\s+", "", text)
    return text


def parse_float(value: Any) -> Optional[float]:
    text = safe_text(value)
    if not text or text in {".", "NA"}:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def parse_day(value: Any) -> Optional[float]:
    text = safe_text(value)
    if not text:
        return None
    match = re.search(r"_d(\d+)|ses-d(\d+)|MR_d(\d+)|UDS[a-z0-9]*_d(\d+)|psychometrics_d(\d+)|d(\d{3,5})", text, re.IGNORECASE)
    if match:
        for group in match.groups():
            if group is not None:
                return float(group)
    return parse_float(text)


def normalize_date(value: Any) -> str:
    text = safe_text(value)
    if not text:
        return ""
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return ""
    return parsed.strftime("%Y-%m-%d")


def first_col(columns: Sequence[str], candidates: Sequence[str]) -> str:
    lower = {str(c).strip().lower(): str(c) for c in columns}
    for cand in candidates:
        if cand in columns:
            return cand
        key = cand.lower()
        if key in lower:
            return lower[key]
    return ""


def join_unique(values: Iterable[Any], max_items: int = 30) -> str:
    seen = set()
    out = []
    for value in values:
        text = safe_text(value)
        if not text:
            continue
        if text not in seen:
            seen.add(text)
            out.append(text)
        if len(out) >= max_items:
            out.append("...")
            break
    return "|".join(out)


def list_metadata_files(roots: Sequence[Path]) -> List[Path]:
    seen = set()
    out = []
    for root in roots:
        root = resolve(root)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() in {".csv", ".tsv", ".json", ".txt", ".pdf", ".xls", ".xlsx"}:
                key = str(path.resolve())
                if key not in seen:
                    seen.add(key)
                    out.append(path)
    return sorted(out, key=lambda p: str(p))


def read_header(path: Path) -> Tuple[List[str], int, int, str]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        try:
            df0 = pd.read_csv(path, sep=sep, nrows=0)
            n_rows = sum(1 for _ in path.open("r", encoding="utf-8", errors="replace")) - 1
            return list(df0.columns), max(n_rows, 0), len(df0.columns), "readable_table"
        except Exception as exc:
            return [], 0, 0, f"failed_header: {exc}"
    if suffix == ".json":
        try:
            obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            cols = list(obj.keys()) if isinstance(obj, dict) else []
            return cols, 1, len(cols), "json_metadata"
        except Exception as exc:
            return [], 0, 0, f"failed_json: {exc}"
    if suffix in {".xlsx", ".xls"}:
        return [], 0, 0, "dictionary_excel_not_tabular_without_openpyxl"
    if suffix == ".pdf":
        return [], 0, 0, "pdf_dictionary_not_ocrd"
    return [], 0, 0, "text_or_other_metadata"


def candidate_label_columns(columns: Sequence[str]) -> List[str]:
    return [col for col in columns if LABEL_COLUMN_REGEX.search(str(col))]


def suggested_role(col: str) -> str:
    low = col.lower()
    if re.search(r"cdrtot|cdr_global|cdrglob|^cdr$|min of cdrtot|max of cdrtot", low):
        return "CDR_GLOBAL"
    if "cdrsum" in low:
        return "CDR_SUM"
    if "demented" in low:
        return "DEMENTIA_STATUS"
    if "normcog" in low or "cognit" in low or "normal" in low:
        return "COGNITIVE_STATUS"
    if re.search(r"dx|diagnos|probad|possad|alzdis|naccalzd|naccadc", low):
        return "DIAGNOSIS"
    if "mci" in low:
        return "COGNITIVE_STATUS"
    return "UNKNOWN"


def why_candidate(col: str) -> str:
    reasons = []
    if CDR_COLUMN_REGEX.search(col):
        reasons.append("matches_CDR_pattern")
    if DX_COLUMN_REGEX.search(col):
        reasons.append("matches_diagnosis_pattern")
    if LABEL_COLUMN_REGEX.search(col):
        reasons.append("matches_label_keyword")
    return "|".join(reasons) or "candidate_by_file_context"


def is_likely_clinical(path: Path, columns: Sequence[str]) -> bool:
    name = path.name.lower()
    if any(pattern in name for pattern in CLINICAL_NAME_PATTERNS):
        return True
    if candidate_label_columns(columns):
        return True
    return False


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    sep = "\t" if suffix == ".tsv" else ","
    return pd.read_csv(path, sep=sep, dtype=str, low_memory=False)


def value_counts_summary(series: pd.Series, max_items: int = 12) -> str:
    counts = series.fillna("").astype(str).str.strip()
    counts = counts[counts != ""]
    vc = counts.value_counts(dropna=False).head(max_items)
    return "; ".join(f"{idx}:{int(val)}" for idx, val in vc.items())


def row_flag(row: pd.Series, columns: Sequence[str], names: Sequence[str]) -> bool:
    lower_to_col = {str(c).lower(): c for c in columns}
    for name in names:
        col = lower_to_col.get(name.lower())
        if col is None:
            continue
        val = parse_float(row.get(col, ""))
        if val is not None and val == 1:
            return True
        text = safe_text(row.get(col, "")).lower()
        if text in {"yes", "true", "present"}:
            return True
    return False


def row_any_prefix_flag(row: pd.Series, columns: Sequence[str], prefixes: Sequence[str]) -> bool:
    for col in columns:
        if any(str(col).upper().startswith(prefix.upper()) for prefix in prefixes):
            val = parse_float(row.get(col, ""))
            if val is not None and val == 1:
                return True
    return False


def row_text(row: pd.Series, columns: Sequence[str]) -> str:
    parts = []
    for col in columns:
        if col in row.index:
            text = safe_text(row.get(col, ""))
            if text and text != ".":
                parts.append(text)
    return " ".join(parts).lower()


def cdr_global_from_row(row: pd.Series, columns: Sequence[str]) -> Optional[float]:
    for preferred in ["CDRTOT", "CDR", "cdr", "Min of CDRTOT", "Max of CDRTOT"]:
        for col in columns:
            if str(col).lower() == preferred.lower():
                val = parse_float(row.get(col, ""))
                if val is not None:
                    return val
    for col in columns:
        if CDR_COLUMN_REGEX.search(str(col)) and "sum" not in str(col).lower():
            val = parse_float(row.get(col, ""))
            if val is not None:
                return val
    return None


def cdr_sum_from_row(row: pd.Series, columns: Sequence[str]) -> Optional[float]:
    for col in columns:
        if "cdrsum" in str(col).lower():
            return parse_float(row.get(col, ""))
    return None


def classify_clinical_row(row: pd.Series, columns: Sequence[str]) -> Tuple[str, str, str, str]:
    cdr = cdr_global_from_row(row, columns)
    cdrsum = cdr_sum_from_row(row, columns)
    text = row_text(row, [c for c in columns if str(c).lower() in {x.lower() for x in DX_TEXT_COLUMNS} or str(c).lower().startswith("dx")])
    normcog = row_flag(row, columns, ["NORMCOG"])
    demented = row_flag(row, columns, ["DEMENTED"])
    ad_flag = row_flag(row, columns, AD_FLAG_COLUMNS)
    mci_flag = row_any_prefix_flag(row, columns, MCI_FLAG_PREFIXES)

    has_cn_text = bool(re.search(r"cognitively normal|normal control|no dementia|non[- ]?demented", text))
    has_ad_text = bool(re.search(r"alzheimer|ad dementia|probable ad|possible ad|dementia of the alzheimer", text))
    has_dementia_text = bool(re.search(r"\bdementia\b", text)) and not has_cn_text
    has_mci_text = bool(re.search(r"\bmci\b|mild cognitive impairment", text))

    evidence = []
    if cdr is not None:
        evidence.append(f"CDR={cdr:g}")
    if cdrsum is not None:
        evidence.append(f"CDRSUM={cdrsum:g}")
    if normcog:
        evidence.append("NORMCOG=1")
    if demented:
        evidence.append("DEMENTED=1")
    if ad_flag:
        evidence.append("AD_flag=1")
    if mci_flag:
        evidence.append("MCI_flag=1")
    if text:
        evidence.append(f"text={text[:120]}")

    if cdr is not None:
        if cdr == 0 and (normcog or has_cn_text) and not (demented or ad_flag or mci_flag or has_ad_text or has_mci_text):
            return "CN", "high", "CDR==0 plus normal cognition evidence", "|".join(evidence)
        if cdr == 0 and not (demented or ad_flag or mci_flag or has_ad_text or has_mci_text):
            return "CN", "medium", "CDR==0 without conflicting impairment evidence", "|".join(evidence)
        if cdr == 0.5 and not (demented or ad_flag or has_ad_text):
            return "MCI", "medium", "CDR==0.5 without dementia/AD evidence", "|".join(evidence)
        if cdr >= 1 and (demented or ad_flag or has_ad_text or has_dementia_text):
            return "AD_DEMENTIA", "high", "CDR>=1 plus dementia/AD evidence", "|".join(evidence)
        if cdr >= 1:
            return "AD_DEMENTIA", "medium", "CDR>=1 interpreted as dementia", "|".join(evidence)

    if has_ad_text or (demented and ad_flag):
        return "AD_DEMENTIA", "high", "explicit AD dementia or dementia with AD flag", "|".join(evidence)
    if demented or has_dementia_text:
        return "AD_DEMENTIA", "medium", "dementia evidence without specific AD confirmation", "|".join(evidence)
    if has_mci_text or mci_flag:
        return "MCI", "medium", "MCI evidence", "|".join(evidence)
    if normcog or has_cn_text:
        return "CN", "high", "explicit normal cognition evidence", "|".join(evidence)
    return "UNKNOWN", "low", "no clear CN/MCI/AD rule matched", "|".join(evidence)


def confidence_rank(conf: str) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get(conf, 0)


def label_priority(label: str) -> int:
    return {"AD_DEMENTIA": 4, "MCI": 3, "CN": 2, "UNKNOWN": 1}.get(label, 0)


def aggregate_subject_label(rows: pd.DataFrame) -> Tuple[str, str, str, str]:
    if rows.empty:
        return "UNKNOWN", "low", "no clinical rows found", ""
    labels = rows["row_label"].tolist()
    high_labels = set(rows.loc[rows["row_label_confidence"] == "high", "row_label"])
    medium_labels = set(rows.loc[rows["row_label_confidence"].isin(["high", "medium"]), "row_label"])
    warning = ""
    if "AD_DEMENTIA" in high_labels:
        label, conf, rule = "AD_DEMENTIA", "high", "at least one high-confidence AD/dementia clinical row"
    elif "MCI" in high_labels and "AD_DEMENTIA" not in medium_labels:
        label, conf, rule = "MCI", "high", "MCI high-confidence evidence and no AD/dementia evidence"
    elif set(labels).issubset({"CN", "UNKNOWN"}) and "CN" in high_labels:
        label, conf, rule = "CN", "high", "normal cognition evidence with no MCI/AD evidence across candidate rows"
    elif "AD_DEMENTIA" in medium_labels:
        label, conf, rule = "AD_DEMENTIA", "medium", "medium-confidence dementia/AD evidence"
    elif "MCI" in medium_labels:
        label, conf, rule = "MCI", "medium", "medium-confidence MCI evidence"
    elif "CN" in medium_labels:
        label, conf, rule = "CN", "medium", "CDR/normal evidence but not fully high-confidence"
    else:
        label, conf, rule = "UNKNOWN", "low", "no clear subject-level label"
    non_unknown = sorted(set(l for l in labels if l != "UNKNOWN"))
    if len(non_unknown) > 1:
        warning = f"longitudinal_or_conflicting_labels:{'|'.join(non_unknown)}"
    return label, conf, rule, warning


def extract_candidate_values_for_subject(rows: pd.DataFrame, candidate_cols: Sequence[str]) -> Dict[str, str]:
    out = {}
    if rows.empty:
        return out
    for col in candidate_cols:
        if col in rows.columns:
            out[f"candidate_{sanitize_col(col)}"] = join_unique(rows[col], max_items=20)
    return out


def sanitize_col(col: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9]+", "_", str(col)).strip("_")
    return clean[:80] or "col"


def build_file_inventory(files: Sequence[Path]) -> Tuple[pd.DataFrame, List[Path], Dict[str, List[str]]]:
    rows = []
    clinical_files = []
    file_candidate_cols: Dict[str, List[str]] = {}
    for path in files:
        cols, n_rows, n_cols, note = read_header(path)
        subject_col = first_col(cols, SUBJECT_CANDIDATES)
        session_col = first_col(cols, SESSION_CANDIDATES)
        date_col = first_col(cols, DATE_CANDIDATES)
        age_col = first_col(cols, AGE_CANDIDATES)
        sex_col = first_col(cols, SEX_CANDIDATES)
        label_cols = candidate_label_columns(cols)
        cdr_cols = [col for col in cols if CDR_COLUMN_REGEX.search(str(col))]
        dx_cols = [col for col in cols if DX_COLUMN_REGEX.search(str(col))]
        if is_likely_clinical(path, cols):
            clinical_files.append(path)
        file_candidate_cols[str(path)] = label_cols
        rows.append(
            {
                "path": str(path),
                "n_rows": n_rows,
                "n_cols": n_cols,
                "subject_col": subject_col,
                "session_col": session_col,
                "date_col": date_col,
                "candidate_label_cols": "|".join(label_cols),
                "candidate_cdr_cols": "|".join(cdr_cols),
                "candidate_dx_cols": "|".join(dx_cols),
                "candidate_age_cols": age_col,
                "candidate_sex_cols": sex_col,
                "notes": note,
            }
        )
    return pd.DataFrame(rows), clinical_files, file_candidate_cols


def build_candidate_column_rankings(clinical_files: Sequence[Path]) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], List[str]]:
    ranking_rows = []
    loaded: Dict[str, pd.DataFrame] = {}
    warnings = []
    for path in clinical_files:
        if path.suffix.lower() not in {".csv", ".tsv"}:
            continue
        try:
            df = load_table(path)
        except Exception as exc:
            warnings.append(f"failed_load_clinical_file:{path}:{exc}")
            continue
        loaded[str(path)] = df
        cols = candidate_label_columns(df.columns)
        for col in cols:
            series = df[col]
            nonempty = series.dropna().astype(str).str.strip()
            nonempty = nonempty[nonempty != ""]
            ranking_rows.append(
                {
                    "file": str(path),
                    "column": col,
                    "dtype": str(series.dtype),
                    "n_nonnull": int(len(nonempty)),
                    "n_unique": int(nonempty.nunique()),
                    "top_values_with_counts": value_counts_summary(series),
                    "why_candidate": why_candidate(col),
                    "suggested_role": suggested_role(col),
                }
            )
    rankings = pd.DataFrame(ranking_rows).sort_values(["suggested_role", "file", "column"]) if ranking_rows else pd.DataFrame(
        columns=["file", "column", "dtype", "n_nonnull", "n_unique", "top_values_with_counts", "why_candidate", "suggested_role"]
    )
    return rankings, loaded, warnings


def clinical_rows_from_loaded(loaded: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, List[str]]:
    rows = []
    warnings = []
    for path, df in loaded.items():
        subject_col = first_col(df.columns, SUBJECT_CANDIDATES)
        if not subject_col:
            continue
        session_col = first_col(df.columns, SESSION_CANDIDATES)
        day_col = first_col(df.columns, DAY_CANDIDATES)
        age_col = first_col(df.columns, AGE_CANDIDATES)
        sex_col = first_col(df.columns, SEX_CANDIDATES)
        candidate_cols = candidate_label_columns(df.columns)
        for idx, row in df.iterrows():
            subject = normalize_subject(row.get(subject_col, ""))
            if not subject:
                continue
            session = normalize_session(row.get(session_col, "")) if session_col else ""
            day = parse_float(row.get(day_col, "")) if day_col else parse_day(session)
            label, conf, rule, evidence = classify_clinical_row(row, df.columns)
            out = {
                "subject_id": subject,
                "clinical_session_label": session,
                "clinical_day": day,
                "age": safe_text(row.get(age_col, "")) if age_col else "",
                "sex": safe_text(row.get(sex_col, "")) if sex_col else "",
                "row_label": label,
                "row_label_confidence": conf,
                "row_label_rule": rule,
                "row_label_evidence": evidence,
                "clinical_source_file": path,
                "clinical_row_index": int(idx),
            }
            for col in candidate_cols:
                out[col] = safe_text(row.get(col, ""))
            rows.append(out)
    if not rows:
        return pd.DataFrame(), warnings
    return pd.DataFrame(rows), warnings


def build_subject_mapping(clinical_rows: pd.DataFrame) -> pd.DataFrame:
    if clinical_rows.empty:
        return pd.DataFrame(
            columns=[
                "subject_id",
                "sex",
                "age",
                "CDR_values",
                "diagnosis_cognitive_status_values",
                "provisional_label_rule",
                "provisional_label",
                "label_confidence",
                "label_warning",
            ]
        )
    all_candidate_cols = sorted([c for c in clinical_rows.columns if LABEL_COLUMN_REGEX.search(c)])
    rows = []
    for subject, sub in clinical_rows.groupby("subject_id"):
        label, conf, rule, warning = aggregate_subject_label(sub)
        cdr_cols = [c for c in all_candidate_cols if CDR_COLUMN_REGEX.search(c)]
        dx_cols = [c for c in all_candidate_cols if DX_COLUMN_REGEX.search(c) or c.lower().startswith("dx")]
        out = {
            "subject_id": subject,
            "sex": join_unique(sub["sex"], max_items=4),
            "age": join_unique(sub["age"], max_items=10),
            "CDR_values": "; ".join(f"{col}={join_unique(sub[col], max_items=10)}" for col in cdr_cols if col in sub.columns and join_unique(sub[col], max_items=10)),
            "diagnosis_cognitive_status_values": "; ".join(f"{col}={join_unique(sub[col], max_items=10)}" for col in dx_cols if col in sub.columns and join_unique(sub[col], max_items=10)),
            "provisional_label_rule": rule,
            "provisional_label": label,
            "label_confidence": conf,
            "label_warning": warning,
            "n_clinical_rows": int(len(sub)),
            "clinical_sources": join_unique(sub["clinical_source_file"], max_items=20),
        }
        out.update(extract_candidate_values_for_subject(sub, all_candidate_cols))
        rows.append(out)
    return pd.DataFrame(rows).sort_values("subject_id")


def parse_session_day(session_id: Any) -> Optional[float]:
    return parse_day(session_id)


def nearest_clinical_row(subject: str, session_day: Optional[float], clinical_rows: pd.DataFrame) -> Tuple[Optional[pd.Series], str, Optional[float]]:
    if clinical_rows.empty:
        return None, "no_clinical_rows", None
    sub = clinical_rows[clinical_rows["subject_id"] == subject].copy()
    if sub.empty:
        return None, "no_subject_clinical_match", None
    if session_day is not None and sub["clinical_day"].notna().any():
        sub["_abs_day_delta"] = (pd.to_numeric(sub["clinical_day"], errors="coerce") - session_day).abs()
        sub["_conf_rank"] = sub["row_label_confidence"].map(confidence_rank)
        sub["_label_rank"] = sub["row_label"].map(label_priority)
        best = sub.sort_values(["_abs_day_delta", "_conf_rank", "_label_rank"], ascending=[True, False, False]).iloc[0]
        delta = float(best["_abs_day_delta"]) if pd.notna(best["_abs_day_delta"]) else None
        return best, "nearest_clinical_day_match", delta
    sub["_conf_rank"] = sub["row_label_confidence"].map(confidence_rank)
    sub["_label_rank"] = sub["row_label"].map(label_priority)
    best = sub.sort_values(["_conf_rank", "_label_rank"], ascending=[False, False]).iloc[0]
    return best, "subject_level_clinical_match_no_day", None


def join_rest_bold_candidates(session_inventory_path: Path, clinical_rows: pd.DataFrame, subject_map: pd.DataFrame) -> pd.DataFrame:
    if not session_inventory_path.exists():
        return pd.DataFrame()
    sessions = pd.read_csv(session_inventory_path, dtype=str, low_memory=False)
    if "has_task_rest_bold" not in sessions.columns:
        return pd.DataFrame()
    rest = sessions[sessions["has_task_rest_bold"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    if rest.empty:
        return pd.DataFrame()
    subject_map_idx = subject_map.set_index("subject_id").to_dict(orient="index") if not subject_map.empty else {}
    rows = []
    for _, row in rest.iterrows():
        subject = safe_text(row.get("subject_id", ""))
        session_id = safe_text(row.get("session_id", ""))
        session_day = parse_session_day(session_id)
        nearest, match_type, day_delta = nearest_clinical_row(subject, session_day, clinical_rows)
        subj = subject_map_idx.get(subject, {})
        if nearest is not None:
            label = nearest["row_label"]
            conf = nearest["row_label_confidence"]
            rule = nearest["row_label_rule"]
            warning = ""
            if day_delta is not None and day_delta > 365:
                conf = "medium" if conf == "high" else conf
                warning = f"nearest_clinical_visit_far:{day_delta:.0f}_days"
        else:
            label = subj.get("provisional_label", "UNKNOWN")
            conf = subj.get("label_confidence", "low")
            rule = subj.get("provisional_label_rule", "")
            warning = "no_session_level_clinical_match"
        tr = safe_text(row.get("TR_RepetitionTime", ""))
        scanner = safe_text(row.get("scanner_manufacturer", ""))
        candidate = (
            label in {"CN", "AD_DEMENTIA"}
            and conf == "high"
            and bool(tr)
            and bool(scanner)
            and (day_delta is None or day_delta <= 365)
        )
        rows.append(
            {
                "subject_id": subject,
                "session_id": session_id,
                "has_task_rest_bold": row.get("has_task_rest_bold", ""),
                "RepetitionTime": tr,
                "scanner_manufacturer": scanner,
                "scanner_model": row.get("scanner_model", ""),
                "bold_file_path": row.get("bold_file_path", ""),
                "json_path": row.get("json_path", ""),
                "session_day": session_day,
                "clinical_match_type": match_type,
                "clinical_day_delta": day_delta,
                "provisional_label": label,
                "label_confidence": conf,
                "provisional_label_rule": rule,
                "candidate_for_smoke_test": bool(candidate),
                "label_warning": warning or subj.get("label_warning", ""),
            }
        )
    return pd.DataFrame(rows)


def select_smoke_candidates(rest_candidates: pd.DataFrame) -> pd.DataFrame:
    if rest_candidates.empty:
        return pd.DataFrame(
            columns=[
                "subject_id",
                "session_id",
                "provisional_label",
                "label_confidence",
                "RepetitionTime",
                "scanner_manufacturer",
                "scanner_model",
                "json_path",
                "bold_file_path",
                "reason",
            ]
        )
    eligible = rest_candidates[rest_candidates["candidate_for_smoke_test"] == True].copy()  # noqa: E712
    if eligible.empty:
        return pd.DataFrame()
    eligible["_session_day_sort"] = pd.to_numeric(eligible["session_day"], errors="coerce").fillna(10**9)
    eligible["_clinical_delta_sort"] = pd.to_numeric(eligible["clinical_day_delta"], errors="coerce").fillna(10**9)
    selected = []
    for label in ["CN", "AD_DEMENTIA"]:
        sub = eligible[eligible["provisional_label"] == label].copy()
        sub = sub.sort_values(["_clinical_delta_sort", "_session_day_sort", "subject_id", "session_id"])
        seen = set()
        for _, row in sub.iterrows():
            if row["subject_id"] in seen:
                continue
            seen.add(row["subject_id"])
            selected.append(
                {
                    "subject_id": row["subject_id"],
                    "session_id": row["session_id"],
                    "provisional_label": row["provisional_label"],
                    "label_confidence": row["label_confidence"],
                    "RepetitionTime": row["RepetitionTime"],
                    "scanner_manufacturer": row["scanner_manufacturer"],
                    "scanner_model": row["scanner_model"],
                    "json_path": row["json_path"],
                    "bold_file_path": row["bold_file_path"],
                    "reason": f"high-confidence {label}; task-rest BOLD; TR/scanner present; one session per subject; clinical day delta={row['clinical_day_delta']}",
                }
            )
            if len([x for x in selected if x["provisional_label"] == label]) >= 5:
                break
    return pd.DataFrame(selected)


def dictionary_text_from_xlsx(path: Path, max_bytes: int) -> str:
    if path.stat().st_size > max_bytes:
        return ""
    parts = []
    try:
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if not (name.endswith(".xml") and ("sharedStrings" in name or "worksheets" in name)):
                    continue
                try:
                    text = zf.read(name).decode("utf-8", errors="replace")
                except Exception:
                    continue
                text = re.sub(r"<[^>]+>", " ", text)
                text = re.sub(r"\s+", " ", text)
                parts.append(text)
    except Exception:
        return ""
    return "\n".join(parts)


def readable_text(path: Path, max_bytes: int) -> str:
    suffix = path.suffix.lower()
    if path.stat().st_size > max_bytes:
        return ""
    if suffix == ".xlsx":
        return dictionary_text_from_xlsx(path, max_bytes)
    if suffix in {".txt", ".csv", ".tsv", ".json", ".xml"}:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""
    if suffix == ".pdf":
        return ""
    return ""


def write_dictionary_matches(path: Path, files: Sequence[Path], candidate_cols: Sequence[str], max_bytes: int) -> None:
    lines = [
        "# OASIS-3 Dictionary Matches",
        "",
        "Text-readable dictionary/metadata files were searched for candidate label column names. PDFs were not OCRed.",
        "",
    ]
    dictionary_files = [p for p in files if "dictionary" in p.name.lower() or p.suffix.lower() in {".xlsx", ".xls", ".pdf", ".txt"}]
    if not dictionary_files:
        lines.append("No dictionary-like files found.")
    for dfile in dictionary_files:
        lines.extend([f"## {dfile}", ""])
        text = readable_text(dfile, max_bytes)
        if not text:
            lines.append("- Not text-readable without OCR/openpyxl, or over size limit.")
            lines.append("")
            continue
        lower = text.lower()
        found_any = False
        for col in sorted(set(candidate_cols), key=str.lower):
            idx = lower.find(str(col).lower())
            if idx < 0:
                continue
            found_any = True
            start = max(0, idx - 180)
            end = min(len(text), idx + len(col) + 280)
            snippet = re.sub(r"\s+", " ", text[start:end]).strip()
            lines.append(f"- `{col}`: {snippet}")
        if not found_any:
            lines.append("- No candidate column names matched.")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(
    path: Path,
    clinical_files: Sequence[Path],
    candidate_rankings: pd.DataFrame,
    subject_map: pd.DataFrame,
    rest_candidates: pd.DataFrame,
    smoke: pd.DataFrame,
    warnings: Sequence[str],
) -> str:
    label_counts = subject_map["provisional_label"].value_counts(dropna=False).to_dict() if not subject_map.empty else {}
    rest_counts = rest_candidates["provisional_label"].value_counts(dropna=False).to_dict() if not rest_candidates.empty else {}
    high_rest = rest_candidates[rest_candidates["label_confidence"] == "high"] if not rest_candidates.empty else pd.DataFrame()
    ready = len(smoke) >= 2 and {"CN", "AD_DEMENTIA"}.issubset(set(smoke.get("provisional_label", [])))
    verdict = "READY_FOR_SMOKE_TEST" if ready else "NEED_MANUAL_LABEL_REVIEW"
    used_files = [p for p in clinical_files if any(token in p.name.lower() for token in ["udsb4", "udsd1", "udsa1", "udsa2", "demographics", "unchanged_cdr"])]
    role_counts = candidate_rankings["suggested_role"].value_counts().to_dict() if not candidate_rankings.empty else {}
    lines = [
        "# OASIS-3 Clinical Mapping Audit",
        "",
        "Read-only metadata audit. No image download, preprocessing, training, checkpoint loading, or large copying was performed.",
        "",
        "## Clinical Files Used",
    ]
    if used_files:
        lines.extend(f"- `{p}`" for p in used_files)
    else:
        lines.append("- No clinical files with usable subject columns were found.")
    lines.extend(
        [
            "",
            "## Candidate Diagnostic/CDR Columns",
            f"- CDR global columns detected: `{int(role_counts.get('CDR_GLOBAL', 0))}`",
            f"- CDR sum columns detected: `{int(role_counts.get('CDR_SUM', 0))}`",
            f"- Diagnosis columns detected: `{int(role_counts.get('DIAGNOSIS', 0))}`",
            f"- Dementia-status columns detected: `{int(role_counts.get('DEMENTIA_STATUS', 0))}`",
            f"- Cognitive-status columns detected: `{int(role_counts.get('COGNITIVE_STATUS', 0))}`",
            "",
            "Primary mapping evidence used by rules includes `CDRTOT`, `CDRSUM`, `dx1..dx5`, `NORMCOG`, `DEMENTED`, `MCI*`, `PROBAD`, `POSSAD`, and `alzdis` when present.",
            "",
            "## Subject-Level Provisional Label Counts",
            f"- CN: `{int(label_counts.get('CN', 0))}`",
            f"- MCI: `{int(label_counts.get('MCI', 0))}`",
            f"- AD_DEMENTIA: `{int(label_counts.get('AD_DEMENTIA', 0))}`",
            f"- UNKNOWN: `{int(label_counts.get('UNKNOWN', 0))}`",
            "",
            "## Rest-BOLD Label-Mapped Sessions",
            f"- Rest-BOLD rows joined to clinical labels: `{len(rest_candidates)}`",
            f"- Rest-BOLD CN rows: `{int(rest_counts.get('CN', 0))}`",
            f"- Rest-BOLD MCI rows: `{int(rest_counts.get('MCI', 0))}`",
            f"- Rest-BOLD AD_DEMENTIA rows: `{int(rest_counts.get('AD_DEMENTIA', 0))}`",
            f"- Rest-BOLD UNKNOWN rows: `{int(rest_counts.get('UNKNOWN', 0))}`",
            f"- High-confidence rest-BOLD rows: `{len(high_rest)}`",
            f"- Smoke-test candidates selected: `{len(smoke)}`",
            "",
            "## Feasibility Verdict",
            f"`{verdict}`",
            "",
            "## Caveats",
            "- These labels are provisional and must be documented before claiming ADNI-compatible CN/AD external validation.",
            "- OASIS clinical status is longitudinal; session-level matching uses nearest clinical day when possible.",
            "- Smoke-test candidates require manual review of temporal alignment between MRI and clinical visit.",
            "- CDR>=1 plus dementia/AD evidence is mapped to `AD_DEMENTIA`; CDR=0 plus normal cognition evidence is mapped to `CN`; CDR=0.5 or MCI evidence is mapped to `MCI`.",
        ]
    )
    if warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {w}" for w in warnings)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return verdict


def main() -> int:
    args = parse_args()
    metadata_raw = resolve(args.metadata_raw)
    symlink_raw = resolve(args.symlink_metadata_raw)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings: List[str] = []

    files = list_metadata_files([metadata_raw, symlink_raw])
    inventory, clinical_files, _ = build_file_inventory(files)
    rankings, loaded, rank_warnings = build_candidate_column_rankings(clinical_files)
    warnings.extend(rank_warnings)
    clinical_rows, clinical_warnings = clinical_rows_from_loaded(loaded)
    warnings.extend(clinical_warnings)
    subject_map = build_subject_mapping(clinical_rows)
    rest_candidates = join_rest_bold_candidates(resolve(args.feasibility_session_inventory), clinical_rows, subject_map)
    smoke = select_smoke_candidates(rest_candidates)

    all_candidate_cols = []
    if not rankings.empty:
        all_candidate_cols = rankings["column"].astype(str).tolist()
    write_dictionary_matches(output_dir / "dictionary_matches.md", files, all_candidate_cols, args.max_dictionary_bytes)

    inventory.to_csv(output_dir / "clinical_file_column_inventory.csv", index=False)
    rankings.to_csv(output_dir / "candidate_label_columns_ranked.csv", index=False)
    subject_map.to_csv(output_dir / "subject_level_clinical_mapping_draft.csv", index=False)
    rest_candidates.to_csv(output_dir / "oasis3_rest_bold_clinical_candidates.csv", index=False)
    smoke.to_csv(output_dir / "oasis3_smoke_test_candidates_clinical.csv", index=False)
    verdict = write_readme(output_dir / "README.md", clinical_files, rankings, subject_map, rest_candidates, smoke, warnings)

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": git_hash(),
        "paths": {
            "metadata_raw": str(metadata_raw),
            "symlink_metadata_raw": str(symlink_raw),
            "feasibility_session_inventory": str(resolve(args.feasibility_session_inventory)),
            "output_dir": str(output_dir),
        },
        "n_metadata_files_seen": len(files),
        "n_clinical_files": len(clinical_files),
        "n_subjects_mapped": int(len(subject_map)),
        "n_rest_bold_rows_joined": int(len(rest_candidates)),
        "n_smoke_candidates": int(len(smoke)),
        "verdict": verdict,
        "warnings": warnings,
        "no_training": True,
        "no_image_download": True,
        "no_preprocessing": True,
        "no_large_copy": True,
    }
    (output_dir / "audit_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print((output_dir / "README.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
