#!/usr/bin/env python3
"""Prepare a metadata-only OASIS-3 smoke-test download review.

This script does not download images, preprocess, train, copy large files, or
load arrays. It joins the selected OASIS-3 smoke-test candidates with existing
clinical and MR JSON metadata so the BOLD download manifest can be manually
reviewed before any imaging data are fetched.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision_bspc_2026 import oasis3_clinical_mapping_audit as clinical  # noqa: E402


DEFAULT_METADATA_RAW = Path("/media/diego/Datos/vae_AD_data/OASIS3/metadata_raw")
DEFAULT_CLINICAL_AUDIT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/oasis3_clinical_mapping_audit"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/oasis3_smoke_test_pre_download_review"
DEFAULT_DOWNLOAD_ROOT = Path("/media/diego/Datos/vae_AD_data/OASIS3/smoke_test_raw")

SMOKE_CANDIDATES_FILE = "oasis3_smoke_test_candidates_clinical.csv"
REST_CANDIDATES_FILE = "oasis3_rest_bold_clinical_candidates.csv"
SUBJECT_MAPPING_FILE = "subject_level_clinical_mapping_draft.csv"

CLINICAL_FILE_BASENAMES = [
    "OASIS3_UDSb4_cdr.csv",
    "OASIS3_UDSd1_diagnoses.csv",
    "OASIS3_UDSa1_participant_demo.csv",
    "OASIS3_demographics.csv",
    "OASIS3_unchanged_CDR_cognitively_healthy.csv",
]

KEY_CLINICAL_FIELDS = [
    "CDRTOT",
    "CDRSUM",
    "DEMENTED",
    "PROBAD",
    "POSSAD",
    "NORMCOG",
    "NORMAL",
    "MCIAMEM",
    "MCIAPLUS",
    "MCINON1",
    "MCINON2",
    "MCIN1ATT",
    "MCIN1EX",
    "MCIN1LAN",
    "MCIN1VIS",
    "MCIN2ATT",
    "MCIN2EX",
    "MCIN2LAN",
    "MCIN2VIS",
    "dx1",
    "dx1_code",
    "dx2",
    "dx2_code",
    "dx3",
    "dx3_code",
    "dx4",
    "dx4_code",
    "dx5",
    "dx5_code",
    "dxmethod",
    "alzdis",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare metadata-only OASIS-3 smoke-test manual review and download manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metadata-raw", type=Path, default=DEFAULT_METADATA_RAW)
    parser.add_argument("--clinical-audit-dir", type=Path, default=DEFAULT_CLINICAL_AUDIT_DIR)
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
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", ""}:
        return False
    return bool(value)


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


def output_path(path: Path) -> Path:
    return resolve(path)


def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    path = output_path(path)
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {path}. Pass --overwrite to replace small audit outputs.")
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


def find_clinical_files(root: Path) -> List[Path]:
    files: List[Path] = []
    seen = set()
    for basename in CLINICAL_FILE_BASENAMES:
        path = find_first_file(root, basename)
        if path and path.resolve() not in seen:
            seen.add(path.resolve())
            files.append(path)
    return files


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, low_memory=False)


def clean_subject(value: Any) -> str:
    return clinical.normalize_subject(value)


def clean_session(value: Any) -> str:
    return clinical.normalize_session(value)


def session_day(session_id: Any) -> Optional[float]:
    return clinical.parse_session_day(session_id)


def select_nearest_rows(
    clinical_rows: pd.DataFrame,
    subject_id: str,
    mr_day: Optional[float],
    wanted_file_hint: str,
) -> pd.DataFrame:
    if clinical_rows.empty:
        return clinical_rows
    sub = clinical_rows[clinical_rows["subject_id"] == subject_id].copy()
    if wanted_file_hint:
        sub = sub[sub["clinical_source_file"].str.contains(wanted_file_hint, case=False, na=False)].copy()
    if sub.empty:
        return sub
    if mr_day is not None and sub["clinical_day"].notna().any():
        sub["_abs_delta"] = (pd.to_numeric(sub["clinical_day"], errors="coerce") - mr_day).abs()
        min_delta = sub["_abs_delta"].min()
        return sub[sub["_abs_delta"] == min_delta].copy()
    return sub.head(1).copy()


def get_field_from_rows(rows: pd.DataFrame, field: str) -> str:
    if rows.empty:
        return ""
    lower = {c.lower(): c for c in rows.columns}
    col = lower.get(field.lower())
    if not col:
        return ""
    return join_unique(rows[col], max_items=10)


def collect_field(rows_by_type: Dict[str, pd.DataFrame], field: str) -> str:
    values = []
    for rows in rows_by_type.values():
        value = get_field_from_rows(rows, field)
        if value:
            values.append(value)
    return join_unique(values, max_items=10)


def clinical_day_summary(rows_by_type: Dict[str, pd.DataFrame]) -> str:
    vals = []
    for rows in rows_by_type.values():
        if not rows.empty and "clinical_day" in rows.columns:
            vals.extend(rows["clinical_day"].dropna().tolist())
    return join_unique(vals, max_items=10)


def clinical_source_summary(rows_by_type: Dict[str, pd.DataFrame]) -> str:
    vals = []
    for rows in rows_by_type.values():
        if not rows.empty and "clinical_source_file" in rows.columns:
            vals.extend(rows["clinical_source_file"].tolist())
    return join_unique(vals, max_items=10)


def clinical_columns_used(rows_by_type: Dict[str, pd.DataFrame]) -> str:
    cols = []
    for rows in rows_by_type.values():
        for field in KEY_CLINICAL_FIELDS:
            if get_field_from_rows(rows, field):
                cols.append(field)
    return join_unique(cols, max_items=40)


def abs_delta_summary(rows_by_type: Dict[str, pd.DataFrame], mr_day: Optional[float]) -> str:
    if mr_day is None:
        return ""
    deltas = []
    for rows in rows_by_type.values():
        if rows.empty or "clinical_day" not in rows.columns:
            continue
        for value in rows["clinical_day"].dropna():
            delta = parse_float(value)
            if delta is not None:
                deltas.append(abs(delta - mr_day))
    if not deltas:
        return ""
    return f"{min(deltas):g}"


def load_selected_clinical_rows(metadata_raw: Path) -> Tuple[pd.DataFrame, List[str]]:
    files = find_clinical_files(metadata_raw)
    loaded: Dict[str, pd.DataFrame] = {}
    warnings = []
    for path in files:
        try:
            loaded[str(path)] = clinical.load_table(path)
        except Exception as exc:
            warnings.append(f"failed_load_clinical_file:{path}:{exc}")
    rows, row_warnings = clinical.clinical_rows_from_loaded(loaded)
    warnings.extend(row_warnings)
    return rows, warnings


def find_mr_json_metadata(metadata_raw: Path) -> Tuple[pd.DataFrame, str]:
    path = find_first_file(metadata_raw, "OASIS3_MR_json.csv")
    if path is None:
        return pd.DataFrame(), ""
    return load_csv(path), str(path)


def match_mr_row(mr_json: pd.DataFrame, candidate: pd.Series) -> pd.DataFrame:
    if mr_json.empty:
        return pd.DataFrame()
    subject_no_prefix = safe_text(candidate["subject_id"]).replace("sub-", "")
    session_id = safe_text(candidate["session_id"])
    json_path = safe_text(candidate.get("json_path", ""))
    sub = mr_json.copy()
    if "subject_id" in sub.columns:
        sub = sub[sub["subject_id"].astype(str) == subject_no_prefix]
    if "label" in sub.columns:
        sub = sub[sub["label"].astype(str) == session_id]
    if json_path and "filename" in sub.columns:
        exact = sub[sub["filename"].astype(str) == json_path]
        if not exact.empty:
            return exact
    if "filename" in sub.columns:
        bold = sub[sub["filename"].astype(str).str.contains("task-rest", case=False, na=False)]
        if not bold.empty:
            return bold
        bold = sub[sub["filename"].astype(str).str.contains("bold", case=False, na=False)]
        if not bold.empty:
            return bold
    return sub


def infer_n_volumes(mr_row: pd.Series) -> str:
    # OASIS3_MR_json.csv has slice timing metadata, but not frame counts.
    # Returning blank avoids inventing a volume count from per-slice metadata.
    return ""


def session_day_bin(day: Any) -> str:
    value = parse_float(day)
    if value is None:
        return "unknown"
    if value == 0:
        return "d0000"
    if value <= 365:
        return "d0001_0365"
    if value <= 1095:
        return "d0366_1095"
    return "d1096_plus"


def manual_flag(row: Dict[str, Any]) -> Tuple[str, str, bool]:
    issues = []
    if row["label_confidence"] != "high":
        issues.append("label_not_high_confidence")
    if row["provisional_label"] not in {"CN", "AD_DEMENTIA"}:
        issues.append("label_not_cn_or_ad_dementia")
    if not bool_from_any(row["has_task_rest_bold"]):
        issues.append("missing_task_rest_bold")
    if not safe_text(row["RepetitionTime"]):
        issues.append("missing_TR")
    if not safe_text(row["manufacturer"]):
        issues.append("missing_manufacturer")
    if not safe_text(row["scanner_model"]):
        issues.append("missing_scanner_model")
    if not safe_text(row["bold_scan_id"]):
        issues.append("missing_bold_scan_id")
    delta = parse_float(row["abs_delta_clinical_to_MR_days"])
    if delta is None:
        issues.append("missing_clinical_mr_day_delta")
    elif delta > 180:
        issues.append("clinical_mr_delta_gt_180_days")
    if issues:
        return "REVIEW_REQUIRED", ";".join(issues), False
    return "READY_FOR_DOWNLOAD_MANUAL_REVIEW", "metadata_complete; manual clinical review still required before download", True


def build_review_tables(
    candidates: pd.DataFrame,
    rest_candidates: pd.DataFrame,
    subject_map: pd.DataFrame,
    mr_json: pd.DataFrame,
    mr_json_source: str,
    clinical_rows: pd.DataFrame,
    download_root: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    warnings = []
    rows = []
    for _, cand in candidates.iterrows():
        subject = clean_subject(cand["subject_id"])
        sess = clean_session(cand["session_id"])
        mr_day = session_day(sess)
        rest_match = rest_candidates[
            (rest_candidates["subject_id"].astype(str) == subject)
            & (rest_candidates["session_id"].astype(str) == sess)
            & (rest_candidates.get("json_path", pd.Series("", index=rest_candidates.index)).astype(str) == safe_text(cand.get("json_path", "")))
        ]
        if rest_match.empty:
            rest_match = rest_candidates[
                (rest_candidates["subject_id"].astype(str) == subject)
                & (rest_candidates["session_id"].astype(str) == sess)
            ]

        cdr_rows = select_nearest_rows(clinical_rows, subject, mr_day, "UDSb4")
        dx_rows = select_nearest_rows(clinical_rows, subject, mr_day, "UDSd1")
        demo_rows = select_nearest_rows(clinical_rows, subject, mr_day, "demo|UDSa1")
        rows_by_type = {"cdr": cdr_rows, "diagnosis": dx_rows, "demo": demo_rows}

        mr_matches = match_mr_row(mr_json, cand)
        if mr_matches.empty:
            warnings.append(f"no_mr_json_match:{subject}:{sess}:{safe_text(cand.get('json_path', ''))}")
            mr_row = pd.Series(dtype=object)
        else:
            if len(mr_matches) > 1:
                warnings.append(f"multiple_mr_json_matches:{subject}:{sess}:{len(mr_matches)}")
            mr_row = mr_matches.iloc[0]

        subject_row = subject_map[subject_map["subject_id"].astype(str) == subject] if not subject_map.empty else pd.DataFrame()
        label_rule = safe_text(subject_row.iloc[0].get("provisional_label_rule", "")) if not subject_row.empty else ""
        label_warning = safe_text(subject_row.iloc[0].get("label_warning", "")) if not subject_row.empty else ""

        out = {
            "subject_id": subject,
            "session_id": sess,
            "provisional_label": safe_text(cand["provisional_label"]),
            "label_confidence": safe_text(cand["label_confidence"]),
            "clinical_source_file": clinical_source_summary(rows_by_type),
            "clinical_label_columns_used": clinical_columns_used(rows_by_type),
            "CDRTOT": collect_field(rows_by_type, "CDRTOT"),
            "CDRSUM": collect_field(rows_by_type, "CDRSUM"),
            "DEMENTED": collect_field(rows_by_type, "DEMENTED"),
            "PROBAD": collect_field(rows_by_type, "PROBAD"),
            "POSSAD": collect_field(rows_by_type, "POSSAD"),
            "NORMCOG": collect_field(rows_by_type, "NORMCOG"),
            "NORMAL": collect_field(rows_by_type, "NORMAL"),
            "MCI_fields": "; ".join(f"{field}={collect_field(rows_by_type, field)}" for field in KEY_CLINICAL_FIELDS if field.upper().startswith("MCI") and collect_field(rows_by_type, field)),
            "dx_fields": "; ".join(f"{field}={collect_field(rows_by_type, field)}" for field in KEY_CLINICAL_FIELDS if field.lower().startswith("dx") and collect_field(rows_by_type, field)),
            "provisional_label_rule": label_rule,
            "label_warning": label_warning,
            "clinical_date_or_days": clinical_day_summary(rows_by_type),
            "MR_session_day": "" if mr_day is None else f"{mr_day:g}",
            "abs_delta_clinical_to_MR_days": abs_delta_summary(rows_by_type, mr_day),
            "has_task_rest_bold": safe_text(cand.get("has_task_rest_bold", "")) or "True",
            "bold_scan_id": safe_text(mr_row.get("acccession", "")),
            "experiment_id": safe_text(mr_row.get("label", "")) or sess,
            "MR_ID": safe_text(mr_row.get("label", "")) or sess,
            "TR": safe_text(mr_row.get("RepetitionTime", "")) or safe_text(cand.get("RepetitionTime", "")),
            "RepetitionTime": safe_text(mr_row.get("RepetitionTime", "")) or safe_text(cand.get("RepetitionTime", "")),
            "n_volumes": infer_n_volumes(mr_row),
            "manufacturer": safe_text(mr_row.get("Manufacturer", "")) or safe_text(cand.get("scanner_manufacturer", "")),
            "scanner_model": safe_text(mr_row.get("ManufacturersModelName", "")) or safe_text(cand.get("scanner_model", "")),
            "json_path": safe_text(mr_row.get("filename", "")) or safe_text(cand.get("json_path", "")),
            "mr_json_metadata_source": mr_json_source,
            "selection_reason": safe_text(cand.get("reason", "")),
            "notes": "",
        }
        status, flag, include = manual_flag(out)
        out["eligibility_status"] = status
        out["manual_review_flag"] = flag
        out["include_in_smoke_test"] = include
        if not out["n_volumes"]:
            out["notes"] = "n_volumes_not_available_in_metadata_csv"
        rows.append(out)

    review = pd.DataFrame(rows)
    manifest = pd.DataFrame(
        [
            {
                "experiment_id": row["experiment_id"],
                "subject_id": row["subject_id"],
                "session_id": row["session_id"],
                "target_label": row["provisional_label"],
                "scan_type_requested": "bold",
                "output_root": str(download_root),
                "include_in_smoke_test": bool(row["include_in_smoke_test"]),
                "reason": row["manual_review_flag"] if not row["include_in_smoke_test"] else "ready_after_manual_review; task-rest BOLD metadata complete",
            }
            for _, row in review.iterrows()
        ]
    )
    summary = build_balance_summary(review)
    return review, manifest, summary, warnings


def build_balance_summary(review: pd.DataFrame) -> pd.DataFrame:
    if review.empty:
        return pd.DataFrame(columns=["label", "manufacturer", "TR", "scanner_model", "session_day_bin", "n"])
    temp = review.copy()
    temp["label"] = temp["provisional_label"]
    temp["session_day_bin"] = temp["MR_session_day"].map(session_day_bin)
    grouped = (
        temp.groupby(["label", "manufacturer", "TR", "scanner_model", "session_day_bin"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["label", "manufacturer", "TR", "scanner_model", "session_day_bin"])
    )
    return grouped


def command_for_next_download(manifest_path: Path, download_root: Path) -> str:
    return (
        "bash oasis-scripts/download_scans/download_oasis_scans_bids.sh "
        f"{manifest_path} {download_root} \"$NITRC_USER\" bold"
    )


def readme_text(
    review: pd.DataFrame,
    manifest: pd.DataFrame,
    warnings: Sequence[str],
    command: str,
    output_dir: Path,
    metadata_raw: Path,
    oasis_scripts_input_path: Path,
    download_root: Path,
) -> str:
    n = len(review)
    n_ready = int(review["include_in_smoke_test"].sum()) if "include_in_smoke_test" in review else 0
    label_counts = review["provisional_label"].value_counts(dropna=False).to_dict() if not review.empty else {}
    tr_complete = bool(review["TR"].astype(str).str.strip().replace("nan", "").ne("").all()) if not review.empty else False
    scanner_complete = bool(
        review["manufacturer"].astype(str).str.strip().replace("nan", "").ne("").all()
        and review["scanner_model"].astype(str).str.strip().replace("nan", "").ne("").all()
    ) if not review.empty else False
    delta_vals = pd.to_numeric(review.get("abs_delta_clinical_to_MR_days", pd.Series(dtype=str)), errors="coerce").dropna()
    max_delta = float(delta_vals.max()) if not delta_vals.empty else np.nan
    aligned = (not delta_vals.empty) and bool((delta_vals <= 180).all())
    manifest_ready = n == 10 and n_ready == n and tr_complete and scanner_complete and aligned
    verdict = "READY_FOR_MANUAL_DOWNLOAD_REVIEW" if manifest_ready else "REVIEW_REQUIRED_BEFORE_DOWNLOAD"

    warnings_block = "\n".join(f"- {w}" for w in warnings) if warnings else "- None."
    lines = [
        "# OASIS-3 Smoke-Test Pre-Download Review",
        "",
        "This is a metadata-only audit. It did not download images, preprocess images, train models, copy large files, or load arrays.",
        "",
        "## Summary",
        "",
        f"- Output directory: `{output_dir}`",
        f"- Metadata root: `{metadata_raw}`",
        f"- Selected candidates reviewed: {n}",
        f"- Included in manifest after automated checks: {n_ready}/{n}",
        f"- Label counts: {json.dumps(label_counts, sort_keys=True)}",
        f"- TR complete: {tr_complete}",
        f"- Scanner/manufacturer complete: {scanner_complete}",
        f"- Maximum absolute clinical-to-MR day delta: {max_delta:g}" if not np.isnan(max_delta) else "- Maximum absolute clinical-to-MR day delta: unavailable",
        f"- Clinical labels temporally aligned within 180 days: {aligned}",
        f"- Feasibility verdict: `{verdict}`",
        "",
        "## Manual Review",
        "",
        "The 10 selected sessions are manually reviewable from the generated CSV. All labels remain provisional OASIS-3 labels and must be reviewed before claiming ADNI-compatible CN/AD external validation.",
        "",
        "## Download Manifest",
        "",
        f"- Review/download manifest: `{output_dir / 'oasis3_smoke_test_download_manifest.csv'}`",
        f"- One-column `oasis-scripts` input: `{oasis_scripts_input_path}`",
        "- Requested scan type: `bold`",
        f"- Target output root for later image download: `{download_root}`",
        "",
        "Do not download images until the manual review table passes review.",
        "",
        "Next command to run manually after review, with `NITRC_USER` set in the shell:",
        "",
        f"```bash\n{command}\n```",
        "",
        "The OASIS script prompts for the NITRC password interactively; do not place credentials in files or logs.",
        "",
        "## Warnings",
        "",
        warnings_block,
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    metadata_raw = resolve(args.metadata_raw)
    clinical_audit_dir = resolve(args.clinical_audit_dir)
    output_dir = prepare_output_dir(args.output_dir, args.overwrite)
    download_root = args.download_root

    candidates_path = require_file(clinical_audit_dir / SMOKE_CANDIDATES_FILE, "smoke candidates")
    rest_path = require_file(clinical_audit_dir / REST_CANDIDATES_FILE, "rest BOLD clinical candidates")
    subject_map_path = require_file(clinical_audit_dir / SUBJECT_MAPPING_FILE, "subject-level clinical mapping")
    if not metadata_raw.exists():
        raise FileNotFoundError(f"Missing metadata raw root: {metadata_raw}")

    candidates = load_csv(candidates_path)
    rest_candidates = load_csv(rest_path)
    subject_map = load_csv(subject_map_path)
    mr_json, mr_json_source = find_mr_json_metadata(metadata_raw)
    clinical_rows, clinical_warnings = load_selected_clinical_rows(metadata_raw)

    review, manifest, summary, warnings = build_review_tables(
        candidates=candidates,
        rest_candidates=rest_candidates,
        subject_map=subject_map,
        mr_json=mr_json,
        mr_json_source=mr_json_source,
        clinical_rows=clinical_rows,
        download_root=download_root,
    )
    warnings.extend(clinical_warnings)

    review_path = output_dir / "oasis3_smoke_test_manual_review_table.csv"
    manifest_path = output_dir / "oasis3_smoke_test_download_manifest.csv"
    oasis_scripts_input_path = output_dir / "oasis3_smoke_test_experiment_ids_for_oasis_scripts.csv"
    summary_path = output_dir / "oasis3_smoke_test_label_balance_summary.csv"
    readme_path = output_dir / "README.md"
    audit_manifest_path = output_dir / "audit_manifest.json"

    review.to_csv(review_path, index=False)
    manifest.to_csv(manifest_path, index=False)
    manifest.loc[manifest["include_in_smoke_test"].astype(bool), ["experiment_id"]].to_csv(oasis_scripts_input_path, index=False)
    summary.to_csv(summary_path, index=False)

    command = command_for_next_download(oasis_scripts_input_path, download_root)
    readme_path.write_text(
        readme_text(review, manifest, warnings, command, output_dir, metadata_raw, oasis_scripts_input_path, download_root),
        encoding="utf-8",
    )

    audit_manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_hash": git_hash(),
        "project_root": str(PROJECT_ROOT),
        "metadata_raw": str(metadata_raw),
        "clinical_audit_dir": str(clinical_audit_dir),
        "output_dir": str(output_dir),
        "download_root_for_later": str(download_root),
        "inputs": {
            "smoke_candidates": str(candidates_path),
            "rest_bold_clinical_candidates": str(rest_path),
            "subject_level_clinical_mapping": str(subject_map_path),
            "mr_json_metadata": mr_json_source,
        },
        "outputs": {
            "manual_review_table": str(review_path),
            "download_manifest": str(manifest_path),
            "oasis_scripts_input": str(oasis_scripts_input_path),
            "label_balance_summary": str(summary_path),
            "readme": str(readme_path),
        },
        "no_image_download": True,
        "no_preprocessing": True,
        "no_training": True,
        "no_large_copy": True,
        "warnings": list(warnings),
    }
    audit_manifest_path.write_text(json.dumps(audit_manifest, indent=2), encoding="utf-8")

    print(f"Wrote {review_path}")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {readme_path}")
    print(f"Candidates reviewed: {len(review)}")
    print(f"Included after automated checks: {int(review['include_in_smoke_test'].sum())}/{len(review)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
