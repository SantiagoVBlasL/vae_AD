#!/usr/bin/env python3
"""Build the live ADNI v5.1 CN-GE data ledger for Martin coordination.

This script creates CSV tracking sheets only. It does not train, build tensors,
or modify any tensor/data arrays.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "final_cn_ge_inventory_before_martin_request"
)
DEFAULT_INPUT = FINAL_DIR / "cn_ge_request_after_filtering_audit_20260513.csv"
MASTER_MANIFEST_DIR = PROJECT_ROOT / "results" / "revision_bspc_2026" / "adni_v5_1_master_manifest"

LEDGER_COLUMNS = [
    "SubjectID",
    "ImageID",
    "Visit",
    "Diagnosis",
    "Manufacturer",
    "ledger_scope",
    "priority_batch",
    "dicom_series_ok",
    "dicom_issue_reason",
    "roisignals_status",
    "processing_status",
    "action_needed",
    "uploaded_batch",
    "uploaded_path",
    "included_in_dataset_version",
    "python_bandpass_requested",
    "notes_martin",
    "notes_santiago",
]

DICOM_NOT_OK_SUBJECTS = {"006_S_6500", "009_S_4388"}
LEGACY_DICOM_ISSUE_ONLY_SUBJECTS = {"006_S_6500"}
DICOM_NOT_OK_REASON = "DICOM series reported not OK by Martin"
LEGACY_DICOM_NOTE = "Reported by Martin: DICOM series not OK; subject was in the first request."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ADNI_v5_1_DATA_LEDGER_CURRENT.csv and dated snapshot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--final-dir", type=Path, default=FINAL_DIR)
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--snapshot-label", default=None)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def first_nonempty(values: Iterable[Any]) -> str:
    for value in values:
        text = clean(value)
        if text:
            return text
    return ""


def metadata_sources(final_dir: Path) -> List[pd.DataFrame]:
    paths = [
        final_dir / "cn_ge_request_compact_for_martin.csv",
        final_dir / "cn_ge_already_included_v5_1_gecn9.csv",
        final_dir / "cn_ge_final_inventory.csv",
        MASTER_MANIFEST_DIR / "adni_v5_1_master_subject_manifest_first_visit.csv",
        MASTER_MANIFEST_DIR / "adni_v5_1_preprocessing_request_for_martin.csv",
    ]
    return [read_csv(path) for path in paths if path.exists()]


def build_metadata_lookup(final_dir: Path) -> Dict[str, Dict[str, str]]:
    lookup: Dict[str, Dict[str, str]] = {}
    fields = {
        "ImageID": ["ImageID"],
        "Visit": ["Visit"],
        "Diagnosis": ["Diagnosis", "ResearchGroup_Mapped", "ResearchGroup"],
        "Manufacturer": ["Manufacturer", "Manufacturer_aux"],
    }
    for df in metadata_sources(final_dir):
        if "SubjectID" not in df.columns:
            continue
        for _, row in df.iterrows():
            sid = clean(row.get("SubjectID"))
            if not sid:
                continue
            current = lookup.setdefault(
                sid,
                {"ImageID": "", "Visit": "", "Diagnosis": "", "Manufacturer": ""},
            )
            for out_col, candidates in fields.items():
                if current[out_col]:
                    continue
                current[out_col] = first_nonempty(row.get(col, "") for col in candidates if col in df.columns)
    return lookup


def roisignals_status(row: pd.Series) -> str:
    request_category = clean(row.get("request_category"))
    action_category = clean(row.get("martin_action_category"))
    if request_category == "no_roisignals_found_needs_preprocessing":
        return "not_found"
    if request_category == "previously_added_to_v5_1_gecn9_but_now_quarantined":
        return "quarantine"
    if action_category == "local_but_martin_stage_confirmation_required":
        return "quarantine"
    if action_category == "local_but_reprocess_required":
        return "qc_fail"
    return "UNKNOWN"


def included_version(row: pd.Series) -> str:
    if clean(row.get("request_category")) == "previously_added_to_v5_1_gecn9_but_now_quarantined":
        return "v5.1_gecn9_provisional_quarantine"
    return ""


def notes_santiago(row: pd.Series) -> str:
    pieces = []
    category = clean(row.get("request_category"))
    action_category = clean(row.get("martin_action_category"))
    reason = clean(row.get("reason"))
    if category:
        pieces.append(f"request_category={category}")
    if action_category and action_category != category:
        pieces.append(f"martin_action_category={action_category}")
    if reason:
        pieces.append(reason)
    return " | ".join(pieces)


def build_ledger(request: pd.DataFrame, metadata: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    for _, row in request.iterrows():
        sid = clean(row.get("SubjectID"))
        meta = metadata.get(sid, {})
        priority_batch = "first_67" if clean(row.get("priority_first_67")) == "YES" else "remaining"
        ledger_row = {
            "SubjectID": sid,
            "ImageID": clean(meta.get("ImageID")),
            "Visit": clean(meta.get("Visit")),
            "Diagnosis": clean(meta.get("Diagnosis")) or "CN",
            "Manufacturer": clean(meta.get("Manufacturer")) or "GE",
            "ledger_scope": "cn_ge_request",
            "priority_batch": priority_batch,
            "dicom_series_ok": "UNKNOWN",
            "dicom_issue_reason": "",
            "roisignals_status": roisignals_status(row),
            "processing_status": "pending",
            "action_needed": clean(row.get("action_needed")),
            "uploaded_batch": "",
            "uploaded_path": "",
            "included_in_dataset_version": included_version(row),
            "python_bandpass_requested": "NO",
            "notes_martin": "",
            "notes_santiago": notes_santiago(row),
        }
        if sid in DICOM_NOT_OK_SUBJECTS:
            ledger_row["dicom_series_ok"] = "NO"
            ledger_row["dicom_issue_reason"] = DICOM_NOT_OK_REASON
            ledger_row["action_needed"] = "exclude_or_replace_dicom"
            ledger_row["processing_status"] = "excluded"
            ledger_row["notes_martin"] = DICOM_NOT_OK_REASON
        rows.append(ledger_row)

    existing_subjects = {row["SubjectID"] for row in rows}
    for sid in sorted(LEGACY_DICOM_ISSUE_ONLY_SUBJECTS - existing_subjects):
        meta = metadata.get(sid, {})
        rows.append(
            {
                "SubjectID": sid,
                "ImageID": clean(meta.get("ImageID")),
                "Visit": clean(meta.get("Visit")),
                "Diagnosis": clean(meta.get("Diagnosis")) or "UNKNOWN",
                "Manufacturer": clean(meta.get("Manufacturer")) or "Philips",
                "ledger_scope": "dicom_issue_only",
                "priority_batch": "legacy_first_request",
                "dicom_series_ok": "NO",
                "dicom_issue_reason": DICOM_NOT_OK_REASON,
                "roisignals_status": "not_applicable",
                "processing_status": "excluded",
                "action_needed": "exclude_or_replace_dicom",
                "uploaded_batch": "",
                "uploaded_path": "",
                "included_in_dataset_version": "",
                "python_bandpass_requested": "NO",
                "notes_martin": LEGACY_DICOM_NOTE,
                "notes_santiago": (
                    "ledger_scope=dicom_issue_only | not part of the 110-row CN-GE first-visit "
                    "request; retained for DICOM exclusion traceability"
                ),
            }
        )

    ledger = pd.DataFrame(rows, columns=LEDGER_COLUMNS)
    expected_rows = len(request) + len(LEGACY_DICOM_ISSUE_ONLY_SUBJECTS - set(request["SubjectID"]))
    if len(ledger) != expected_rows:
        raise RuntimeError(f"Ledger row count mismatch: {len(ledger)} vs expected {expected_rows}")
    if ledger["SubjectID"].duplicated().any():
        duplicated = sorted(ledger.loc[ledger["SubjectID"].duplicated(), "SubjectID"].unique())
        raise RuntimeError(f"Duplicate SubjectID rows in ledger: {duplicated}")
    if set(ledger["python_bandpass_requested"]) != {"NO"}:
        raise RuntimeError("python_bandpass_requested must be NO in every row")
    return ledger


def write_readme_update(path: Path, ledger: pd.DataFrame, snapshot_path: Path) -> None:
    scope_counts = ledger["ledger_scope"].value_counts().sort_index()
    dicom_counts = ledger["dicom_series_ok"].value_counts().sort_index()
    processing_counts = ledger["processing_status"].value_counts().sort_index()
    row_006 = ledger[ledger["SubjectID"].eq("006_S_6500")].iloc[0].to_dict()
    row_009 = ledger[ledger["SubjectID"].eq("009_S_4388")].iloc[0].to_dict()
    lines = [
        "# ADNI v5.1 ledger update: 006_S_6500",
        "",
        "This update adds `006_S_6500` for DICOM-exclusion traceability without changing the 110-row CN-GE request scope.",
        "",
        "## What changed",
        "",
        "- Added/ensured `ledger_scope`.",
        "- Existing request rows are marked `ledger_scope=cn_ge_request`.",
        "- Added `006_S_6500` as `ledger_scope=dicom_issue_only` because Martin reported the DICOM series was not OK.",
        "- Kept `009_S_4388` as `dicom_series_ok=NO`, `processing_status=excluded`.",
        "- Python bandpass remains `NO` for every row.",
        "",
        "## 006_S_6500 row",
        "",
        f"- SubjectID: `{row_006.get('SubjectID', '')}`",
        f"- ImageID: `{row_006.get('ImageID', '')}`",
        f"- Visit: `{row_006.get('Visit', '')}`",
        f"- Diagnosis: `{row_006.get('Diagnosis', '')}`",
        f"- Manufacturer: `{row_006.get('Manufacturer', '')}`",
        f"- priority_batch: `{row_006.get('priority_batch', '')}`",
        f"- ledger_scope: `{row_006.get('ledger_scope', '')}`",
        f"- dicom_series_ok: `{row_006.get('dicom_series_ok', '')}`",
        f"- processing_status: `{row_006.get('processing_status', '')}`",
        f"- action_needed: `{row_006.get('action_needed', '')}`",
        "",
        "## 009_S_4388 check",
        "",
        f"- dicom_series_ok: `{row_009.get('dicom_series_ok', '')}`",
        f"- processing_status: `{row_009.get('processing_status', '')}`",
        f"- action_needed: `{row_009.get('action_needed', '')}`",
        "",
        "## Counts",
        "",
        f"- total rows: `{len(ledger)}`",
    ]
    for key, value in scope_counts.items():
        lines.append(f"- ledger_scope `{key}`: `{int(value)}`")
    for key, value in dicom_counts.items():
        lines.append(f"- dicom_series_ok `{key}`: `{int(value)}`")
    for key, value in processing_counts.items():
        lines.append(f"- processing_status `{key}`: `{int(value)}`")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `ADNI_v5_1_DATA_LEDGER_CURRENT.csv`",
            f"- `{snapshot_path.name}`",
            "- `README_update_006_S_6500.md`",
            "",
            "No training, no tensor construction, and no tensor modification were performed.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def count_block(df: pd.DataFrame, column: str) -> str:
    counts = df[column].value_counts(dropna=False).sort_index()
    lines = [f"{column}:"]
    for key, value in counts.items():
        lines.append(f"  {key}: {int(value)}")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    request = read_csv(args.input)
    metadata = build_metadata_lookup(args.final_dir)
    ledger = build_ledger(request, metadata)

    current_path = args.final_dir / "ADNI_v5_1_DATA_LEDGER_CURRENT.csv"
    snapshot_label = args.snapshot_label or args.date
    snapshot_path = args.final_dir / f"ADNI_v5_1_DATA_LEDGER_{snapshot_label}.csv"
    readme_path = args.final_dir / "README_update_006_S_6500.md"
    ledger.to_csv(current_path, index=False)
    ledger.to_csv(snapshot_path, index=False)
    write_readme_update(readme_path, ledger, snapshot_path)

    print(f"Wrote {current_path}")
    print(f"Wrote {snapshot_path}")
    print(f"Wrote {readme_path}")
    print(f"rows: {len(ledger)}")
    for column in ["ledger_scope", "dicom_series_ok", "processing_status", "action_needed", "priority_batch"]:
        print(count_block(ledger, column))
    print("No training. No tensor construction. No tensor modification. Python bandpass requested: NO.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
