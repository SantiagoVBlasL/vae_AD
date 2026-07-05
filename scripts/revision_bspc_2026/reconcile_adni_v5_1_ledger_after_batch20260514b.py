#!/usr/bin/env python3
"""Reconcile the live ADNI v5.1 CN-GE ledger after batch 20260514b.

This script only writes lightweight ledger/QC files under results/.
It does not modify tensors, does not replace the live Drive ledger, and does not train.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_DATASET_VERSION = "v5.1_batch20260514b"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "final_cn_ge_inventory_before_martin_request"
)

DEFAULT_DATASET_ROOT = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass"
)
DEFAULT_SUBJECT_METADATA = (
    DEFAULT_DATASET_ROOT / "subject_metadata_v5_1_batch20260514b_no_pybandpass.csv"
)
DEFAULT_TRAINING_METADATA = (
    DEFAULT_DATASET_ROOT / "training_ready_metadata_v5_1_batch20260514b_no_pybandpass.csv"
)
DEFAULT_LEDGER_ORIGINAL = OUTPUT_DIR / "ADNI_v5_1_DATA_LEDGER_CURRENT.csv"
DEFAULT_LEDGER_LAST_CANDIDATE = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "martin_bandpass_batch_20260514b"
    / "ADNI_v5_1_DATA_LEDGER_20260514b_imported_candidate.csv"
)
DEFAULT_BATCH_DECISIONS = {
    "20260513_bandpass_batch1": PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "martin_bandpass_batch_20260513"
    / "batch_import_decision.csv",
    "20260514_bandpass_batch2": PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "martin_bandpass_batch_20260514"
    / "batch_import_decision.csv",
    "20260514b_bandpass_batch3": PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "martin_bandpass_batch_20260514b"
    / "batch_import_decision.csv",
}

OUT_RECONCILED = OUTPUT_DIR / "ADNI_v5_1_DATA_LEDGER_CURRENT_RECONCILED_20260514b.csv"
OUT_SNAPSHOT = OUTPUT_DIR / "ADNI_v5_1_DATA_LEDGER_20260514b_snapshot.csv"
OUT_REMAINING = OUTPUT_DIR / "remaining_pending_after_20260514b.csv"
OUT_README = OUTPUT_DIR / "README_reconciled_after_20260514b.md"
OUT_SUMMARY_JSON = OUTPUT_DIR / "ADNI_v5_1_DATA_LEDGER_20260514b_reconciliation_summary.json"

REQUIRED_LEDGER_COLUMNS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconcile ADNI v5.1 ledger after Martin batch 20260514b.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ledger-original", type=Path, default=DEFAULT_LEDGER_ORIGINAL)
    parser.add_argument("--subject-metadata", type=Path, default=DEFAULT_SUBJECT_METADATA)
    parser.add_argument("--training-ready-metadata", type=Path, default=DEFAULT_TRAINING_METADATA)
    parser.add_argument("--ledger-last-candidate", type=Path, default=DEFAULT_LEDGER_LAST_CANDIDATE)
    parser.add_argument("--batch-20260513", type=Path, default=DEFAULT_BATCH_DECISIONS["20260513_bandpass_batch1"])
    parser.add_argument("--batch-20260514", type=Path, default=DEFAULT_BATCH_DECISIONS["20260514_bandpass_batch2"])
    parser.add_argument("--batch-20260514b", type=Path, default=DEFAULT_BATCH_DECISIONS["20260514b_bandpass_batch3"])
    return parser.parse_args()


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def normalize_diagnosis(value: Any) -> str:
    text = clean(value).upper()
    if text in {"AD", "CN", "MCI"}:
        return text
    if "CONTROL" in text or text == "NORMAL":
        return "CN"
    if "DEMENT" in text or "ALZ" in text:
        return "AD"
    if "MCI" in text:
        return "MCI"
    return clean(value)


def normalize_manufacturer(value: Any) -> str:
    text = clean(value).upper()
    if not text:
        return ""
    if "GE" in text:
        return "GE"
    if "SIEMENS" in text:
        return "SIEMENS"
    if "PHILIPS" in text:
        return "Philips"
    return clean(value)


def read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = ""
    return out


def index_by_subject(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    if df.empty or "SubjectID" not in df.columns:
        return {}
    return df.drop_duplicates("SubjectID", keep="first").set_index("SubjectID", drop=False).to_dict(orient="index")


def append_note(existing: Any, addition: str) -> str:
    base = clean(existing)
    if not addition:
        return base
    if addition in base:
        return base
    return addition if not base else f"{base} | {addition}"


def batch_ready_value(row: Mapping[str, Any]) -> str:
    for col in ["import_ready_new", "import_ready"]:
        if col in row:
            return clean(row.get(col, ""))
    return ""


def load_batch_decisions(paths: Mapping[str, Path]) -> Tuple[Dict[str, Dict[str, Dict[str, str]]], Dict[str, List[Dict[str, str]]]]:
    by_batch: Dict[str, Dict[str, Dict[str, str]]] = {}
    by_subject: Dict[str, List[Dict[str, str]]] = {}
    for batch_name, path in paths.items():
        df = read_csv(path)
        df = ensure_columns(
            df,
            [
                "SubjectID",
                "selected_path",
                "selected_relative_path",
                "stage_guess",
                "spectral_class",
                "qc_signal_pass",
                "qc_signal_reason",
                "qc_signal_warning",
                "decision_reason",
                "decision_warning",
                "python_bandpass_requested",
                "python_bandpass_final",
            ],
        )
        records: Dict[str, Dict[str, str]] = {}
        for raw in df.to_dict(orient="records"):
            row = {str(k): clean(v) for k, v in raw.items()}
            sid = row.get("SubjectID", "")
            if not sid:
                continue
            row["source_batch"] = batch_name
            row["import_ready_any"] = batch_ready_value(row)
            records.setdefault(sid, row)
            by_subject.setdefault(sid, []).append(row)
        by_batch[batch_name] = records
    return by_batch, by_subject


def choose_decision(
    sid: str,
    final_row: Mapping[str, Any],
    decisions_by_batch: Mapping[str, Mapping[str, Mapping[str, str]]],
    decisions_by_subject: Mapping[str, List[Mapping[str, str]]],
) -> Dict[str, str]:
    source_batch = clean(final_row.get("source_batch", ""))
    if source_batch and sid in decisions_by_batch.get(source_batch, {}):
        return dict(decisions_by_batch[source_batch][sid])
    for row in decisions_by_subject.get(sid, []):
        if clean(row.get("import_ready_any", "")).lower() == "yes":
            return dict(row)
    if decisions_by_subject.get(sid):
        return dict(decisions_by_subject[sid][-1])
    return {}


def imported_note(final_row: Mapping[str, Any], decision: Mapping[str, Any]) -> str:
    parts = [
        "reconciled_after_batch20260514b=present_in_final_dataset",
        f"source_batch={clean(final_row.get('source_batch', decision.get('source_batch', '')))}",
        f"stage={clean(final_row.get('stage_guess', decision.get('stage_guess', '')))}",
        f"spectral={clean(final_row.get('spectral_class', decision.get('spectral_class', '')))}",
        f"qc_signal_pass={clean(decision.get('qc_signal_pass', ''))}",
        f"decision_reason={clean(decision.get('decision_reason', ''))}",
    ]
    return "; ".join(part for part in parts if not part.endswith("="))


def pending_note(decision: Mapping[str, Any]) -> str:
    if not decision:
        return "reconciled_after_batch20260514b=not_in_final_dataset; status=pending; no_import_decision_match"
    return (
        "reconciled_after_batch20260514b=not_in_final_dataset; status=pending; "
        f"latest_batch={clean(decision.get('source_batch', ''))}; "
        f"decision_reason={clean(decision.get('decision_reason', ''))}; "
        f"stage={clean(decision.get('stage_guess', ''))}; "
        f"qc_signal_pass={clean(decision.get('qc_signal_pass', ''))}"
    )


def reconcile_ledger(
    ledger: pd.DataFrame,
    final_metadata: pd.DataFrame,
    decisions_by_batch: Mapping[str, Mapping[str, Mapping[str, str]]],
    decisions_by_subject: Mapping[str, List[Mapping[str, str]]],
) -> pd.DataFrame:
    ledger = ensure_columns(ledger, REQUIRED_LEDGER_COLUMNS)
    if ledger["SubjectID"].duplicated().any():
        duplicated = sorted(ledger.loc[ledger["SubjectID"].duplicated(), "SubjectID"].tolist())
        raise RuntimeError(f"Ledger has duplicated SubjectID rows: {duplicated}")
    final_map = index_by_subject(final_metadata)
    out = ledger.copy()
    for idx, row in out.iterrows():
        sid = clean(row.get("SubjectID", ""))
        dicom_no = clean(row.get("dicom_series_ok", "")).upper() == "NO"
        final_row = final_map.get(sid, {})
        present = bool(final_row)
        decision = choose_decision(sid, final_row, decisions_by_batch, decisions_by_subject)
        out.at[idx, "python_bandpass_requested"] = "NO"
        if dicom_no:
            out.at[idx, "processing_status"] = "excluded"
            out.at[idx, "roisignals_status"] = "not_applicable"
            out.at[idx, "action_needed"] = "exclude_or_replace_dicom"
            out.at[idx, "included_in_dataset_version"] = ""
            out.at[idx, "uploaded_batch"] = ""
            out.at[idx, "uploaded_path"] = ""
            out.at[idx, "notes_santiago"] = append_note(
                row.get("notes_santiago", ""),
                "reconciled_after_batch20260514b=dicom_series_ok_NO; kept_excluded",
            )
        elif present:
            uploaded_batch = clean(final_row.get("source_batch", decision.get("source_batch", "")))
            uploaded_path = clean(final_row.get("roisignals_path", decision.get("selected_path", "")))
            if not uploaded_path:
                uploaded_path = clean(decision.get("selected_path", ""))
            out.at[idx, "processing_status"] = "imported"
            out.at[idx, "roisignals_status"] = "qc_pass"
            out.at[idx, "included_in_dataset_version"] = FINAL_DATASET_VERSION
            out.at[idx, "uploaded_batch"] = uploaded_batch
            out.at[idx, "uploaded_path"] = uploaded_path
            out.at[idx, "action_needed"] = "none_imported"
            out.at[idx, "notes_santiago"] = append_note(row.get("notes_santiago", ""), imported_note(final_row, decision))
        else:
            latest_decision = decisions_by_subject.get(sid, [{}])[-1]
            out.at[idx, "processing_status"] = "pending"
            out.at[idx, "included_in_dataset_version"] = ""
            out.at[idx, "python_bandpass_requested"] = "NO"
            out.at[idx, "notes_santiago"] = append_note(row.get("notes_santiago", ""), pending_note(latest_decision))
    return out[REQUIRED_LEDGER_COLUMNS + [col for col in out.columns if col not in REQUIRED_LEDGER_COLUMNS]]


def status_counts(df: pd.DataFrame) -> Dict[str, int]:
    return {str(k): int(v) for k, v in df["processing_status"].value_counts(dropna=False).to_dict().items()}


def build_summary(
    reconciled: pd.DataFrame,
    training_ready: pd.DataFrame,
    ledger_last_candidate: pd.DataFrame,
) -> Dict[str, Any]:
    dx = reconciled["Diagnosis"].map(normalize_diagnosis)
    manufacturer = reconciled["Manufacturer"].map(normalize_manufacturer)
    imported = reconciled["processing_status"].eq("imported")
    pending = reconciled["processing_status"].eq("pending")
    excluded = reconciled["processing_status"].eq("excluded")
    dicom_no = reconciled["dicom_series_ok"].map(lambda x: clean(x).upper()).eq("NO")
    train = training_ready.copy()
    train["ResearchGroup_Mapped"] = train["ResearchGroup_Mapped"].map(normalize_diagnosis)
    train["Manufacturer"] = train["Manufacturer"].map(normalize_manufacturer)
    final_training_cn_ge = int((train["ResearchGroup_Mapped"].eq("CN") & train["Manufacturer"].eq("GE")).sum())
    imported_cn_ge = int((imported & dx.eq("CN") & manufacturer.eq("GE")).sum())
    return {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "final_dataset_version": FINAL_DATASET_VERSION,
        "total_rows": int(len(reconciled)),
        "processing_status_counts": status_counts(reconciled),
        "imported": int(imported.sum()),
        "pending": int(pending.sum()),
        "excluded": int(excluded.sum()),
        "imported_CN_GE": imported_cn_ge,
        "pending_CN_GE": int((pending & dx.eq("CN") & manufacturer.eq("GE")).sum()),
        "excluded_DICOM": int((excluded & dicom_no).sum()),
        "final_training_ready_CN_GE": final_training_cn_ge,
        "imported_CN_GE_matches_final_metadata_CN_GE": bool(imported_cn_ge == final_training_cn_ge),
        "python_bandpass_requested_non_NO": int(
            (~reconciled["python_bandpass_requested"].map(lambda x: clean(x).upper()).eq("NO")).sum()
        ),
        "last_batch_candidate_rows": int(len(ledger_last_candidate)),
        "last_batch_candidate_imported": int(ledger_last_candidate.get("processing_status", pd.Series(dtype=str)).eq("imported").sum())
        if not ledger_last_candidate.empty
        else 0,
        "pending_subjects": reconciled.loc[pending, "SubjectID"].astype(str).tolist(),
        "excluded_subjects": reconciled.loc[excluded, "SubjectID"].astype(str).tolist(),
    }


def write_readme(summary: Mapping[str, Any], pending: pd.DataFrame) -> None:
    pending_subjects = ", ".join(pending["SubjectID"].astype(str).tolist()) if not pending.empty else "none"
    lines = [
        "# ADNI v5.1 Ledger Reconciled After Batch 20260514b",
        "",
        f"Generated: `{summary['generated']}`",
        "",
        "This is a reconciled ledger for Drive review. It does not replace the live Drive copy by itself.",
        "",
        "## Answers",
        "",
        f"- Total ledger rows: `{summary['total_rows']}`.",
        f"- Imported: `{summary['imported']}`.",
        f"- Pending: `{summary['pending']}`.",
        f"- Excluded: `{summary['excluded']}`.",
        f"- Imported CN-GE: `{summary['imported_CN_GE']}`.",
        f"- Pending CN-GE: `{summary['pending_CN_GE']}`.",
        f"- Excluded DICOM rows: `{summary['excluded_DICOM']}`.",
        f"- Final training-ready metadata CN-GE: `{summary['final_training_ready_CN_GE']}`.",
        f"- Imported CN-GE matches final metadata CN-GE=101: `{summary['imported_CN_GE_matches_final_metadata_CN_GE']}`.",
        f"- Python bandpass requested values different from NO: `{summary['python_bandpass_requested_non_NO']}`.",
        "- Python bandpass final path: `OFF`.",
        "- Training run: `False`.",
        "- Tensor modification: `False`.",
        "- Drive replacement: `False`.",
        "",
        "## Remaining Pending Subjects",
        "",
        pending_subjects,
        "",
        "## Files",
        "",
        f"- Reconciled ledger: `{OUT_RECONCILED}`",
        f"- Snapshot: `{OUT_SNAPSHOT}`",
        f"- Remaining pending: `{OUT_REMAINING}`",
        f"- Summary JSON: `{OUT_SUMMARY_JSON}`",
    ]
    OUT_README.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ledger = read_csv(args.ledger_original)
    final_metadata = read_csv(args.subject_metadata)
    training_ready = read_csv(args.training_ready_metadata)
    last_candidate = read_csv(args.ledger_last_candidate, required=False)
    decisions_by_batch, decisions_by_subject = load_batch_decisions(
        {
            "20260513_bandpass_batch1": args.batch_20260513,
            "20260514_bandpass_batch2": args.batch_20260514,
            "20260514b_bandpass_batch3": args.batch_20260514b,
        }
    )
    reconciled = reconcile_ledger(ledger, final_metadata, decisions_by_batch, decisions_by_subject)
    pending = reconciled[reconciled["processing_status"].eq("pending")].copy()
    summary = build_summary(reconciled, training_ready, last_candidate)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reconciled.to_csv(OUT_RECONCILED, index=False)
    reconciled.to_csv(OUT_SNAPSHOT, index=False)
    pending.to_csv(OUT_REMAINING, index=False)
    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(summary, pending)

    print(f"total_rows={summary['total_rows']}")
    print(f"imported={summary['imported']}")
    print(f"pending={summary['pending']}")
    print(f"excluded={summary['excluded']}")
    print(f"imported_CN_GE={summary['imported_CN_GE']}")
    print(f"pending_CN_GE={summary['pending_CN_GE']}")
    print(f"excluded_DICOM={summary['excluded_DICOM']}")
    print(f"final_training_ready_CN_GE={summary['final_training_ready_CN_GE']}")
    print(f"imported_CN_GE_matches_final_metadata_CN_GE={summary['imported_CN_GE_matches_final_metadata_CN_GE']}")
    print(f"python_bandpass_requested_non_NO={summary['python_bandpass_requested_non_NO']}")
    print("training_run=False")
    print("tensor_modification=False")
    print("drive_replacement=False")


if __name__ == "__main__":
    main()
