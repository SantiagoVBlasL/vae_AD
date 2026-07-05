#!/usr/bin/env python3
"""Build compact CN-GE request tables for Martin.

Read-only with respect to data products:
- no training;
- no tensor construction;
- no source data modification;
- Python bandpass remains OFF.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FINAL_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "final_cn_ge_inventory_before_martin_request"
)

REJECTED_CN_GE = [
    "135_S_4446",
    "135_S_4598",
    "135_S_5113",
    "135_S_6104",
    "135_S_6359",
    "135_S_6360",
    "135_S_6411",
    "135_S_6473",
    "135_S_6509",
    "135_S_6510",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create short CN-GE QC rejection and request tables for Martin.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--final-dir", type=Path, default=DEFAULT_FINAL_DIR)
    return parser.parse_args()


def clean(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def numeric_for_display(value: Any, digits: int = 4) -> str:
    text = clean(value)
    if not text:
        return ""
    try:
        num = float(text)
    except ValueError:
        return text
    if abs(num - round(num)) < 1e-9:
        return str(int(round(num)))
    return f"{num:.{digits}f}".rstrip("0").rstrip(".")


def subject_candidate_rows(inventory: pd.DataFrame, sid: str) -> pd.DataFrame:
    sub = inventory[
        inventory["SubjectID"].eq(sid)
        & inventory["path"].ne("NO_LOCAL_FILE")
        & inventory["path_exists_now"].astype(str).str.lower().isin({"true", "1", "yes"})
    ].copy()
    return sub


def summarize_failure_reasons(sub: pd.DataFrame, final_reason: str) -> str:
    reasons = sorted({clean(x) for x in sub.get("reason", pd.Series(dtype=str)) if clean(x)})
    all_reasons = set()
    for reason in reasons:
        for part in reason.split(";"):
            part = clean(part)
            if part:
                all_reasons.add(part)
    if final_reason:
        for part in final_reason.split(";"):
            part = clean(part)
            if part:
                all_reasons.add(part)

    phrases: List[str] = []
    if "roi_count_not_170" in all_reasons:
        bad_counts = sorted(
            {
                numeric_for_display(x, digits=0)
                for x in sub.loc[sub["reason"].astype(str).str.contains("roi_count_not_170", na=False), "n_rois"]
            }
        )
        bad_counts = [x for x in bad_counts if x]
        if bad_counts:
            phrases.append(f"local ROI count is not AAL3-170 in some candidates (n_rois={','.join(bad_counts)})")
        else:
            phrases.append("local ROI count is not AAL3-170")
    if "scale_not_around_10000" in all_reasons:
        phrases.append("at least one local 170-ROI candidate has scale not compatible with 10000-level DPARSF outputs")
    if "stage_or_spectral_not_sufficient_for_direct_import" in all_reasons:
        phrases.append("170-ROI candidate exists but is CovRegressed/non-F-stage or not spectrally sufficient for direct no-Python-bandpass import")
    if "finite_fraction_lt_0.95" in all_reasons:
        phrases.append("finite signal fraction is below 0.95")
    if "load_failed" in all_reasons:
        phrases.append("local file failed to load")
    if not phrases:
        phrases = [final_reason or "; ".join(sorted(all_reasons)) or "failed v5.1 no-Python-bandpass QC"]
    return "; ".join(phrases)


def best_row_from_request(request: pd.DataFrame, sid: str) -> pd.Series:
    rows = request[request["SubjectID"].eq(sid)]
    if rows.empty:
        raise RuntimeError(f"Subject missing from request CSV: {sid}")
    return rows.iloc[0]


def build_rejected_summary(inventory: pd.DataFrame, request: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    missing = sorted(set(REJECTED_CN_GE) - set(request["SubjectID"]))
    if missing:
        raise RuntimeError(f"Rejected subjects missing from request CSV: {missing}")

    for sid in REJECTED_CN_GE:
        req = best_row_from_request(request, sid)
        sub = subject_candidate_rows(inventory, sid)
        final_reason = clean(req.get("final_reason", ""))
        rows.append(
            {
                "SubjectID": sid,
                "best_local_path": clean(req.get("best_path", "")),
                "stage_guess": clean(req.get("best_stage_guess", "")),
                "n_rois": numeric_for_display(req.get("best_n_rois", "")),
                "n_timepoints": numeric_for_display(req.get("best_n_timepoints", "")),
                "finite_fraction": numeric_for_display(req.get("best_finite_fraction", "")),
                "scale_label": clean(req.get("best_scale_label", "")),
                "qc_failure_reason": summarize_failure_reasons(sub, final_reason),
                "action_needed": "re-export/reprocess first-visit AAL3 ROISignals with 170 ROIs, finite signal, 10000-compatible scale, DPARSF-compatible stage, Python bandpass OFF",
            }
        )
    return pd.DataFrame(rows)


def request_action(row: pd.Series) -> str:
    cls = clean(row.get("final_classification", ""))
    if cls == "already_local_but_rejected_qc":
        return "re-export/reprocess local QC-rejected CN-GE"
    if cls == "no_roisignals_found_needs_preprocessing":
        return "provide/process missing first-visit CN-GE ROISignals"
    if cls == "already_local_but_stage_confirmation_needed":
        return "confirm local stage or re-export if incompatible"
    return clean(row.get("martin_request", "")) or "review"


def build_compact_request(request: pd.DataFrame, rejected_summary: pd.DataFrame) -> pd.DataFrame:
    rejected_reasons = rejected_summary.set_index("SubjectID")["qc_failure_reason"].to_dict()
    rows: List[Dict[str, Any]] = []
    for idx, (_, row) in enumerate(request.reset_index(drop=True).iterrows(), start=1):
        sid = clean(row.get("SubjectID", ""))
        cls = clean(row.get("final_classification", ""))
        local_status = (
            "local_file_rejected_qc"
            if cls == "already_local_but_rejected_qc"
            else "no_local_roisignals"
            if cls == "no_roisignals_found_needs_preprocessing"
            else "local_stage_confirmation_needed"
        )
        rows.append(
            {
                "priority_rank": idx,
                "priority_group": "priority_first_67" if idx <= 67 else "remaining_after_first_67",
                "SubjectID": sid,
                "ImageID": clean(row.get("ImageID", "")),
                "ResearchGroup_Mapped": clean(row.get("ResearchGroup_Mapped", "")),
                "Manufacturer": clean(row.get("Manufacturer", "")),
                "Visit": clean(row.get("Visit", "")),
                "StudyDate": clean(row.get("StudyDate", "")),
                "Age": clean(row.get("Age", "")),
                "Sex": clean(row.get("Sex", "")),
                "request_category": cls,
                "local_status": local_status,
                "best_local_path": clean(row.get("best_path", "")),
                "qc_or_missing_reason": rejected_reasons.get(sid, clean(row.get("final_reason", ""))),
                "action_needed": request_action(row),
                "python_bandpass_requested": "NO",
            }
        )
    return pd.DataFrame(rows)


def write_readme(
    path: Path,
    rejected_summary: pd.DataFrame,
    compact: pd.DataFrame,
    included_n: int,
    rejected_n: int,
    missing_n: int,
) -> None:
    reason_counts = (
        rejected_summary["qc_failure_reason"]
        .str.split("; ")
        .explode()
        .value_counts()
        .to_dict()
    )
    reason_lines = "\n".join(f"- `{reason}`: `{count}` subjects/cases." for reason, count in reason_counts.items())
    priority_first = int(compact["priority_group"].eq("priority_first_67").sum())
    priority_remaining = int(compact["priority_group"].eq("remaining_after_first_67").sum())
    lines = [
        "# CN-GE v5.1 Update For Martin",
        "",
        "Read-only communication summary. No training, no tensor construction, and no data products were modified.",
        "",
        "## Current Status",
        "",
        f"- CN-GE already incorporated into `v5.1_gecn9`: `{included_n}`.",
        f"- CN-GE local files rejected by QC: `{rejected_n}`.",
        f"- CN-GE with no local ROISignals found: `{missing_n}`.",
        "- Python bandpass: `OFF`. We are not asking for, nor applying, an extra Python bandpass step.",
        "",
        "## Why The 10 Local CN-GE Were Not Incorporated",
        "",
        reason_lines or "- All 10 failed v5.1 no-Python-bandpass QC.",
        "",
        "The short per-subject table is `cn_ge_qc_rejected_reason_summary_for_martin.csv`.",
        "",
        "## What We Need From Martin",
        "",
        "Please re-export/reprocess only the remaining CN-GE subjects in `cn_ge_request_compact_for_martin.csv`:",
        "",
        "- the 10 local QC-rejected CN-GE need fresh first-visit AAL3 ROISignals;",
        "- the 91 missing CN-GE need first-visit AAL3 ROISignals if available;",
        "- all outputs should be DPARSF-compatible, 170 ROIs, finite signal, scale compatible with the 10000-level DPARSF outputs, and Python bandpass OFF.",
        "",
        f"The compact request marks `{priority_first}` rows as `priority_first_67` and `{priority_remaining}` as `remaining_after_first_67`, so the first 67 Martin mentioned can be handled first.",
        "",
        "## Files",
        "",
        "- `cn_ge_qc_rejected_reason_summary_for_martin.csv`",
        "- `cn_ge_request_compact_for_martin.csv`",
        "- `README_for_martin_update.md`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    final_dir = args.final_dir
    inventory_path = final_dir / "cn_ge_final_inventory.csv"
    request_path = final_dir / "cn_ge_to_ask_martin.csv"

    inventory = read_csv(inventory_path)
    request = read_csv(request_path)

    rejected_request = request[request["final_classification"].eq("already_local_but_rejected_qc")].copy()
    rejected_ids = rejected_request["SubjectID"].tolist()
    if rejected_ids != REJECTED_CN_GE:
        raise RuntimeError(
            "Unexpected rejected CN-GE list/order. "
            f"Got {rejected_ids}, expected {REJECTED_CN_GE}"
        )

    rejected_summary = build_rejected_summary(inventory, request)
    compact = build_compact_request(request, rejected_summary)

    rejected_summary_path = final_dir / "cn_ge_qc_rejected_reason_summary_for_martin.csv"
    compact_path = final_dir / "cn_ge_request_compact_for_martin.csv"
    readme_path = final_dir / "README_for_martin_update.md"

    rejected_summary.to_csv(rejected_summary_path, index=False)
    compact.to_csv(compact_path, index=False)

    missing_n = int(request["final_classification"].eq("no_roisignals_found_needs_preprocessing").sum())
    rejected_n = int(request["final_classification"].eq("already_local_but_rejected_qc").sum())
    included_n = 9
    included_path = final_dir / "cn_ge_already_included_v5_1_gecn9.csv"
    if included_path.exists():
        included_n = int(read_csv(included_path)["SubjectID"].nunique())

    write_readme(readme_path, rejected_summary, compact, included_n, rejected_n, missing_n)

    command = {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "script": str(Path(__file__).resolve()),
        "inventory_path": str(inventory_path),
        "request_path": str(request_path),
        "outputs": [
            str(rejected_summary_path),
            str(compact_path),
            str(readme_path),
        ],
        "training_run": False,
        "tensor_construction": False,
        "python_bandpass_applied": False,
    }
    (final_dir / "cn_ge_martin_update_command_log.json").write_text(
        json.dumps(command, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {rejected_summary_path}")
    print(f"Wrote {compact_path}")
    print(f"Wrote {readme_path}")
    print(f"summary: included={included_n} rejected_qc={rejected_n} missing_local={missing_n}")
    print("No training. No tensor construction. Python bandpass OFF.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
