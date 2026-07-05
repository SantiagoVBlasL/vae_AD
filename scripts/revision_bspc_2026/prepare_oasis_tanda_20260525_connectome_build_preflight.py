#!/usr/bin/env python
"""Dry-run/read-only OASIS connectome build preflight.

This script plans an external-validation OASIS connectome tensor build using the
audited 170-to-131 AAL3 ROI mapping. It does not compute connectomes and does
not modify input data.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "Tanda_2026_05_25"
DEFAULT_OASIS_AUDIT = (
    PROJECT_ROOT / "results" / "revision_bspc_2026" / "oasis_tanda_2026_05_25_audit"
)
DEFAULT_ROI_MAPPING_AUDIT = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "oasis_tanda_2026_05_25_roi_mapping_audit"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "oasis_tanda_2026_05_25_connectome_build_preflight"
)

CHANNELS = [
    {
        "channel_index": 0,
        "channel_name": "Pearson_Full_FisherZ_Signed",
        "planned_source": "ROI time series after AAL3 170-to-ADNI-131 extraction and ADNI Yeo17 order",
    },
    {
        "channel_index": 1,
        "channel_name": "Pearson_OMST_GCE_Signed_Weighted",
        "planned_source": "Pearson full matrix transformed by locked ADNI OMST/GCE graph filtering",
    },
    {
        "channel_index": 2,
        "channel_name": "MI_KNN_Symmetric",
        "planned_source": "kNN mutual-information channel, symmetric, using locked ADNI channel convention",
    },
]

BUILD_CANDIDATES = [
    {
        "build_candidate": "runwise_connectome_average",
        "description": "Compute each channel per QC-usable run, Fisher-z/static-transform at run level as applicable, then average connectomes per subject/session.",
    },
    {
        "build_candidate": "concatenated_timeseries",
        "description": "Concatenate QC-usable ROI time series within subject/session, then compute one connectome per subject/session.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--oasis-audit-dir", type=Path, default=DEFAULT_OASIS_AUDIT)
    parser.add_argument("--roi-mapping-audit-dir", type=Path, default=DEFAULT_ROI_MAPPING_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--confirm-build",
        action="store_true",
        help="Reserved for a future build script. This preflight remains read-only and will not compute connectomes.",
    )
    return parser.parse_args()


def require_paths(args: argparse.Namespace) -> None:
    required = [
        args.input_dir,
        args.oasis_audit_dir / "run_manifest.csv",
        args.oasis_audit_dir / "subject_session_manifest.csv",
        args.roi_mapping_audit_dir / "roi_mapping_170_to_131.csv",
        args.roi_mapping_audit_dir / "final_recommendation.md",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required input(s): " + ", ".join(missing))


def write_csv_md(df: pd.DataFrame, csv_path: Path, md_path: Path, title: str) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        if df.empty:
            f.write("_No rows._\n")
        else:
            f.write(df.to_markdown(index=False))
            f.write("\n")


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    run_manifest = pd.read_csv(args.oasis_audit_dir / "run_manifest.csv")
    subject_manifest = pd.read_csv(args.oasis_audit_dir / "subject_session_manifest.csv")
    roi_mapping_raw = pd.read_csv(args.roi_mapping_audit_dir / "roi_mapping_170_to_131.csv")
    final_recommendation = (args.roi_mapping_audit_dir / "final_recommendation.md").read_text(
        encoding="utf-8"
    )
    return run_manifest, subject_manifest, roi_mapping_raw, final_recommendation


def build_roi_mapping_used(roi_mapping_raw: pd.DataFrame) -> pd.DataFrame:
    keep = roi_mapping_raw[roi_mapping_raw["keep_drop"].astype(str).eq("keep")].copy()
    keep["ADNI_final_index_131"] = keep["ADNI_final_index_131"].astype(int)
    keep["oasis_col_idx_0based"] = keep["oasis_col_idx_0based"].astype(int)
    keep["oasis_col_idx_1based"] = keep["oasis_col_idx_1based"].astype(int)
    keep = keep.sort_values("ADNI_final_index_131").reset_index(drop=True)
    keep.insert(0, "planned_extraction_order", keep.index.astype(int))
    keep["planned_extraction_order_1based"] = keep["planned_extraction_order"] + 1
    keep["matches_adni_final_order"] = keep["planned_extraction_order"].eq(keep["ADNI_final_index_131"])
    cols = [
        "planned_extraction_order",
        "planned_extraction_order_1based",
        "oasis_col_idx_0based",
        "oasis_col_idx_1based",
        "AAL3_ROI_name",
        "AAL3_short_name",
        "ADNI_final_ROI_name",
        "ADNI_final_index_131",
        "ADNI_final_index_131_1based",
        "adni_original_index_131_before_yeo_reorder",
        "matches_adni_final_order",
        "reason",
    ]
    return keep[cols]


def build_run_selection_table(run_manifest: pd.DataFrame, roi_order_status: str) -> pd.DataFrame:
    runs = run_manifest.copy()
    runs["roi_qc_ok_bool"] = bool_series(runs["roi_qc_ok"])
    runs["selected_for_preflight"] = runs["roi_qc_ok_bool"]
    runs["blocked_until_roi_order_confirmation"] = roi_order_status != "confirmed"
    runs["build_candidates"] = runs["selected_for_preflight"].map(
        lambda ok: "runwise_connectome_average;concatenated_timeseries" if ok else ""
    )
    runs["planned_roi_source"] = runs.apply(
        lambda r: r.get("roi_txt_path") if r["selected_for_preflight"] else "", axis=1
    )
    keep_cols = [
        "subject_id",
        "session_id",
        "run_id",
        "experiment_id",
        "diagnosis",
        "diagnosis_confidence",
        "TR_seconds",
        "Manufacturer",
        "ScannerModel",
        "age_at_MR",
        "sex",
        "raw_bold_expected",
        "has_roi_mat",
        "has_roi_txt",
        "roi_qc_ok",
        "n_timepoints",
        "n_rois",
        "finite_fraction",
        "run_availability_status",
        "selected_for_preflight",
        "blocked_until_roi_order_confirmation",
        "build_candidates",
        "roi_mat_path",
        "roi_txt_path",
    ]
    return runs[keep_cols].sort_values(["subject_id", "session_id", "run_id"]).reset_index(drop=True)


def build_subject_level_manifest(run_selection: pd.DataFrame, subject_manifest: pd.DataFrame) -> pd.DataFrame:
    selected = run_selection[run_selection["selected_for_preflight"]].copy()
    selected["n_timepoints"] = pd.to_numeric(selected["n_timepoints"], errors="coerce").fillna(0).astype(int)
    grouped = (
        selected.groupby(["subject_id", "session_id"], dropna=False)
        .agg(
            selected_qc_runs=("run_id", "count"),
            selected_run_ids=("run_id", lambda s: ";".join(s.astype(str))),
            selected_total_timepoints=("n_timepoints", "sum"),
            min_run_timepoints=("n_timepoints", "min"),
            max_run_timepoints=("n_timepoints", "max"),
        )
        .reset_index()
    )
    base_cols = [
        "subject_id",
        "session_id",
        "experiment_id",
        "diagnosis",
        "diagnosis_confidence",
        "TR_seconds",
        "Manufacturer",
        "ScannerModel",
        "age_at_MR",
        "sex",
        "expected_raw_bold_runs",
        "processed_roi_runs",
        "roi_qc_ok_runs",
        "has_multiple_qc_ok_runs",
    ]
    base = subject_manifest[base_cols].copy()
    out = base.merge(grouped, on=["subject_id", "session_id"], how="left")
    out["selected_qc_runs"] = out["selected_qc_runs"].fillna(0).astype(int)
    out["selected_total_timepoints"] = out["selected_total_timepoints"].fillna(0).astype(int)
    out["selected_run_ids"] = out["selected_run_ids"].fillna("")
    out["planned_include_subject_session"] = out["selected_qc_runs"] >= 1
    out["runwise_connectome_average_plan"] = out["selected_qc_runs"].map(
        lambda n: f"average {n} per-run connectome(s)" if n >= 1 else "exclude: no QC-usable ROI run"
    )
    out["concatenated_timeseries_plan"] = out.apply(
        lambda r: f"concatenate {int(r['selected_qc_runs'])} run(s), total_timepoints={int(r['selected_total_timepoints'])}"
        if r["selected_qc_runs"] >= 1
        else "exclude: no QC-usable ROI run",
        axis=1,
    )
    return out.sort_values(["subject_id", "session_id"]).reset_index(drop=True)


def get_roi_order_status(final_recommendation_text: str) -> str:
    if "ready_for_oasis_connectome_build" in final_recommendation_text:
        return "confirmed"
    if "needs_martin_roi_order_confirmation" in final_recommendation_text:
        return "needs_martin_roi_order_confirmation"
    return "unresolved"


def build_planned_tensor_shape(
    subject_level: pd.DataFrame, roi_mapping_used: pd.DataFrame, roi_order_status: str
) -> str:
    included = subject_level[subject_level["planned_include_subject_session"]].copy()
    diagnosis_counts = included["diagnosis"].value_counts(dropna=False).to_dict()
    manufacturer_counts = included["Manufacturer"].value_counts(dropna=False).to_dict()
    n_subject_sessions = int(len(included))
    n_runs = int(subject_level["selected_qc_runs"].sum())
    n_rois = int(len(roi_mapping_used))
    n_channels = len(CHANNELS)
    channel_lines = "\n".join(
        f"- channel {c['channel_index']}: `{c['channel_name']}`" for c in CHANNELS
    )
    candidate_lines = "\n".join(
        f"- `{c['build_candidate']}`: planned tensor shape `({n_subject_sessions}, {n_channels}, {n_rois}, {n_rois})`"
        for c in BUILD_CANDIDATES
    )
    return f"""# Planned Tensor Shape

This is a dry-run plan only. No connectomes were computed.

## ROI Order Status

`{roi_order_status}`

The build remains blocked until Martin confirms the OASIS AAL3 ROI column order, unless that confirmation has already been documented externally.

## Included Subject/Sessions

- Planned included subject/sessions: {n_subject_sessions}
- Planned selected QC-usable runs: {n_runs}
- Diagnosis counts: {diagnosis_counts}
- Manufacturer counts: {manufacturer_counts}

## Channels

{channel_lines}

## Candidate Build Strategies

{candidate_lines}

## Intermediate Shapes

- `runwise_connectome_average`: first compute per-run tensors with shape `({n_runs}, {n_channels}, {n_rois}, {n_rois})`, then aggregate to subject/session tensor shape `({n_subject_sessions}, {n_channels}, {n_rois}, {n_rois})`.
- `concatenated_timeseries`: concatenate ROI time series within each subject/session, then compute directly to tensor shape `({n_subject_sessions}, {n_channels}, {n_rois}, {n_rois})`.

## Non-Actions

- No ROI time-series matrices were loaded for connectivity computation.
- No connectomes were computed.
- No input data were modified.
"""


def build_dry_run_report(
    run_selection: pd.DataFrame,
    subject_level: pd.DataFrame,
    roi_mapping_used: pd.DataFrame,
    roi_order_status: str,
) -> str:
    selected = run_selection[run_selection["selected_for_preflight"]]
    included = subject_level[subject_level["planned_include_subject_session"]]
    multi = included[included["selected_qc_runs"] >= 2]
    single = included[included["selected_qc_runs"] == 1]
    return f"""# Dry-Run Report

Status: `dry_run_only`

## Validation Summary

- ROI mapping rows retained for ADNI order: {len(roi_mapping_used)}
- ROI mapping order exact: {bool(roi_mapping_used["matches_adni_final_order"].all())}
- Selected QC-usable runs: {len(selected)}
- Included subject/sessions: {len(included)}
- Subject/sessions with two or more runs: {len(multi)}
- Subject/sessions with one run: {len(single)}
- ROI-order status from mapping audit: `{roi_order_status}`

## Planned Build Candidates

1. `runwise_connectome_average`
   - compute each connectome channel per QC-usable run
   - combine run-level connectomes per subject/session after connectivity

2. `concatenated_timeseries`
   - concatenate QC-usable ROI time series within subject/session
   - compute one connectome per subject/session

## Guardrails For Future Build

- Require explicit ROI-order confirmation before running a real build.
- Require `--confirm-build` in any future build-capable script.
- Keep OASIS external-only; do not merge OASIS into ADNI training or threshold selection.
- Use the exact `roi_mapping_used.csv` extraction order.
"""


def main() -> None:
    args = parse_args()
    if args.confirm_build:
        raise RuntimeError(
            "This is a read-only preflight script. Real connectome computation is intentionally not implemented here."
        )
    require_paths(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_manifest, subject_manifest, roi_mapping_raw, final_recommendation = load_inputs(args)
    roi_order_status = get_roi_order_status(final_recommendation)

    roi_mapping_used = build_roi_mapping_used(roi_mapping_raw)
    if len(roi_mapping_used) != 131:
        raise ValueError(f"Expected 131 retained ROIs, got {len(roi_mapping_used)}")
    if not bool(roi_mapping_used["matches_adni_final_order"].all()):
        raise ValueError("ROI mapping is not sorted in exact ADNI final 131 order")

    run_selection = build_run_selection_table(run_manifest, roi_order_status)
    subject_level = build_subject_level_manifest(run_selection, subject_manifest)

    write_csv_md(
        run_selection,
        args.output_dir / "run_selection_table.csv",
        args.output_dir / "run_selection_table.md",
        "Run Selection Table",
    )
    write_csv_md(
        subject_level,
        args.output_dir / "subject_level_manifest.csv",
        args.output_dir / "subject_level_manifest.md",
        "Subject-Level Manifest",
    )
    write_csv_md(
        roi_mapping_used,
        args.output_dir / "roi_mapping_used.csv",
        args.output_dir / "roi_mapping_used.md",
        "ROI Mapping Used",
    )

    (args.output_dir / "planned_tensor_shape.md").write_text(
        build_planned_tensor_shape(subject_level, roi_mapping_used, roi_order_status),
        encoding="utf-8",
    )
    (args.output_dir / "dry_run_report.md").write_text(
        build_dry_run_report(run_selection, subject_level, roi_mapping_used, roi_order_status),
        encoding="utf-8",
    )

    selected = run_selection[run_selection["selected_for_preflight"]]
    included = subject_level[subject_level["planned_include_subject_session"]]
    command_log: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "mode": "read_only_dry_run",
        "confirm_build": False,
        "computed_connectomes": False,
        "modified_input_data": False,
        "inputs": {
            "input_dir": str(args.input_dir),
            "oasis_audit_dir": str(args.oasis_audit_dir),
            "roi_mapping_audit_dir": str(args.roi_mapping_audit_dir),
            "output_dir": str(args.output_dir),
        },
        "roi_order_status": roi_order_status,
        "planned_candidates": [c["build_candidate"] for c in BUILD_CANDIDATES],
        "planned_channels": [c["channel_name"] for c in CHANNELS],
        "counts": {
            "run_manifest_rows": int(len(run_manifest)),
            "selected_qc_runs": int(len(selected)),
            "subject_sessions_included": int(len(included)),
            "rois": int(len(roi_mapping_used)),
            "channels": int(len(CHANNELS)),
        },
        "outputs": [
            "run_selection_table.csv",
            "run_selection_table.md",
            "subject_level_manifest.csv",
            "subject_level_manifest.md",
            "roi_mapping_used.csv",
            "roi_mapping_used.md",
            "planned_tensor_shape.md",
            "dry_run_report.md",
            "command_log.json",
        ],
    }
    (args.output_dir / "command_log.json").write_text(
        json.dumps(command_log, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "mode": "read_only_dry_run",
                "roi_order_status": roi_order_status,
                "selected_qc_runs": int(len(selected)),
                "subject_sessions_included": int(len(included)),
                "planned_tensor_shape": [int(len(included)), int(len(CHANNELS)), int(len(roi_mapping_used)), int(len(roi_mapping_used))],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
