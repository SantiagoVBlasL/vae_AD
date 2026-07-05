#!/usr/bin/env python3
"""Prepare the OASIS next-batch 60CN/60AD download/package helper.

No external download is attempted. The script packages the locked selection
manifests, verifies non-overlap with the Tanda_2026_05_25 pilot, writes QC
tables and manual instructions for Martin, and creates a zip archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SELECTED = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/oasis_next_batch_selection_audit/selected_ideal_60CN_60AD.csv"
)
DEFAULT_DOWNLOAD_MANIFEST = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/oasis_next_batch_selection_audit/download_manifest_for_martin.csv"
)
DEFAULT_PILOT_DIR = PROJECT_ROOT / "data/Tanda_2026_05_25"
DEFAULT_PILOT_AUDIT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/oasis_tanda_2026_05_25_audit"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/OASIS_next_60CN_60AD_2026_05_26"
DEFAULT_ARCHIVE = PROJECT_ROOT / "data/OASIS_next_60CN_60AD_2026_05_26.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--selected-csv", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--download-manifest", type=Path, default=DEFAULT_DOWNLOAD_MANIFEST)
    parser.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT_DIR)
    parser.add_argument("--pilot-audit-dir", type=Path, default=DEFAULT_PILOT_AUDIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--skip-zip", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def write_csv_md(df: pd.DataFrame, csv_path: Path, title: str) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    with csv_path.with_suffix(".md").open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        if df.empty:
            f.write("_No rows._\n")
        else:
            f.write(df.to_markdown(index=False))
            f.write("\n")


def normalize_subject(value: Any) -> str:
    text = str(value).strip()
    match = re.search(r"OAS\d{5}", text)
    return f"sub-{match.group(0)}" if match else text


def normalize_session(value: Any) -> str:
    text = str(value).strip()
    if text.startswith("ses-"):
        return text
    match = re.search(r"MR_d(\d+)", text)
    return f"ses-d{match.group(1)}" if match else text


def normalize_experiment(value: Any) -> str:
    return str(value).strip()


def load_pilot_sets(pilot_dir: Path, audit_dir: Path) -> dict[str, set[str]]:
    subject_ids: set[str] = set()
    session_ids: set[str] = set()
    subject_session_pairs: set[str] = set()
    experiment_ids: set[str] = set()
    candidate_files = [
        pilot_dir / "oasis3_pilot_30cn_30ad_subjects.csv",
        audit_dir / "subject_session_manifest.csv",
        audit_dir / "run_manifest.csv",
    ]
    for path in candidate_files:
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        if "subject_id" in df.columns:
            subjects = df["subject_id"].map(normalize_subject)
            subject_ids.update(subjects)
        else:
            subjects = pd.Series([""] * len(df))
        if "session_id" in df.columns:
            sessions = df["session_id"].map(normalize_session)
            session_ids.update(sessions)
        else:
            sessions = pd.Series([""] * len(df))
        subject_session_pairs.update(
            f"{sid}|{session}" for sid, session in zip(subjects, sessions) if sid and session
        )
        if "experiment_id" in df.columns:
            experiment_ids.update(df["experiment_id"].map(normalize_experiment))
    return {
        "subject_ids": {x for x in subject_ids if x},
        "session_ids": {x for x in session_ids if x},
        "subject_session_pairs": {x for x in subject_session_pairs if x},
        "experiment_ids": {x for x in experiment_ids if x},
    }


def selected_subjects(selected: pd.DataFrame, manifest: pd.DataFrame, pilot_sets: dict[str, set[str]]) -> pd.DataFrame:
    selected = selected.copy()
    manifest = manifest.copy()
    for df in (selected, manifest):
        df["subject_id_norm"] = df["subject_id"].map(normalize_subject)
        df["session_id_norm"] = df["session_id"].map(normalize_session)
        df["experiment_id_norm"] = df["experiment_id"].map(normalize_experiment)
    manifest_cols = [
        c
        for c in [
            "subject_id_norm",
            "session_id_norm",
            "experiment_id_norm",
            "included_in_minimum_30CN_30AD",
            "included_in_ideal_60CN_60AD",
            "scanner_manufacturer_model",
            "expected_runs",
            "expected_tr2_rest_runs",
            "has_two_or_more_expected_runs",
            "reason_selected",
        ]
        if c in manifest.columns
    ]
    merged = selected.merge(
        manifest[manifest_cols].drop_duplicates(["subject_id_norm", "session_id_norm", "experiment_id_norm"]),
        on=["subject_id_norm", "session_id_norm", "experiment_id_norm"],
        how="left",
        suffixes=("", "_download_manifest"),
    )
    merged["pilot_subject_overlap_recomputed"] = merged["subject_id_norm"].isin(pilot_sets["subject_ids"])
    merged["subject_session_pair"] = merged["subject_id_norm"] + "|" + merged["session_id_norm"]
    merged["pilot_session_overlap_recomputed"] = merged["subject_session_pair"].isin(
        pilot_sets["subject_session_pairs"]
    )
    merged["pilot_session_day_overlap_descriptive"] = merged["session_id_norm"].isin(
        pilot_sets["session_ids"]
    )
    merged["pilot_experiment_overlap_recomputed"] = merged["experiment_id_norm"].isin(pilot_sets["experiment_ids"])
    merged["download_status"] = "not_downloaded_external_credentials_or_commands_not_available"
    merged["local_package_path"] = ""
    return merged


def build_download_qc(selected_used: pd.DataFrame, output_dir: Path, archive: Path) -> pd.DataFrame:
    dx_counts = selected_used["diagnosis"].value_counts().to_dict()
    rows = [
        {"qc_item": "target_rows", "value": len(selected_used), "status": "ok", "details": "selected_ideal_60CN_60AD"},
        {"qc_item": "unique_subjects", "value": selected_used["subject_id_norm"].nunique(), "status": "ok", "details": ""},
        {"qc_item": "unique_experiment_ids", "value": selected_used["experiment_id_norm"].nunique(), "status": "ok", "details": ""},
        {"qc_item": "CN_count", "value": dx_counts.get("CN", 0), "status": "ok" if dx_counts.get("CN", 0) == 60 else "check", "details": ""},
        {"qc_item": "AD_DEMENTIA_count", "value": dx_counts.get("AD_DEMENTIA", 0), "status": "ok" if dx_counts.get("AD_DEMENTIA", 0) == 60 else "check", "details": ""},
        {"qc_item": "pilot_subject_overlap", "value": int(selected_used["pilot_subject_overlap_recomputed"].sum()), "status": "ok" if not selected_used["pilot_subject_overlap_recomputed"].any() else "fail", "details": ""},
        {"qc_item": "pilot_subject_session_pair_overlap", "value": int(selected_used["pilot_session_overlap_recomputed"].sum()), "status": "ok" if not selected_used["pilot_session_overlap_recomputed"].any() else "fail", "details": "SubjectID+session_id pair; session day alone is not globally unique."},
        {"qc_item": "pilot_session_day_overlap_descriptive", "value": int(selected_used["pilot_session_day_overlap_descriptive"].sum()), "status": "informational", "details": "Same ses-d#### can occur for different subjects and is not treated as overlap."},
        {"qc_item": "pilot_experiment_overlap", "value": int(selected_used["pilot_experiment_overlap_recomputed"].sum()), "status": "ok" if not selected_used["pilot_experiment_overlap_recomputed"].any() else "fail", "details": ""},
        {"qc_item": "download_attempted", "value": False, "status": "manual_required", "details": "No OASIS credentials/external download command available in this workspace."},
        {"qc_item": "output_dir", "value": str(output_dir), "status": "ok", "details": ""},
        {"qc_item": "archive_path", "value": str(archive), "status": "planned", "details": ""},
    ]
    return pd.DataFrame(rows)


def missing_downloads(selected_used: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "subject_id",
        "session_id",
        "experiment_id",
        "diagnosis",
        "age_at_MR",
        "sex",
        "Manufacturer",
        "ScannerModel",
        "TR_seconds",
        "expected_tr2_rest_runs",
        "expected_runs",
        "target_tr2_series_descriptions",
        "reason_selected",
        "download_status",
    ]
    out = selected_used[[c for c in cols if c in selected_used.columns]].copy()
    out["failure_or_missing_reason"] = "not_downloaded_no_credentials_or_external_download_command_available"
    out["manual_action"] = "Martin should download/process this experiment from OASIS using the package manifest."
    return out


def file_inventory(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(root)
        rows.append(
            {
                "relative_path": str(rel),
                "path": str(path),
                "extension": path.suffix,
                "size_bytes": path.stat().st_size,
                "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
            }
        )
    return pd.DataFrame(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_checksums(root: Path) -> None:
    checksum_path = root / "checksums_sha256.txt"
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_dir() or path == checksum_path:
            continue
        lines.append(f"{sha256_file(path)}  {path.relative_to(root)}")
    checksum_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_readme(root: Path, selected_used: pd.DataFrame, archive: Path) -> None:
    text = f"""# OASIS Next Batch 60CN/60AD Download Package

Created: `{datetime.now().isoformat(timespec='seconds')}`

This package contains the selected non-overlapping OASIS 60CN + 60 AD_DEMENTIA
batch manifests for Martin. No external download was attempted in this
workspace because OASIS credentials/download commands are not available here.

## Target Counts

- Rows: `{len(selected_used)}`
- Unique subjects: `{selected_used['subject_id_norm'].nunique()}`
- CN: `{int((selected_used['diagnosis'] == 'CN').sum())}`
- AD_DEMENTIA: `{int((selected_used['diagnosis'] == 'AD_DEMENTIA').sum())}`
- Pilot subject/session/experiment overlap: `0/0/0` expected and verified in `download_qc.csv`.

## Manual Download/Processing Instructions

1. Use `selected_subjects_used.csv` or `download_manifest_for_martin.csv` as the locked target list.
2. Retrieve exactly the listed `experiment_id` / `session_id` rows.
3. Do not include any subject/session from `data/Tanda_2026_05_25`.
4. Prioritize resting-state fMRI TR approximately 2.2 s, Siemens TrioTim, with the expected rest runs listed in the manifest.
5. Keep all available run-level files separated by subject/session/run.
6. If processed ROI signals are returned, use AAL3 ROI signal matrices with the same convention used for the Tanda batch.
7. Do not merge this batch into ADNI training; it is for external OASIS calibration/test only.

## Archive

The local package archive is planned/written at:

`{archive}`

The archive contains manifests and instructions only unless downloaded files are
manually added later.
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def copy_inputs(root: Path, selected_csv: Path, download_manifest: Path) -> None:
    shutil.copy2(selected_csv, root / "selected_ideal_60CN_60AD.csv")
    shutil.copy2(download_manifest, root / "download_manifest_for_martin.csv")


def create_archive(root: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(root.name / path.relative_to(root)) if isinstance(root.name, Path) else str(Path(root.name) / path.relative_to(root)))


def write_command_log(args: argparse.Namespace, root: Path, archive: Path, selected_used: pd.DataFrame) -> None:
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "argv": __import__("sys").argv,
        "inputs": {
            "selected_csv": str(args.selected_csv),
            "download_manifest": str(args.download_manifest),
            "pilot_dir": str(args.pilot_dir),
            "pilot_audit_dir": str(args.pilot_audit_dir),
        },
        "outputs": {
            "output_dir": str(root),
            "archive": str(archive),
        },
        "counts": {
            "rows": int(len(selected_used)),
            "unique_subjects": int(selected_used["subject_id_norm"].nunique()),
            "unique_experiments": int(selected_used["experiment_id_norm"].nunique()),
            "CN": int((selected_used["diagnosis"] == "CN").sum()),
            "AD_DEMENTIA": int((selected_used["diagnosis"] == "AD_DEMENTIA").sum()),
            "pilot_subject_overlap": int(selected_used["pilot_subject_overlap_recomputed"].sum()),
            "pilot_subject_session_pair_overlap": int(selected_used["pilot_session_overlap_recomputed"].sum()),
            "pilot_experiment_overlap": int(selected_used["pilot_experiment_overlap_recomputed"].sum()),
        },
        "download_attempted": False,
        "reason_no_download": "No OASIS credentials or external download command is available in this workspace.",
        "model_training": False,
        "connectome_computation": False,
        "previous_oasis_pilot_modified": False,
    }
    (root / "command_log.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    selected = read_csv(args.selected_csv)
    manifest = read_csv(args.download_manifest)
    pilot_sets = load_pilot_sets(args.pilot_dir, args.pilot_audit_dir)
    selected_used = selected_subjects(selected, manifest, pilot_sets)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    copy_inputs(args.output_dir, args.selected_csv, args.download_manifest)
    write_csv_md(selected_used, args.output_dir / "selected_subjects_used.csv", "Selected Subjects Used")
    qc = build_download_qc(selected_used, args.output_dir, args.archive)
    write_csv_md(qc, args.output_dir / "download_qc.csv", "Download QC")
    write_csv_md(missing_downloads(selected_used), args.output_dir / "missing_or_failed_downloads.csv", "Missing or Failed Downloads")
    write_readme(args.output_dir, selected_used, args.archive)
    write_command_log(args, args.output_dir, args.archive, selected_used)
    inventory = file_inventory(args.output_dir)
    write_csv_md(inventory, args.output_dir / "file_inventory.csv", "File Inventory")
    write_checksums(args.output_dir)
    inventory = file_inventory(args.output_dir)
    write_csv_md(inventory, args.output_dir / "file_inventory.csv", "File Inventory")
    write_checksums(args.output_dir)
    if not args.skip_zip:
        create_archive(args.output_dir, args.archive)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "archive": str(args.archive),
                "rows": int(len(selected_used)),
                "unique_subjects": int(selected_used["subject_id_norm"].nunique()),
                "CN": int((selected_used["diagnosis"] == "CN").sum()),
                "AD_DEMENTIA": int((selected_used["diagnosis"] == "AD_DEMENTIA").sum()),
                "pilot_overlap_subjects": int(selected_used["pilot_subject_overlap_recomputed"].sum()),
                "pilot_overlap_subject_session_pairs": int(selected_used["pilot_session_overlap_recomputed"].sum()),
                "pilot_overlap_experiments": int(selected_used["pilot_experiment_overlap_recomputed"].sum()),
                "download_attempted": False,
                "zip_created": bool(args.archive.exists()) if not args.skip_zip else False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
