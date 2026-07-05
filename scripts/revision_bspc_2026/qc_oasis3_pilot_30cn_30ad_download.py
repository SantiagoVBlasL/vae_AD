#!/usr/bin/env python3
"""Read-only QC for the OASIS3 pilot 30 CN / 30 AD download.

The script compares expected experiment_id values from the pilot manifest
against the local download folder, then reports BOLD/T1w NIfTI availability,
run counts, missing experiments, and suspicious tiny files.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/oasis3_pilot_30cn_30ad_download_manifest/oasis3_pilot_30cn_30ad_subjects.csv"
)
DEFAULT_DOWNLOAD_ROOT = Path("/media/diego/Datos/OASIS3/pilot_30cn_30ad_bold_T1w")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/oasis3_pilot_30cn_30ad_download_qc"

NIFTI_TINY_BYTES = 1_000_000
JSON_TINY_BYTES = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--download-root", type=Path, default=DEFAULT_DOWNLOAD_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def md_table(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        text = "_No rows._"
    else:
        text = df.to_markdown(index=False)
    path.write_text(text + "\n", encoding="utf-8")


def is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def is_bold_nifti(path: Path) -> bool:
    name = path.name.lower()
    parts = {p.name.lower() for p in path.parents}
    return is_nifti(path) and ("bold" in name or any(p.startswith("func") for p in parts))


def is_t1w_nifti(path: Path) -> bool:
    name = path.name.lower()
    parts = {p.name.lower() for p in path.parents}
    return is_nifti(path) and ("t1w" in name or any(p.startswith("anat") for p in parts))


def immediate_experiment_dirs(root: Path) -> dict[str, Path]:
    if not root.exists():
        raise FileNotFoundError(f"Download root not found: {root}")
    return {p.name: p for p in root.iterdir() if p.is_dir()}


def scan_experiment(folder: Path | None) -> dict[str, Any]:
    if folder is None or not folder.exists():
        return {
            "folder_exists": False,
            "folder_path": "",
            "n_total_files": 0,
            "total_size_bytes": 0,
            "n_bold_nifti": 0,
            "n_t1w_nifti": 0,
            "bold_nifti_paths": "",
            "t1w_nifti_paths": "",
            "suspicious_tiny_file_count": 0,
            "suspicious_tiny_files": "",
            "zero_byte_file_count": 0,
            "zero_byte_files": "",
        }
    files = [p for p in folder.rglob("*") if p.is_file()]
    bold = sorted([p for p in files if is_bold_nifti(p)])
    t1w = sorted([p for p in files if is_t1w_nifti(p)])
    tiny: list[str] = []
    zero: list[str] = []
    total_size = 0
    for path in files:
        size = path.stat().st_size
        total_size += size
        rel = str(path.relative_to(folder))
        if size == 0:
            zero.append(rel)
        if is_nifti(path) and size < NIFTI_TINY_BYTES:
            tiny.append(f"{rel} ({size} B)")
        elif path.suffix.lower() == ".json" and size < JSON_TINY_BYTES:
            tiny.append(f"{rel} ({size} B)")
    return {
        "folder_exists": True,
        "folder_path": str(folder),
        "n_total_files": len(files),
        "total_size_bytes": total_size,
        "n_bold_nifti": len(bold),
        "n_t1w_nifti": len(t1w),
        "bold_nifti_paths": "; ".join(str(p.relative_to(folder)) for p in bold),
        "t1w_nifti_paths": "; ".join(str(p.relative_to(folder)) for p in t1w),
        "suspicious_tiny_file_count": len(tiny),
        "suspicious_tiny_files": "; ".join(tiny),
        "zero_byte_file_count": len(zero),
        "zero_byte_files": "; ".join(zero),
    }


def status_and_notes(row: dict[str, Any]) -> tuple[str, str]:
    notes: list[str] = []
    if not row["folder_exists"]:
        return "missing_folder", "experiment folder not downloaded"
    if row["n_bold_nifti"] == 0:
        notes.append("no BOLD NIfTI")
    if row["n_t1w_nifti"] == 0:
        notes.append("no T1w NIfTI")
    if row["suspicious_tiny_file_count"] > 0:
        notes.append("suspicious tiny file(s)")
    if row["zero_byte_file_count"] > 0:
        notes.append("zero-byte file(s)")
    if not notes:
        return "complete_bold_and_t1w", "has at least one BOLD NIfTI and one T1w NIfTI"
    return "incomplete_or_suspicious", "; ".join(notes)


def build_qc(manifest: pd.DataFrame, download_dirs: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    expected_ids = set(manifest["experiment_id"].astype(str))
    downloaded_ids = set(download_dirs)
    all_ids = sorted(expected_ids | downloaded_ids)
    manifest_by_id = manifest.drop_duplicates("experiment_id").set_index("experiment_id", drop=False)
    rows: list[dict[str, Any]] = []
    for experiment_id in all_ids:
        expected = experiment_id in expected_ids
        mrow = manifest_by_id.loc[experiment_id].to_dict() if expected else {}
        scan = scan_experiment(download_dirs.get(experiment_id))
        row: dict[str, Any] = {
            "experiment_id": experiment_id,
            "expected_in_manifest": expected,
            "downloaded_folder_present": experiment_id in downloaded_ids,
            "subject_id": mrow.get("subject_id", ""),
            "diagnosis": mrow.get("diagnosis", ""),
            "session_id": mrow.get("session_id", ""),
            "TR_seconds": mrow.get("TR_seconds", ""),
            "ScannerModel": mrow.get("ScannerModel", ""),
            "Manufacturer": mrow.get("Manufacturer", ""),
            "diagnosis_confidence": mrow.get("diagnosis_confidence", ""),
        }
        row.update(scan)
        row["has_bold_nifti"] = row["n_bold_nifti"] > 0
        row["has_t1w_nifti"] = row["n_t1w_nifti"] > 0
        row["incomplete_folder"] = bool(row["folder_exists"]) and (not row["has_bold_nifti"] or not row["has_t1w_nifti"])
        row["qc_status"], row["qc_notes"] = status_and_notes(row)
        rows.append(row)
    qc = pd.DataFrame(rows)
    missing = qc[qc["expected_in_manifest"] & ~qc["downloaded_folder_present"]].copy()
    counts_cols = [
        "experiment_id",
        "expected_in_manifest",
        "downloaded_folder_present",
        "diagnosis",
        "n_bold_nifti",
        "n_t1w_nifti",
        "has_bold_nifti",
        "has_t1w_nifti",
        "n_total_files",
        "suspicious_tiny_file_count",
        "zero_byte_file_count",
        "qc_status",
        "qc_notes",
    ]
    run_counts = qc[counts_cols].copy()
    return qc, missing, run_counts


def write_readme(outdir: Path, manifest: pd.DataFrame, qc: pd.DataFrame, missing: pd.DataFrame, download_root: Path) -> None:
    expected = int(manifest["experiment_id"].nunique())
    downloaded_expected = int((qc["expected_in_manifest"] & qc["downloaded_folder_present"]).sum())
    downloaded_total = int(qc["downloaded_folder_present"].sum())
    extra = int((~qc["expected_in_manifest"] & qc["downloaded_folder_present"]).sum())
    has_bold = int((qc["expected_in_manifest"] & qc["has_bold_nifti"]).sum())
    has_t1w = int((qc["expected_in_manifest"] & qc["has_t1w_nifti"]).sum())
    complete = int((qc["expected_in_manifest"] & qc["has_bold_nifti"] & qc["has_t1w_nifti"]).sum())
    suspicious = int((qc["expected_in_manifest"] & (qc["suspicious_tiny_file_count"] > 0)).sum())
    zero = int((qc["expected_in_manifest"] & (qc["zero_byte_file_count"] > 0)).sum())
    missing_count = int(len(missing))
    dx_counts = manifest["diagnosis"].value_counts(dropna=False).to_dict()
    lines = [
        "# OASIS3 Pilot 30CN/30AD Download QC",
        "",
        "Read-only QC of the downloaded BOLD/T1w pilot folder.",
        "",
        "| Item | Count |",
        "|---|---:|",
        f"| Expected unique experiment_id values | {expected} |",
        f"| Downloaded expected experiment folders | {downloaded_expected} |",
        f"| Downloaded folders total | {downloaded_total} |",
        f"| Extra downloaded folders not in manifest | {extra} |",
        f"| Missing expected experiment folders | {missing_count} |",
        f"| Expected experiments with at least one BOLD NIfTI | {has_bold} |",
        f"| Expected experiments with at least one T1w NIfTI | {has_t1w} |",
        f"| Expected experiments complete for BOLD+T1w | {complete} |",
        f"| Expected experiments with suspicious tiny files | {suspicious} |",
        f"| Expected experiments with zero-byte files | {zero} |",
        "",
        f"Download root: `{download_root}`",
        "",
        "Manifest diagnosis counts:",
        "",
    ]
    for key, value in dx_counts.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "QC criteria:",
            "",
            f"- BOLD NIfTI: `.nii`/`.nii.gz` file with `bold` in filename or inside a `func*` folder.",
            f"- T1w NIfTI: `.nii`/`.nii.gz` file with `T1w` in filename or inside an `anat*` folder.",
            f"- Suspicious tiny NIfTI threshold: `< {NIFTI_TINY_BYTES}` bytes.",
            f"- Suspicious tiny JSON threshold: `< {JSON_TINY_BYTES}` bytes.",
            "",
            "Generated files:",
            "",
            "- `download_qc.csv/.md`",
            "- `missing_experiments.csv/.md`",
            "- `run_counts_by_experiment.csv/.md`",
            "- `command_log.json`",
            "",
            "Safety: no downloaded data were modified.",
        ]
    )
    (outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    manifest_path = resolve(args.manifest)
    download_root = args.download_root
    outdir = resolve(args.output_dir)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not download_root.exists():
        raise FileNotFoundError(download_root)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path)
    if "experiment_id" not in manifest.columns:
        raise RuntimeError(f"Manifest is missing experiment_id column: {manifest_path}")
    manifest["experiment_id"] = manifest["experiment_id"].astype(str)
    download_dirs = immediate_experiment_dirs(download_root)
    qc, missing, run_counts = build_qc(manifest, download_dirs)
    qc.to_csv(outdir / "download_qc.csv", index=False)
    missing.to_csv(outdir / "missing_experiments.csv", index=False)
    run_counts.to_csv(outdir / "run_counts_by_experiment.csv", index=False)
    md_table(qc, outdir / "download_qc.md")
    md_table(missing, outdir / "missing_experiments.md")
    md_table(run_counts, outdir / "run_counts_by_experiment.md")
    write_readme(outdir, manifest, qc, missing, download_root)
    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "manifest": str(manifest_path),
        "download_root": str(download_root),
        "output_dir": str(outdir),
        "expected_unique_experiment_ids": int(manifest["experiment_id"].nunique()),
        "downloaded_experiment_folders": len(download_dirs),
        "missing_experiment_ids": missing["experiment_id"].astype(str).tolist(),
        "tiny_nifti_threshold_bytes": NIFTI_TINY_BYTES,
        "tiny_json_threshold_bytes": JSON_TINY_BYTES,
        "training_launched": False,
        "downloaded_data_modified": False,
    }
    (outdir / "command_log.json").write_text(json.dumps(command_log, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote OASIS3 pilot download QC to {outdir}")
    print(
        qc[qc["expected_in_manifest"]][
            ["experiment_id", "downloaded_folder_present", "n_bold_nifti", "n_t1w_nifti", "qc_status"]
        ]
        .head(20)
        .to_string(index=False)
    )
    print(f"Expected={manifest['experiment_id'].nunique()} downloaded_expected={(qc['expected_in_manifest'] & qc['downloaded_folder_present']).sum()} missing={len(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
