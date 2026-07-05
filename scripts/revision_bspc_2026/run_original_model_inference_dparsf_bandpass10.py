#!/usr/bin/env python3
"""Inventory and guarded original-model inference for DPARSF-bandpass ROI signals.

This wrapper is intentionally conservative:
- no retraining;
- no V4 checkpoints;
- no raw/NIfTI copying;
- dry-run performs inventory/metadata validation and prints the command only;
- real inference is blocked unless subject inventory and metadata checks pass,
  or the operator explicitly opts into the documented exceptions.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "data/OneDrive_1_2-5-2026"
DEFAULT_TRAINING_DIR = PROJECT_ROOT / "results/vae_3channels_beta25"
DEFAULT_OUTPUT_SYMLINK = PROJECT_ROOT / "results/revision_bspc_2026/original_model_inference_dparsf_bandpass10"
DEFAULT_BIG_DISK_TARGET = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026/original_model_inference_dparsf_bandpass10"
)
DEFAULT_EXTERNAL_LOG_DIR = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026/run_logs")
DEFAULT_AAL3_ROI_METADATA_PATH = PROJECT_ROOT / "data/ROI_MNI_V7_vol.txt"
EXPECTED_FINAL_ROIS = 131
SUBJECT_RE = re.compile(r"[0-9]{3}_S_[0-9]{4}")

CHANNEL_NAMES_MASTER = [
    "Pearson_OMST_GCE_Signed_Weighted",
    "Pearson_Full_FisherZ_Signed",
    "MI_KNN_Symmetric",
    "dFC_AbsDiffMean",
    "dFC_StdDev",
    "DistanceCorr",
    "Granger_F_lag1",
]
SELECTED_CHANNELS = [1, 0, 2]
SELECTED_CHANNEL_NAMES = [
    CHANNEL_NAMES_MASTER[idx] for idx in SELECTED_CHANNELS
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Guarded original-paper-model inference for DPARSF-bandpass ADNI controls.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--training-output-dir", type=Path, default=DEFAULT_TRAINING_DIR)
    parser.add_argument("--output-symlink", type=Path, default=DEFAULT_OUTPUT_SYMLINK)
    parser.add_argument("--big-disk-target", type=Path, default=DEFAULT_BIG_DISK_TARGET)
    parser.add_argument("--external-log-dir", type=Path, default=DEFAULT_EXTERNAL_LOG_DIR)
    parser.add_argument("--aal3-roi-metadata-path", type=Path, default=DEFAULT_AAL3_ROI_METADATA_PATH)
    parser.add_argument("--expected-n", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-one-subject", action="store_true")
    parser.add_argument("--allow-non10", action="store_true")
    parser.add_argument("--allow-missing-metadata", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--classifier-types", nargs="+", default=["logreg", "svm"])
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--ensemble-method", choices=["mean", "median"], default="mean")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def run_capture(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
        text = (completed.stdout or "") + (completed.stderr or "")
        return text.strip()
    except Exception as exc:
        return f"ERROR running {shlex.join(command)}: {exc}"


def disk_status_text(input_root: Path, big_root: Path) -> str:
    lines = [
        "$ df -hT -x squashfs -x tmpfs -x devtmpfs",
        run_capture(["df", "-hT", "-x", "squashfs", "-x", "tmpfs", "-x", "devtmpfs"]),
        "",
        f"$ du -sh {input_root}",
        run_capture(["du", "-sh", str(input_root)]),
        "",
        f"$ du -sh {big_root}",
        run_capture(["du", "-sh", str(big_root)]),
    ]
    try:
        usage = shutil.disk_usage(big_root if big_root.exists() else big_root.parent)
        lines.extend(
            [
                "",
                "Big-disk target filesystem:",
                f"total={usage.total} used={usage.used} free={usage.free}",
            ]
        )
    except Exception:
        pass
    return "\n".join(lines) + "\n"


def ensure_output_layout(output_symlink: Path, big_target: Path, dry_run: bool) -> Tuple[Path, Path, Path]:
    output_symlink = resolve_path(output_symlink)
    big_target = big_target.resolve() if big_target.exists() else big_target
    big_target.mkdir(parents=True, exist_ok=True)
    audit_dir = big_target / "audit"
    tables_dir = big_target / "Tables"
    features_dir = big_target / "features_or_tensor"
    for path in [audit_dir, tables_dir, features_dir]:
        path.mkdir(parents=True, exist_ok=True)

    if output_symlink.exists() or output_symlink.is_symlink():
        if not output_symlink.is_symlink():
            raise RuntimeError(f"Output path exists but is not a symlink: {output_symlink}")
        if output_symlink.resolve() != big_target.resolve():
            raise RuntimeError(
                f"Output symlink target mismatch: {output_symlink.resolve()} != {big_target.resolve()}"
            )
    elif not dry_run:
        raise RuntimeError(
            "Refusing real run because local output symlink is missing. Prepare it first:\n"
            f"  ln -s {shlex.quote(str(big_target))} {shlex.quote(str(output_symlink))}"
        )
    return audit_dir, tables_dir, features_dir


def cleanup_generated_outputs_for_overwrite(big_target: Path) -> None:
    """Remove only generated payload dirs for this guarded run."""
    for child_name in ["features_or_tensor", "Tables"]:
        child = big_target / child_name
        if child.exists():
            shutil.rmtree(child)
        child.mkdir(parents=True, exist_ok=True)


def output_has_existing_outputs(big_target: Path) -> bool:
    if not big_target.exists():
        return False
    for child in big_target.iterdir():
        if child.name.startswith("."):
            continue
        return True
    return False


def read_txt_shape(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "txt_shape": "",
        "txt_rows": np.nan,
        "txt_cols": np.nan,
        "txt_read_status": "not_read",
        "txt_min": np.nan,
        "txt_max": np.nan,
    }
    try:
        data = np.loadtxt(path, delimiter=",")
        out.update(
            {
                "txt_shape": str(tuple(data.shape)),
                "txt_rows": int(data.shape[0]) if data.ndim >= 1 else 1,
                "txt_cols": int(data.shape[1]) if data.ndim >= 2 else 1,
                "txt_read_status": "ok_comma_delimited",
                "txt_min": float(np.nanmin(data)),
                "txt_max": float(np.nanmax(data)),
            }
        )
    except Exception as comma_exc:
        try:
            data = np.loadtxt(path)
            out.update(
                {
                    "txt_shape": str(tuple(data.shape)),
                    "txt_rows": int(data.shape[0]) if data.ndim >= 1 else 1,
                    "txt_cols": int(data.shape[1]) if data.ndim >= 2 else 1,
                    "txt_read_status": "ok_whitespace_delimited",
                    "txt_min": float(np.nanmin(data)),
                    "txt_max": float(np.nanmax(data)),
                }
            )
        except Exception as ws_exc:
            out["txt_read_status"] = f"failed: comma={comma_exc}; whitespace={ws_exc}"
    return out


def write_file_inventory(input_root: Path, audit_dir: Path) -> None:
    files = sorted(path for path in input_root.rglob("*") if path.is_file())
    rows = []
    for path in files:
        rows.append(
            f"{path.relative_to(input_root)}\t{path.stat().st_size}\t{datetime.fromtimestamp(path.stat().st_mtime).isoformat()}"
        )
    (audit_dir / "file_inventory.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")

    tree_lines = []
    for idx, path in enumerate(sorted(input_root.rglob("*"))):
        if idx >= 250:
            tree_lines.append("... truncated at 250 entries ...")
            break
        rel = path.relative_to(input_root)
        marker = "/" if path.is_dir() else ""
        size = "" if path.is_dir() else f" ({path.stat().st_size} bytes)"
        tree_lines.append(f"{rel}{marker}{size}")
    (audit_dir / "directory_tree_head.txt").write_text("\n".join(tree_lines) + "\n", encoding="utf-8")


def inventory_input(input_root: Path, audit_dir: Path, expected_n: int) -> pd.DataFrame:
    write_file_inventory(input_root, audit_dir)
    file_rows: Dict[str, Dict[str, Any]] = {}
    for path in sorted(input_root.rglob("*")):
        if not path.is_file():
            continue
        match = SUBJECT_RE.search(path.name)
        if not match:
            continue
        sid = match.group(0)
        row = file_rows.setdefault(
            sid,
            {
                "SubjectID": sid,
                "txt_count": 0,
                "mat_count": 0,
                "png_count": 0,
                "gif_count": 0,
                "other_count": 0,
                "txt_paths": [],
                "mat_paths": [],
                "other_paths": [],
                "min_file_size_bytes": np.nan,
                "suspicious_small_file_count": 0,
            },
        )
        suffix = path.suffix.lower()
        size = path.stat().st_size
        row["min_file_size_bytes"] = size if pd.isna(row["min_file_size_bytes"]) else min(row["min_file_size_bytes"], size)
        if size < 1024:
            row["suspicious_small_file_count"] += 1
        if suffix == ".txt" and path.name.startswith("ROISignals_"):
            row["txt_count"] += 1
            row["txt_paths"].append(str(path))
        elif suffix == ".mat" and path.name.startswith("ROISignals_"):
            row["mat_count"] += 1
            row["mat_paths"].append(str(path))
        elif suffix == ".png":
            row["png_count"] += 1
            row["other_paths"].append(str(path))
        elif suffix == ".gif":
            row["gif_count"] += 1
            row["other_paths"].append(str(path))
        else:
            row["other_count"] += 1
            row["other_paths"].append(str(path))

    rows = []
    for sid, row in sorted(file_rows.items()):
        txt_path = Path(row["txt_paths"][0]) if row["txt_paths"] else None
        shape_info = read_txt_shape(txt_path) if txt_path else {}
        issue_parts = []
        if row["txt_count"] != 1:
            issue_parts.append(f"txt_count={row['txt_count']}")
        if row["mat_count"] != 1:
            issue_parts.append(f"mat_count={row['mat_count']}")
        if row["suspicious_small_file_count"]:
            issue_parts.append("suspicious_small_files")
        if shape_info.get("txt_cols") not in [131, 166, 170]:
            issue_parts.append(f"unexpected_txt_cols={shape_info.get('txt_cols')}")
        row_out = {
            "SubjectID": sid,
            "txt_count": row["txt_count"],
            "mat_count": row["mat_count"],
            "png_count": row["png_count"],
            "gif_count": row["gif_count"],
            "has_txt": row["txt_count"] > 0,
            "has_mat": row["mat_count"] > 0,
            "has_both_txt_and_mat": row["txt_count"] > 0 and row["mat_count"] > 0,
            "txt_path": row["txt_paths"][0] if row["txt_paths"] else "",
            "mat_path": row["mat_paths"][0] if row["mat_paths"] else "",
            "all_txt_paths": "|".join(row["txt_paths"]),
            "all_mat_paths": "|".join(row["mat_paths"]),
            "min_file_size_bytes": row["min_file_size_bytes"],
            "suspicious_small_file_count": row["suspicious_small_file_count"],
            "issue": ";".join(issue_parts) if issue_parts else "OK",
        }
        row_out.update(shape_info)
        rows.append(row_out)
    detected = pd.DataFrame(rows)
    detected.to_csv(audit_dir / "detected_subjects.csv", index=False)

    issue_df = detected[detected["issue"] != "OK"].copy() if not detected.empty else pd.DataFrame()
    if len(detected) != expected_n:
        expected_row = {
            "SubjectID": "__EXPECTED_N_MISMATCH__",
            "issue": f"expected_n={expected_n};detected_n={len(detected)}",
        }
        issue_df = pd.concat([issue_df, pd.DataFrame([expected_row])], ignore_index=True)
    issue_df.to_csv(audit_dir / "missing_or_duplicate_subjects.csv", index=False)
    return detected


def parse_metadata_path_value(value: Any) -> Optional[Path]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def recover_original_model_metadata_paths(training_dir: Path) -> List[Path]:
    paths: List[Path] = []
    run_config = training_dir / "run_config.json"
    if run_config.exists():
        try:
            cfg = json.loads(run_config.read_text(encoding="utf-8"))
            for value in [cfg.get("metadata_path"), cfg.get("args", {}).get("metadata_path")]:
                path = parse_metadata_path_value(value)
                if path is not None:
                    paths.append(path)
        except Exception:
            pass
    for text_file in sorted(training_dir.glob("*.txt")) + sorted((training_dir / "Logs").glob("*.log")):
        try:
            text = text_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for match in re.finditer(r"['\"]metadata_path['\"]\s*:\s*['\"]([^'\"]+)['\"]", text):
            path = parse_metadata_path_value(match.group(1))
            if path is not None:
                paths.append(path)
    return unique_paths(paths)


def csv_header(path: Path) -> List[str]:
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return []


def looks_like_metadata_csv(path: Path) -> bool:
    cols = csv_header(path)
    if not cols:
        return False
    return normalize_subject_id_column_from_columns(cols) is not None and "Age" in cols and "Sex" in cols


def discover_metadata_csvs_under_project() -> List[Path]:
    paths: List[Path] = []
    for base in [PROJECT_ROOT / "data", PROJECT_ROOT / "results"]:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.csv")):
            if is_current_output_path(path):
                continue
            if looks_like_metadata_csv(path):
                paths.append(path)
    return paths


def unique_paths(paths: Iterable[Path]) -> List[Path]:
    unique = []
    seen = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def path_is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except (OSError, ValueError):
        return False


def is_current_output_path(path: Path) -> bool:
    return path_is_under(path, DEFAULT_OUTPUT_SYMLINK) or path_is_under(path, DEFAULT_BIG_DISK_TARGET)


def candidate_metadata_paths(training_dir: Path) -> List[Path]:
    paths = []
    paths.extend(recover_original_model_metadata_paths(training_dir))
    paths.extend(
        [
            PROJECT_ROOT / "data/SubjectsData_AAL3_procesado2.csv",
            PROJECT_ROOT / "data/revision_bspc_2026/adni_expanded_v4_all_available/subject_metadata_adni_expanded_v4_all_available.csv",
            PROJECT_ROOT / "data/revision_bspc_2026/adni_expanded_v3_all_available/subject_metadata_adni_expanded_v3_all_available.csv",
        ]
    )
    paths.extend(discover_metadata_csvs_under_project())
    return unique_paths(paths)


def normalize_subject_id_column_from_columns(columns: Sequence[str]) -> Optional[str]:
    for col in ["SubjectID", "Subject ID", "Subject", "subject_id", "PTID"]:
        if col in columns:
            return col
    return None


def normalize_subject_id_column(df: pd.DataFrame) -> Optional[str]:
    return normalize_subject_id_column_from_columns(list(df.columns))


def first_present(row: Mapping[str, Any], names: Sequence[str], default: Any = np.nan) -> Any:
    for name in names:
        if name in row and not pd.isna(row[name]) and str(row[name]).strip() != "":
            return row[name]
    return default


def parse_manufacturer_from_protocol(value: Any) -> Any:
    if pd.isna(value):
        return np.nan
    match = re.search(r"Manufacturer=([^;]+)", str(value))
    return match.group(1).strip() if match else np.nan


def normalize_metadata_hit(hit: Mapping[str, Any], sid: str) -> Dict[str, Any]:
    research = first_present(
        hit,
        ["ResearchGroup_Mapped", "ResearchGroup", "Research Group", "Group", "DX", "Diagnosis"],
        "CN",
    )
    manufacturer = first_present(hit, ["Manufacturer"], np.nan)
    if pd.isna(manufacturer):
        manufacturer = parse_manufacturer_from_protocol(first_present(hit, ["Imaging Protocol"], np.nan))
    if pd.isna(manufacturer) or str(manufacturer).strip() == "":
        manufacturer = "Unknown"
    site3 = first_present(hit, ["Site3"], np.nan)
    if pd.isna(site3):
        site3 = sid.split("_")[0]
    image_id = first_present(hit, ["ImageID", "Image ID", "Image Data ID"], np.nan)
    if not pd.isna(image_id):
        image_id = str(image_id).strip().lstrip("I")
    return {
        "ResearchGroup_Mapped": research,
        "ResearchGroup": first_present(hit, ["ResearchGroup", "Research Group", "Group"], research),
        "Manufacturer": manufacturer,
        "Site3": site3,
        "Age": first_present(hit, ["Age"], np.nan),
        "Sex": first_present(hit, ["Sex"], np.nan),
        "ImageID": image_id,
    }


def metadata_quality(hit: Mapping[str, Any], sid: str) -> int:
    normalized = normalize_metadata_hit(hit, sid)
    score = 0
    if not pd.isna(normalized["Age"]):
        score += 1
    if not pd.isna(normalized["Sex"]) and str(normalized["Sex"]).strip():
        score += 1
    if normalized["Manufacturer"] != "Unknown":
        score += 2
    if not pd.isna(normalized["ResearchGroup_Mapped"]) and str(normalized["ResearchGroup_Mapped"]).strip():
        score += 1
    if not pd.isna(normalized["Site3"]) and str(normalized["Site3"]).strip():
        score += 1
    return score


def build_metadata(
    subjects: pd.DataFrame,
    audit_dir: Path,
    input_root: Path,
    training_dir: Path,
) -> Tuple[pd.DataFrame, List[str]]:
    subject_ids = subjects["SubjectID"].astype(str).tolist()
    hits: Dict[str, Dict[str, Any]] = {}
    raw_hits = []
    search_paths = candidate_metadata_paths(training_dir)
    pd.DataFrame({"metadata_search_order": [str(path) for path in search_paths]}).to_csv(
        audit_dir / "metadata_search_order.csv", index=False
    )
    for path in search_paths:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        sid_col = normalize_subject_id_column(df)
        if sid_col is None:
            continue
        df[sid_col] = df[sid_col].astype(str).str.strip()
        sub = df[df[sid_col].isin(subject_ids)].copy()
        if sub.empty:
            continue
        for _, row in sub.iterrows():
            sid = str(row[sid_col])
            candidate = {**row.to_dict(), "metadata_source_file": str(path)}
            raw_hits.append({"SubjectID": sid, **candidate})
            if sid not in hits or metadata_quality(candidate, sid) > metadata_quality(hits[sid], sid):
                hits[sid] = candidate

    rows = []
    missing = []
    by_subject = subjects.set_index("SubjectID").to_dict(orient="index")
    for sid in subject_ids:
        hit = hits.get(sid, {})
        if not hit:
            missing.append(sid)
        normalized = normalize_metadata_hit(hit, sid) if hit else normalize_metadata_hit({}, sid)
        rows.append(
            {
                "SubjectID": sid,
                "found_metadata": "yes" if hit else "no",
                "metadata_source_file": hit.get("metadata_source_file", "MISSING"),
                "ResearchGroup_Mapped": normalized["ResearchGroup_Mapped"],
                "ResearchGroup": normalized["ResearchGroup"],
                "Manufacturer": normalized["Manufacturer"],
                "Site3": normalized["Site3"],
                "Age": normalized["Age"],
                "Sex": normalized["Sex"],
                "ImageID": normalized["ImageID"],
                "source_preprocessing": "DPARSF_original_bandpass",
                "input_root": str(input_root),
                "roi_signal_txt_path": by_subject.get(sid, {}).get("txt_path", ""),
                "roi_signal_mat_path": by_subject.get(sid, {}).get("mat_path", ""),
                "metadata_found": bool(hit),
            }
        )
    meta = pd.DataFrame(rows)
    meta.to_csv(audit_dir / "dparsf_bandpass10_metadata_for_inference.csv", index=False)
    meta[
        [
            "SubjectID",
            "found_metadata",
            "metadata_source_file",
            "ResearchGroup_Mapped",
            "Age",
            "Sex",
            "Manufacturer",
            "Site3",
        ]
    ].to_csv(audit_dir / "subject_metadata_recovery_report.csv", index=False)
    pd.DataFrame(raw_hits).to_csv(audit_dir / "metadata_raw_hits.csv", index=False)
    pd.DataFrame({"SubjectID": missing, "issue": "metadata_missing"}).to_csv(
        audit_dir / "missing_metadata_subjects.csv", index=False
    )
    return meta, missing


def load_run_config(training_dir: Path) -> Dict[str, Any]:
    path = training_dir / "run_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Original model run_config.json missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_original_model_config(run_config: Mapping[str, Any]) -> None:
    master = run_config.get("channel_names_master_in_tensor_order") or run_config.get("DEFAULT_CHANNEL_NAMES")
    channels = run_config.get("channels_to_use_indices") or run_config.get("args", {}).get("channels_to_use")
    if list(master) != CHANNEL_NAMES_MASTER:
        raise RuntimeError(f"Unexpected original model channel order: {master}")
    if list(channels) != SELECTED_CHANNELS:
        raise RuntimeError(f"Unexpected original model selected channels: {channels}")


def write_validation_report(
    audit_dir: Path,
    subjects: pd.DataFrame,
    expected_n: int,
    missing_metadata: Sequence[str],
    run_config: Mapping[str, Any],
    input_root: Path,
    output_symlink: Path,
    big_target: Path,
    disk_status: str,
) -> None:
    n_detected = len(subjects)
    blockers = []
    if n_detected != expected_n:
        blockers.append(f"Expected {expected_n} subjects but detected {n_detected}.")
    if missing_metadata:
        blockers.append(f"Missing metadata for {len(missing_metadata)} subjects: {', '.join(missing_metadata)}.")
    bad_subjects = subjects[subjects["issue"] != "OK"] if not subjects.empty else subjects
    if not bad_subjects.empty:
        blockers.append(f"Inventory issues in {len(bad_subjects)} rows; see missing_or_duplicate_subjects.csv.")
    args_cfg = run_config.get("args", {})
    lines = [
        "# DPARSF Bandpass10 Input Validation",
        "",
        f"Input root: `{input_root}`",
        f"Detected subjects: `{n_detected}`",
        f"Expected subjects: `{expected_n}`",
        f"Blocking inventory issue: `{'YES' if n_detected != expected_n else 'NO'}`",
        "Inventory explanation: recursive scan found the subject IDs below in ROI signal, check-image, and atlas-preview files. No additional filename matching `[0-9]{3}_S_[0-9]{4}` was found under the input root.",
        f"Detected SubjectIDs: `{', '.join(subjects['SubjectID'].astype(str).tolist()) if not subjects.empty else 'none'}`",
        "",
        "## Original Model",
        f"Training output dir: `{DEFAULT_TRAINING_DIR}`",
        f"Recovered original metadata path: `{run_config.get('metadata_path', args_cfg.get('metadata_path', 'not_found'))}`",
        f"Selected channel indices: `{args_cfg.get('channels_to_use', SELECTED_CHANNELS)}`",
        f"Selected channel names: `{args_cfg.get('selected_channel_names', SELECTED_CHANNEL_NAMES)}`",
        "Python bandpass during tensor adapter: `SKIPPED`.",
        "Reason: input ROI signals are already DPARSF-bandpass filtered; applying Python bandpass would double-filter.",
        "TR assumption: `3.0 s`; target homogenized length: `140 TRs`.",
        "",
        "## Output",
        f"Local symlink: `{output_symlink}`",
        f"Big-disk target: `{big_target}`",
        f"Symlink valid: `{output_symlink.is_symlink() and output_symlink.resolve() == big_target.resolve() if output_symlink.exists() or output_symlink.is_symlink() else False}`",
        "",
        "## Metadata Recovery",
        "Per-subject metadata recovery is written to `subject_metadata_recovery_report.csv`. Metadata source search order is written to `metadata_search_order.csv`.",
        "",
        "## Previous Prediction Search",
        "Previous prediction matches are written to `previous_predictions_matched.csv`.",
        "",
        "## Blockers",
    ]
    if blockers:
        lines.extend(f"- {item}" for item in blockers)
    else:
        lines.append("- None.")
    lines.extend(["", "## Disk Status", "```", disk_status.strip(), "```", ""])
    (audit_dir / "input_validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def preprocessing_blocked(
    subjects: pd.DataFrame,
    expected_n: int,
    missing_metadata: Sequence[str],
    allow_non10: bool,
    allow_missing_metadata: bool,
) -> List[str]:
    blockers = []
    if len(subjects) != expected_n and not allow_non10:
        blockers.append(f"detected_n={len(subjects)} expected_n={expected_n}; pass --allow-non10 only after documenting this.")
    if missing_metadata and not allow_missing_metadata:
        blockers.append(
            f"metadata missing for {len(missing_metadata)} subjects; pass --allow-missing-metadata only if imputation is acceptable."
        )
    if not subjects.empty and (subjects["issue"] != "OK").any():
        blockers.append("one or more subjects have inventory issues in missing_or_duplicate_subjects.csv")
    return blockers


def configure_aal3_roi_processing(aal3_metadata_path: Path) -> Any:
    """Force feature_extraction_manual to use this repo's AAL3 ROI metadata."""
    from scripts import feature_extraction_manual as fem

    aal3_metadata_path = resolve_path(aal3_metadata_path)
    if not aal3_metadata_path.exists():
        raise FileNotFoundError(
            "AAL3 ROI metadata is required for 170->131 ROI reduction but was not found: "
            f"{aal3_metadata_path}"
        )

    manual_order_path = aal3_metadata_path.parent / "aal3_131_manual_network_order.csv"
    if not manual_order_path.exists():
        raise FileNotFoundError(
            "AAL3 131 ROI manual Yeo17 order file is required for historical ROI reordering but was not found: "
            f"{manual_order_path}"
        )

    fem.PROJECT_ROOT = PROJECT_ROOT
    fem.BASE_PATH_AAL3 = aal3_metadata_path.parent
    fem.AAL3_META_PATH = aal3_metadata_path
    fem.AAL3_MANUAL_NETWORK_MAPPING_CSV = manual_order_path
    fem._initialize_aal3_roi_processing_info()

    mapping = fem.AAL3_ROI_ORDER_MAPPING or {}
    if fem.FINAL_N_ROIS_EXPECTED != EXPECTED_FINAL_ROIS:
        raise RuntimeError(
            f"AAL3 ROI reduction did not initialize to {EXPECTED_FINAL_ROIS} ROIs; "
            f"got {fem.FINAL_N_ROIS_EXPECTED}. Metadata path: {aal3_metadata_path}"
        )
    if fem.AAL3_MISSING_INDICES_0BASED is None or fem.INDICES_OF_SMALL_ROIS_TO_DROP_FROM_166 is None:
        raise RuntimeError("AAL3 ROI reduction indices are not initialized; refusing to proceed with 170 ROIs.")
    if not mapping.get("new_order_indices") or len(mapping["new_order_indices"]) != EXPECTED_FINAL_ROIS:
        raise RuntimeError(
            "AAL3 ROI reordering is inactive or invalid; original model expects the historical "
            "131-ROI Yeo17-reordered tensor order."
        )
    return fem


def read_roi_signal_txt(path: Path) -> np.ndarray:
    try:
        return np.loadtxt(path, delimiter=",")
    except Exception:
        return np.loadtxt(path)


def reduce_and_reorder_rois(raw_170: np.ndarray, subject_id: str, fem: Any) -> np.ndarray:
    reduced = fem._orient_and_reduce_rois(
        raw_170,
        subject_id,
        fem.RAW_DATA_EXPECTED_COLUMNS,
        fem.AAL3_MISSING_INDICES_0BASED,
        fem.INDICES_OF_SMALL_ROIS_TO_DROP_FROM_166,
        fem.FINAL_N_ROIS_EXPECTED,
    )
    if reduced is None:
        raise RuntimeError(f"{subject_id}: ROI orientation/reduction failed")
    if reduced.shape[1] != EXPECTED_FINAL_ROIS:
        raise RuntimeError(f"{subject_id}: expected {EXPECTED_FINAL_ROIS} reduced ROIs, got {reduced.shape[1]}")

    mapping = fem.AAL3_ROI_ORDER_MAPPING or {}
    reduced = fem._reorder_rois_by_network_for_timeseries(
        reduced, mapping["new_order_indices"], subject_id
    )
    if reduced.shape[1] != EXPECTED_FINAL_ROIS:
        raise RuntimeError(f"{subject_id}: expected {EXPECTED_FINAL_ROIS} final ROIs after reordering, got {reduced.shape[1]}")
    return reduced


def write_roi_failure_readme(audit_dir: Path, aal3_metadata_path: Path, validation: pd.DataFrame, error: str = "") -> None:
    lines = [
        "# ROI Reduction Validation Failure",
        "",
        "The wrapper refused to continue because the DPARSF ROI signals could not be validated as the historical AAL3 131-ROI tensor format.",
        "",
        f"AAL3 ROI metadata path: `{aal3_metadata_path}`",
        f"Expected final ROIs: `{EXPECTED_FINAL_ROIS}`",
        "",
        "The original paper model expects 7-channel tensors with matrices shaped `(131, 131)` in the historical Yeo17-reordered AAL3 order.",
        "The wrapper must not silently proceed with 170 ROI signals.",
    ]
    if error:
        lines.extend(["", "## Error", f"`{error}`"])
    if not validation.empty:
        bad = validation[validation["status"] != "OK"]
        lines.extend(["", "## Failed Rows"])
        if bad.empty:
            lines.append("- No failed rows in CSV, but validation did not complete cleanly.")
        else:
            for _, row in bad.iterrows():
                lines.append(
                    f"- {row.get('SubjectID', '')}: input={row.get('input_n_rois', '')}, "
                    f"output={row.get('output_n_rois', '')}, status={row.get('status', '')}"
                )
    (audit_dir / "failure_roi_reduction_readme.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_roi_reduction(
    subjects: pd.DataFrame,
    audit_dir: Path,
    aal3_metadata_path: Path,
    fem: Any,
    smoke_one_subject: bool = False,
) -> pd.DataFrame:
    validation_subjects = subjects.sort_values("SubjectID").head(1) if smoke_one_subject else subjects.sort_values("SubjectID")
    rows = []
    for _, row in validation_subjects.iterrows():
        sid = str(row["SubjectID"])
        input_n_rois: Any = np.nan
        output_n_rois: Any = np.nan
        status = "OK"
        try:
            raw = read_roi_signal_txt(Path(row["txt_path"]))
            input_n_rois = int(raw.shape[1]) if raw.ndim == 2 and raw.shape[1] in [131, 166, 170] else int(raw.shape[0])
            reduced = reduce_and_reorder_rois(raw, sid, fem)
            output_n_rois = int(reduced.shape[1])
            if output_n_rois != EXPECTED_FINAL_ROIS:
                status = f"FAIL: expected_output_rois={EXPECTED_FINAL_ROIS};got={output_n_rois}"
        except Exception as exc:
            status = f"FAIL: {exc}"
        rows.append(
            {
                "SubjectID": sid,
                "input_n_rois": input_n_rois,
                "output_n_rois": output_n_rois,
                "expected_n_rois": EXPECTED_FINAL_ROIS,
                "aal3_roi_metadata_path": str(aal3_metadata_path),
                "roi_order_mapping_path": str(aal3_metadata_path.parent / "aal3_131_manual_network_order.csv"),
                "roi_reordering_active": bool((fem.AAL3_ROI_ORDER_MAPPING or {}).get("new_order_indices")),
                "status": status,
            }
        )
    validation = pd.DataFrame(rows)
    validation.to_csv(audit_dir / "roi_reduction_validation.csv", index=False)
    if validation.empty or (validation["status"] != "OK").any() or (validation["output_n_rois"] != EXPECTED_FINAL_ROIS).any():
        write_roi_failure_readme(audit_dir, aal3_metadata_path, validation)
        raise RuntimeError(
            "ROI reduction validation failed; see audit/roi_reduction_validation.csv and "
            "audit/failure_roi_reduction_readme.md"
        )
    return validation


def already_filtered_preprocess(raw_170: np.ndarray, subject_id: str, fem: Any) -> np.ndarray:
    from scipy.interpolate import interp1d
    from sklearn.preprocessing import StandardScaler

    reduced = reduce_and_reorder_rois(raw_170, subject_id, fem)
    scaled = StandardScaler().fit_transform(np.nan_to_num(reduced, nan=0.0, posinf=0.0, neginf=0.0))
    target_len = int(fem.TARGET_LEN_TS)
    if scaled.shape[0] > target_len:
        scaled = scaled[:target_len, :]
    elif scaled.shape[0] < target_len:
        old = np.linspace(0, 1, scaled.shape[0])
        new = np.linspace(0, 1, target_len)
        out = np.zeros((target_len, scaled.shape[1]), dtype=np.float32)
        for idx in range(scaled.shape[1]):
            out[:, idx] = interp1d(old, scaled[:, idx], kind="linear", fill_value="extrapolate")(new)
        scaled = out
    return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def normalize_subject_tensor(matrices: Mapping[str, np.ndarray], fem: Any) -> np.ndarray:
    from sklearn.preprocessing import RobustScaler

    stacked = []
    for channel_name in fem.CONNECTIVITY_CHANNEL_NAMES:
        matrix = matrices.get(channel_name)
        if matrix is None:
            raise RuntimeError(f"Missing connectivity matrix for channel {channel_name}")
        if matrix.shape != (131, 131):
            raise RuntimeError(f"Channel {channel_name} has shape {matrix.shape}; expected (131, 131)")
        scaled = np.zeros_like(matrix, dtype=np.float32)
        off_diag = ~np.eye(matrix.shape[0], dtype=bool)
        vals = matrix[off_diag]
        if vals.size > 0 and np.nanstd(vals) > 1e-12:
            scaled[off_diag] = RobustScaler().fit_transform(vals.reshape(-1, 1)).ravel()
        else:
            scaled = matrix.astype(np.float32)
        np.fill_diagonal(scaled, 0.0)
        stacked.append(scaled)
    return np.stack(stacked, axis=0).astype(np.float32)


def generate_tensor_from_roi_txt(
    subjects: pd.DataFrame,
    features_dir: Path,
    overwrite: bool,
    run_config: Mapping[str, Any],
    fem: Any,
    tensor_filename: str = "GLOBAL_TENSOR_DPARSF_original_bandpass10_AAL3_131ROIs.npz",
) -> Path:
    if list(fem.CONNECTIVITY_CHANNEL_NAMES) != CHANNEL_NAMES_MASTER:
        raise RuntimeError(f"Feature extractor channel order mismatch: {fem.CONNECTIVITY_CHANNEL_NAMES}")
    tensor_path = features_dir / tensor_filename
    if tensor_path.exists() and not overwrite:
        return tensor_path
    individual_dir = features_dir / "individual_subject_tensors"
    individual_dir.mkdir(parents=True, exist_ok=True)
    tensors = []
    subject_ids = []
    qc_rows = []
    for _, row in subjects.sort_values("SubjectID").iterrows():
        sid = str(row["SubjectID"])
        raw = read_roi_signal_txt(Path(row["txt_path"]))
        processed = already_filtered_preprocess(raw, sid, fem)
        conn = fem.calculate_all_connectivity_modalities_for_subject(
            sid,
            processed,
            fem.N_NEIGHBORS_MI,
            fem.DFC_WIN_POINTS,
            fem.DFC_STEP,
            fem.GRANGER_MAX_LAG,
            fem.AAL3_ROI_ORDER_MAPPING,
        )
        tensor = normalize_subject_tensor(conn["matrices"], fem)
        out_path = individual_dir / f"tensor_7ch_131rois_{sid}.npz"
        np.savez_compressed(
            out_path,
            tensor_data=tensor,
            subject_id=sid,
            channel_names=np.array(CHANNEL_NAMES_MASTER, dtype=str),
            rois_count=131,
            target_len_ts=140,
            tr_seconds=3.0,
            python_bandpass_applied=False,
            preprocessing_note="DPARSF bandpass already applied; Python bandpass intentionally skipped.",
        )
        tensors.append(tensor)
        subject_ids.append(sid)
        qc_rows.append(
            {
                "SubjectID": sid,
                "raw_txt_shape": str(tuple(raw.shape)),
                "processed_shape": str(tuple(processed.shape)),
                "tensor_shape": str(tuple(tensor.shape)),
                "python_bandpass_applied": False,
                "tensor_path": str(out_path),
                "status": "OK",
            }
        )
    global_tensor = np.stack(tensors, axis=0).astype(np.float32)
    np.savez_compressed(
        tensor_path,
        global_tensor_data=global_tensor,
        subject_ids=np.array(subject_ids, dtype=str),
        channel_names=np.array(CHANNEL_NAMES_MASTER, dtype=str),
        roi_names_in_order=np.array(run_config.get("roi_names_in_order", [f"ROI_{i+1:03d}" for i in range(131)]), dtype=str),
        network_labels_in_order=np.array(run_config.get("network_labels_in_order", []), dtype=str),
        target_len_ts=140,
        tr_seconds=3.0,
        filter_low_hz=np.nan,
        filter_high_hz=np.nan,
        python_bandpass_applied=False,
        source_preprocessing="DPARSF_original_bandpass",
        notes="ROI signals were already DPARSF-bandpass filtered; no additional Python bandpass was applied.",
    )
    pd.DataFrame(qc_rows).to_csv(features_dir / "tensor_generation_qc.csv", index=False)
    return tensor_path


def build_inference_command(
    python_executable: str,
    training_dir: Path,
    tensor_path: Path,
    metadata_path: Path,
    output_symlink: Path,
    classifier_types: Sequence[str],
    ensemble_method: str,
    decision_threshold: float,
) -> List[str]:
    return [
        python_executable,
        str(PROJECT_ROOT / "scripts/inference_covid_from_adcn.py"),
        "--training_output_dir",
        str(training_dir),
        "--covid_tensor_path",
        str(tensor_path),
        "--covid_metadata_path",
        str(metadata_path),
        "--output_dir",
        str(output_symlink),
        "--classifier_types",
        *classifier_types,
        "--ensemble_method",
        ensemble_method,
        "--decision_threshold",
        str(decision_threshold),
        "--skip_signature",
    ]


def write_command_and_manifest(
    target: Path,
    command: Sequence[str],
    args: argparse.Namespace,
    subjects: pd.DataFrame,
    metadata: pd.DataFrame,
    blockers: Sequence[str],
    tensor_path: Path,
    dry_run: bool,
) -> None:
    shell_command = shlex.join(command)
    (target / "command.txt").write_text(shell_command + "\n", encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "input_root": str(resolve_path(args.input_root)),
        "expected_n": args.expected_n,
        "detected_n": int(len(subjects)),
        "detected_subjects": subjects["SubjectID"].astype(str).tolist(),
        "blockers": list(blockers),
        "training_output_dir": str(resolve_path(args.training_output_dir)),
        "model_type": "original_paper_model",
        "channels_to_use": SELECTED_CHANNELS,
        "selected_channel_names": SELECTED_CHANNEL_NAMES,
        "smoke_one_subject": bool(args.smoke_one_subject),
        "aal3_roi_metadata_path": str(resolve_path(args.aal3_roi_metadata_path)),
        "expected_final_rois": EXPECTED_FINAL_ROIS,
        "python_bandpass_applied": False,
        "python_bandpass_note": "Skipped to avoid double filtering because DPARSF already applied bandpass.",
        "tensor_path": str(tensor_path),
        "metadata_path": str(target / "audit/dparsf_bandpass10_metadata_for_inference.csv"),
        "output_symlink": str(resolve_path(args.output_symlink)),
        "output_realpath": str(resolve_path(args.output_symlink).resolve()) if resolve_path(args.output_symlink).exists() else None,
        "big_disk_target": str(args.big_disk_target),
        "classifier_types": args.classifier_types,
        "command": list(command),
        "command_shell": shell_command,
        "missing_metadata_subjects": metadata.loc[~metadata["metadata_found"], "SubjectID"].astype(str).tolist(),
    }
    (target / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def previous_prediction_paths() -> List[Path]:
    paths = [
        Path("/media/diego/Datos/adni_expansion/MARTIN59/inference_outputs/Tables/martin59_predictions_with_metadata.csv"),
        Path("/media/diego/Datos/adni_expansion/MARTIN59/inference_outputs/Tables/covid_predictions_ensemble.csv"),
        Path("/media/diego/Datos/adni_expansion/MARTIN_20260429_PHILIPS10/inference_original_model/Tables/philips7_original_model_predictions_with_metadata.csv"),
        Path("/media/diego/Datos/adni_expansion/PHILIPS_CN_STRESS/inference_outputs_plus2/Tables/philips_plus2_predictions_with_metadata.csv"),
    ]
    for base in [
        Path("/media/diego/Datos/adni_expansion"),
        Path("/media/diego/Datos/vae_AD_results"),
        PROJECT_ROOT / "results/revision_bspc_2026",
    ]:
        if base.exists():
            for path in sorted(base.rglob("*.csv")):
                if is_current_output_path(path):
                    continue
                paths.append(path)
    return [path for path in unique_paths(paths) if not is_current_output_path(path)]


def is_previous_inference_prediction_path(path: Path) -> bool:
    low_path = str(path).lower()
    low_name = path.name.lower()
    parts = [part.lower() for part in path.parts]

    if is_current_output_path(path):
        return False
    if "ablation_runs" in low_path:
        return False
    if any(part.startswith("fold_") for part in parts):
        return False
    if low_name.startswith(("all_folds_", "pooled_test_predictions", "test_predictions_", "train_predictions_", "dev_predictions_")):
        return False
    if "adni_expanded_v" in low_path and "inference" not in low_path:
        return False

    if "inference" in low_path:
        return True
    if low_name.endswith("_predictions_with_metadata.csv"):
        return True
    if low_name in {"covid_predictions_ensemble.csv", "covid_predictions_per_fold.csv"}:
        return True
    return False


def prediction_like_columns(columns: Sequence[str]) -> List[str]:
    keep = []
    for col in columns:
        low = col.lower()
        if (
            col in ["SubjectID", "classifier", "fold", "Age", "Sex", "Manufacturer", "Site3", "SourceCohort"]
            or low.startswith("y_score")
            or low.startswith("y_pred")
            or low in ["pred", "predicted_label", "proba_ad", "prob_ad", "ad_probability", "y_true"]
        ):
            keep.append(col)
    return keep


def is_prediction_table(path: Path, columns: Sequence[str]) -> bool:
    low_name = path.name.lower()
    if "prediction" in low_name or "predictions" in low_name:
        return True
    prediction_cols = {
        "y_score_ensemble",
        "y_pred_ensemble",
        "y_pred_majority_vote",
        "y_score",
        "y_score_final",
        "y_score_raw",
        "y_score_cal",
        "y_pred",
        "pred",
        "predicted_label",
        "proba_ad",
        "prob_ad",
        "ad_probability",
    }
    return bool(prediction_cols.intersection(set(columns)))


def collect_previous_predictions(subject_ids: Sequence[str]) -> pd.DataFrame:
    rows = []
    subject_set = set(subject_ids)
    for path in previous_prediction_paths():
        if not path.exists():
            continue
        if not is_previous_inference_prediction_path(path):
            continue
        try:
            columns = csv_header(path)
        except Exception:
            continue
        sid_col = normalize_subject_id_column_from_columns(columns)
        if sid_col is None or not is_prediction_table(path, columns):
            continue
        usecols = list(dict.fromkeys([sid_col] + prediction_like_columns(columns)))
        try:
            df = pd.read_csv(path, usecols=usecols)
        except Exception:
            continue
        df[sid_col] = df[sid_col].astype(str).str.strip()
        hit = df[df[sid_col].isin(subject_set)].copy()
        if hit.empty:
            continue
        for _, row in hit.iterrows():
            rows.append(
                {
                    "SubjectID": str(row[sid_col]),
                    "classifier": row.get("classifier", ""),
                    "previous_predictions_source_file": str(path),
                    "previous_preproc_source": str(path),
                    "previous_y_score_ensemble": row.get("y_score_ensemble", np.nan),
                    "previous_y_pred_ensemble": row.get("y_pred_ensemble", np.nan),
                    "previous_y_pred_majority_vote": row.get("y_pred_majority_vote", np.nan),
                    "previous_y_score": row.get("y_score", row.get("y_score_final", row.get("proba_ad", np.nan))),
                    "previous_y_pred": row.get("y_pred", row.get("pred", np.nan)),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "SubjectID",
                "classifier",
                "previous_predictions_source_file",
                "previous_y_score_ensemble",
                "previous_y_pred_ensemble",
                "previous_y_pred_majority_vote",
                "previous_y_score",
                "previous_y_pred",
            ]
        )
    prev = pd.DataFrame(rows)
    prev = prev.sort_values(["SubjectID", "classifier", "previous_predictions_source_file"])
    return prev


def postprocess_inference_outputs(target: Path, metadata: pd.DataFrame) -> None:
    tables_dir = target / "Tables"
    ens_path = tables_dir / "covid_predictions_ensemble.csv"
    if not ens_path.exists():
        return
    ens = pd.read_csv(ens_path)
    merged = ens.merge(metadata, on="SubjectID", how="left")
    merged.to_csv(tables_dir / "dparsf_bandpass10_predictions_with_metadata.csv", index=False)
    previous = collect_previous_predictions(metadata["SubjectID"].astype(str).tolist())
    comparison = previous.merge(
        merged[
            [
                "SubjectID",
                "classifier",
                "y_score_ensemble",
                "y_pred_ensemble",
                "y_pred_majority_vote",
                "y_score_std",
                "Age",
                "Sex",
                "Manufacturer",
            ]
        ],
        on=["SubjectID", "classifier"],
        how="outer",
    )
    comparison = comparison.rename(
        columns={
            "y_score_ensemble": "dparsf_bandpass_y_score_ensemble",
            "y_pred_ensemble": "dparsf_bandpass_y_pred_ensemble",
            "y_pred_majority_vote": "dparsf_bandpass_y_pred_majority_vote",
        }
    )
    comparison["delta_score"] = pd.to_numeric(comparison["dparsf_bandpass_y_score_ensemble"], errors="coerce") - pd.to_numeric(
        comparison["previous_y_score_ensemble"], errors="coerce"
    )
    comparison["changed_prediction_0p5"] = (
        pd.to_numeric(comparison["previous_y_pred_ensemble"], errors="coerce")
        != pd.to_numeric(comparison["dparsf_bandpass_y_pred_ensemble"], errors="coerce")
    )
    comparison["changed_majority_vote"] = (
        pd.to_numeric(comparison["previous_y_pred_majority_vote"], errors="coerce")
        != pd.to_numeric(comparison["dparsf_bandpass_y_pred_majority_vote"], errors="coerce")
    )
    comparison.to_csv(tables_dir / "subject_level_comparison_vs_previous_preprocessing.csv", index=False)
    write_prediction_summary(target / "audit/summary_original_model_dparsf_bandpass10.md", merged, comparison, len(metadata))


def write_prediction_summary(path: Path, predictions: pd.DataFrame, comparison: pd.DataFrame, n_detected: int) -> None:
    lines = [
        "# Original Model Inference: DPARSF Bandpass10",
        "",
        f"N detected: `{n_detected}`",
        "",
        "## AD-like Counts",
    ]
    for classifier, sub in predictions.groupby("classifier"):
        lines.append(
            f"- {classifier}: AD-like ensemble={int((sub['y_pred_ensemble'] == 1).sum())}/{len(sub)}, "
            f"majority-vote={int((sub['y_pred_majority_vote'] == 1).sum())}/{len(sub)}"
        )
    lines.extend(["", "## Comparison With Previous Preprocessing"])
    if comparison.empty or comparison["previous_y_score_ensemble"].isna().all():
        lines.append("- No previous original-model predictions found for these subjects, except any rows with non-missing previous fields in the CSV.")
    else:
        for classifier, sub in comparison.dropna(subset=["previous_y_score_ensemble"]).groupby("classifier"):
            delta = pd.to_numeric(sub["delta_score"], errors="coerce")
            lines.append(
                f"- {classifier}: mean delta={delta.mean():.4f}, median delta={delta.median():.4f}, "
                f"changed 0.5 prediction={int(sub['changed_prediction_0p5'].sum())}/{len(sub)}, "
                f"changed majority vote={int(sub['changed_majority_vote'].sum())}/{len(sub)}"
            )
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "- If DPARSF-bandpass scores are lower than previous scores, preprocessing/filter mismatch is likely contributing.",
            "- If scores are similar, scanner/cohort/model confounding is more likely than filter mismatch.",
            "- If scores increase or are unstable, inspect ROI signal QC and subject-level preprocessing.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pending_summary(path: Path, subjects: pd.DataFrame, blockers: Sequence[str]) -> None:
    lines = [
        "# Original Model Inference: DPARSF Bandpass10",
        "",
        f"N detected: `{len(subjects)}`",
        "Inference status: `NOT_RUN`",
        "",
        "## Blockers",
    ]
    lines.extend(f"- {item}" for item in blockers) if blockers else lines.append("- None for dry-run.")
    lines.extend(
        [
            "",
            "No predictions are available yet. Dry-run does not generate tensor features or call the original inference script.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_root = resolve_path(args.input_root)
    training_dir = resolve_path(args.training_output_dir)
    output_symlink = resolve_path(args.output_symlink)
    big_target = args.big_disk_target
    audit_dir, tables_dir, features_dir = ensure_output_layout(output_symlink, big_target, args.dry_run)
    if args.overwrite and not args.dry_run:
        cleanup_generated_outputs_for_overwrite(big_target)
        tables_dir = big_target / "Tables"
        features_dir = big_target / "features_or_tensor"
    disk_status = disk_status_text(input_root, Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026"))

    run_config = load_run_config(training_dir)
    validate_original_model_config(run_config)
    aal3_roi_metadata_path = resolve_path(args.aal3_roi_metadata_path)
    try:
        fem = configure_aal3_roi_processing(aal3_roi_metadata_path)
    except Exception as exc:
        write_roi_failure_readme(audit_dir, aal3_roi_metadata_path, pd.DataFrame(), str(exc))
        raise
    subjects = inventory_input(input_root, audit_dir, args.expected_n)
    metadata, missing_metadata = build_metadata(subjects, audit_dir, input_root, training_dir)
    previous_predictions = collect_previous_predictions(subjects["SubjectID"].astype(str).tolist())
    previous_predictions.to_csv(audit_dir / "previous_predictions_matched.csv", index=False)
    roi_validation = validate_roi_reduction(
        subjects,
        audit_dir,
        aal3_roi_metadata_path,
        fem,
        smoke_one_subject=args.smoke_one_subject,
    )
    tensor_path = features_dir / "GLOBAL_TENSOR_DPARSF_original_bandpass10_AAL3_131ROIs.npz"
    command = build_inference_command(
        args.python_executable,
        training_dir,
        tensor_path,
        audit_dir / "dparsf_bandpass10_metadata_for_inference.csv",
        output_symlink,
        args.classifier_types,
        args.ensemble_method,
        args.decision_threshold,
    )
    blockers = preprocessing_blocked(
        subjects,
        args.expected_n,
        missing_metadata,
        args.allow_non10,
        args.allow_missing_metadata,
    )

    write_validation_report(
        audit_dir,
        subjects,
        args.expected_n,
        missing_metadata,
        run_config,
        input_root,
        output_symlink,
        big_target,
        disk_status,
    )
    write_command_and_manifest(big_target, command, args, subjects, metadata, blockers, tensor_path, args.dry_run)
    write_pending_summary(audit_dir / "summary_original_model_dparsf_bandpass10.md", subjects, blockers)

    print(disk_status)
    print(f"Detected subjects ({len(subjects)}/{args.expected_n} expected):")
    print(subjects[["SubjectID", "txt_count", "mat_count", "txt_shape", "issue"]].to_string(index=False))
    print(f"Original model path: {training_dir}")
    print(f"Selected channels: {SELECTED_CHANNELS}")
    print(f"Selected channel names: {', '.join(SELECTED_CHANNEL_NAMES)}")
    print("Python bandpass: SKIPPED (DPARSF already applied bandpass; avoids double filtering)")
    print(f"AAL3 ROI metadata path: {aal3_roi_metadata_path}")
    print(f"ROI reduction/reordering active: {bool((fem.AAL3_ROI_ORDER_MAPPING or {}).get('new_order_indices'))}")
    print("ROI reduction validation:")
    print(roi_validation.to_string(index=False))
    print(f"Metadata path: {audit_dir / 'dparsf_bandpass10_metadata_for_inference.csv'}")
    print(f"Missing metadata subjects: {missing_metadata}")
    print("Metadata recovery:")
    print(
        metadata[
            [
                "SubjectID",
                "found_metadata",
                "metadata_source_file",
                "ResearchGroup_Mapped",
                "Age",
                "Sex",
                "Manufacturer",
                "Site3",
            ]
        ].to_string(index=False)
    )
    print(f"Previous prediction matches: {len(previous_predictions)} rows")
    if not previous_predictions.empty:
        print(
            previous_predictions[
                [
                    "SubjectID",
                    "classifier",
                    "previous_predictions_source_file",
                    "previous_y_score_ensemble",
                    "previous_y_pred_ensemble",
                    "previous_y_pred_majority_vote",
                    "previous_y_score",
                    "previous_y_pred",
                ]
            ].head(30).to_string(index=False)
        )
    print(f"Output symlink path: {output_symlink}")
    if output_symlink.exists() or output_symlink.is_symlink():
        print(f"Output realpath: {output_symlink.resolve()}")
    else:
        print("Output realpath: symlink missing")
    print(f"Big-disk target: {big_target}")
    print("\nCommand:")
    print(shlex.join(command))

    if args.dry_run:
        print("\nDry-run requested. Inference was not launched and tensor generation was not run.")
        if blockers:
            print("Blocking issues for real run:")
            for item in blockers:
                print(f"- {item}")
        return 0

    if output_has_existing_outputs(big_target) and not args.overwrite:
        raise RuntimeError(f"Refusing real run because target has existing outputs; pass --overwrite: {big_target}")
    if blockers:
        raise RuntimeError("Refusing real run due to blocking issues:\n" + "\n".join(f"- {item}" for item in blockers))
    if not output_symlink.is_symlink() or output_symlink.resolve() != big_target.resolve():
        raise RuntimeError(f"Refusing real run because output symlink is invalid: {output_symlink}")

    if args.smoke_one_subject:
        smoke_features_dir = features_dir / "smoke_one_subject"
        smoke_features_dir.mkdir(parents=True, exist_ok=True)
        smoke_subjects = subjects.sort_values("SubjectID").head(1)
        smoke_tensor_path = generate_tensor_from_roi_txt(
            smoke_subjects,
            smoke_features_dir,
            args.overwrite,
            run_config,
            fem,
            tensor_filename="GLOBAL_TENSOR_SMOKE_ONE_SUBJECT_DPARSF_original_bandpass10_AAL3_131ROIs.npz",
        )
        smoke_tensor = np.load(smoke_tensor_path, allow_pickle=False)["global_tensor_data"]
        write_command_and_manifest(big_target, command, args, smoke_subjects, metadata, [], smoke_tensor_path, False)
        print("\nSmoke-one-subject requested. Classifier inference was not launched.")
        print(f"Smoke subject: {smoke_subjects['SubjectID'].iloc[0]}")
        print(f"Smoke tensor path: {smoke_tensor_path}")
        print(f"Smoke tensor shape: {tuple(smoke_tensor.shape)}")
        return 0

    tensor_path = generate_tensor_from_roi_txt(subjects, features_dir, args.overwrite, run_config, fem)
    command = build_inference_command(
        args.python_executable,
        training_dir,
        tensor_path,
        audit_dir / "dparsf_bandpass10_metadata_for_inference.csv",
        output_symlink,
        args.classifier_types,
        args.ensemble_method,
        args.decision_threshold,
    )
    write_command_and_manifest(big_target, command, args, subjects, metadata, [], tensor_path, False)

    args.external_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.external_log_dir / f"run_original_model_inference_dparsf_bandpass10_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    with log_path.open("w", encoding="utf-8") as log_f:
        completed = subprocess.run(command, cwd=str(PROJECT_ROOT), text=True, stdout=log_f, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Inference command failed with exit code {completed.returncode}; see {log_path}")
    postprocess_inference_outputs(big_target, metadata)
    print(f"Real inference completed. Log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
