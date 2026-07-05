#!/usr/bin/env python3
"""Read-only QC audit for the OASIS next 60CN/60AD RAW download."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data/OASIS_next_60CN_60AD_2026_05_26_RAW"
LOCKED_MANIFEST = PROJECT_ROOT / "data/OASIS_next_60CN_60AD_2026_05_26/download_manifest_for_martin.csv"
SELECTED_IDEAL = PROJECT_ROOT / "data/OASIS_next_60CN_60AD_2026_05_26/selected_ideal_60CN_60AD.csv"
PILOT_DIR = PROJECT_ROOT / "data/Tanda_2026_05_25"
OUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/oasis_next_60cn_60ad_raw_download_qc"

BIDS_BOLD_RE = re.compile(
    r"sub-(?P<subject>OAS\d+)_ses-(?P<session>d\d+)_task-(?P<task>[^_]+)"
    r"(?:_run-(?P<run>\d+))?_bold(?P<suffix>\.nii\.gz|\.nii|\.json)$"
)
EXPERIMENT_RE = re.compile(r"(?P<experiment>OAS\d+_MR_d\d+)")
SUBJECT_RE = re.compile(r"sub-(?P<subject>OAS\d+)|(?P<subject2>OAS\d+)")
SESSION_RE = re.compile(r"ses-(?P<session>d\d+)|_MR_(?P<session2>d\d+)")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--locked-manifest", type=Path, default=LOCKED_MANIFEST)
    parser.add_argument("--selected-ideal", type=Path, default=SELECTED_IDEAL)
    parser.add_argument("--pilot-dir", type=Path, default=PILOT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--tiny-nifti-threshold-bytes", type=int, default=1_000_000)
    parser.add_argument("--tiny-json-threshold-bytes", type=int, default=100)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json_file(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_json_read_error": str(exc)}


def md_table(df: pd.DataFrame, max_rows: int = 120) -> str:
    if df.empty:
        return "_No rows._\n"
    sub = df.head(max_rows).copy()
    lines = [
        "| " + " | ".join(sub.columns) + " |",
        "| " + " | ".join(["---"] * len(sub.columns)) + " |",
    ]
    for _, row in sub.iterrows():
        vals: List[str] = []
        for col in sub.columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                vals.append(f"{value:.6g}" if np.isfinite(value) else "")
            elif pd.isna(value):
                vals.append("")
            else:
                vals.append(str(value).replace("\n", " ").replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Showing first {max_rows} of {len(df)} rows._")
    return "\n".join(lines) + "\n"


def write_pair(out_dir: Path, stem: str, df: pd.DataFrame, max_rows: int = 120) -> None:
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    (out_dir / f"{stem}.md").write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def experiment_from_subject_session(subject: Optional[str], session: Optional[str]) -> Optional[str]:
    if not subject or not session:
        return None
    return f"{subject}_MR_{session}"


def parse_path_ids(path: Path) -> Dict[str, Optional[str]]:
    text = str(path)
    m = BIDS_BOLD_RE.search(path.name)
    if m:
        subject = m.group("subject")
        session = m.group("session")
        return {
            "subject_id": f"sub-{subject}",
            "subject_core": subject,
            "session_id": f"ses-{session}",
            "session_core": session,
            "experiment_id": experiment_from_subject_session(subject, session),
            "task": m.group("task"),
            "run": m.group("run"),
            "is_bids_bold": "true",
        }
    exp = EXPERIMENT_RE.search(text)
    subj = SUBJECT_RE.search(text)
    sess = SESSION_RE.search(text)
    subject_core = None
    session_core = None
    if subj:
        subject_core = subj.group("subject") or subj.group("subject2")
    if sess:
        session_core = sess.group("session") or sess.group("session2")
    return {
        "subject_id": f"sub-{subject_core}" if subject_core else None,
        "subject_core": subject_core,
        "session_id": f"ses-{session_core}" if session_core else None,
        "session_core": session_core,
        "experiment_id": exp.group("experiment") if exp else experiment_from_subject_session(subject_core, session_core),
        "task": None,
        "run": None,
        "is_bids_bold": "false",
    }


def extension_for(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    return path.suffix


def inventory_files(raw_dir: Path, tiny_nifti: int, tiny_json: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path
        ids = parse_path_ids(path)
        ext = extension_for(path)
        size = path.stat().st_size
        is_json = ext == ".json"
        is_nifti = ext in {".nii", ".nii.gz"}
        modality = "BOLD_JSON" if is_json and "_bold" in path.name else ("BOLD_NIFTI" if is_nifti and "_bold" in path.name else "unknown")
        suspicious_reason = ""
        if is_nifti and size < tiny_nifti:
            suspicious_reason = f"nifti_size_below_{tiny_nifti}"
        elif is_json and size < tiny_json:
            suspicious_reason = f"json_size_below_{tiny_json}"
        elif size == 0:
            suspicious_reason = "zero_size"
        rows.append(
            {
                "file_path": str(rel),
                "extension": ext,
                "size_bytes": int(size),
                "size_mb": size / 1_000_000.0,
                "experiment_id": ids["experiment_id"],
                "subject_id": ids["subject_id"],
                "session_id": ids["session_id"],
                "task": ids["task"],
                "run": ids["run"],
                "modality_type": modality,
                "is_bids_bold": ids["is_bids_bold"],
                "suspicious_reason": suspicious_reason,
            }
        )
    return pd.DataFrame(rows)


def bold_run_qc(raw_dir: Path, inventory: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    json_rows: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    nifti_rows: Set[Tuple[str, str, str, str]] = set()
    size_by_key: Dict[Tuple[str, str, str, str], int] = {}
    for _, row in inventory.iterrows():
        if row["modality_type"] not in {"BOLD_JSON", "BOLD_NIFTI"}:
            continue
        exp = row["experiment_id"]
        subj = row["subject_id"]
        sess = row["session_id"]
        task = row["task"] or ""
        run = str(row["run"] or "")
        key = (str(exp), str(subj), str(sess), f"{task}__{run}")
        if row["modality_type"] == "BOLD_NIFTI":
            nifti_rows.add(key)
            size_by_key[key] = int(row["size_bytes"])
        elif row["modality_type"] == "BOLD_JSON":
            meta = read_json_file(resolve(Path(str(row["file_path"]))))
            json_rows[key] = meta

    manifest_by_exp = manifest.drop_duplicates("experiment_id").set_index("experiment_id")
    all_keys = sorted(set(json_rows) | nifti_rows)
    rows: List[Dict[str, Any]] = []
    for key in all_keys:
        exp, subj, sess, task_run = key
        task, run = task_run.split("__", 1)
        meta = json_rows.get(key, {})
        manifest_row = manifest_by_exp.loc[exp].to_dict() if exp in manifest_by_exp.index else {}
        rows.append(
            {
                "experiment_id": exp,
                "subject_id": subj,
                "session_id": sess,
                "task_from_filename": task,
                "run": run,
                "has_nifti": key in nifti_rows,
                "has_json": key in json_rows,
                "nifti_size_bytes": size_by_key.get(key),
                "RepetitionTime": meta.get("RepetitionTime"),
                "TaskName": meta.get("TaskName", task),
                "SeriesDescription": meta.get("SeriesDescription"),
                "Manufacturer": meta.get("Manufacturer"),
                "ManufacturersModelName": meta.get("ManufacturersModelName"),
                "AcquisitionTime": meta.get("AcquisitionTime"),
                "json_read_error": meta.get("_json_read_error"),
                "manifest_diagnosis": manifest_row.get("diagnosis"),
                "manifest_expected_runs": manifest_row.get("expected_runs"),
                "manifest_expected_tr2_rest_runs": manifest_row.get("expected_tr2_rest_runs"),
            }
        )
    return pd.DataFrame(rows)


def experiment_coverage(raw_dir: Path, manifest: pd.DataFrame, inventory: pd.DataFrame, run_qc: pd.DataFrame) -> pd.DataFrame:
    expected = set(manifest["experiment_id"].dropna().astype(str))
    found_dirs = {p.name for p in raw_dir.iterdir() if p.is_dir()}
    found_from_files = set(inventory["experiment_id"].dropna().astype(str))
    found = found_dirs | found_from_files
    rows: List[Dict[str, Any]] = []
    run_counts = run_qc.groupby("experiment_id").agg(
        n_bold_runs=("experiment_id", "size"),
        n_bold_nifti=("has_nifti", "sum"),
        n_bold_json=("has_json", "sum"),
    )
    manifest_by_exp = manifest.drop_duplicates("experiment_id").set_index("experiment_id")
    for exp in sorted(expected | found):
        m = manifest_by_exp.loc[exp].to_dict() if exp in manifest_by_exp.index else {}
        rc = run_counts.loc[exp].to_dict() if exp in run_counts.index else {}
        rows.append(
            {
                "experiment_id": exp,
                "expected": exp in expected,
                "found_experiment_folder": exp in found_dirs,
                "found_any_file": exp in found_from_files,
                "status": "found_expected" if exp in expected and exp in found else ("missing" if exp in expected else "extra"),
                "subject_id": m.get("subject_id"),
                "diagnosis": m.get("diagnosis"),
                "expected_runs": m.get("expected_runs"),
                "expected_tr2_rest_runs": m.get("expected_tr2_rest_runs"),
                "n_bold_runs_detected": int(rc.get("n_bold_runs", 0) or 0),
                "n_bold_nifti": int(rc.get("n_bold_nifti", 0) or 0),
                "n_bold_json": int(rc.get("n_bold_json", 0) or 0),
            }
        )
    return pd.DataFrame(rows)


def tr_distribution(run_qc: pd.DataFrame) -> pd.DataFrame:
    if run_qc.empty:
        return pd.DataFrame()
    df = run_qc.copy()
    df["RepetitionTime"] = pd.to_numeric(df["RepetitionTime"], errors="coerce")
    group_cols = ["RepetitionTime", "task_from_filename", "SeriesDescription"]
    out = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_runs=("experiment_id", "size"),
            n_experiments=("experiment_id", "nunique"),
            n_subjects=("subject_id", "nunique"),
        )
        .reset_index()
        .sort_values(["RepetitionTime", "task_from_filename", "SeriesDescription"], na_position="last")
    )
    return out


def diagnosis_balance(manifest: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    selected = manifest.drop_duplicates("experiment_id").copy()
    expected_counts = selected["diagnosis"].value_counts().rename_axis("diagnosis").reset_index(name="expected_n")
    found_exp = set(coverage.loc[coverage["status"].eq("found_expected"), "experiment_id"].astype(str))
    found_counts = (
        selected[selected["experiment_id"].astype(str).isin(found_exp)]["diagnosis"]
        .value_counts()
        .rename_axis("diagnosis")
        .reset_index(name="found_n")
    )
    out = expected_counts.merge(found_counts, on="diagnosis", how="outer").fillna(0)
    out["expected_n"] = out["expected_n"].astype(int)
    out["found_n"] = out["found_n"].astype(int)
    out["missing_n"] = out["expected_n"] - out["found_n"]
    out["target_n"] = out["diagnosis"].map({"CN": 60, "AD_DEMENTIA": 60}).fillna(np.nan)
    out["target_met_expected"] = out["expected_n"].eq(out["target_n"])
    out["target_met_found"] = out["found_n"].eq(out["target_n"])
    return out.sort_values("diagnosis")


def collect_ids_from_paths(root: Path) -> Dict[str, Set[str]]:
    subjects: Set[str] = set()
    experiments: Set[str] = set()
    subject_sessions: Set[str] = set()
    for path in root.rglob("*"):
        ids = parse_path_ids(path)
        subj = ids.get("subject_id")
        sess = ids.get("session_id")
        exp = ids.get("experiment_id")
        if subj:
            subjects.add(str(subj))
        if exp:
            experiments.add(str(exp))
        if subj and sess:
            subject_sessions.add(f"{subj}|{sess}")
    return {"subject": subjects, "experiment_id": experiments, "subject_session": subject_sessions}


def pilot_overlap(manifest: pd.DataFrame, coverage: pd.DataFrame, pilot_dir: Path, raw_dir: Path) -> pd.DataFrame:
    pilot = collect_ids_from_paths(pilot_dir)
    raw = collect_ids_from_paths(raw_dir)
    manifest_subjects = set(manifest["subject_id"].dropna().astype(str))
    manifest_exps = set(manifest["experiment_id"].dropna().astype(str))
    manifest_pairs = set()
    if "session_id" in manifest.columns:
        manifest_pairs = set(manifest["subject_id"].astype(str) + "|" + "ses-" + manifest["session_id"].astype(str).str.extract(r"(d\d+)", expand=False).fillna(""))
    found = set(coverage.loc[coverage["status"].eq("found_expected"), "experiment_id"].astype(str))
    found_manifest = manifest[manifest["experiment_id"].astype(str).isin(found)]
    found_subjects = set(found_manifest["subject_id"].dropna().astype(str))
    found_pairs = set()
    if "session_id" in found_manifest.columns:
        found_pairs = set(found_manifest["subject_id"].astype(str) + "|" + "ses-" + found_manifest["session_id"].astype(str).str.extract(r"(d\d+)", expand=False).fillna(""))
    rows = []
    for scope, selected_ids, raw_ids, pilot_ids in [
        ("subject", manifest_subjects, raw["subject"], pilot["subject"]),
        ("experiment_id", manifest_exps, raw["experiment_id"], pilot["experiment_id"]),
        ("subject_session", manifest_pairs, raw["subject_session"], pilot["subject_session"]),
    ]:
        rows.append(
            {
                "scope": scope,
                "selected_manifest_n": len(selected_ids),
                "raw_detected_n": len(raw_ids),
                "pilot_detected_n": len(pilot_ids),
                "selected_vs_pilot_overlap_n": len(selected_ids & pilot_ids),
                "raw_vs_pilot_overlap_n": len(raw_ids & pilot_ids),
                "found_expected_vs_pilot_overlap_n": len((found_subjects if scope == "subject" else found if scope == "experiment_id" else found_pairs) & pilot_ids),
                "selected_vs_pilot_overlap_values": ";".join(sorted(selected_ids & pilot_ids)[:100]),
                "raw_vs_pilot_overlap_values": ";".join(sorted(raw_ids & pilot_ids)[:100]),
            }
        )
    return pd.DataFrame(rows)


def suspicious_files(inventory: pd.DataFrame) -> pd.DataFrame:
    if inventory.empty:
        return pd.DataFrame()
    return inventory[inventory["suspicious_reason"].astype(str).ne("")].copy().sort_values(["suspicious_reason", "size_bytes", "file_path"])


def write_readme(out_dir: Path, raw_dir: Path, coverage: pd.DataFrame, run_qc: pd.DataFrame, overlap: pd.DataFrame, diag: pd.DataFrame, suspicious: pd.DataFrame) -> None:
    n_expected = int(coverage["expected"].sum()) if not coverage.empty else 0
    n_found = int(coverage["found_experiment_folder"].sum()) if not coverage.empty else 0
    n_missing = int(coverage["status"].eq("missing").sum()) if not coverage.empty else 0
    n_extra = int(coverage["status"].eq("extra").sum()) if not coverage.empty else 0
    n_nifti = int(run_qc["has_nifti"].sum()) if not run_qc.empty else 0
    n_json = int(run_qc["has_json"].sum()) if not run_qc.empty else 0
    overlap_any = int(overlap.filter(like="_overlap_n").sum().sum()) if not overlap.empty else 0
    lines = [
        "# OASIS Next 60CN/60AD RAW Download QC",
        "",
        "Read-only audit of the newly downloaded RAW OASIS batch. No preprocessing, connectome construction, model scoring, or training was performed.",
        "",
        f"- Raw folder: `{raw_dir}`",
        f"- Expected experiments: `{n_expected}`",
        f"- Found top-level experiment folders: `{n_found}`",
        f"- Missing expected experiments: `{n_missing}`",
        f"- Extra experiment folders/files: `{n_extra}`",
        f"- BOLD NIfTI runs detected: `{n_nifti}`",
        f"- BOLD JSON sidecars detected: `{n_json}`",
        f"- Pilot overlap count aggregate: `{overlap_any}`",
        f"- Suspicious tiny/zero files: `{len(suspicious)}`",
        "",
        "## Diagnosis Balance",
        "",
        md_table(diag),
        "",
        "## Pilot Overlap",
        "",
        md_table(overlap),
    ]
    write_text(out_dir / "README.md", "\n".join(lines))


def final_recommendation_text(coverage: pd.DataFrame, run_qc: pd.DataFrame, overlap: pd.DataFrame, suspicious: pd.DataFrame) -> str:
    n_missing = int(coverage["status"].eq("missing").sum()) if not coverage.empty else 0
    n_extra = int(coverage["status"].eq("extra").sum()) if not coverage.empty else 0
    n_bad_pairs = int((~(run_qc["has_nifti"] & run_qc["has_json"])).sum()) if not run_qc.empty else 0
    overlap_any = int(overlap.filter(like="_overlap_n").sum().sum()) if not overlap.empty else 0
    tr_values = sorted(pd.to_numeric(run_qc.get("RepetitionTime", pd.Series(dtype=float)), errors="coerce").dropna().unique().tolist()) if not run_qc.empty else []
    ready = n_missing == 0 and n_extra == 0 and n_bad_pairs == 0 and overlap_any == 0 and suspicious.empty
    status = "ready_for_martin_packaging" if ready else "needs_download_completion_or_review_before_processing"
    lines = [
        "# Final Recommendation",
        "",
        f"Decision: `{status}`.",
        "",
        "This RAW batch should not proceed to ROI extraction/connectome construction until the missing/extra coverage and run-sidecar checks are resolved.",
        "",
        "Summary:",
        f"- Missing expected experiments: `{n_missing}`.",
        f"- Extra experiments/files: `{n_extra}`.",
        f"- BOLD run rows lacking either NIfTI or JSON: `{n_bad_pairs}`.",
        f"- Pilot overlap aggregate count: `{overlap_any}`.",
        f"- Suspicious tiny/zero files: `{len(suspicious)}`.",
        f"- RepetitionTime values detected: `{tr_values}`.",
        "",
        "Upload/package recommendation for Martin:",
        "1. Complete or explain all missing expected experiment folders.",
        "2. Keep each experiment in the current `OASxxxxx_MR_dyyyy/func*` BIDS-like structure.",
        "3. Include both `*_bold.nii.gz` and matching `*_bold.json` for every usable run.",
        "4. Preserve the locked selected manifest with diagnosis/age/sex/TR metadata in the package.",
        "5. Do not mix this batch with `Tanda_2026_05_25`; overlap should remain zero.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    raw_dir = resolve(args.raw_dir)
    locked_manifest_path = resolve(args.locked_manifest)
    selected_ideal_path = resolve(args.selected_ideal)
    pilot_dir = resolve(args.pilot_dir)
    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in [raw_dir, locked_manifest_path, selected_ideal_path, pilot_dir]:
        if not path.exists():
            raise FileNotFoundError(path)

    locked = pd.read_csv(locked_manifest_path)
    selected = pd.read_csv(selected_ideal_path)
    if len(locked) != 120:
        raise RuntimeError(f"Locked manifest expected 120 rows, found {len(locked)}")
    if len(selected) != 120:
        raise RuntimeError(f"Selected ideal expected 120 rows, found {len(selected)}")

    inventory = inventory_files(raw_dir, args.tiny_nifti_threshold_bytes, args.tiny_json_threshold_bytes)
    run_qc = bold_run_qc(raw_dir, inventory, locked)
    coverage = experiment_coverage(raw_dir, locked, inventory, run_qc)
    tr = tr_distribution(run_qc)
    diag = diagnosis_balance(locked, coverage)
    overlap = pilot_overlap(locked, coverage, pilot_dir, raw_dir)
    suspicious = suspicious_files(inventory)

    write_pair(out_dir, "raw_file_inventory", inventory, max_rows=250)
    write_pair(out_dir, "experiment_coverage_qc", coverage, max_rows=160)
    write_pair(out_dir, "bold_run_qc", run_qc, max_rows=250)
    write_pair(out_dir, "tr_distribution", tr)
    write_pair(out_dir, "diagnosis_balance_qc", diag)
    write_pair(out_dir, "pilot_overlap_qc", overlap)
    write_pair(out_dir, "suspicious_files", suspicious, max_rows=200)
    write_readme(out_dir, raw_dir, coverage, run_qc, overlap, diag, suspicious)
    write_text(out_dir / "final_recommendation.md", final_recommendation_text(coverage, run_qc, overlap, suspicious))
    command_log = {
        "timestamp": now(),
        "raw_dir": str(raw_dir),
        "locked_manifest": str(locked_manifest_path),
        "selected_ideal": str(selected_ideal_path),
        "pilot_dir": str(pilot_dir),
        "output_dir": str(out_dir),
        "read_only": True,
        "preprocessing_performed": False,
        "connectomes_built": False,
        "models_trained_or_scored": False,
        "summary": {
            "expected_experiments": int(coverage["expected"].sum()),
            "found_experiment_folders": int(coverage["found_experiment_folder"].sum()),
            "missing_experiments": int(coverage["status"].eq("missing").sum()),
            "extra_experiments": int(coverage["status"].eq("extra").sum()),
            "bold_nifti_runs": int(run_qc["has_nifti"].sum()) if not run_qc.empty else 0,
            "bold_json_sidecars": int(run_qc["has_json"].sum()) if not run_qc.empty else 0,
            "suspicious_files": int(len(suspicious)),
        },
        "outputs": [
            "README.md",
            "raw_file_inventory.csv/.md",
            "experiment_coverage_qc.csv/.md",
            "bold_run_qc.csv/.md",
            "tr_distribution.csv/.md",
            "diagnosis_balance_qc.csv/.md",
            "pilot_overlap_qc.csv/.md",
            "suspicious_files.csv/.md",
            "final_recommendation.md",
        ],
    }
    write_text(out_dir / "command_log.json", json.dumps(command_log, indent=2, sort_keys=True))
    print(f"Wrote OASIS RAW QC audit to {out_dir.relative_to(PROJECT_ROOT)}")
    print(json.dumps(command_log["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
