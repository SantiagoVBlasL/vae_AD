#!/usr/bin/env python3
"""Final read-only CN-GE local inventory before asking Martin for reprocessing.

This script is intentionally conservative:
- no model training;
- no connectivity tensor construction;
- no Python bandpass;
- no writes outside the final audit output directory;
- no modifications to existing v5 or v5.1 data products.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_and_import_local_ge_cn_roisignals_v5_1 import (  # noqa: E402
    RAW_ROIS,
    candidate_compatibility,
    canonical_manufacturer,
    clean_string,
    explicit_has_f_stage,
    normalize_group,
    normalize_subject,
    qc_candidate,
    rank_candidate,
    stage_guess,
)


DEFAULT_MASTER_FIRST_VISIT = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_master_manifest"
    / "adni_v5_1_master_subject_manifest_first_visit.csv"
)
DEFAULT_PREPROCESSING_REQUEST = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_master_manifest"
    / "adni_v5_1_preprocessing_request_for_martin.csv"
)
DEFAULT_AUGMENTED_DIR = (
    PROJECT_ROOT / "results" / "revision_bspc_2026" / "adni_v5_1_gecn_augmented_manifest"
)
DEFAULT_LOCAL_AUDIT_DIR = (
    PROJECT_ROOT / "results" / "revision_bspc_2026" / "adni_v5_1_gecn_local_roisignals_audit"
)
DEFAULT_GECN9_QC_DIR = (
    PROJECT_ROOT / "results" / "revision_bspc_2026" / "adni_v5_1_gecn9_full_build_qc"
)
DEFAULT_V5_QC_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_expanded_v5_dparsf10000_no_pybandpass_pretraining_qc"
)
DEFAULT_V5_1_DATA_LINK = (
    PROJECT_ROOT / "data" / "revision_bspc_2026" / "adni_expanded_v5_1_gecn9_no_pybandpass"
)
DEFAULT_V5_1_BIG_DATA = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/adni_expanded_v5_1_gecn9_no_pybandpass"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "final_cn_ge_inventory_before_martin_request"
)

SEARCH_ROOTS = [
    PROJECT_ROOT,
    Path("/media/diego/Datos"),
    Path("/media/diego/My_Book_Diego"),
]

SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".ipynb_checkpoints",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
}

STAGE_ORDER = ["ARWSDCFN", "ARWSDCF", "ARWSDC", "CovRegressed", "historical_10000", "unknown"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final CN-GE local ROISignals inventory before requesting anything else from Martin.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--master-first-visit", type=Path, default=DEFAULT_MASTER_FIRST_VISIT)
    parser.add_argument("--preprocessing-request", type=Path, default=DEFAULT_PREPROCESSING_REQUEST)
    parser.add_argument("--augmented-dir", type=Path, default=DEFAULT_AUGMENTED_DIR)
    parser.add_argument("--local-audit-dir", type=Path, default=DEFAULT_LOCAL_AUDIT_DIR)
    parser.add_argument("--gecn9-qc-dir", type=Path, default=DEFAULT_GECN9_QC_DIR)
    parser.add_argument("--v5-qc-dir", type=Path, default=DEFAULT_V5_QC_DIR)
    parser.add_argument("--v5-1-data-link", type=Path, default=DEFAULT_V5_1_DATA_LINK)
    parser.add_argument("--v5-1-big-data", type=Path, default=DEFAULT_V5_1_BIG_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(f"Output directory exists and is not empty; pass --overwrite: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def yes_mask(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.lower().isin({"yes", "true", "1"})


def boolish(value: Any) -> bool:
    return clean_string(value).lower() in {"yes", "true", "1"}


def normalize_stage(value: Any, path: Any = "") -> str:
    text = f"{clean_string(value)} {clean_string(path)}".upper()
    if "ARWSDCFN" in text:
        return "ARWSDCFN"
    if "ARWSDCF" in text:
        return "ARWSDCF"
    if "COVREGRESSED" in text:
        return "CovRegressed"
    if "ARWSDC" in text or "FUNIMGARWSDCOVS" in text:
        return "ARWSDC"
    if "HISTORICAL_10000" in text or "/DESDE_CERO/" in text or "ROISIGNALSAAL3" in text:
        return "historical_10000"
    return "unknown"


def load_cn_ge_universe(master_first_visit: Path, preprocessing_request: Path) -> pd.DataFrame:
    master = safe_read_csv(master_first_visit)
    if master.empty:
        raise FileNotFoundError(f"Master first-visit manifest missing or empty: {master_first_visit}")
    master["SubjectID"] = master["SubjectID"].map(normalize_subject)
    master["ResearchGroup_Mapped"] = master.get("ResearchGroup_Mapped", "").map(normalize_group)
    master["Manufacturer"] = master.get("Manufacturer", "").map(canonical_manufacturer)
    mask = (
        master["SubjectID"].astype(bool)
        & master["ResearchGroup_Mapped"].eq("CN")
        & master["Manufacturer"].eq("GE")
    )
    out = master.loc[mask].drop_duplicates("SubjectID", keep="first").copy()
    request = safe_read_csv(preprocessing_request)
    request_subjects: Set[str] = set()
    if not request.empty and "SubjectID" in request.columns:
        req = request.copy()
        req["SubjectID"] = req["SubjectID"].map(normalize_subject)
        req["ResearchGroup_Mapped"] = req.get("ResearchGroup_Mapped", "").map(normalize_group)
        req["Manufacturer"] = req.get("Manufacturer", "").map(canonical_manufacturer)
        request_subjects = set(
            req.loc[
                req["ResearchGroup_Mapped"].eq("CN") & req["Manufacturer"].eq("GE"),
                "SubjectID",
            ]
        )
    out["in_preprocessing_request_for_martin"] = out["SubjectID"].isin(request_subjects)
    out["first_visit_universe_source"] = str(master_first_visit)
    return out.sort_values("SubjectID").reset_index(drop=True)


def source_root_for(path: Path, roots: Sequence[Path]) -> str:
    path_str = str(path)
    matches = [str(root) for root in roots if path_str.startswith(str(root))]
    return max(matches, key=len) if matches else ""


def dedupe_paths(paths: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    seen: Set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def effective_search_roots(v5_1_data_link: Path, v5_1_big_data: Path) -> List[Path]:
    roots = list(SEARCH_ROOTS)
    for extra in [v5_1_data_link, v5_1_big_data]:
        if extra.exists() or extra.is_symlink():
            roots.append(extra)
    return dedupe_paths(roots)


def path_exists_now(value: Any) -> bool:
    text = clean_string(value)
    return bool(text) and Path(text).exists()


def path_key(path: Path) -> str:
    try:
        return str(path.resolve())
    except Exception:
        return str(path)


def discover_roisignals_all_roots(target_subjects: Sequence[str], roots: Sequence[Path]) -> pd.DataFrame:
    targets = set(target_subjects)
    rows: List[Dict[str, Any]] = []
    seen_keys: Set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = [
                d for d in dirnames
                if d not in SKIP_DIR_NAMES and not d.startswith(".Trash")
            ]
            for filename in filenames:
                lower = filename.lower()
                if not (lower.endswith(".mat") or lower.endswith(".txt")):
                    continue
                if "roisignals" not in lower:
                    continue
                sid = normalize_subject(filename)
                if sid not in targets:
                    continue
                path = Path(dirpath) / filename
                key = path_key(path)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                try:
                    stat = path.stat()
                    size = int(stat.st_size)
                    mtime = pd.Timestamp.fromtimestamp(stat.st_mtime).isoformat()
                except OSError:
                    size = np.nan
                    mtime = ""
                stage = normalize_stage(stage_guess(path), path)
                rows.append(
                    {
                        "SubjectID": sid,
                        "path": str(path),
                        "resolved_path": key,
                        "source_root": source_root_for(path, roots),
                        "suffix": path.suffix.lower(),
                        "file_size_bytes": size,
                        "mtime": mtime,
                        "stage_guess": stage,
                        "explicit_has_F_stage": explicit_has_f_stage(stage, path),
                        "discovered_by_full_root_scan": True,
                    }
                )
    return pd.DataFrame(rows)


def load_previous_candidates(local_audit_dir: Path, target_subjects: Sequence[str]) -> pd.DataFrame:
    targets = set(target_subjects)
    frames: List[pd.DataFrame] = []
    for filename, label in [
        ("ge_cn_candidate_roisignals_long.csv", "previous_candidate_long"),
        ("ge_cn_recommended_roisignals.csv", "previous_recommended"),
        ("ge_cn_rejected_or_needs_confirmation.csv", "previous_rejected_or_needs_confirmation"),
    ]:
        df = safe_read_csv(local_audit_dir / filename)
        if df.empty or "SubjectID" not in df.columns:
            continue
        df["SubjectID"] = df["SubjectID"].map(normalize_subject)
        path_col = "path" if "path" in df.columns else "recommended_path" if "recommended_path" in df.columns else ""
        if not path_col:
            continue
        df["path"] = df[path_col].map(clean_string)
        df = df[df["SubjectID"].isin(targets) & df["path"].astype(bool)].copy()
        if df.empty:
            continue
        df["previous_audit_source_file"] = filename
        df["previous_audit_source_label"] = label
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["resolved_path"] = [path_key(Path(p)) for p in out["path"]]
    out = out.drop_duplicates(["SubjectID", "resolved_path"], keep="first").copy()
    return out


def qc_or_reuse_candidate(base: Dict[str, Any], previous_by_key: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, Any]:
    key = (base["SubjectID"], base["resolved_path"])
    previous = previous_by_key.get(key)
    if previous:
        row = dict(previous)
        for col in [
            "path",
            "resolved_path",
            "source_root",
            "suffix",
            "file_size_bytes",
            "mtime",
            "discovered_by_full_root_scan",
        ]:
            if col in base and clean_string(base.get(col)):
                row[col] = base[col]
        row["qc_source"] = "previous_local_audit_reused"
    else:
        row = qc_candidate(Path(str(base["path"])), base)
        row["qc_source"] = "computed_in_final_audit"
    row["stage_guess"] = normalize_stage(row.get("stage_guess", ""), row.get("path", ""))
    row["path_exists_now"] = path_exists_now(row.get("path"))
    row["seen_in_previous_local_audit"] = bool(previous)
    compat = clean_string(row.get("suspected_stage_compatibility", ""))
    if not compat:
        compat, reason = candidate_compatibility(row)
        row["suspected_stage_compatibility"] = compat
        row["compatibility_reason"] = reason
    row["compatible_for_no_pybandpass_v5_1"] = {
        "direct_compatible": "yes",
        "direct_compatible_with_warning": "yes_with_warning",
    }.get(row["suspected_stage_compatibility"], "no")
    row["reason"] = clean_string(row.get("compatibility_reason", ""))
    return row


def build_candidate_long(targets: pd.DataFrame, roots: Sequence[Path], local_audit_dir: Path) -> pd.DataFrame:
    subjects = targets["SubjectID"].tolist()
    discovered = discover_roisignals_all_roots(subjects, roots)
    previous = load_previous_candidates(local_audit_dir, subjects)
    previous_by_key = {
        (str(r["SubjectID"]), str(r["resolved_path"])): r.to_dict()
        for _, r in previous.iterrows()
    } if not previous.empty else {}

    bases: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()
    if not discovered.empty:
        for _, r in discovered.iterrows():
            key = (str(r["SubjectID"]), str(r["resolved_path"]))
            seen.add(key)
            bases.append(r.to_dict())
    if not previous.empty:
        for _, r in previous.iterrows():
            key = (str(r["SubjectID"]), str(r["resolved_path"]))
            if key in seen:
                continue
            row = r.to_dict()
            row.setdefault("source_root", "")
            row.setdefault("suffix", Path(clean_string(row.get("path"))).suffix.lower())
            row.setdefault("resolved_path", path_key(Path(clean_string(row.get("path")))))
            row["discovered_by_full_root_scan"] = False
            bases.append(row)

    rows = [qc_or_reuse_candidate(base, previous_by_key) for base in bases]
    long_df = pd.DataFrame(rows)
    if long_df.empty:
        long_df = pd.DataFrame(columns=["SubjectID", "path"])

    no_file_rows: List[Dict[str, Any]] = []
    subjects_with_current_file = set(
        long_df.loc[long_df.get("path_exists_now", pd.Series(dtype=bool)).fillna(False).astype(bool), "SubjectID"]
    ) if not long_df.empty and "path_exists_now" in long_df.columns else set()
    for _, target in targets.iterrows():
        sid = target["SubjectID"]
        if sid in subjects_with_current_file:
            continue
        no_file_rows.append(
            {
                "SubjectID": sid,
                "path": "NO_LOCAL_FILE",
                "resolved_path": "",
                "source_root": "",
                "suffix": "",
                "file_size_bytes": "",
                "mtime": "",
                "stage_guess": "unknown",
                "n_rois": "",
                "n_timepoints": "",
                "finite_fraction": "",
                "scale_label": "",
                "compatible_for_no_pybandpass_v5_1": "no",
                "reason": "no_local_roisignals_found",
                "path_exists_now": False,
                "qc_source": "not_applicable",
                "seen_in_previous_local_audit": False,
                "discovered_by_full_root_scan": False,
            }
        )
    if no_file_rows:
        long_df = pd.concat([long_df, pd.DataFrame(no_file_rows)], ignore_index=True, sort=False)
    return long_df


def load_v5_current_subjects(v5_qc_dir: Path) -> Set[str]:
    subjects: Set[str] = set()
    for filename in ["tensor_metadata_alignment.csv", "training_ready_manifest.csv"]:
        df = safe_read_csv(v5_qc_dir / filename)
        if df.empty or "SubjectID" not in df.columns:
            continue
        df["SubjectID"] = df["SubjectID"].map(normalize_subject)
        if "in_tensor" in df.columns:
            mask = df["in_tensor"].map(boolish)
            subjects.update(df.loc[mask, "SubjectID"])
        else:
            subjects.update(df["SubjectID"])
    return subjects


def load_gecn9_inclusion(
    gecn9_qc_dir: Path,
    v5_1_data_link: Path,
    v5_1_big_data: Path,
) -> Tuple[Set[str], Set[str], pd.DataFrame]:
    tensor_subjects: Set[str] = set()
    metadata_subjects: Set[str] = set()
    alignment = safe_read_csv(gecn9_qc_dir / "v5_1_gecn9_subject_alignment.csv")
    if not alignment.empty and "SubjectID" in alignment.columns:
        alignment["SubjectID"] = alignment["SubjectID"].map(normalize_subject)
        tensor_subjects.update(alignment["SubjectID"])
        if "training_ready" in alignment.columns:
            metadata_subjects.update(alignment.loc[alignment["training_ready"].map(boolish), "SubjectID"])
        else:
            metadata_subjects.update(alignment["SubjectID"])

    for root in [v5_1_data_link, v5_1_big_data]:
        metadata_path = root / "training_ready_metadata_v5_1_gecn9_no_pybandpass.csv"
        meta = safe_read_csv(metadata_path)
        if not meta.empty and "SubjectID" in meta.columns:
            meta["SubjectID"] = meta["SubjectID"].map(normalize_subject)
            metadata_subjects.update(meta["SubjectID"])
        npz_path = root / "subject_tensors" / "GLOBAL_TENSOR_ADNI_expanded_v5_1_gecn9_no_pybandpass.npz"
        if npz_path.exists():
            try:
                with np.load(npz_path, allow_pickle=False) as zf:
                    tensor_subjects.update(str(x).upper() for x in zf["subject_ids"].astype(str))
            except Exception:
                pass
    return tensor_subjects, metadata_subjects, alignment


def best_candidate_per_subject(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()
    real = long_df[long_df["path"].ne("NO_LOCAL_FILE")].copy()
    if real.empty:
        return pd.DataFrame()
    real["_rank"] = real.apply(rank_candidate, axis=1)
    return real.sort_values(["SubjectID", "_rank"], kind="mergesort").drop_duplicates("SubjectID", keep="first").drop(columns=["_rank"])


def classify_subject(
    sid: str,
    included_v5_1: bool,
    candidate_rows: pd.DataFrame,
    best: Optional[pd.Series],
) -> Tuple[str, str, str]:
    if included_v5_1:
        return "already_included_v5_1_gecn9", "already present in v5.1_gecn9 tensor/metadata", ""
    current = candidate_rows[candidate_rows["path"].ne("NO_LOCAL_FILE") & candidate_rows["path_exists_now"].fillna(False).astype(bool)]
    if current.empty:
        return (
            "no_roisignals_found_needs_preprocessing",
            "no local ROISignals .mat/.txt found in search roots",
            "Ask Martin for first-visit DPARSF-compatible AAL3 ROISignals with 170 ROIs, finite signal, scale compatible with 10000, and no extra Python bandpass.",
        )
    compat_values = set(current["compatible_for_no_pybandpass_v5_1"].fillna("").astype(str))
    if compat_values <= {"no"}:
        reasons = sorted(set(current["reason"].fillna("").astype(str)))
        reject_like = any(
            token in "|".join(reasons)
            for token in ["load_failed", "finite_fraction", "roi_count", "scale_not"]
        )
        if reject_like:
            return (
                "already_local_but_rejected_qc",
                "; ".join(r for r in reasons if r) or "local file failed QC",
                "Ask Martin to re-export/reprocess this first visit because local ROISignals exist but fail QC for v5.1 no-Python-bandpass inclusion.",
            )
    if best is not None:
        return (
            "already_local_but_stage_confirmation_needed",
            clean_string(best.get("reason")) or "local ROISignals exist but stage evidence is not definitive",
            "Ask Martin to confirm whether the existing local ROISignals are from a DPARSF-filtered AAL3 stage compatible with v5.1 no-Python-bandpass; otherwise ask for reprocessed first-visit ROISignals.",
        )
    return (
        "already_local_but_stage_confirmation_needed",
        "local ROISignals exist but no selected best candidate could be established",
        "Ask Martin for stage confirmation or a clean first-visit export.",
    )


def build_subject_tables(
    targets: pd.DataFrame,
    long_df: pd.DataFrame,
    v5_current_subjects: Set[str],
    gecn9_tensor_subjects: Set[str],
    gecn9_metadata_subjects: Set[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    best_df = best_candidate_per_subject(long_df)
    best_by_subject = {
        str(row["SubjectID"]): row
        for _, row in best_df.iterrows()
    } if not best_df.empty else {}
    rows: List[Dict[str, Any]] = []
    grouped = {sid: sub.copy() for sid, sub in long_df.groupby("SubjectID")} if not long_df.empty else {}
    for _, target in targets.iterrows():
        sid = target["SubjectID"]
        sub = grouped.get(sid, pd.DataFrame())
        best = best_by_subject.get(sid)
        in_v5 = sid in v5_current_subjects
        in_gecn9_tensor = sid in gecn9_tensor_subjects
        in_gecn9_metadata = sid in gecn9_metadata_subjects
        included_v5_1 = in_gecn9_tensor and in_gecn9_metadata
        final_class, final_reason, martin_request = classify_subject(sid, included_v5_1, sub, best)
        local_file_count = int(
            (sub["path"].ne("NO_LOCAL_FILE") & sub["path_exists_now"].fillna(False).astype(bool)).sum()
        ) if not sub.empty and "path_exists_now" in sub.columns else 0
        compat_counts = (
            sub.loc[sub["path"].ne("NO_LOCAL_FILE"), "compatible_for_no_pybandpass_v5_1"]
            .fillna("no")
            .value_counts()
            .to_dict()
        ) if not sub.empty and "compatible_for_no_pybandpass_v5_1" in sub.columns else {}
        row = target.to_dict()
        row.update(
            {
                "in_v5_current": in_v5,
                "in_v5_1_gecn9_tensor": in_gecn9_tensor,
                "in_v5_1_gecn9_metadata": in_gecn9_metadata,
                "included_safely_in_v5_1_gecn9": included_v5_1,
                "local_roisignals_file_count": local_file_count,
                "local_compatible_yes_count": int(compat_counts.get("yes", 0)),
                "local_compatible_yes_with_warning_count": int(compat_counts.get("yes_with_warning", 0)),
                "local_compatible_no_count": int(compat_counts.get("no", 0)),
                "best_path": clean_string(best.get("path")) if best is not None else "",
                "best_stage_guess": clean_string(best.get("stage_guess")) if best is not None else "",
                "best_n_rois": clean_string(best.get("n_rois")) if best is not None else "",
                "best_n_timepoints": clean_string(best.get("n_timepoints")) if best is not None else "",
                "best_finite_fraction": clean_string(best.get("finite_fraction")) if best is not None else "",
                "best_scale_label": clean_string(best.get("scale_label")) if best is not None else "",
                "best_compatible_for_no_pybandpass_v5_1": clean_string(best.get("compatible_for_no_pybandpass_v5_1")) if best is not None else "no",
                "final_classification": final_class,
                "final_reason": final_reason,
                "martin_request": martin_request,
                "python_bandpass_applied": False,
            }
        )
        rows.append(row)
    subject_df = pd.DataFrame(rows)
    included = subject_df[subject_df["included_safely_in_v5_1_gecn9"].astype(bool)].copy()
    not_included = subject_df[~subject_df["included_safely_in_v5_1_gecn9"].astype(bool)].copy()
    ask = not_included.copy()
    return included, not_included, ask


def add_subject_context_to_long(
    long_df: pd.DataFrame,
    targets: pd.DataFrame,
    included: pd.DataFrame,
    not_included: pd.DataFrame,
) -> pd.DataFrame:
    subject_context = pd.concat([included, not_included], ignore_index=True, sort=False)
    identity_cols = [
        "SubjectID",
        "ImageID",
        "Age",
        "Sex",
        "Visit",
        "Description",
    ]
    status_cols = [
        "SubjectID",
        "in_v5_current",
        "in_v5_1_gecn9_tensor",
        "in_v5_1_gecn9_metadata",
        "included_safely_in_v5_1_gecn9",
        "final_classification",
        "martin_request",
    ]
    id_cols = [c for c in identity_cols if c in subject_context.columns]
    status_cols = [c for c in status_cols if c in subject_context.columns]
    out = long_df.copy()
    id_context = subject_context[id_cols].drop_duplicates("SubjectID") if id_cols else pd.DataFrame()
    if not id_context.empty:
        out = out.merge(id_context, on="SubjectID", how="left", suffixes=("", "_subject"))
        for col in [c for c in id_cols if c != "SubjectID"]:
            sub_col = f"{col}_subject"
            if sub_col not in out.columns:
                continue
            if col in out.columns:
                out[col] = out[col].where(out[col].map(clean_string).astype(bool), out[sub_col])
                out = out.drop(columns=[sub_col])
            else:
                out = out.rename(columns={sub_col: col})
    status_context = subject_context[status_cols].drop_duplicates("SubjectID") if status_cols else pd.DataFrame()
    if not status_context.empty:
        drop_status = [c for c in status_context.columns if c != "SubjectID" and c in out.columns]
        out = out.drop(columns=drop_status)
        out = out.merge(status_context, on="SubjectID", how="left")
    desired = [
        "SubjectID",
        "ImageID",
        "Age",
        "Sex",
        "Visit",
        "Description",
        "in_v5_current",
        "in_v5_1_gecn9_tensor",
        "in_v5_1_gecn9_metadata",
        "included_safely_in_v5_1_gecn9",
        "final_classification",
        "path",
        "stage_guess",
        "n_rois",
        "n_timepoints",
        "finite_fraction",
        "scale_label",
        "compatible_for_no_pybandpass_v5_1",
        "reason",
        "path_exists_now",
        "qc_source",
        "seen_in_previous_local_audit",
        "discovered_by_full_root_scan",
        "source_root",
        "suffix",
        "file_size_bytes",
        "mtime",
        "load_status",
        "shape",
        "orientation_status",
        "mean",
        "std",
        "median",
        "min",
        "max",
        "bandpassed_like",
        "energy_below_0p01",
        "energy_0p01_0p08",
        "energy_above_0p08",
        "spectral_status",
        "suspected_stage_compatibility",
        "martin_request",
    ]
    ordered = [c for c in desired if c in out.columns]
    rest = [c for c in out.columns if c not in ordered]
    return out[ordered + rest].sort_values(["SubjectID", "path"]).reset_index(drop=True)


def count_local_subjects(long_df: pd.DataFrame) -> int:
    if long_df.empty:
        return 0
    mask = long_df["path"].ne("NO_LOCAL_FILE") & long_df["path_exists_now"].fillna(False).astype(bool)
    return int(long_df.loc[mask, "SubjectID"].nunique())


def write_readme(
    output_dir: Path,
    cn_ge_inventory: pd.DataFrame,
    included: pd.DataFrame,
    not_included: pd.DataFrame,
    ask: pd.DataFrame,
    roots: Sequence[Path],
    command: Dict[str, Any],
) -> None:
    universe_n = int(cn_ge_inventory["SubjectID"].nunique()) if not cn_ge_inventory.empty else 0
    local_n = count_local_subjects(cn_ge_inventory)
    safe_n = int(included["SubjectID"].nunique()) if not included.empty else 0
    class_counts = not_included["final_classification"].value_counts().to_dict() if not not_included.empty else {}
    ask_lines = []
    for cls, sub in ask.groupby("final_classification", dropna=False):
        ask_lines.append(f"- `{cls}`: `{len(sub)}` subjects.")
    ask_text = "\n".join(ask_lines) if ask_lines else "- No remaining CN-GE request rows."
    roots_text = "\n".join(f"- `{root}`: `{'available' if root.exists() else 'missing'}`" for root in roots)
    lines = [
        "# Final CN-GE Local Inventory Before Martin Request",
        "",
        f"Generated: `{command['created']}`.",
        "",
        "Read-only audit. No training, no connectivity tensor construction, no Python bandpass, and no modification of existing v5/v5.1 tensors were performed.",
        "",
        "## Explicit Answers",
        "",
        f"- CN-GE subjects in the desired first-visit universe: `{universe_n}`.",
        f"- CN-GE subjects with local ROISignals `.mat`/`.txt`: `{local_n}`.",
        f"- CN-GE subjects safely included in v5.1 GECN9 tensor/metadata: `{safe_n}`.",
        f"- CN-GE subjects not included yet: `{int(not_included['SubjectID'].nunique()) if not not_included.empty else 0}`.",
        "- Are we applying Python bandpass? `No. Python bandpass is OFF in this audit and in the requested v5.1 path.`",
        "",
        "## Why Others Were Not Included",
        "",
        ask_text,
        "",
        "Classification meanings:",
        "",
        "- `already_local_but_stage_confirmation_needed`: ROISignals exist locally, but provenance/stage evidence is not enough for direct inclusion without confirmation.",
        "- `already_local_but_rejected_qc`: ROISignals exist locally but fail load, finite fraction, ROI-count, or scale QC.",
        "- `no_roisignals_found_needs_preprocessing`: no local ROISignals file was found in the audited roots.",
        "",
        "## What Exactly To Ask Martin",
        "",
        "Use `cn_ge_to_ask_martin.csv` as the subject-level request list.",
        "",
        "For `no_roisignals_found_needs_preprocessing`, ask Martin to provide first-visit DPARSF-compatible AAL3 `ROISignals` for those CN-GE subjects: 170 ROIs, finite signal, scale compatible with the 10000-level DPARSF outputs, and no extra Python bandpass.",
        "",
        "For `already_local_but_stage_confirmation_needed`, ask Martin to confirm whether the existing local ROISignals path is from a DPARSF-filtered AAL3 stage compatible with v5.1 no-Python-bandpass. If not, ask for a clean first-visit export.",
        "",
        "For `already_local_but_rejected_qc`, ask Martin for a fresh first-visit export/reprocess because the local file exists but fails the v5.1 inclusion QC.",
        "",
        "## Search Roots",
        "",
        roots_text,
        "",
        "## Output Files",
        "",
        "- `cn_ge_final_inventory.csv`: long inventory, one row per candidate file plus explicit `NO_LOCAL_FILE` rows.",
        "- `cn_ge_already_included_v5_1_gecn9.csv`: subjects already included in v5.1 GECN9 tensor/metadata.",
        "- `cn_ge_local_but_not_included.csv`: subject-level CN-GE not included in v5.1 GECN9.",
        "- `cn_ge_to_ask_martin.csv`: final subject-level ask list.",
        "- `command_log.json`: input paths, search roots, and no-training/no-bandpass flags.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    prepare_output_dir(args.output_dir, args.overwrite)
    roots = effective_search_roots(args.v5_1_data_link, args.v5_1_big_data)

    targets = load_cn_ge_universe(args.master_first_visit, args.preprocessing_request)
    long_df = build_candidate_long(targets, roots, args.local_audit_dir)

    v5_current_subjects = load_v5_current_subjects(args.v5_qc_dir)
    gecn9_tensor_subjects, gecn9_metadata_subjects, _alignment = load_gecn9_inclusion(
        args.gecn9_qc_dir,
        args.v5_1_data_link,
        args.v5_1_big_data,
    )
    included, not_included, ask = build_subject_tables(
        targets,
        long_df,
        v5_current_subjects,
        gecn9_tensor_subjects,
        gecn9_metadata_subjects,
    )
    final_inventory = add_subject_context_to_long(long_df, targets, included, not_included)

    final_inventory.to_csv(args.output_dir / "cn_ge_final_inventory.csv", index=False)
    included.to_csv(args.output_dir / "cn_ge_already_included_v5_1_gecn9.csv", index=False)
    not_included.to_csv(args.output_dir / "cn_ge_local_but_not_included.csv", index=False)
    ask.to_csv(args.output_dir / "cn_ge_to_ask_martin.csv", index=False)

    command = {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "script": str(Path(__file__).resolve()),
        "master_first_visit": str(args.master_first_visit),
        "preprocessing_request": str(args.preprocessing_request),
        "augmented_dir": str(args.augmented_dir),
        "local_audit_dir": str(args.local_audit_dir),
        "gecn9_qc_dir": str(args.gecn9_qc_dir),
        "v5_qc_dir": str(args.v5_qc_dir),
        "v5_1_data_link": str(args.v5_1_data_link),
        "v5_1_big_data": str(args.v5_1_big_data),
        "output_dir": str(args.output_dir),
        "search_roots": [str(root) for root in roots],
        "python_bandpass_applied": False,
        "training_run": False,
        "connectivity_tensor_constructed": False,
        "existing_tensors_modified": False,
    }
    (args.output_dir / "command_log.json").write_text(
        json.dumps(command, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_readme(args.output_dir, final_inventory, included, not_included, ask, roots, command)

    class_counts = not_included["final_classification"].value_counts().to_dict() if not not_included.empty else {}
    print(f"Wrote final CN-GE inventory to {args.output_dir}")
    print(f"cn_ge_universe={targets['SubjectID'].nunique()}")
    print(f"local_roisignals_subjects={count_local_subjects(final_inventory)}")
    print(f"included_v5_1_gecn9={included['SubjectID'].nunique() if not included.empty else 0}")
    print(f"not_included_classes={class_counts}")
    print("Python bandpass applied: NO. No training run. No tensors modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
