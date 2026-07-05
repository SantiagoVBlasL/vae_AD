#!/usr/bin/env python3
"""Trace historical ADNI ROI/BOLD signal sources used around the paper tensor.

Read-only audit:
- no moving/copying/deleting source data;
- no connectivity computation;
- no training;
- global tensors are inspected only for small metadata keys/NPZ headers.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_dparsf_only_rebuild"
    / "historical_roi_signal_trace"
)

SEARCH_ROOTS = [
    Path("/home/diego/proyectos"),
    Path("/media/diego/My_Book_Diego"),
]

REFERENCE_METADATA = [
    Path("/home/diego/proyectos/betavae-xai-ad/data/SubjectsData_AAL3_procesado2.csv"),
    PROJECT_ROOT / "data" / "SubjectsData_AAL3_procesado2.csv",
]

HISTORICAL_TENSOR_DIRS = [
    Path(
        "/home/diego/proyectos/betavae-xai-ad/data/"
        "AAL3_dynamicROIs_fmri_tensor_NeuroEnhanced_v6.5.17_AAL3_131ROIs_"
        "OMST_GCE_Signed_GrangerLag1_ChNorm_ROIreorderedYeo17_ParallelTuned"
    ),
    PROJECT_ROOT
    / "data"
    / "AAL3_dynamicROIs_fmri_tensor_NeuroEnhanced_v6.5.17_AAL3_131ROIs_"
    "OMST_GCE_Signed_GrangerLag1_ChNorm_ROIreorderedYeo17_ParallelTuned",
]

HISTORICAL_MODULE = Path("/home/diego/proyectos/betavae-xai-ad/src/betavae_xai/feature_extraction_manual.py")
HISTORICAL_DEFAULT_DIRNAME = "ROISignals_AAL3_NiftiPreprocessedAllBatchesNorm"
HISTORICAL_TENSOR_DIR_TOKEN = "AAL3_dynamicROIs_fmri_tensor_NeuroEnhanced_v6.5.17"

SUBJECT_RE = re.compile(r"(?<!\d)(\d{3}_S_\d{4})(?!\d)", re.IGNORECASE)
ROI_FILE_RE = re.compile(r"^ROISignals_.*?(\d{3}_S_\d{4}).*\.(mat|txt)$", re.IGNORECASE)
TENSOR_RE = re.compile(r"GLOBAL_TENSOR_from_.*\.npz$", re.IGNORECASE)

SMALL_NPZ_KEYS = {
    "subject_ids",
    "SubjectID",
    "SubjectIDs",
    "subjects",
    "channel_names",
    "roi_names_in_order",
    "network_labels_in_order",
    "rois_count",
    "target_len_ts",
    "tr_seconds",
    "filter_low_hz",
    "filter_high_hz",
    "python_bandpass_applied",
    "source_preprocessing",
}

PRUNE_DIR_NAMES = {
    ".git",
    ".cache",
    "__pycache__",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    "wandb",
    "checkpoints",
    "checkpoint",
    "model_checkpoints",
    "trained_models",
    "plots",
    "figures",
    "lost+found",
}

SKIP_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".joblib",
    ".pkl",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".html",
    ".pdf",
    ".nii",
    ".gz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace historical ADNI ROI signal source folders and coverage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--roots", type=Path, nargs="*", default=SEARCH_ROOTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-shape-samples-per-folder", type=int, default=3)
    parser.add_argument("--max-txt-shape-mb", type=float, default=12.0)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(f"Output directory exists and is not empty; pass --overwrite: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def normalize_subject(value: Any) -> str:
    if pd.isna(value):
        return ""
    match = SUBJECT_RE.search(str(value).strip().upper())
    return match.group(1).upper() if match else ""


def first_existing_col(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower = {str(col).strip().lower(): str(col) for col in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        hit = lower.get(candidate.lower())
        if hit:
            return hit
    return None


def read_reference_subjects() -> Tuple[set[str], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    subjects: set[str] = set()
    for path in REFERENCE_METADATA:
        row = {"path": str(path), "exists": path.exists(), "n_subjects": 0, "status": ""}
        if not path.exists():
            row["status"] = "missing"
            rows.append(row)
            continue
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False)
            sid_col = first_existing_col(df.columns, ["SubjectID", "Subject", "PTID", "subject_id"])
            if sid_col is None:
                row["status"] = "no_subject_column"
            else:
                vals = set(df[sid_col].map(normalize_subject).replace("", np.nan).dropna().astype(str))
                row["n_subjects"] = len(vals)
                row["status"] = "ok"
                subjects.update(vals)
        except Exception as exc:
            row["status"] = f"failed: {exc}"
        rows.append(row)
    return subjects, rows


def should_prune(path: Path) -> bool:
    if path.name in PRUNE_DIR_NAMES:
        return True
    low = str(path).lower()
    if "/site-packages/" in low or "/.conda/" in low or "/envs/" in low:
        return True
    return False


def is_roi_signal_file(path: Path) -> bool:
    if path.suffix.lower() not in {".mat", ".txt"}:
        return False
    if ROI_FILE_RE.match(path.name):
        return True
    return "ROISignals" in str(path) and SUBJECT_RE.search(path.name) is not None


def is_candidate_tensor(path: Path) -> bool:
    return path.suffix.lower() == ".npz" and bool(TENSOR_RE.match(path.name))


def candidate_dir_from_file(path: Path) -> Path:
    for parent in [path.parent, *path.parents]:
        if parent.name.startswith("ROISignals"):
            return parent
        if parent.name == "ResultsAAL3":
            return path.parent
    return path.parent


def path_tokens(path: Path) -> List[str]:
    terms = [
        HISTORICAL_DEFAULT_DIRNAME,
        "ROISignals_AAL3",
        "ROISignals",
        "ResultsAAL3",
        "FunImgARWSDCFN",
        "FunImgARWSDCF",
        "ARWSDCFN",
        "ARWSDCF",
        "ARWSDC",
        "NiftiPreprocessedAllBatchesNorm",
        "bandpass",
        "passband",
        "pasabandas",
        "adni_passband_20260510",
        "MARTIN59",
        "PHILIPS",
        "smoke",
        "test",
        "AAL3",
    ]
    low = str(path).lower()
    return list(dict.fromkeys(term for term in terms if term.lower() in low))


def classify_folder(path: Path, tokens: Sequence[str]) -> str:
    low = str(path).lower()
    token_low = {t.lower() for t in tokens}
    if HISTORICAL_DEFAULT_DIRNAME.lower() in low:
        return "historical_original_default"
    if "smoke" in token_low or "smoke" in low or re.search(r"/test(_|/|$)", low):
        return "smoke_test"
    if {"funimgarwsdcfn", "arwsdcfn", "bandpass", "passband", "pasabandas"}.intersection(token_low):
        return "dparsf_passband"
    if "niftipreprocessedallbatchesnorm" in token_low or "cov" in low or "regress" in low:
        return "covregressed_no_bandpass"
    if {"martin59", "philips", "adni_passband_20260510"}.intersection(token_low):
        return "expansion_batch"
    return "unknown"


def scan_files(roots: Sequence[Path]) -> Tuple[List[Path], List[Path], List[Path]]:
    roi_files: List[Path] = []
    tensor_files: List[Path] = []
    named_dirs: List[Path] = []
    for root in roots:
        root = resolve(root)
        if not root.exists():
            continue
        for current, dirs, files in os.walk(root, followlinks=False):
            current_path = Path(current)
            if (
                current_path.name == HISTORICAL_DEFAULT_DIRNAME
                or current_path.name.startswith("ROISignals_AAL3")
                or "ResultsAAL3" in current_path.parts and current_path.name.startswith("ROISignals")
                or HISTORICAL_TENSOR_DIR_TOKEN in current_path.name
            ):
                named_dirs.append(current_path)
            dirs[:] = [name for name in dirs if not should_prune(current_path / name)]
            for name in files:
                path = current_path / name
                suffix = path.suffix.lower()
                if suffix in SKIP_SUFFIXES and suffix != ".npz":
                    continue
                if is_roi_signal_file(path):
                    roi_files.append(path)
                elif is_candidate_tensor(path):
                    tensor_files.append(path)
    for tensor_dir in HISTORICAL_TENSOR_DIRS:
        if tensor_dir.exists():
            named_dirs.append(tensor_dir)
            tensor_files.extend(sorted(tensor_dir.glob("GLOBAL_TENSOR_from_*.npz")))
    return sorted(set(roi_files)), sorted(set(tensor_files)), sorted(set(named_dirs))


def read_txt_shape(path: Path, max_mb: float) -> Dict[str, Any]:
    try:
        if path.stat().st_size > max_mb * 1024 * 1024:
            return {"shape": "", "rows": np.nan, "cols": np.nan, "status": f"not_read_gt_{max_mb:g}mb"}
        try:
            arr = np.loadtxt(path, delimiter=",")
            status = "ok_comma"
        except Exception:
            arr = np.loadtxt(path)
            status = "ok_whitespace"
        return {
            "shape": str(tuple(int(x) for x in arr.shape)),
            "rows": int(arr.shape[0]) if arr.ndim >= 1 else 1,
            "cols": int(arr.shape[1]) if arr.ndim >= 2 else 1,
            "status": status,
        }
    except Exception as exc:
        return {"shape": "", "rows": np.nan, "cols": np.nan, "status": f"failed: {exc}"}


def read_mat_shape(path: Path) -> Dict[str, Any]:
    try:
        import scipy.io  # type: ignore

        entries = scipy.io.whosmat(path)
        if not entries:
            return {"shape": "", "rows": np.nan, "cols": np.nan, "status": "mat_no_vars"}
        preferred = sorted(entries, key=lambda item: (0 if "signal" in item[0].lower() or "roi" in item[0].lower() else 1, item[0]))
        name, shape, klass = preferred[0]
        return {
            "shape": str(tuple(int(x) for x in shape)),
            "rows": int(shape[0]) if len(shape) >= 1 else 1,
            "cols": int(shape[1]) if len(shape) >= 2 else 1,
            "status": f"ok_whosmat:{name}:{klass}",
        }
    except Exception as exc:
        return {"shape": "", "rows": np.nan, "cols": np.nan, "status": f"failed: {exc}"}


def build_folder_tables(
    roi_files: Sequence[Path],
    named_dirs: Sequence[Path],
    tensor_subjects: set[str],
    metadata_subjects: set[str],
    max_samples: int,
    max_txt_mb: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grouped: Dict[Path, List[Path]] = {}
    for file_path in roi_files:
        grouped.setdefault(candidate_dir_from_file(file_path), []).append(file_path)
    for folder in named_dirs:
        grouped.setdefault(folder, [])

    dir_rows: List[Dict[str, Any]] = []
    shape_rows: List[Dict[str, Any]] = []
    coverage_rows: List[Dict[str, Any]] = []
    missing_rows: List[Dict[str, Any]] = []
    reference_431 = tensor_subjects or metadata_subjects

    for folder, files in sorted(grouped.items(), key=lambda item: str(item[0])):
        subjects = sorted({normalize_subject(path.name) for path in files if normalize_subject(path.name)})
        tokens = path_tokens(folder)
        txt_files = [path for path in files if path.suffix.lower() == ".txt"]
        mat_files = [path for path in files if path.suffix.lower() == ".mat"]
        mtimes = [path.stat().st_mtime for path in files if path.exists()]
        folder_exists = folder.exists()
        folder_mtime = folder.stat().st_mtime if folder_exists else np.nan
        classification = classify_folder(folder, tokens)
        matching = sorted(set(subjects).intersection(reference_431))
        missing = sorted(reference_431 - set(subjects))
        extra = sorted(set(subjects) - reference_431)
        dir_rows.append(
            {
                "folder": str(folder),
                "exists": folder_exists,
                "classification": classification,
                "provenance_tokens": "|".join(tokens),
                "n_files": len(files),
                "n_txt": len(txt_files),
                "n_mat": len(mat_files),
                "n_subjects_found": len(subjects),
                "mtime_folder": timestamp(folder_mtime),
                "mtime_files_min": timestamp(min(mtimes)) if mtimes else "",
                "mtime_files_max": timestamp(max(mtimes)) if mtimes else "",
                "sample_subjects": "|".join(subjects[:10]),
            }
        )
        coverage_rows.append(
            {
                "folder": str(folder),
                "classification": classification,
                "n_subjects_found": len(subjects),
                "reference_source": "historical_tensor_subject_ids" if tensor_subjects else "SubjectsData_AAL3_procesado2",
                "n_reference_subjects": len(reference_431),
                "n_matching_original_431": len(matching),
                "n_missing_original_431": len(missing),
                "n_extra": len(extra),
                "coverage_fraction": (len(matching) / len(reference_431)) if reference_431 else np.nan,
            }
        )
        for sid in missing[:1000]:
            missing_rows.append(
                {
                    "folder": str(folder),
                    "classification": classification,
                    "SubjectID": sid,
                }
            )
        sample_files = sorted(txt_files)[:max_samples] + sorted(mat_files)[:max_samples]
        for sample in sample_files:
            shape = read_txt_shape(sample, max_txt_mb) if sample.suffix.lower() == ".txt" else read_mat_shape(sample)
            shape_rows.append(
                {
                    "folder": str(folder),
                    "SubjectID": normalize_subject(sample.name),
                    "path": str(sample),
                    "suffix": sample.suffix.lower(),
                    "file_size_bytes": sample.stat().st_size,
                    "shape": shape["shape"],
                    "rows": shape["rows"],
                    "cols": shape["cols"],
                    "status": shape["status"],
                }
            )
    return pd.DataFrame(dir_rows), pd.DataFrame(coverage_rows), pd.DataFrame(missing_rows), pd.DataFrame(shape_rows)


def timestamp(value: Any) -> str:
    if pd.isna(value):
        return ""
    try:
        return pd.Timestamp(float(value), unit="s").isoformat()
    except Exception:
        return ""


def _read_npy_header_from_npz(npz_path: Path, key: str) -> Tuple[str, str]:
    member = f"{key}.npy"
    try:
        with zipfile.ZipFile(npz_path) as zf:
            if member not in zf.namelist():
                return "", ""
            with zf.open(member) as handle:
                version = np.lib.format.read_magic(handle)
                if version == (1, 0):
                    shape, _fortran, dtype = np.lib.format.read_array_header_1_0(handle)
                elif version == (2, 0):
                    shape, _fortran, dtype = np.lib.format.read_array_header_2_0(handle)
                else:
                    shape, _fortran, dtype = np.lib.format._read_array_header(handle, version)
                return str(tuple(int(x) for x in shape)), str(dtype)
    except Exception:
        return "", ""


def read_small_npz_metadata(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        with np.load(path, allow_pickle=False) as zf:
            for key in zf.files:
                if key not in SMALL_NPZ_KEYS:
                    continue
                arr = zf[key]
                if arr.shape == ():
                    out[key] = str(arr.tolist())
                elif arr.ndim == 1 and arr.size <= 600:
                    out[key] = "|".join(map(str, arr.tolist()))
                else:
                    out[key] = f"shape={arr.shape}"
    except Exception as exc:
        out["small_metadata_status"] = f"failed: {exc}"
    return out


def inspect_tensors(tensor_files: Sequence[Path]) -> Tuple[pd.DataFrame, set[str], str]:
    rows: List[Dict[str, Any]] = []
    reference_candidates: List[Tuple[int, str, set[str]]] = []
    explicit_reference_paths = []
    for folder in HISTORICAL_TENSOR_DIRS:
        explicit_reference_paths.extend(sorted(folder.glob("GLOBAL_TENSOR_from_*.npz")) if folder.exists() else [])
        if not folder.exists():
            rows.append(
                {
                    "path": str(folder / "GLOBAL_TENSOR_from_*.npz"),
                    "exists": False,
                    "is_explicit_reference": True,
                    "status": "reference_dir_missing",
                }
            )
    explicit_set = {str(path) for path in explicit_reference_paths}
    for path in sorted(set(tensor_files) | set(explicit_reference_paths)):
        meta = read_small_npz_metadata(path)
        tensor_shape = ""
        tensor_dtype = ""
        for key in ["global_tensor_data", "tensor_data", "X", "arr_0"]:
            tensor_shape, tensor_dtype = _read_npy_header_from_npz(path, key)
            if tensor_shape:
                break
        subject_ids = []
        for key in ["subject_ids", "SubjectID", "SubjectIDs", "subjects"]:
            if key in meta:
                subject_ids = [normalize_subject(x) for x in str(meta[key]).split("|")]
                subject_ids = [sid for sid in subject_ids if sid]
                break
        is_explicit_reference = str(path) in explicit_set or any(str(path).startswith(str(folder)) for folder in HISTORICAL_TENSOR_DIRS)
        if is_explicit_reference and path.name.startswith("GLOBAL_TENSOR_from_") and len(subject_ids) >= 100:
            reference_candidates.append((len(subject_ids), str(path), set(subject_ids)))
        rows.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "is_explicit_reference": is_explicit_reference,
                "used_as_subject_reference": False,
                "status": "ok" if path.exists() else "missing",
                "tensor_shape_header": tensor_shape,
                "tensor_dtype_header": tensor_dtype,
                "n_subject_ids": len(subject_ids),
                "subject_ids_preview": "|".join(subject_ids[:12]),
                "channel_names": meta.get("channel_names", ""),
                "rois_count": meta.get("rois_count", ""),
                "target_len_ts": meta.get("target_len_ts", ""),
                "tr_seconds": meta.get("tr_seconds", ""),
                "filter_low_hz": meta.get("filter_low_hz", ""),
                "filter_high_hz": meta.get("filter_high_hz", ""),
                "python_bandpass_applied": meta.get("python_bandpass_applied", ""),
                "source_preprocessing": meta.get("source_preprocessing", ""),
            }
        )
    reference_subjects: set[str] = set()
    reference_path = ""
    if reference_candidates:
        _count, reference_path, reference_subjects = sorted(reference_candidates, reverse=True)[0]
        for row in rows:
            if row["path"] == reference_path:
                row["used_as_subject_reference"] = True
    return pd.DataFrame(rows), reference_subjects, reference_path


def rank_candidate_folders(dirs: pd.DataFrame, coverage: pd.DataFrame, shapes: pd.DataFrame) -> pd.DataFrame:
    if dirs.empty:
        return pd.DataFrame()
    df = dirs.merge(coverage, on=["folder", "classification"], how="left", suffixes=("", "_coverage"))
    shape_summary = shapes.groupby("folder", dropna=False).agg(
        sample_cols=("cols", lambda s: "|".join(sorted({str(int(x)) for x in pd.to_numeric(s, errors="coerce").dropna()}))),
        sample_shapes=("shape", lambda s: "|".join(sorted({str(x) for x in s if str(x)}))),
    ).reset_index() if not shapes.empty else pd.DataFrame(columns=["folder", "sample_cols", "sample_shapes"])
    df = df.merge(shape_summary, on="folder", how="left")
    class_bonus = {
        "historical_original_default": 10000,
        "dparsf_passband": 100,
        "expansion_batch": 50,
        "unknown": 0,
        "covregressed_no_bandpass": -500,
        "smoke_test": -10000,
    }
    df["rank_score"] = (
        df["classification"].map(class_bonus).fillna(0)
        + pd.to_numeric(df["n_matching_original_431"], errors="coerce").fillna(0) * 100
        + pd.to_numeric(df["coverage_fraction"], errors="coerce").fillna(0) * 1000
        + pd.to_numeric(df["n_subjects_found"], errors="coerce").fillna(0)
        + pd.to_numeric(df["n_mat"], errors="coerce").fillna(0) * 0.1
    )
    return df.sort_values(["rank_score", "n_matching_original_431", "n_subjects_found"], ascending=False)


def write_readme(
    path: Path,
    dirs: pd.DataFrame,
    tensors: pd.DataFrame,
    ranked: pd.DataFrame,
    metadata_rows: List[Dict[str, Any]],
    tensor_subjects: set[str],
    tensor_reference_path: str,
    metadata_subjects: set[str],
) -> None:
    ref_n = len(tensor_subjects) or len(metadata_subjects)
    historical_defaults = ranked[ranked["classification"].eq("historical_original_default")] if not ranked.empty else pd.DataFrame()
    fullish_coverage = ranked[pd.to_numeric(ranked["coverage_fraction"], errors="coerce").fillna(0) >= 0.95] if not ranked.empty else pd.DataFrame()
    confirmed_original = historical_defaults.iloc[0].to_dict() if not historical_defaults.empty else {}
    best_available = ranked.iloc[0].to_dict() if not ranked.empty else {}
    if confirmed_original:
        original_answer = f"`{confirmed_original.get('folder')}`"
        original_exists = bool(confirmed_original.get("exists", False))
        original_location = "media" if "/media/" in str(confirmed_original.get("folder", "")) else (
            "home" if "/home/" in str(confirmed_original.get("folder", "")) else "unknown"
        )
        original_match = int(confirmed_original.get("n_matching_original_431", 0) or 0)
        original_missing = int(confirmed_original.get("n_missing_original_431", 0) or 0)
        original_cols = str(confirmed_original.get("sample_cols", ""))
    elif not fullish_coverage.empty:
        possible = fullish_coverage.iloc[0].to_dict()
        original_answer = f"`{possible.get('folder')}` (possible moved/renamed source; expected dirname not found)"
        original_exists = bool(possible.get("exists", False))
        original_location = "media" if "/media/" in str(possible.get("folder", "")) else (
            "home" if "/home/" in str(possible.get("folder", "")) else "unknown"
        )
        original_match = int(possible.get("n_matching_original_431", 0) or 0)
        original_missing = int(possible.get("n_missing_original_431", 0) or 0)
        original_cols = str(possible.get("sample_cols", ""))
    else:
        original_answer = f"`NOT FOUND` (expected `{HISTORICAL_DEFAULT_DIRNAME}` was not detected)"
        original_exists = False
        original_location = "not_found"
        original_match = 0
        original_missing = ref_n
        original_cols = ""

    best_folder = best_available.get("folder", "NONE")
    best_class = best_available.get("classification", "")
    best_match = int(best_available.get("n_matching_original_431", 0) or 0) if best_available else 0
    best_missing = int(best_available.get("n_missing_original_431", 0) or 0) if best_available else 0
    best_cols = str(best_available.get("sample_cols", ""))
    historical_tensor_found = bool(tensor_reference_path)
    module_exists = HISTORICAL_MODULE.exists()
    lines = [
        "# Historical ROI Signal Source Trace",
        "",
        "Read-only audit. No files were moved/copied/deleted, no connectivity was computed, and no training was run.",
        "",
        "## Explicit Answers",
        "",
        f"- Most likely original ROI signal folder: {original_answer}",
        f"- Historical default module path exists? `{'YES' if module_exists else 'NO'}` (`{HISTORICAL_MODULE}`)",
        f"- Does the confirmed original folder still exist? `{'YES' if original_exists else 'NO'}`",
        f"- Confirmed-original location appears to be: `{original_location}`",
        f"- Historical tensor metadata found? `{'YES' if historical_tensor_found else 'NO'}`",
        f"- Historical tensor subject reference used: `{tensor_reference_path or 'NONE'}`",
        f"- Reference subject count used for coverage: `{ref_n}` (`{'tensor subject_ids' if tensor_subjects else 'SubjectsData_AAL3_procesado2.csv'}`)",
        f"- Confirmed-original coverage: `{original_match}/{ref_n}` matched; missing `{original_missing}`",
        f"- Confirmed-original sample ROI columns: `{original_cols or 'unknown'}`",
        f"- Best available ROI folder by conservative rank: `{best_folder}`",
        f"- Best-available classification: `{best_class}`",
        f"- Best-available coverage: `{best_match}/{ref_n}` matched; missing `{best_missing}`",
        f"- Best-available sample ROI columns: `{best_cols or 'unknown'}`",
        f"- Signals appear 170 raw ROIs or 131 reduced? `{'170_raw_like' if '170' in (original_cols or best_cols) else ('131_reduced_like' if '131' in (original_cols or best_cols) else 'unknown')}`",
        f"- Should the historical original folder be used for v5 DPARSF-bandpass-only? `NO`: it is absent here, and even if recovered it belongs to the historical Python-bandpass branch unless Martin confirms it is DPARSF-bandpass.",
        "- Missing data for DPARSF-bandpass-only rebuild: DPARSF-bandpass ROI signals for every retained final-training subject, especially all v4 subjects currently without confirmed DPARSF-bandpass `.txt` signals.",
        "",
        "## Recommended Next Commands",
        "",
        "```bash",
        "/home/diego/anaconda3/envs/vae_ad/bin/python scripts/revision_bspc_2026/find_all_adni_roi_signal_sources.py --overwrite",
        "/home/diego/anaconda3/envs/vae_ad/bin/python scripts/revision_bspc_2026/build_adni_v5_dparsf_only_subject_manifest.py --overwrite",
        "```",
        "",
        "## Reference Metadata Paths",
        "",
    ]
    for row in metadata_rows:
        lines.append(f"- `{row['path']}` exists=`{row['exists']}` status=`{row['status']}` n_subjects=`{row['n_subjects']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    prepare_output_dir(output_dir, args.overwrite)
    roots = [resolve(root) for root in args.roots]

    metadata_subjects, metadata_rows = read_reference_subjects()
    roi_files, tensor_files, named_dirs = scan_files(roots)
    tensors, tensor_subjects, tensor_reference_path = inspect_tensors(tensor_files)
    dirs, coverage, missing, shapes = build_folder_tables(
        roi_files,
        named_dirs,
        tensor_subjects,
        metadata_subjects,
        args.max_shape_samples_per_folder,
        args.max_txt_shape_mb,
    )
    ranked = rank_candidate_folders(dirs, coverage, shapes)

    dirs.to_csv(output_dir / "roi_signal_source_directories.csv", index=False)
    tensors.to_csv(output_dir / "historical_tensor_locations.csv", index=False)
    coverage.to_csv(output_dir / "folder_coverage_vs_original431.csv", index=False)
    ranked.to_csv(output_dir / "candidate_original_roi_folder_ranked.csv", index=False)
    missing.to_csv(output_dir / "missing_original_subjects_by_folder.csv", index=False)
    shapes.to_csv(output_dir / "sample_shapes_by_folder.csv", index=False)
    pd.DataFrame(metadata_rows).to_csv(output_dir / "reference_metadata_status.csv", index=False)
    write_readme(output_dir / "README.md", dirs, tensors, ranked, metadata_rows, tensor_subjects, tensor_reference_path, metadata_subjects)

    print(f"Wrote historical ROI trace to {output_dir}")
    print(f"ROI files found: {len(roi_files)}")
    print(f"Candidate folders: {len(dirs)}")
    print(f"Tensor candidates: {len(tensors)}")
    if not ranked.empty:
        print("Top candidate:")
        print(ranked[["folder", "classification", "n_subjects_found", "n_matching_original_431", "n_missing_original_431", "rank_score"]].head(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
