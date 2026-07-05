#!/usr/bin/env python3
"""Audit local CN-GE ROI signals for an ADNI v5.1 no-Python-bandpass rebuild.

Read-only audit:
- no extraction from zip files;
- no full connectivity tensor construction;
- no model training;
- no source file modifications.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.io as scipy_io
from scipy.signal import welch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from build_v5_dparsf10000_no_pybandpass_manifest_and_extract import (
        build_roi_reduction_and_order,
        pearson_full,
        preprocess_timeseries_no_pybandpass,
    )
except Exception as exc:  # pragma: no cover - reported in output if needed
    build_roi_reduction_and_order = None
    pearson_full = None
    preprocess_timeseries_no_pybandpass = None
    IMPORT_ERROR = str(exc)
else:
    IMPORT_ERROR = ""


DEFAULT_FIRST_VISIT = (
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
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_gecn_local_roisignals_audit"
)
DEFAULT_V4_TENSOR = (
    PROJECT_ROOT
    / "data"
    / "AAL3_dynamicROIs_fmri_tensor_NeuroEnhanced_v6.5.17_AAL3_131ROIs_OMST_GCE_Signed_GrangerLag1_ChNorm_ROIreorderedYeo17_ParallelTuned"
    / "GLOBAL_TENSOR_from_AAL3_dynamicROIs_fmri_tensor_NeuroEnhanced_v6.5.17_AAL3_131ROIs_OMST_GCE_Signed_GrangerLag1_ChNorm_ROIreorderedYeo17_ParallelTuned.npz"
)

ROISIGNAL_ROOTS = [
    Path("/media/diego/Datos/adni_expansion/MARTIN_20260429_PHILIPS10/OneDrive_2_29-4-2026/ResultsAAL3/ROISignals_AAL3_FunImgARWSDCF"),
    Path("/media/diego/Datos/adni_expansion/GE_batch7/ROISignals_AAL3_from_CovRegressed_GE_batch7"),
    Path("/media/diego/Datos/adni_expansion/GE_smoketest3/ROISignals_AAL3_from_CovRegressed_GE_smoketest3"),
    Path("/media/diego/Datos/adni_bridge_expansion/dparsf_single/GE/FunImgARWSDCovs"),
    Path("/media/diego/Datos/adni_bridge_expansion/dparsf_single/GE/Results/ROISignals_FunImgARWSDC"),
    Path("/home/diego/proyectos/vae_AD/data/OneDrive_1_27-4-2026/ResultsAAL3/ROISignals_AAL3_FunImgARWSDCF"),
    Path("/media/diego/Datos/desde_cero/ROISignalsAAL3"),
    Path("/media/diego/Datos/AAL3/ROISignalsAAL3"),
    Path("/media/diego/Datos/AAL3_paper/ROISignalsAAL3"),
    Path("/media/diego/Datos/june_paper/ROISignalsAAL3"),
    Path("/media/diego/My_Book_Diego/vae_AD_data/adni_passband_20260510/ResultsAAL3/ROISignals_AAL3_FunImgARWSDCFN"),
]

ZIP_SEARCH_ROOTS = [
    PROJECT_ROOT / "data",
    Path("/media/diego/Datos/adni_expansion"),
    Path("/media/diego/Datos/adni_bridge_expansion"),
    Path("/media/diego/Datos/desde_cero"),
    Path("/media/diego/Datos/AAL3"),
    Path("/media/diego/Datos/AAL3_paper"),
    Path("/media/diego/Datos/june_paper"),
    Path("/media/diego/My_Book_Diego/vae_AD_data"),
]

PRIORITY19_CN_GE = [
    "005_S_0602",
    "005_S_0610",
    "005_S_6084",
    "005_S_6093",
    "009_S_0751",
    "009_S_6163",
    "009_S_6212",
    "009_S_6286",
    "010_S_6567",
    "135_S_6473",
    "135_S_6509",
    "135_S_6510",
    "135_S_4446",
    "135_S_4598",
    "135_S_5113",
    "135_S_6104",
    "135_S_6359",
    "135_S_6360",
    "135_S_6411",
]

SUBJECT_RE = re.compile(r"(?<!\d)(\d{3}_S_\d{4})(?!\d)", re.IGNORECASE)
TR_SECONDS = 3.0
RAW_ROIS = 170


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit local CN-GE ROISignals for ADNI v5.1 no-Python-bandpass.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--first-visit-manifest", type=Path, default=DEFAULT_FIRST_VISIT)
    parser.add_argument("--preprocessing-request", type=Path, default=DEFAULT_PREPROCESSING_REQUEST)
    parser.add_argument("--v4-tensor", type=Path, default=DEFAULT_V4_TENSOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-zip-scan", action="store_true")
    parser.add_argument("--skip-v4-smoke", action="store_true")
    return parser.parse_args()


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(f"Output directory exists and is not empty; pass --overwrite: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def clean_string(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def normalize_subject(value: Any) -> str:
    match = SUBJECT_RE.search(clean_string(value).upper())
    return match.group(1).upper() if match else ""


def canonical_manufacturer(value: Any) -> str:
    text = clean_string(value).upper()
    if "GE" in text:
        return "GE"
    if "PHILIPS" in text:
        return "Philips"
    if "SIEMENS" in text:
        return "SIEMENS"
    return clean_string(value) or "UNKNOWN"


def normalize_group(value: Any) -> str:
    text = clean_string(value).upper()
    if text in {"CN", "AD", "MCI"}:
        return text
    if "CONTROL" in text or text == "NORMAL":
        return "CN"
    if "DEMENT" in text or "ALZ" in text:
        return "AD"
    if "MCI" in text:
        return "MCI"
    return ""


def load_target_cn_ge(first_visit: Path, preprocessing_request: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path, source in [(first_visit, "first_visit_manifest"), (preprocessing_request, "preprocessing_request")]:
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        df["SubjectID"] = df["SubjectID"].map(normalize_subject)
        df["ResearchGroup_Mapped"] = df.get("ResearchGroup_Mapped", "").map(normalize_group)
        df["Manufacturer"] = df.get("Manufacturer", "").map(canonical_manufacturer)
        df["target_source"] = source
        frames.append(df)
    if frames:
        all_rows = pd.concat(frames, ignore_index=True, sort=False)
    else:
        all_rows = pd.DataFrame(columns=["SubjectID", "ResearchGroup_Mapped", "Manufacturer"])
    mask = all_rows["SubjectID"].astype(bool) & all_rows["ResearchGroup_Mapped"].eq("CN") & all_rows["Manufacturer"].eq("GE")
    targets = all_rows.loc[mask].drop_duplicates("SubjectID", keep="first").copy()
    for sid in PRIORITY19_CN_GE:
        if sid not in set(targets["SubjectID"]):
            targets = pd.concat(
                [
                    targets,
                    pd.DataFrame(
                        [
                            {
                                "SubjectID": sid,
                                "ResearchGroup_Mapped": "CN",
                                "Manufacturer": "GE",
                                "target_source": "priority19_fallback",
                            }
                        ]
                    ),
                ],
                ignore_index=True,
                sort=False,
            )
    targets["is_priority19_v4_cn_ge"] = targets["SubjectID"].isin(PRIORITY19_CN_GE)
    return targets.sort_values("SubjectID").reset_index(drop=True)


def stage_guess(path: Path) -> str:
    text = str(path)
    upper = text.upper()
    if "ARWSDCFN" in upper:
        return "ARWSDCFN"
    if "ARWSDCF" in upper:
        return "ARWSDCF"
    if "COVREGRESSED" in upper:
        return "CovRegressed"
    if "ARWSDC" in upper or "FUNIMGARWSDCOVS" in upper:
        return "ARWSDC"
    if any(token in upper for token in ["/DESDE_CERO/", "/AAL3/ROISIGNALSAAL3", "/AAL3_PAPER/", "/JUNE_PAPER/"]):
        return "historical_10000"
    return "unknown"


def explicit_has_f_stage(stage: str, path: Path) -> bool:
    text = str(path).upper()
    return stage in {"ARWSDCFN", "ARWSDCF"} or "FUNIMGARWSDCF" in text


def source_root_for(path: Path, roots: Sequence[Path]) -> str:
    path_str = str(path)
    matches = [str(root) for root in roots if path_str.startswith(str(root))]
    return max(matches, key=len) if matches else ""


def dedupe_roots(roots: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            out.append(root)
    return out


def discover_files(target_subjects: Sequence[str], roots: Sequence[Path]) -> pd.DataFrame:
    targets = set(target_subjects)
    rows: List[Dict[str, Any]] = []
    for root in dedupe_roots(roots):
        if not root.exists():
            continue
        for suffix in ("*.mat", "*.txt"):
            for path in sorted(root.rglob(f"ROISignals_{suffix}")):
                sid = normalize_subject(path.name)
                if sid not in targets:
                    continue
                try:
                    stat = path.stat()
                    size = stat.st_size
                    mtime = pd.Timestamp.fromtimestamp(stat.st_mtime).isoformat()
                except OSError:
                    size = np.nan
                    mtime = ""
                rows.append(
                    {
                        "SubjectID": sid,
                        "path": str(path),
                        "source_root": source_root_for(path, roots),
                        "suffix": path.suffix.lower(),
                        "file_size_bytes": size,
                        "mtime": mtime,
                        "stage_guess": stage_guess(path),
                        "explicit_has_F_stage": explicit_has_f_stage(stage_guess(path), path),
                    }
                )
    return pd.DataFrame(rows)


def scan_zip_hits(target_subjects: Sequence[str], zip_roots: Sequence[Path]) -> pd.DataFrame:
    targets = set(target_subjects)
    rows: List[Dict[str, Any]] = []
    zip_paths: List[Path] = []
    for root in dedupe_roots(zip_roots):
        if root.exists():
            zip_paths.extend(sorted(root.rglob("*.zip")))
    for zpath in sorted(set(zip_paths)):
        try:
            with zipfile.ZipFile(zpath) as zf:
                for info in zf.infolist():
                    name = info.filename
                    if "ROISignals_" not in name or not name.lower().endswith((".mat", ".txt")):
                        continue
                    sid = normalize_subject(Path(name).name)
                    if sid not in targets:
                        continue
                    rows.append(
                        {
                            "SubjectID": sid,
                            "zip_path": str(zpath),
                            "member_name": name,
                            "member_size": info.file_size,
                            "compress_size": info.compress_size,
                            "date_time": "-".join(f"{x:02d}" for x in info.date_time),
                            "stage_guess": stage_guess(Path(name)),
                            "note": "zipinfo_only_not_extracted",
                        }
                    )
        except Exception as exc:
            rows.append({"zip_path": str(zpath), "status": f"zip_scan_failed:{exc}"})
    return pd.DataFrame(rows)


def choose_main_2d_entry(entries: Sequence[Tuple[str, Tuple[int, ...], str]]) -> Optional[Tuple[str, Tuple[int, ...], str]]:
    numeric = {"double", "single", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"}
    candidates = []
    for name, shape, klass in entries:
        if len(shape) == 2 and klass in numeric:
            name_score = 1 if ("signal" in name.lower() or "roi" in name.lower()) else 0
            size_score = int(shape[0]) * int(shape[1])
            candidates.append((name_score, size_score, name, shape, klass))
    if not candidates:
        return None
    _name_score, _size_score, name, shape, klass = sorted(candidates, reverse=True)[0]
    return name, shape, klass


def load_signal(path: Path) -> Tuple[Optional[np.ndarray], str, str, str]:
    try:
        if path.suffix.lower() == ".mat":
            entries = [(name, tuple(int(x) for x in shape), klass) for name, shape, klass in scipy_io.whosmat(path)]
            chosen = choose_main_2d_entry(entries)
            if chosen is None:
                return None, "", "", f"no_numeric_2d_var:{entries}"
            name, _shape, klass = chosen
            data = scipy_io.loadmat(path, variable_names=[name], squeeze_me=False)
            arr = np.asarray(data[name], dtype=np.float64)
            return arr, name, str(tuple(int(x) for x in arr.shape)), f"ok_mat:{klass}"
        if path.suffix.lower() == ".txt":
            try:
                arr = np.loadtxt(path, delimiter=",", dtype=np.float64)
                status = "ok_txt_comma"
            except Exception:
                arr = np.loadtxt(path, dtype=np.float64)
                status = "ok_txt_whitespace"
            return arr, "", str(tuple(int(x) for x in arr.shape)), status
    except Exception as exc:
        return None, "", "", f"failed:{exc}"
    return None, "", "", f"unsupported_suffix:{path.suffix}"


def orient_time_by_roi(arr: np.ndarray) -> Tuple[Optional[np.ndarray], int, int, str]:
    if arr.ndim != 2:
        return None, 0, 0, "not_2d"
    if arr.shape[1] in {170, 166, 131}:
        return arr, int(arr.shape[0]), int(arr.shape[1]), "time_by_roi"
    if arr.shape[0] in {170, 166, 131}:
        return arr.T, int(arr.shape[1]), int(arr.shape[0]), "roi_by_time_transposed"
    return arr, int(arr.shape[0]), int(arr.shape[1]), "unknown_orientation_assumed_time_by_col"


def classify_scale(vals: np.ndarray) -> str:
    if vals.size == 0:
        return "other"
    mean = float(np.nanmean(vals))
    median = float(np.nanmedian(vals))
    std = float(np.nanstd(vals))
    p01, p99 = np.nanpercentile(vals, [1, 99])
    frac_neg = float(np.mean(vals < 0))
    typical = np.nanmedian(np.abs([mean, median]))
    if np.isfinite(typical) and 1000 <= typical <= 50000 and frac_neg < 0.01:
        return "around_10000_global_scaled"
    centered_tol = max(1e-6, 0.10 * std) if np.isfinite(std) else 1e-6
    if np.isfinite(mean) and abs(mean) <= centered_tol and frac_neg > 0.01 and p01 < 0 < p99:
        return "zero_centered"
    return "other"


def spectral_energy(arr_time_roi: Optional[np.ndarray], tr: float = TR_SECONDS) -> Dict[str, Any]:
    out = {
        "energy_below_0p01": np.nan,
        "energy_0p01_0p08": np.nan,
        "energy_above_0p08": np.nan,
        "bandpassed_like": "uncertain",
        "spectral_status": "not_computed",
        "spectral_valid_rois": 0,
    }
    if arr_time_roi is None or arr_time_roi.ndim != 2 or arr_time_roi.shape[0] < 8:
        out["spectral_status"] = "invalid_shape_or_short_timeseries"
        return out
    arr = np.asarray(arr_time_roi, dtype=np.float64)
    valid_cols = []
    for idx in range(arr.shape[1]):
        col = arr[:, idx]
        finite = np.isfinite(col)
        if finite.mean() < 0.95 or finite.sum() < 8:
            continue
        fill = col.copy()
        fill[~finite] = np.nanmean(fill[finite])
        fill = fill - np.mean(fill)
        if np.nanstd(fill) <= 1e-9:
            continue
        valid_cols.append(fill)
    if not valid_cols:
        out["spectral_status"] = "no_valid_roi_columns"
        return out
    mat = np.asarray(valid_cols, dtype=np.float64).T
    fs = 1.0 / tr
    nperseg = min(128, mat.shape[0])
    freqs, pxx = welch(mat, fs=fs, axis=0, nperseg=nperseg, detrend=False)
    total = np.sum(pxx, axis=0)
    good = total > 0
    if not np.any(good):
        out["spectral_status"] = "zero_psd_energy"
        return out
    pxx = pxx[:, good]
    total = total[good]
    below = np.sum(pxx[freqs < 0.01, :], axis=0) / total
    band = np.sum(pxx[(freqs >= 0.01) & (freqs <= 0.08), :], axis=0) / total
    above = np.sum(pxx[freqs > 0.08, :], axis=0) / total
    below_m = float(np.nanmedian(below))
    band_m = float(np.nanmedian(band))
    above_m = float(np.nanmedian(above))
    outside = below_m + above_m
    if band_m >= 0.85 and below_m <= 0.10 and above_m <= 0.10:
        label = "yes"
    elif outside >= 0.30 or band_m < 0.70:
        label = "no"
    else:
        label = "uncertain"
    out.update(
        {
            "energy_below_0p01": below_m,
            "energy_0p01_0p08": band_m,
            "energy_above_0p08": above_m,
            "bandpassed_like": label,
            "spectral_status": "ok",
            "spectral_valid_rois": int(good.sum()),
        }
    )
    return out


def candidate_compatibility(row: Dict[str, Any]) -> Tuple[str, str]:
    if not str(row.get("load_status", "")).startswith("ok"):
        return "reject", "load_failed"
    if float(row.get("finite_fraction", 0.0) or 0.0) < 0.95:
        return "reject", "finite_fraction_lt_0.95"
    if int(row.get("n_rois", 0) or 0) != RAW_ROIS:
        return "reject", "roi_count_not_170"
    if row.get("scale_label") != "around_10000_global_scaled":
        return "reject", "scale_not_around_10000"
    stage = str(row.get("stage_guess", ""))
    band = str(row.get("bandpassed_like", "uncertain"))
    if stage in {"ARWSDCFN", "ARWSDCF"} and band == "yes":
        return "direct_compatible", "explicit_F_stage_qc_scale_spectral_ok"
    if stage in {"ARWSDCFN", "ARWSDCF"}:
        return "direct_compatible_with_warning", f"explicit_F_stage_qc_scale_spectral_{band}"
    if stage in {"ARWSDC", "CovRegressed", "historical_10000"} and band == "yes":
        return "direct_compatible_with_warning", "non_explicit_F_stage_but_qc_scale_spectral_strong"
    return "needs_stage_confirmation", "stage_or_spectral_not_sufficient_for_direct_import"


def qc_candidate(path: Path, base: Dict[str, Any]) -> Dict[str, Any]:
    arr, var_name, shape, status = load_signal(path)
    row = dict(base)
    row.update(
        {
            "main_variable": var_name,
            "shape": shape,
            "load_status": status,
            "n_timepoints": 0,
            "n_rois": 0,
            "orientation_status": "",
            "finite_fraction": 0.0,
            "nan_count": np.nan,
            "columns_all_nan": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
            "scale_label": "other",
        }
    )
    if arr is None or arr.ndim != 2 or arr.size == 0:
        row.update(spectral_energy(None))
        compat, reason = candidate_compatibility(row)
        row["suspected_stage_compatibility"] = compat
        row["compatibility_reason"] = reason
        return row
    oriented, n_tp, n_rois, orient_status = orient_time_by_roi(arr)
    finite = np.isfinite(arr)
    vals = arr[finite]
    row.update(
        {
            "n_timepoints": n_tp,
            "n_rois": n_rois,
            "orientation_status": orient_status,
            "finite_fraction": float(finite.mean()),
            "nan_count": int(np.isnan(arr).sum()),
            "columns_all_nan": int(np.isnan(oriented).all(axis=0).sum()) if oriented is not None else np.nan,
            "mean": float(np.nanmean(vals)) if vals.size else np.nan,
            "std": float(np.nanstd(vals)) if vals.size else np.nan,
            "median": float(np.nanmedian(vals)) if vals.size else np.nan,
            "min": float(np.nanmin(vals)) if vals.size else np.nan,
            "max": float(np.nanmax(vals)) if vals.size else np.nan,
            "scale_label": classify_scale(vals),
        }
    )
    row.update(spectral_energy(oriented))
    compat, reason = candidate_compatibility(row)
    row["suspected_stage_compatibility"] = compat
    row["compatibility_reason"] = reason
    return row


def rank_candidate(row: pd.Series) -> Tuple[int, int, int, int, int, float, int, str]:
    compat_rank = {
        "direct_compatible": 0,
        "direct_compatible_with_warning": 1,
        "needs_stage_confirmation": 2,
        "reject": 3,
    }.get(str(row.get("suspected_stage_compatibility")), 9)
    stage_rank = {
        "ARWSDCFN": 0,
        "ARWSDCF": 1,
        "historical_10000": 2,
        "ARWSDC": 3,
        "CovRegressed": 4,
        "unknown": 5,
    }.get(str(row.get("stage_guess")), 6)
    path = str(row.get("path", ""))
    path_rank = 0 if ("ResultsAAL3" in path and ("FunImgARWSDCF" in path or "ROISignals_AAL3_FunImgARWSDCF" in path)) else 1
    if path.startswith("/media/diego/Datos/adni_expansion/MARTIN_20260429_PHILIPS10"):
        storage_rank = 0
    elif path.startswith("/media/"):
        storage_rank = 1
    elif path.startswith("/home/"):
        storage_rank = 2
    else:
        storage_rank = 3
    suffix_rank = 0 if str(row.get("suffix")) == ".mat" else 1
    finite_rank = -float(row.get("finite_fraction", 0.0) or 0.0)
    size_rank = -int(float(row.get("file_size_bytes", 0) or 0))
    return compat_rank, stage_rank, path_rank, storage_rank, suffix_rank, finite_rank, size_rank, path


def build_recommendations(targets: pd.DataFrame, long_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    rejected_rows: List[Dict[str, Any]] = []
    by_subject = {sid: sub.copy() for sid, sub in long_df.groupby("SubjectID")} if not long_df.empty else {}
    for _, target in targets.iterrows():
        sid = target["SubjectID"]
        sub = by_subject.get(sid, pd.DataFrame())
        if sub.empty:
            row = target.to_dict()
            row.update(
                {
                    "recommended_path": "",
                    "recommended_stage_guess": "",
                    "recommended_compatibility": "needs_preprocessing",
                    "recommended_to_add": "no",
                    "selection_reason": "no_local_roisignals_found",
                }
            )
            rows.append(row)
            rejected_rows.append(row.copy())
            continue
        sub = sub.copy()
        sub["_rank"] = sub.apply(rank_candidate, axis=1)
        sub = sub.sort_values("_rank", kind="mergesort")
        best = sub.iloc[0].drop(labels=["_rank"]).to_dict()
        recommended_to_add = best["suspected_stage_compatibility"] in {"direct_compatible", "direct_compatible_with_warning"}
        row = target.to_dict()
        for key, value in best.items():
            row[f"recommended_{key}" if key in {"path", "stage_guess"} else key] = value
        row["recommended_path"] = best["path"]
        row["recommended_stage_guess"] = best["stage_guess"]
        row["recommended_compatibility"] = best["suspected_stage_compatibility"]
        row["recommended_to_add"] = "yes" if recommended_to_add else "no"
        row["selection_reason"] = (
            "best_rank_by_compatibility_stage_qc_spectral_path; "
            f"{int(len(sub))} local candidate files inspected"
        )
        rows.append(row)
        for _, cand in sub.iloc[1:].iterrows():
            rej = target.to_dict()
            rej.update(cand.drop(labels=["_rank"]).to_dict())
            rej["rejection_or_confirmation_reason"] = "not_selected_lower_priority_duplicate_or_alternative"
            rejected_rows.append(rej)
        if not recommended_to_add:
            rej = target.to_dict()
            rej.update(best)
            rej["rejection_or_confirmation_reason"] = best.get("compatibility_reason", "not_direct_compatible")
            rejected_rows.append(rej)
    return pd.DataFrame(rows), pd.DataFrame(rejected_rows)


def pearson_flat(a: np.ndarray, b: np.ndarray) -> float:
    mask = ~np.eye(a.shape[0], dtype=bool)
    av = np.asarray(a[mask], dtype=np.float64)
    bv = np.asarray(b[mask], dtype=np.float64)
    valid = np.isfinite(av) & np.isfinite(bv)
    if valid.sum() < 2:
        return np.nan
    return float(np.corrcoef(av[valid], bv[valid])[0, 1])


def robustscale_offdiag(matrix: np.ndarray) -> np.ndarray:
    out = np.zeros_like(matrix, dtype=np.float32)
    if matrix.shape[0] <= 1:
        return matrix.astype(np.float32)
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    vals = matrix[mask].astype(np.float64)
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    med = np.median(vals)
    q25, q75 = np.percentile(vals, [25, 75])
    iqr = q75 - q25
    if not np.isfinite(iqr) or iqr <= 1e-12:
        out[mask] = vals.astype(np.float32)
    else:
        out[mask] = ((vals - med) / iqr).astype(np.float32)
    return out


def v4_smoke_comparison(recommended: pd.DataFrame, v4_tensor: Path) -> pd.DataFrame:
    if IMPORT_ERROR:
        return pd.DataFrame([{"status": f"extractor_import_failed:{IMPORT_ERROR}"}])
    if build_roi_reduction_and_order is None or preprocess_timeseries_no_pybandpass is None or pearson_full is None:
        return pd.DataFrame([{"status": "extractor_functions_unavailable"}])
    if not v4_tensor.exists():
        return pd.DataFrame([{"status": f"v4_tensor_missing:{v4_tensor}"}])
    candidates = recommended[recommended["recommended_path"].fillna("").astype(bool)].copy()
    if candidates.empty:
        return pd.DataFrame([{"status": "no_recommended_physical_candidates"}])
    roi_info = build_roi_reduction_and_order()
    rows: List[Dict[str, Any]] = []
    with np.load(v4_tensor, allow_pickle=False) as zf:
        subjects = [str(x) for x in zf["subject_ids"]]
        channels = [str(x) for x in zf["channel_names"]]
        channel_idx = channels.index("Pearson_Full_FisherZ_Signed") if "Pearson_Full_FisherZ_Signed" in channels else 1
        tensor = zf["global_tensor_data"]
        subject_to_idx = {sid: idx for idx, sid in enumerate(subjects)}
        for _, rec in candidates.iterrows():
            sid = rec["SubjectID"]
            row: Dict[str, Any] = {
                "SubjectID": sid,
                "recommended_path": rec.get("recommended_path", ""),
                "recommended_compatibility": rec.get("recommended_compatibility", ""),
                "in_v4_tensor": sid in subject_to_idx,
                "python_bandpass": "OFF",
                "historical_channel_index": channel_idx,
                "historical_channel_name": channels[channel_idx] if channel_idx < len(channels) else "",
            }
            if sid not in subject_to_idx:
                row["status"] = "subject_not_in_v4_tensor"
                rows.append(row)
                continue
            raw, _var, raw_shape, load_status = load_signal(Path(str(rec["recommended_path"])))
            row["raw_shape"] = raw_shape
            row["load_status"] = load_status
            if raw is None:
                row["status"] = "raw_not_loaded"
                rows.append(row)
                continue
            ts, pre_status, qc = preprocess_timeseries_no_pybandpass(raw, roi_info)
            row["preprocess_status"] = pre_status
            row.update({f"preprocess_{k}": v for k, v in qc.items()})
            if ts is None:
                row["status"] = "preprocess_failed"
                rows.append(row)
                continue
            pearson = pearson_full(ts)
            pearson_scaled = robustscale_offdiag(pearson)
            hist = np.asarray(tensor[subject_to_idx[sid], channel_idx], dtype=np.float32)
            mask = ~np.eye(hist.shape[0], dtype=bool)
            scaled_diff = pearson_scaled - hist
            raw_diff = pearson - hist
            row.update(
                {
                    "status": "ok",
                    "corr": pearson_flat(pearson_scaled, hist),
                    "MAE": float(np.nanmean(np.abs(scaled_diff[mask]))),
                    "max_abs_diff": float(np.nanmax(np.abs(scaled_diff[mask]))),
                    "comparison_matrix": "robustscale_offdiag(Pearson_Full_FisherZ_OFF)",
                    "raw_pearson_vs_v4_corr": pearson_flat(pearson, hist),
                    "raw_pearson_vs_v4_MAE": float(np.nanmean(np.abs(raw_diff[mask]))),
                    "raw_pearson_vs_v4_max_abs_diff": float(np.nanmax(np.abs(raw_diff[mask]))),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def write_readme(
    output_dir: Path,
    targets: pd.DataFrame,
    long_df: pd.DataFrame,
    recommended: pd.DataFrame,
    rejected: pd.DataFrame,
    smoke: pd.DataFrame,
    zip_hits: pd.DataFrame,
) -> None:
    found_subjects = set(long_df["SubjectID"]) if not long_df.empty else set()
    clean = int(recommended["recommended_compatibility"].eq("direct_compatible").sum()) if not recommended.empty else 0
    warn = int(recommended["recommended_compatibility"].eq("direct_compatible_with_warning").sum()) if not recommended.empty else 0
    needs = int(recommended["recommended_compatibility"].isin(["needs_stage_confirmation", "needs_preprocessing"]).sum()) if not recommended.empty else 0
    reject = int(recommended["recommended_compatibility"].eq("reject").sum()) if not recommended.empty else 0
    v4_ok = smoke[smoke.get("status", pd.Series(dtype=str)).eq("ok")] if not smoke.empty else pd.DataFrame()
    v4_in = int(smoke.get("in_v4_tensor", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not smoke.empty and "in_v4_tensor" in smoke.columns else 0
    v4_checked = int(smoke["SubjectID"].nunique()) if not smoke.empty and "SubjectID" in smoke.columns else 0
    lines = [
        "# CN-GE Local ROISignals Audit for ADNI v5.1",
        "",
        "Read-only audit. No zip extraction, no full connectivity tensor, and no training were run.",
        "",
        "## Explicit Answers",
        "",
        f"- CN-GE first-visit universe audited: `{targets['SubjectID'].nunique()}`.",
        f"- CN-GE subjects with physical local ROISignals: `{len(found_subjects)}`.",
        f"- Direct-compatible clean CN-GE: `{clean}`.",
        f"- Direct-compatible with warning CN-GE: `{warn}`.",
        f"- CN-GE needing confirmation/preprocessing: `{needs}`.",
        f"- CN-GE rejected by QC: `{reject}`.",
        f"- Zip hits found by zipinfo only: `{zip_hits['SubjectID'].nunique() if not zip_hits.empty and 'SubjectID' in zip_hits else 0}` subjects.",
        f"- Recommended physical CN-GE candidates present in paper v4 tensor: `{v4_in}/{v4_checked}`.",
        f"- V4 channel-1 smoke comparisons computed: `{len(v4_ok)}` ok rows.",
        "- Python bandpass applied in this audit: `False`.",
        "",
        "## Interpretation Rules",
        "",
        "- `direct_compatible`: explicit ARWSDCF/ARWSDCFN path, 170 ROIs, high finite fraction, around-10000 scale, and spectral bandpass-like signal.",
        "- `direct_compatible_with_warning`: usable local signal, but source stage is non-explicit or spectral evidence is not fully decisive.",
        "- `needs_stage_confirmation`: local signal exists, but stage/spectral evidence is not sufficient for direct import.",
        "- `reject`: load, finite fraction, ROI count, or scale failed QC.",
        "",
        "## Outputs",
        "",
        "- `ge_cn_candidate_roisignals_long.csv`",
        "- `ge_cn_recommended_roisignals.csv`",
        "- `ge_cn_rejected_or_needs_confirmation.csv`",
        "- `ge_cn_v4_channel1_smoke_comparison.csv`",
        "- `ge_cn_zipinfo_roisignals_hits.csv`",
        "- `ge_cn_targets.csv`",
        "",
        "## Next Command",
        "",
        "`/home/diego/anaconda3/envs/vae_ad/bin/python scripts/revision_bspc_2026/build_v5_1_gecn_augmented_manifest_and_tensor_plan.py --overwrite`",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    prepare_output_dir(args.output_dir, args.overwrite)
    roots = dedupe_roots(ROISIGNAL_ROOTS)
    targets = load_target_cn_ge(args.first_visit_manifest, args.preprocessing_request)
    targets.to_csv(args.output_dir / "ge_cn_targets.csv", index=False)

    discovered = discover_files(targets["SubjectID"].tolist(), roots)
    if discovered.empty:
        long_df = pd.DataFrame()
    else:
        rows = []
        for _, row in discovered.iterrows():
            rows.append(qc_candidate(Path(row["path"]), row.to_dict()))
        long_df = pd.DataFrame(rows)
    long_df.to_csv(args.output_dir / "ge_cn_candidate_roisignals_long.csv", index=False)

    if args.skip_zip_scan:
        zip_hits = pd.DataFrame()
    else:
        zip_hits = scan_zip_hits(targets["SubjectID"].tolist(), ZIP_SEARCH_ROOTS)
    zip_hits.to_csv(args.output_dir / "ge_cn_zipinfo_roisignals_hits.csv", index=False)

    recommended, rejected = build_recommendations(targets, long_df)
    recommended.to_csv(args.output_dir / "ge_cn_recommended_roisignals.csv", index=False)
    rejected.to_csv(args.output_dir / "ge_cn_rejected_or_needs_confirmation.csv", index=False)

    if args.skip_v4_smoke:
        smoke = pd.DataFrame([{"status": "skipped_by_user"}])
    else:
        smoke = v4_smoke_comparison(recommended, args.v4_tensor)
    smoke.to_csv(args.output_dir / "ge_cn_v4_channel1_smoke_comparison.csv", index=False)

    command = {
        "script": str(Path(__file__).resolve()),
        "first_visit_manifest": str(args.first_visit_manifest),
        "preprocessing_request": str(args.preprocessing_request),
        "v4_tensor": str(args.v4_tensor),
        "output_dir": str(args.output_dir),
        "overwrite": bool(args.overwrite),
        "skip_zip_scan": bool(args.skip_zip_scan),
        "skip_v4_smoke": bool(args.skip_v4_smoke),
        "python_bandpass_applied": False,
        "tr_seconds": TR_SECONDS,
    }
    (args.output_dir / "command_log.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
    write_readme(args.output_dir, targets, long_df, recommended, rejected, smoke, zip_hits)

    print(f"Wrote CN-GE local ROISignals audit to {args.output_dir}")
    print(f"cn_ge_targets={targets['SubjectID'].nunique()} physical_found={long_df['SubjectID'].nunique() if not long_df.empty else 0}")
    if not recommended.empty:
        print(f"recommended_counts={recommended['recommended_compatibility'].value_counts(dropna=False).to_dict()}")
    print("No full connectivity tensor computed. No training run.")


if __name__ == "__main__":
    main()
