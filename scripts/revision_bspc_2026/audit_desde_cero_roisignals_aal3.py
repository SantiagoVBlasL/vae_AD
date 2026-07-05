#!/usr/bin/env python3
"""Audit a candidate AAL3 ROI-signal folder for ADNI v5 reconstruction.

Read-only with respect to source data:
- no source files are moved/copied/deleted;
- no connectivity is computed;
- no model training is run;
- the historical global tensor is inspected only for small metadata/header keys.
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ROI_ROOT = Path("/media/diego/Datos/desde_cero/ROISignalsAAL3")
DEFAULT_V4_METADATA = (
    PROJECT_ROOT
    / "data"
    / "revision_bspc_2026"
    / "adni_expanded_v4_all_available"
    / "subject_metadata_adni_expanded_v4_all_available.csv"
)
DEFAULT_HISTORICAL_TENSOR = (
    PROJECT_ROOT
    / "data"
    / "AAL3_dynamicROIs_fmri_tensor_NeuroEnhanced_v6.5.17_AAL3_131ROIs_"
    "OMST_GCE_Signed_GrangerLag1_ChNorm_ROIreorderedYeo17_ParallelTuned"
    / "GLOBAL_TENSOR_from_AAL3_dynamicROIs_fmri_tensor_NeuroEnhanced_v6.5.17_AAL3_131ROIs_"
    "OMST_GCE_Signed_GrangerLag1_ChNorm_ROIreorderedYeo17_ParallelTuned.npz"
)
DEFAULT_NEW_PASSBAND_ROOT = Path(
    "/media/diego/My_Book_Diego/vae_AD_data/adni_passband_20260510/"
    "ResultsAAL3/ROISignals_AAL3_FunImgARWSDCFN"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_dparsf_only_rebuild"
    / "desde_cero_roisignals_aal3_audit"
)

SUBJECT_RE = re.compile(r"(?<!\d)(\d{3}_S_\d{4})(?!\d)", re.IGNORECASE)


@dataclass(frozen=True)
class SignalLoadResult:
    array: Optional[np.ndarray]
    variable_name: str
    shape: str
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit /media/diego/Datos/desde_cero/ROISignalsAAL3 as a candidate ADNI ROI source.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--roi-root", type=Path, default=DEFAULT_ROI_ROOT)
    parser.add_argument("--v4-metadata", type=Path, default=DEFAULT_V4_METADATA)
    parser.add_argument("--historical-tensor", type=Path, default=DEFAULT_HISTORICAL_TENSOR)
    parser.add_argument("--new-passband-root", type=Path, default=DEFAULT_NEW_PASSBAND_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--stats-sample-size",
        type=int,
        default=0,
        help="0 means compute scale statistics for all readable subjects.",
    )
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


def timestamp(value: float) -> str:
    try:
        return pd.Timestamp(float(value), unit="s").isoformat()
    except Exception:
        return ""


def discover_roisignal_mat_files(root: Path) -> Dict[str, Path]:
    if not root.exists():
        return {}
    out: Dict[str, Path] = {}
    for path in sorted(root.glob("ROISignals_*.mat")):
        sid = normalize_subject(path.name)
        if sid and sid not in out:
            out[sid] = path
    return out


def discover_signal_files(root: Path) -> Dict[str, Path]:
    """Discover passband files, preferring .mat and falling back to .txt."""
    if not root.exists():
        return {}
    out: Dict[str, Path] = {}
    for suffix in [".mat", ".txt"]:
        for path in sorted(root.glob(f"ROISignals_*{suffix}")):
            sid = normalize_subject(path.name)
            if sid and sid not in out:
                out[sid] = path
    return out


def read_v4_subjects(path: Path) -> Tuple[set[str], str]:
    if not path.exists():
        return set(), "missing"
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        sid_col = first_existing_col(df.columns, ["SubjectID", "Subject", "PTID", "subject_id"])
        if sid_col is None:
            return set(), "no_subject_column"
        subjects = set(df[sid_col].map(normalize_subject).replace("", np.nan).dropna().astype(str))
        return subjects, "ok"
    except Exception as exc:
        return set(), f"failed: {exc}"


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


def read_historical_tensor_subjects(path: Path) -> Tuple[set[str], Dict[str, Any]]:
    info: Dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "status": "",
        "tensor_shape_header": "",
        "tensor_dtype_header": "",
        "n_subjects": 0,
    }
    if not path.exists():
        info["status"] = "missing"
        return set(), info
    for key in ["global_tensor_data", "tensor_data", "X", "arr_0"]:
        shape, dtype = _read_npy_header_from_npz(path, key)
        if shape:
            info["tensor_shape_header"] = shape
            info["tensor_dtype_header"] = dtype
            break
    try:
        try:
            npz = np.load(path, allow_pickle=False)
            close_npz = True
            subject_raw = npz["subject_ids"] if "subject_ids" in npz.files else None
        except Exception:
            npz = np.load(path, allow_pickle=True)
            close_npz = True
            subject_raw = npz["subject_ids"] if "subject_ids" in npz.files else None
        if subject_raw is None:
            info["status"] = "no_subject_ids_key"
            return set(), info
        subjects = {sid for sid in (normalize_subject(x) for x in subject_raw.tolist()) if sid}
        info["n_subjects"] = len(subjects)
        info["status"] = "ok"
        return subjects, info
    except Exception as exc:
        info["status"] = f"failed: {exc}"
        return set(), info
    finally:
        if "close_npz" in locals() and close_npz:
            npz.close()


def whosmat_entries(path: Path) -> List[Tuple[str, Tuple[int, ...], str]]:
    import scipy.io  # type: ignore

    return [(name, tuple(int(x) for x in shape), klass) for name, shape, klass in scipy.io.whosmat(path)]


def choose_main_2d_entry(entries: Sequence[Tuple[str, Tuple[int, ...], str]]) -> Optional[Tuple[str, Tuple[int, ...], str]]:
    candidates = []
    for name, shape, klass in entries:
        if len(shape) != 2:
            continue
        if klass not in {"double", "single", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"}:
            continue
        size = int(shape[0]) * int(shape[1])
        name_bonus = 1 if ("signal" in name.lower() or "roi" in name.lower()) else 0
        candidates.append((name_bonus, size, name, shape, klass))
    if not candidates:
        return None
    _bonus, _size, name, shape, klass = sorted(candidates, reverse=True)[0]
    return name, shape, klass


def load_mat_signal(path: Path) -> SignalLoadResult:
    try:
        import scipy.io  # type: ignore

        entries = whosmat_entries(path)
        chosen = choose_main_2d_entry(entries)
        if chosen is None:
            return SignalLoadResult(None, "", "", f"no_numeric_2d_var:{entries}")
        name, shape, klass = chosen
        mat = scipy.io.loadmat(path, variable_names=[name], squeeze_me=False)
        arr = np.asarray(mat[name], dtype=np.float64)
        if arr.ndim != 2:
            return SignalLoadResult(None, name, str(tuple(arr.shape)), "loaded_var_not_2d")
        return SignalLoadResult(arr, name, str(tuple(int(x) for x in arr.shape)), f"ok:{klass}")
    except Exception as exc:
        return SignalLoadResult(None, "", "", f"failed: {exc}")


def load_txt_signal(path: Path) -> SignalLoadResult:
    try:
        try:
            arr = np.loadtxt(path, delimiter=",", dtype=np.float64)
            status = "ok_comma"
        except Exception:
            arr = np.loadtxt(path, dtype=np.float64)
            status = "ok_whitespace"
        if arr.ndim != 2:
            return SignalLoadResult(None, "", str(tuple(arr.shape)), "txt_not_2d")
        return SignalLoadResult(arr, "", str(tuple(int(x) for x in arr.shape)), status)
    except Exception as exc:
        return SignalLoadResult(None, "", "", f"failed: {exc}")


def load_signal(path: Path) -> SignalLoadResult:
    if path.suffix.lower() == ".mat":
        return load_mat_signal(path)
    if path.suffix.lower() == ".txt":
        return load_txt_signal(path)
    return SignalLoadResult(None, "", "", f"unsupported_suffix:{path.suffix}")


def inventory_rows(files: Dict[str, Path]) -> Tuple[pd.DataFrame, Dict[str, SignalLoadResult]]:
    rows: List[Dict[str, Any]] = []
    loaded_meta: Dict[str, SignalLoadResult] = {}
    for sid, path in sorted(files.items()):
        stat = path.stat()
        entries_text = ""
        try:
            entries = whosmat_entries(path)
            entries_text = "|".join(f"{name}:{shape}:{klass}" for name, shape, klass in entries)
        except Exception as exc:
            entries_text = f"failed: {exc}"
        loaded = load_mat_signal(path)
        loaded_meta[sid] = loaded
        rows.append(
            {
                "SubjectID": sid,
                "path": str(path),
                "file_size_bytes": stat.st_size,
                "mtime": timestamp(stat.st_mtime),
                "mat_entries": entries_text,
                "main_variable": loaded.variable_name,
                "main_shape": loaded.shape,
                "load_status": loaded.status,
            }
        )
    return pd.DataFrame(rows), loaded_meta


def finite_values(arr: np.ndarray) -> np.ndarray:
    vals = np.asarray(arr, dtype=np.float64).ravel()
    return vals[np.isfinite(vals)]


def scale_label(stats: Dict[str, float]) -> str:
    median = stats.get("median", np.nan)
    mean = stats.get("mean", np.nan)
    std = stats.get("std", np.nan)
    frac_negative = stats.get("fraction_negative", np.nan)
    p05 = stats.get("p05", np.nan)
    p95 = stats.get("p95", np.nan)
    typical_abs = np.nanmedian(np.abs([median, mean]))
    if np.isfinite(typical_abs) and 1000 <= typical_abs <= 50000 and (not np.isfinite(frac_negative) or frac_negative < 0.01):
        return "around_10000_global_scaled"
    centered_tolerance = max(1e-6, 0.10 * std) if np.isfinite(std) else 1e-6
    if (
        np.isfinite(mean)
        and np.isfinite(std)
        and abs(mean) <= centered_tolerance
        and np.isfinite(frac_negative)
        and frac_negative > 0.01
        and np.isfinite(p05)
        and np.isfinite(p95)
        and p05 < 0 < p95
    ):
        return "zero_centered_or_regressed"
    return "unknown"


def compute_scale_statistics(files: Dict[str, Path], sample_size: int) -> pd.DataFrame:
    items = sorted(files.items())
    if sample_size and sample_size > 0 and sample_size < len(items):
        rng = np.random.default_rng(20260511)
        keep_idx = sorted(rng.choice(len(items), size=sample_size, replace=False).tolist())
        items = [items[idx] for idx in keep_idx]
    rows: List[Dict[str, Any]] = []
    for sid, path in items:
        loaded = load_mat_signal(path)
        row: Dict[str, Any] = {
            "SubjectID": sid,
            "path": str(path),
            "shape": loaded.shape,
            "load_status": loaded.status,
            "stats_sampled": bool(sample_size and sample_size > 0),
        }
        if loaded.array is None:
            row["n_values"] = np.nan
            row["n_finite_values"] = 0
            row["fraction_finite"] = 0.0
            row["heuristic_scale_label"] = "unknown"
            rows.append(row)
            continue
        arr = loaded.array
        vals = finite_values(arr)
        row["n_values"] = int(arr.size)
        row["n_finite_values"] = int(vals.size)
        row["fraction_finite"] = float(vals.size / arr.size) if arr.size else np.nan
        if vals.size == 0:
            row["heuristic_scale_label"] = "unknown"
            rows.append(row)
            continue
        roi_means = np.nanmean(arr, axis=0)
        timepoint_means = np.nanmean(arr, axis=1)
        stats = {
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals)),
            "min": float(np.nanmin(vals)),
            "max": float(np.nanmax(vals)),
            "p01": float(np.nanpercentile(vals, 1)),
            "p05": float(np.nanpercentile(vals, 5)),
            "median": float(np.nanmedian(vals)),
            "p95": float(np.nanpercentile(vals, 95)),
            "p99": float(np.nanpercentile(vals, 99)),
            "median_roi_mean": float(np.nanmedian(roi_means)),
            "median_timepoint_mean": float(np.nanmedian(timepoint_means)),
            "fraction_negative": float(np.mean(vals < 0)),
            "fraction_near_zero": float(np.mean(np.abs(vals) < 1e-8)),
        }
        row.update(stats)
        row["heuristic_scale_label"] = scale_label(stats)
        rows.append(row)
    return pd.DataFrame(rows)


def coverage_rows(
    desde_subjects: set[str],
    historical_subjects: set[str],
    v4_subjects: set[str],
    passband_subjects: set[str],
) -> pd.DataFrame:
    references = [
        ("historical_tensor_431", historical_subjects),
        ("v4_metadata", v4_subjects),
        ("new_passband_20260510", passband_subjects),
    ]
    rows: List[Dict[str, Any]] = []
    for name, ref in references:
        overlap = sorted(desde_subjects & ref)
        missing = sorted(ref - desde_subjects)
        extra = sorted(desde_subjects - ref)
        rows.append(
            {
                "reference": name,
                "n_desde_cero_subjects": len(desde_subjects),
                "n_reference_subjects": len(ref),
                "n_overlap": len(overlap),
                "n_missing_from_desde_cero": len(missing),
                "n_extra_vs_reference": len(extra),
                "coverage_fraction": (len(overlap) / len(ref)) if ref else np.nan,
                "overlap_preview": "|".join(overlap[:20]),
                "missing_preview": "|".join(missing[:20]),
                "extra_preview": "|".join(extra[:20]),
            }
        )
    sentinel_subjects = ["114_S_6039", "035_S_6927", "094_S_6736", "301_S_6592"]
    for sid in sentinel_subjects:
        rows.append(
            {
                "reference": f"sentinel_{sid}",
                "n_desde_cero_subjects": len(desde_subjects),
                "n_reference_subjects": 1,
                "n_overlap": int(sid in desde_subjects),
                "n_missing_from_desde_cero": int(sid not in desde_subjects),
                "n_extra_vs_reference": np.nan,
                "coverage_fraction": float(sid in desde_subjects),
                "overlap_preview": sid if sid in desde_subjects else "",
                "missing_preview": "" if sid in desde_subjects else sid,
                "extra_preview": "",
            }
        )
    return pd.DataFrame(rows)


def missing_subject_rows(
    desde_subjects: set[str],
    historical_subjects: set[str],
    v4_subjects: set[str],
    passband_subjects: set[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for reference, subjects in [
        ("historical_tensor_431", historical_subjects),
        ("v4_metadata", v4_subjects),
        ("new_passband_20260510", passband_subjects),
    ]:
        for sid in sorted(subjects - desde_subjects):
            rows.append({"reference": reference, "SubjectID": sid, "missing_if_used_for_v5": True})
    return pd.DataFrame(rows)


def align_arrays(a: np.ndarray, b: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    if a.shape == b.shape:
        return a, b, "same_orientation"
    if a.shape == b.T.shape:
        return a, b.T, "transposed_new_passband"
    if a.T.shape == b.shape:
        return a.T, b, "transposed_desde_cero"
    return None, None, "shape_mismatch"


def pearson_flat(a: np.ndarray, b: np.ndarray) -> float:
    av = finite_values(a)
    bv = finite_values(b)
    if av.size != bv.size or av.size < 2:
        return np.nan
    mask = np.isfinite(av) & np.isfinite(bv)
    av = av[mask]
    bv = bv[mask]
    if av.size < 2 or np.nanstd(av) == 0 or np.nanstd(bv) == 0:
        return np.nan
    return float(np.corrcoef(av, bv)[0, 1])


def compare_overlap(desde_files: Dict[str, Path], passband_files: Dict[str, Path]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for sid in sorted(set(desde_files) & set(passband_files)):
        desde_path = desde_files[sid]
        passband_path = passband_files[sid]
        desde = load_signal(desde_path)
        passband = load_signal(passband_path)
        row: Dict[str, Any] = {
            "SubjectID": sid,
            "desde_cero_path": str(desde_path),
            "new_passband_path": str(passband_path),
            "desde_shape": desde.shape,
            "new_passband_shape": passband.shape,
            "desde_status": desde.status,
            "new_passband_status": passband.status,
        }
        if desde.array is None or passband.array is None:
            row["alignment"] = "not_loaded"
            row["comparison_decision"] = "different_signal"
            rows.append(row)
            continue
        a, b, alignment = align_arrays(desde.array, passband.array)
        row["alignment"] = alignment
        if a is None or b is None:
            row["comparison_decision"] = "different_signal"
            rows.append(row)
            continue
        diff = a - b
        abs_diff = np.abs(diff[np.isfinite(diff)])
        corr = pearson_flat(a, b)
        med_a = float(np.nanmedian(np.abs(a)))
        med_b = float(np.nanmedian(np.abs(b)))
        scale_ratio = med_a / med_b if med_b not in {0.0, -0.0} and np.isfinite(med_b) else np.nan
        mean_abs_diff = float(np.nanmean(abs_diff)) if abs_diff.size else np.nan
        max_abs_diff = float(np.nanmax(abs_diff)) if abs_diff.size else np.nan
        row.update(
            {
                "aligned_shape": str(tuple(int(x) for x in a.shape)),
                "mean_absolute_difference": mean_abs_diff,
                "max_absolute_difference": max_abs_diff,
                "pearson_flat": corr,
                "scale_ratio_median_abs_desde_over_new": scale_ratio,
            }
        )
        if np.allclose(a, b, rtol=1e-6, atol=1e-8, equal_nan=True):
            decision = "identical_or_near_identical"
        elif np.isfinite(corr) and corr >= 0.999 and np.isfinite(scale_ratio):
            decision = "same_shape_different_scale"
        else:
            decision = "different_signal"
        row["comparison_decision"] = decision
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_scale(stats: pd.DataFrame) -> Dict[str, Any]:
    if stats.empty:
        return {"majority_scale_label": "unknown", "n_scale_rows": 0}
    counts = stats["heuristic_scale_label"].value_counts(dropna=False).to_dict()
    majority = str(stats["heuristic_scale_label"].mode(dropna=False).iloc[0])
    out: Dict[str, Any] = {"majority_scale_label": majority, "n_scale_rows": len(stats)}
    for label, count in counts.items():
        out[f"n_scale_{label}"] = int(count)
    for col in ["mean", "std", "median", "fraction_negative", "fraction_near_zero"]:
        if col in stats.columns:
            out[f"median_subject_{col}"] = float(pd.to_numeric(stats[col], errors="coerce").median())
    if "n_finite_values" in stats.columns:
        finite = pd.to_numeric(stats["n_finite_values"], errors="coerce").fillna(0)
        out["n_subjects_no_finite_values"] = int((finite <= 0).sum())
    if "fraction_finite" in stats.columns:
        frac_finite = pd.to_numeric(stats["fraction_finite"], errors="coerce").fillna(0)
        out["n_subjects_fraction_finite_lt_0_99"] = int((frac_finite < 0.99).sum())
    return out


def summarize_comparison(comparison: pd.DataFrame) -> Dict[str, Any]:
    if comparison.empty:
        return {"n_overlap_compared": 0, "majority_comparison_decision": "none"}
    counts = comparison["comparison_decision"].value_counts(dropna=False).to_dict()
    majority = str(comparison["comparison_decision"].mode(dropna=False).iloc[0])
    out: Dict[str, Any] = {"n_overlap_compared": len(comparison), "majority_comparison_decision": majority}
    for label, count in counts.items():
        out[f"n_comparison_{label}"] = int(count)
    for col in ["mean_absolute_difference", "max_absolute_difference", "pearson_flat", "scale_ratio_median_abs_desde_over_new"]:
        if col in comparison.columns:
            out[f"median_{col}"] = float(pd.to_numeric(comparison[col], errors="coerce").median())
    return out


def provenance_heuristics(root: Path, coverage: pd.DataFrame, stats: pd.DataFrame, comparison: pd.DataFrame) -> Dict[str, Any]:
    low_path = str(root).lower()
    scale_summary = summarize_scale(stats)
    comparison_summary = summarize_comparison(comparison)
    comparison_counts = comparison["comparison_decision"].value_counts().to_dict() if not comparison.empty else {}
    n_identical = int(comparison_counts.get("identical_or_near_identical", 0))
    n_compared = int(len(comparison))
    if any(token in low_path for token in ["arwsdcfn", "bandpass", "passband", "pasabandas"]):
        possible_dparsf_bandpass = "likely_from_path"
    elif n_compared and n_identical / n_compared >= 0.80:
        possible_dparsf_bandpass = "likely_from_overlap_with_new_passband"
    else:
        possible_dparsf_bandpass = "unknown"
    possible_global_10000 = scale_summary.get("majority_scale_label") == "around_10000_global_scaled"
    v4_row = coverage[coverage["reference"].eq("v4_metadata")]
    covers_v4 = bool(not v4_row.empty and float(v4_row.iloc[0]["coverage_fraction"]) >= 0.98)
    if possible_dparsf_bandpass.startswith("likely") and covers_v4:
        compatible = "likely_but_confirm_with_martin"
    elif possible_dparsf_bandpass == "unknown":
        compatible = "unknown"
    else:
        compatible = "no_or_incomplete"
    return {
        "possible_dparsf_bandpass": possible_dparsf_bandpass,
        "possible_global_10000_normalization": bool(possible_global_10000),
        "compatible_with_python_bandpass_off_v5": compatible,
        **scale_summary,
        **comparison_summary,
    }


def write_readme(
    path: Path,
    roi_root: Path,
    v4_metadata: Path,
    historical_tensor_info: Dict[str, Any],
    new_passband_root: Path,
    inventory: pd.DataFrame,
    coverage: pd.DataFrame,
    stats: pd.DataFrame,
    comparison: pd.DataFrame,
    heuristics: Dict[str, Any],
) -> None:
    def row_for(reference: str) -> Dict[str, Any]:
        hit = coverage[coverage["reference"].eq(reference)]
        return hit.iloc[0].to_dict() if not hit.empty else {}

    historical = row_for("historical_tensor_431")
    v4 = row_for("v4_metadata")
    passband = row_for("new_passband_20260510")
    subjects = set(inventory["SubjectID"].astype(str)) if "SubjectID" in inventory.columns else set()
    present_114 = "YES" if "114_S_6039" in subjects else "NO"
    present_new_ads = {sid: ("YES" if sid in subjects else "NO") for sid in ["035_S_6927", "094_S_6736", "301_S_6592"]}
    scale_majority = heuristics.get("majority_scale_label", "unknown")
    comparison_majority = heuristics.get("majority_comparison_decision", "none")
    n_identical = heuristics.get("n_comparison_identical_or_near_identical", 0)
    n_compared = heuristics.get("n_overlap_compared", 0)
    invalid_subjects: List[str] = []
    low_finite_subjects: List[str] = []
    if not stats.empty and "n_finite_values" in stats.columns:
        finite = pd.to_numeric(stats["n_finite_values"], errors="coerce").fillna(0)
        invalid_subjects = stats.loc[finite <= 0, "SubjectID"].astype(str).tolist()
    if not stats.empty and "fraction_finite" in stats.columns:
        frac_finite = pd.to_numeric(stats["fraction_finite"], errors="coerce").fillna(0)
        low_finite_subjects = stats.loc[frac_finite < 0.99, "SubjectID"].astype(str).tolist()
    finite_114 = "UNKNOWN"
    if not stats.empty and "SubjectID" in stats.columns and "n_finite_values" in stats.columns:
        row_114 = stats[stats["SubjectID"].astype(str).eq("114_S_6039")]
        if not row_114.empty:
            finite_count_114 = pd.to_numeric(pd.Series([row_114.iloc[0]["n_finite_values"]]), errors="coerce").fillna(0).iloc[0]
            finite_114 = "YES" if int(finite_count_114) > 0 else "NO"
    likely_missing_source = (
        historical.get("n_overlap", 0) >= 400
        or v4.get("n_overlap", 0) >= 400
        or "114_S_6039" in subjects
    )
    can_rebuild = heuristics.get("compatible_with_python_bandpass_off_v5", "unknown")
    lines = [
        "# desde_cero ROISignalsAAL3 Audit",
        "",
        "Read-only audit. Source `.mat`/`.txt` files were not moved, copied, deleted, or rewritten. No connectivity was computed and no training was run.",
        "",
        "## Inputs",
        "",
        f"- ROI root: `{roi_root}` exists=`{roi_root.exists()}`",
        f"- v4 metadata: `{v4_metadata}` exists=`{v4_metadata.exists()}`",
        f"- historical tensor: `{historical_tensor_info.get('path')}` exists=`{historical_tensor_info.get('exists')}` status=`{historical_tensor_info.get('status')}` shape=`{historical_tensor_info.get('tensor_shape_header')}`",
        f"- new passband root: `{new_passband_root}` exists=`{new_passband_root.exists()}`",
        "",
        "## Explicit Answers",
        "",
        f"- Is this likely the missing ROI source folder? `{'POSSIBLY' if likely_missing_source else 'UNKNOWN'}`. It contains many historical-style `ROISignals_<SubjectID>.mat` files, but path/provenance does not by itself prove DPARSF-passband.",
        f"- How many subjects does it cover? `{len(subjects)}` unique subjects.",
        f"- Does it cover the historical 431? `{int(historical.get('n_overlap', 0))}/{int(historical.get('n_reference_subjects', 0))}` overlap; missing `{int(historical.get('n_missing_from_desde_cero', 0))}`.",
        f"- Does it cover v4 515? `{int(v4.get('n_overlap', 0))}/{int(v4.get('n_reference_subjects', 0))}` overlap; missing `{int(v4.get('n_missing_from_desde_cero', 0))}`.",
        f"- Is `114_S_6039` present? `{present_114}`; finite numeric signal values? `{finite_114}`.",
        f"- New AD sentinels present? `035_S_6927={present_new_ads['035_S_6927']}`, `094_S_6736={present_new_ads['094_S_6736']}`, `301_S_6592={present_new_ads['301_S_6592']}`.",
        f"- Are values around 10000 or zero-centered? Majority scale label: `{scale_majority}`.",
        f"- Subjects with no finite signal values: `{len(invalid_subjects)}`" + (f" (`{'|'.join(invalid_subjects[:20])}`)" if invalid_subjects else "."),
        f"- Subjects with fraction_finite < 0.99: `{len(low_finite_subjects)}`" + (f" (`{'|'.join(low_finite_subjects[:20])}`)" if low_finite_subjects else "."),
        f"- Overlap with new DPARSF-passband folder: `{int(passband.get('n_overlap', 0))}/{int(passband.get('n_reference_subjects', 0))}` subjects.",
        f"- Are overlapping subjects identical/compatible with new DPARSF-passband? Majority comparison: `{comparison_majority}`; identical/near-identical `{int(n_identical)}/{int(n_compared)}`.",
        f"- possible_dparsf_bandpass: `{heuristics.get('possible_dparsf_bandpass')}`.",
        f"- possible_global_10000_normalization: `{heuristics.get('possible_global_10000_normalization')}`.",
        f"- Can we use this folder to rebuild v5 DPARSF-only? `{can_rebuild}`. Treat as blocked unless provenance or overlap comparison confirms these are the DPARSF-bandpass signals Martin wants.",
        "- What remains to ask Martin: whether `/media/diego/Datos/desde_cero/ROISignalsAAL3` came from the same DPARSF passband stage as `FunImgARWSDCFN`, whether any Python/post-DPARSF filtering was applied, and which missing v4/new AD subjects should define the final manifest.",
        "",
        "## Recommended Next Step",
        "",
        "Review `overlap_comparison_with_new_passband.csv` and confirm provenance with Martin before computing connectivity.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    roi_root = resolve(args.roi_root)
    v4_metadata = resolve(args.v4_metadata)
    historical_tensor = resolve(args.historical_tensor)
    new_passband_root = resolve(args.new_passband_root)
    output_dir = resolve(args.output_dir)
    prepare_output_dir(output_dir, args.overwrite)

    desde_files = discover_roisignal_mat_files(roi_root)
    passband_files = discover_signal_files(new_passband_root)
    v4_subjects, v4_status = read_v4_subjects(v4_metadata)
    historical_subjects, historical_tensor_info = read_historical_tensor_subjects(historical_tensor)
    historical_tensor_info["v4_metadata_status"] = v4_status

    inventory, _loaded_meta = inventory_rows(desde_files)
    stats = compute_scale_statistics(desde_files, args.stats_sample_size)
    coverage = coverage_rows(set(desde_files), historical_subjects, v4_subjects, set(passband_files))
    comparison = compare_overlap(desde_files, passband_files)
    missing = missing_subject_rows(set(desde_files), historical_subjects, v4_subjects, set(passband_files))
    heuristics = provenance_heuristics(roi_root, coverage, stats, comparison)
    invalid = pd.DataFrame()
    if not stats.empty and "n_finite_values" in stats.columns:
        finite = pd.to_numeric(stats["n_finite_values"], errors="coerce").fillna(0)
        frac_finite = pd.to_numeric(stats.get("fraction_finite", pd.Series(np.nan, index=stats.index)), errors="coerce").fillna(0)
        invalid = stats.loc[(finite <= 0) | (frac_finite < 0.99)].copy()

    inventory.to_csv(output_dir / "roisignals_inventory.csv", index=False)
    stats.to_csv(output_dir / "roisignals_scale_statistics.csv", index=False)
    coverage.to_csv(output_dir / "coverage_vs_historical_v4.csv", index=False)
    comparison.to_csv(output_dir / "overlap_comparison_with_new_passband.csv", index=False)
    missing.to_csv(output_dir / "missing_subjects_if_used_for_v5.csv", index=False)
    invalid.to_csv(output_dir / "invalid_or_low_finite_signal_subjects.csv", index=False)
    pd.DataFrame([historical_tensor_info | heuristics]).to_csv(output_dir / "provenance_heuristics_summary.csv", index=False)
    write_readme(
        output_dir / "README.md",
        roi_root,
        v4_metadata,
        historical_tensor_info,
        new_passband_root,
        inventory,
        coverage,
        stats,
        comparison,
        heuristics,
    )

    print(f"Wrote desde_cero ROISignalsAAL3 audit to {output_dir}")
    print(f"desde_cero .mat subjects: {len(desde_files)}")
    print(f"historical tensor subjects: {len(historical_subjects)}")
    print(f"v4 metadata subjects: {len(v4_subjects)} status={v4_status}")
    print(f"new passband subjects: {len(passband_files)}")
    if not coverage.empty:
        print(coverage[["reference", "n_overlap", "n_reference_subjects", "n_missing_from_desde_cero", "coverage_fraction"]].to_string(index=False))
    if not comparison.empty:
        print(comparison["comparison_decision"].value_counts().to_string())
    print(f"possible_dparsf_bandpass={heuristics.get('possible_dparsf_bandpass')}")
    print(f"compatible_with_python_bandpass_off_v5={heuristics.get('compatible_with_python_bandpass_off_v5')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
