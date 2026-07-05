#!/usr/bin/env python3
"""Audit signal stage/filtering provenance for /media/diego/Datos/desde_cero/ROISignalsAAL3.

Read-only audit:
- no source files are moved/copied/deleted;
- no final connectivity tensor is computed;
- no model training is run;
- only a 5-subject Pearson channel-1 smoke-test is computed for provenance.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import io as scipy_io
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, welch, windows


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DESDE_CERO_ROOT = Path("/media/diego/Datos/desde_cero/ROISignalsAAL3")
DEFAULT_NEW_PASSBAND_ROOT = Path(
    "/media/diego/My_Book_Diego/vae_AD_data/adni_passband_20260510/"
    "ResultsAAL3/ROISignals_AAL3_FunImgARWSDCFN"
)
DEFAULT_HISTORICAL_TENSOR = (
    PROJECT_ROOT
    / "data"
    / "AAL3_dynamicROIs_fmri_tensor_NeuroEnhanced_v6.5.17_AAL3_131ROIs_"
    "OMST_GCE_Signed_GrangerLag1_ChNorm_ROIreorderedYeo17_ParallelTuned"
    / "GLOBAL_TENSOR_from_AAL3_dynamicROIs_fmri_tensor_NeuroEnhanced_v6.5.17_AAL3_131ROIs_"
    "OMST_GCE_Signed_GrangerLag1_ChNorm_ROIreorderedYeo17_ParallelTuned.npz"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_dparsf_only_rebuild"
    / "desde_cero_signal_stage_audit"
)

ROI_META_PATH = PROJECT_ROOT / "data" / "ROI_MNI_V7_vol.txt"
ROI_ORDER_CSV = PROJECT_ROOT / "data" / "aal3_131_manual_network_order.csv"

SUBJECT_RE = re.compile(r"(?<!\d)(\d{3}_S_\d{4})(?!\d)", re.IGNORECASE)
PROVENANCE_KEYWORDS = [
    "Filter",
    "BandPass",
    "FunImg",
    "ARW",
    "ARWSDC",
    "ARWSDCF",
    "ARWSDCFN",
    "CovRegressed",
    "Normalized",
    "Regressed",
    "DPARSF",
    "DPABI",
    "ROISignals",
    "AAL3",
]

TR_SECONDS = 3.0
LOW_CUT_HZ = 0.01
HIGH_CUT_HZ = 0.08
FILTER_ORDER = 2
TAPER_ALPHA = 0.1
RAW_ROIS = 170
TARGET_LEN_TS = 140
AAL3_MISSING_1BASED = [35, 36, 81, 82]
SMALL_ROI_VOXEL_THRESHOLD = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether desde_cero ROISignalsAAL3 signals are historical/pre-passband or DPARSF-passband.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--desde-cero-root", type=Path, default=DEFAULT_DESDE_CERO_ROOT)
    parser.add_argument("--new-passband-root", type=Path, default=DEFAULT_NEW_PASSBAND_ROOT)
    parser.add_argument("--historical-tensor", type=Path, default=DEFAULT_HISTORICAL_TENSOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--spectral-random-n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260511)
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


def timestamp(value: Any) -> str:
    try:
        return pd.Timestamp(float(value), unit="s").isoformat()
    except Exception:
        return ""


def discover_signal_files(root: Path, suffixes: Sequence[str] = (".mat", ".txt")) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if not root.exists():
        return out
    for suffix in suffixes:
        for path in sorted(root.glob(f"ROISignals_*{suffix}")):
            sid = normalize_subject(path.name)
            if sid and sid not in out:
                out[sid] = path
    return out


def provenance_inventory(desde_cero_root: Path, max_depth: int = 3) -> pd.DataFrame:
    base = desde_cero_root.parent
    rows: List[Dict[str, Any]] = []
    if not base.exists():
        return pd.DataFrame(rows)
    for current, dirs, files in os.walk(base, followlinks=False):
        current_path = Path(current)
        try:
            rel_dir = current_path.relative_to(base)
            depth = 0 if str(rel_dir) == "." else len(rel_dir.parts)
        except ValueError:
            depth = 0
        if depth > max_depth:
            dirs[:] = []
            continue
        names: List[Tuple[Path, str]] = [(current_path, "dir")]
        names.extend((current_path / name, "file") for name in files)
        for path, kind in names:
            rel = path.relative_to(base) if path != base else Path(".")
            low = str(path).lower()
            matched = [kw for kw in PROVENANCE_KEYWORDS if kw.lower() in low]
            stat = path.stat() if path.exists() else None
            rows.append(
                {
                    "base": str(base),
                    "relative_path": str(rel),
                    "path": str(path),
                    "kind": kind,
                    "depth": depth if kind == "dir" else min(depth + 1, max_depth + 1),
                    "size_bytes": stat.st_size if stat and path.is_file() else np.nan,
                    "mtime": timestamp(stat.st_mtime) if stat else "",
                    "matched_keywords": "|".join(matched),
                    "is_candidate_provenance": bool(matched),
                    "suffix": path.suffix.lower() if kind == "file" else "",
                }
            )
        dirs[:] = [name for name in dirs if len((current_path / name).relative_to(base).parts) <= max_depth]
    return pd.DataFrame(rows)


def whosmat_entries(path: Path) -> List[Tuple[str, Tuple[int, ...], str]]:
    return [(name, tuple(int(x) for x in shape), klass) for name, shape, klass in scipy_io.whosmat(path)]


def choose_main_2d_entry(entries: Sequence[Tuple[str, Tuple[int, ...], str]]) -> Optional[Tuple[str, Tuple[int, ...], str]]:
    numeric = {"double", "single", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"}
    candidates = []
    for name, shape, klass in entries:
        if len(shape) == 2 and klass in numeric:
            score = (1 if ("signal" in name.lower() or "roi" in name.lower()) else 0, int(shape[0]) * int(shape[1]))
            candidates.append((*score, name, shape, klass))
    if not candidates:
        return None
    _name_score, _size, name, shape, klass = sorted(candidates, reverse=True)[0]
    return name, shape, klass


def load_mat_signal(path: Path) -> Tuple[Optional[np.ndarray], str, str, str]:
    try:
        entries = whosmat_entries(path)
        chosen = choose_main_2d_entry(entries)
        if chosen is None:
            return None, "", "", f"no_numeric_2d_var:{entries}"
        name, shape, klass = chosen
        data = scipy_io.loadmat(path, variable_names=[name], squeeze_me=False)
        arr = np.asarray(data[name], dtype=np.float64)
        if arr.ndim != 2:
            return None, name, str(tuple(arr.shape)), "loaded_var_not_2d"
        return arr, name, str(tuple(int(x) for x in arr.shape)), f"ok:{klass}"
    except Exception as exc:
        return None, "", "", f"failed:{exc}"


def load_txt_signal(path: Path) -> Tuple[Optional[np.ndarray], str, str, str]:
    try:
        try:
            arr = np.loadtxt(path, delimiter=",", dtype=np.float64)
            status = "ok_comma"
        except Exception:
            arr = np.loadtxt(path, dtype=np.float64)
            status = "ok_whitespace"
        if arr.ndim != 2:
            return None, "", str(tuple(arr.shape)), "txt_not_2d"
        return arr, "", str(tuple(int(x) for x in arr.shape)), status
    except Exception as exc:
        return None, "", "", f"failed:{exc}"


def load_signal(path: Path) -> Tuple[Optional[np.ndarray], str, str, str]:
    if path.suffix.lower() == ".mat":
        return load_mat_signal(path)
    if path.suffix.lower() == ".txt":
        return load_txt_signal(path)
    return None, "", "", f"unsupported_suffix:{path.suffix}"


def finite_values(arr: np.ndarray) -> np.ndarray:
    vals = np.asarray(arr, dtype=np.float64).ravel()
    return vals[np.isfinite(vals)]


def signal_scale_label(mean: float, median: float, std: float, fraction_negative: float, p01: float, p99: float) -> str:
    typical = np.nanmedian(np.abs([mean, median]))
    if np.isfinite(typical) and 1000 <= typical <= 50000 and (not np.isfinite(fraction_negative) or fraction_negative < 0.01):
        return "around_10000_global_scaled"
    centered_tol = max(1e-6, 0.10 * std) if np.isfinite(std) else 1e-6
    if np.isfinite(mean) and abs(mean) <= centered_tol and np.isfinite(fraction_negative) and fraction_negative > 0.01 and p01 < 0 < p99:
        return "zero_centered_or_regressed"
    return "unknown"


def signal_statistics(files: Dict[str, Path]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for sid, path in sorted(files.items()):
        arr, var_name, shape, status = load_mat_signal(path)
        stat = path.stat()
        row: Dict[str, Any] = {
            "SubjectID": sid,
            "path": str(path),
            "file_size_bytes": stat.st_size,
            "mtime": timestamp(stat.st_mtime),
            "main_variable": var_name,
            "shape": shape,
            "load_status": status,
            "is_114_S_6039": sid == "114_S_6039",
        }
        if arr is None:
            row.update(
                {
                    "finite_fraction": 0.0,
                    "fraction_nan_cols": np.nan,
                    "flag_finite_fraction_lt_0_95": True,
                    "heuristic_scale_label": "unknown",
                }
            )
            rows.append(row)
            continue
        vals = finite_values(arr)
        row["n_values"] = int(arr.size)
        row["n_finite_values"] = int(vals.size)
        row["finite_fraction"] = float(vals.size / arr.size) if arr.size else np.nan
        finite_by_col = np.all(np.isfinite(arr), axis=0)
        any_nonfinite_by_col = np.any(~np.isfinite(arr), axis=0)
        row["fraction_nan_cols"] = float(np.mean(~finite_by_col)) if arr.shape[1] else np.nan
        row["fraction_any_nonfinite_cols"] = float(np.mean(any_nonfinite_by_col)) if arr.shape[1] else np.nan
        row["flag_finite_fraction_lt_0_95"] = bool(row["finite_fraction"] < 0.95)
        if vals.size == 0:
            row["heuristic_scale_label"] = "unknown"
            rows.append(row)
            continue
        stats = {
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals)),
            "min": float(np.nanmin(vals)),
            "p01": float(np.nanpercentile(vals, 1)),
            "median": float(np.nanmedian(vals)),
            "p99": float(np.nanpercentile(vals, 99)),
            "max": float(np.nanmax(vals)),
            "fraction_negative": float(np.mean(vals < 0)),
        }
        row.update(stats)
        row["heuristic_scale_label"] = signal_scale_label(
            stats["mean"], stats["median"], stats["std"], stats["fraction_negative"], stats["p01"], stats["p99"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def select_spectral_subjects(files: Dict[str, Path], random_n: int, seed: int) -> List[str]:
    required = ["002_S_0295", "002_S_0685", "002_S_2010", "114_S_6039", "301_S_6592"]
    available = sorted(files)
    rng = np.random.default_rng(seed)
    random_subjects = []
    if available and random_n > 0:
        size = min(random_n, len(available))
        random_subjects = [available[idx] for idx in sorted(rng.choice(len(available), size=size, replace=False).tolist())]
    return list(dict.fromkeys([sid for sid in required if sid in files] + random_subjects))


def spectral_energy_for_subject(sid: str, path: Path, tr: float) -> Dict[str, Any]:
    arr, _var, shape, status = load_mat_signal(path)
    row: Dict[str, Any] = {"SubjectID": sid, "path": str(path), "shape": shape, "load_status": status}
    if arr is None:
        row["spectral_status"] = "not_loaded"
        return row
    if arr.shape[0] == RAW_ROIS and arr.shape[1] != RAW_ROIS:
        arr = arr.T
    if arr.ndim != 2:
        row["spectral_status"] = "not_2d"
        return row
    valid_cols = []
    for idx in range(arr.shape[1]):
        col = arr[:, idx]
        if np.all(np.isfinite(col)) and np.nanstd(col) > 1e-9:
            valid_cols.append(idx)
    if not valid_cols:
        row["spectral_status"] = "no_valid_finite_nonconstant_rois"
        row["n_valid_rois"] = 0
        return row
    fs = 1.0 / tr
    below_fracs: List[float] = []
    pass_fracs: List[float] = []
    above_fracs: List[float] = []
    peak_freqs: List[float] = []
    for idx in valid_cols:
        ts = arr[:, idx].astype(np.float64)
        ts = ts - np.nanmean(ts)
        nperseg = min(128, len(ts))
        freqs, pxx = welch(ts, fs=fs, nperseg=nperseg, detrend=False)
        if pxx.size == 0 or np.nansum(pxx) <= 0:
            continue
        total = float(np.nansum(pxx))
        below_fracs.append(float(np.nansum(pxx[freqs < LOW_CUT_HZ]) / total))
        pass_fracs.append(float(np.nansum(pxx[(freqs >= LOW_CUT_HZ) & (freqs <= HIGH_CUT_HZ)]) / total))
        above_fracs.append(float(np.nansum(pxx[freqs > HIGH_CUT_HZ]) / total))
        peak_freqs.append(float(freqs[int(np.nanargmax(pxx))]))
    if not pass_fracs:
        row["spectral_status"] = "no_valid_psd"
        row["n_valid_rois"] = len(valid_cols)
        return row
    med_below = float(np.nanmedian(below_fracs))
    med_pass = float(np.nanmedian(pass_fracs))
    med_above = float(np.nanmedian(above_fracs))
    if med_below < 0.05 and med_above < 0.15 and med_pass > 0.80:
        label = "bandpassed_like"
    elif med_below > 0.10 or med_above > 0.25:
        label = "not_strictly_bandpassed_like"
    else:
        label = "ambiguous"
    row.update(
        {
            "spectral_status": "ok",
            "n_timepoints": int(arr.shape[0]),
            "n_rois": int(arr.shape[1]),
            "n_valid_rois": int(len(valid_cols)),
            "median_energy_below_0p01": med_below,
            "median_energy_0p01_0p08": med_pass,
            "median_energy_above_0p08": med_above,
            "median_peak_frequency_hz": float(np.nanmedian(peak_freqs)),
            "heuristic_temporal_filter_label": label,
        }
    )
    return row


def spectral_audit(files: Dict[str, Path], random_n: int, seed: int) -> pd.DataFrame:
    subjects = select_spectral_subjects(files, random_n, seed)
    return pd.DataFrame([spectral_energy_for_subject(sid, files[sid], TR_SECONDS) for sid in subjects])


def align_arrays(a: np.ndarray, b: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    if a.shape == b.shape:
        return a, b, "same_orientation"
    if a.shape == b.T.shape:
        return a, b.T, "transposed_new_passband"
    if a.T.shape == b.shape:
        return a.T, b, "transposed_desde_cero"
    return None, None, "shape_mismatch"


def pearson_flat(a: np.ndarray, b: np.ndarray, offdiag_only: bool = False) -> float:
    if offdiag_only and a.ndim == 2 and a.shape[0] == a.shape[1]:
        mask2d = ~np.eye(a.shape[0], dtype=bool)
        av = a[mask2d].ravel()
        bv = b[mask2d].ravel()
    else:
        av = a.ravel()
        bv = b.ravel()
    mask = np.isfinite(av) & np.isfinite(bv)
    av = av[mask]
    bv = bv[mask]
    if av.size < 2 or np.nanstd(av) == 0 or np.nanstd(bv) == 0:
        return np.nan
    return float(np.corrcoef(av, bv)[0, 1])


def compare_desde_vs_new(desde_files: Dict[str, Path], passband_files: Dict[str, Path]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for sid in sorted(set(desde_files) & set(passband_files)):
        a, _avar, ashape, astatus = load_signal(desde_files[sid])
        b, _bvar, bshape, bstatus = load_signal(passband_files[sid])
        row: Dict[str, Any] = {
            "SubjectID": sid,
            "desde_cero_path": str(desde_files[sid]),
            "new_passband_path": str(passband_files[sid]),
            "desde_shape": ashape,
            "new_passband_shape": bshape,
            "desde_status": astatus,
            "new_passband_status": bstatus,
        }
        if a is None or b is None:
            row["comparison_decision"] = "not_loaded"
            rows.append(row)
            continue
        aa, bb, alignment = align_arrays(a, b)
        row["alignment"] = alignment
        if aa is None or bb is None:
            row["comparison_decision"] = "different_shape"
            rows.append(row)
            continue
        diff = aa - bb
        finite_diff = diff[np.isfinite(diff)]
        med_a = float(np.nanmedian(np.abs(aa)))
        med_b = float(np.nanmedian(np.abs(bb)))
        scale_ratio = med_a / med_b if np.isfinite(med_b) and med_b != 0 else np.nan
        corr = pearson_flat(aa, bb)
        row.update(
            {
                "aligned_shape": str(tuple(int(x) for x in aa.shape)),
                "desde_mean": float(np.nanmean(aa)),
                "new_passband_mean": float(np.nanmean(bb)),
                "desde_std": float(np.nanstd(aa)),
                "new_passband_std": float(np.nanstd(bb)),
                "pearson_flat": corr,
                "mean_abs_diff": float(np.nanmean(np.abs(finite_diff))) if finite_diff.size else np.nan,
                "max_abs_diff": float(np.nanmax(np.abs(finite_diff))) if finite_diff.size else np.nan,
                "scale_ratio_median_abs_desde_over_new": scale_ratio,
            }
        )
        if np.allclose(aa, bb, rtol=1e-6, atol=1e-8, equal_nan=True):
            decision = "identical_or_near_identical"
        elif np.isfinite(corr) and corr >= 0.999 and np.isfinite(scale_ratio):
            decision = "same_stage_scaling_difference"
        else:
            decision = "different_signals_or_stages"
        row["comparison_decision"] = decision
        rows.append(row)
    return pd.DataFrame(rows)


def load_historical_subject_ids(path: Path) -> Tuple[List[str], List[str], Dict[str, Any]]:
    info = {"path": str(path), "exists": path.exists(), "status": ""}
    if not path.exists():
        info["status"] = "missing"
        return [], [], info
    with np.load(path, allow_pickle=False) as zf:
        subjects = [normalize_subject(x) for x in zf["subject_ids"].tolist()]
        channels = [str(x) for x in zf["channel_names"].tolist()]
        info["status"] = "ok"
        info["n_subjects"] = len(subjects)
        info["channel_names"] = "|".join(channels)
    return subjects, channels, info


def build_roi_reduction_and_order() -> Dict[str, Any]:
    meta = pd.read_csv(ROI_META_PATH, sep="\t")
    meta["color"] = pd.to_numeric(meta["color"], errors="coerce")
    meta = meta.dropna(subset=["color"]).copy()
    meta["color"] = meta["color"].astype(int)
    missing_0based = [idx - 1 for idx in AAL3_MISSING_1BASED]
    valid_166 = meta[~meta["color"].isin(AAL3_MISSING_1BASED)].copy()
    valid_166 = valid_166.sort_values("color").reset_index(drop=True)
    small_indices_166 = valid_166[valid_166["vol_vox"] < SMALL_ROI_VOXEL_THRESHOLD].index.tolist()
    final_131 = valid_166.drop(index=small_indices_166).reset_index(drop=True)
    mapping = pd.read_csv(ROI_ORDER_CSV)
    mapping = mapping.sort_values("Index_131").reset_index(drop=True)
    label_col = "Yeo17_Label_manual"
    network_col = "Yeo17_Network_manual"
    labs = mapping[label_col].astype(int)
    mapping["__sort_is_bg"] = (labs <= 0).astype(int)
    hemi = mapping["Hemi"].astype(str).str.upper() if "Hemi" in mapping.columns else pd.Series(["U"] * len(mapping))
    mapping["__sort_hemi"] = hemi.map({"L": 0, "R": 1}).fillna(2).astype(int)
    name_col = "nom_l" if "nom_l" in mapping.columns else "nom_c"
    mapping["__sort_name"] = mapping[name_col].astype(str)
    sorted_mapping = mapping.sort_values(["__sort_is_bg", label_col, "__sort_hemi", "__sort_name"], kind="mergesort")
    new_order_indices = sorted_mapping["Index_131"].astype(int).tolist()
    return {
        "missing_0based": missing_0based,
        "small_indices_166": small_indices_166,
        "new_order_indices": new_order_indices,
        "final_roi_names_new_order": sorted_mapping[name_col].astype(str).tolist(),
        "network_labels_new_order": sorted_mapping[network_col].astype(str).tolist(),
        "n_final_rois": len(final_131),
    }


def orient_reduce_reorder(raw: np.ndarray, roi_info: Dict[str, Any]) -> Optional[np.ndarray]:
    if raw.ndim != 2:
        return None
    arr = raw.copy()
    if arr.shape[0] == RAW_ROIS and arr.shape[1] != RAW_ROIS:
        arr = arr.T
    if arr.shape[1] != RAW_ROIS:
        return None
    arr = np.delete(arr, roi_info["missing_0based"], axis=1)
    if arr.shape[1] != RAW_ROIS - len(AAL3_MISSING_1BASED):
        return None
    arr = np.delete(arr, roi_info["small_indices_166"], axis=1)
    if arr.shape[1] != roi_info["n_final_rois"]:
        return None
    return arr[:, roi_info["new_order_indices"]]


def bandpass_filter_signals(sigs: np.ndarray) -> np.ndarray:
    fs = 1.0 / TR_SECONDS
    nyq = 0.5 * fs
    b, a = butter(FILTER_ORDER, [LOW_CUT_HZ / nyq, HIGH_CUT_HZ / nyq], btype="band")
    filtered = np.zeros_like(sigs, dtype=np.float64)
    padlen_required = 3 * max(len(a), len(b))
    for idx in range(sigs.shape[1]):
        roi = sigs[:, idx].copy()
        if len(roi) > padlen_required:
            roi = roi * windows.tukey(len(roi), alpha=TAPER_ALPHA)
        if len(roi) <= padlen_required or np.all(np.isclose(roi, roi[0] if len(roi) else 0.0)):
            filtered[:, idx] = roi
        else:
            filtered[:, idx] = filtfilt(b, a, roi)
    return filtered


def standardize_timeseries(sigs: np.ndarray) -> np.ndarray:
    sigs = np.nan_to_num(sigs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    mean = np.mean(sigs, axis=0, keepdims=True)
    std = np.std(sigs, axis=0, keepdims=True)
    std[std <= 1e-9] = 1.0
    return (sigs - mean) / std


def homogenize_length(sigs: np.ndarray, target_len: int = TARGET_LEN_TS) -> np.ndarray:
    if sigs.shape[0] == target_len:
        return sigs.astype(np.float32)
    if sigs.shape[0] > target_len:
        return sigs[:target_len, :].astype(np.float32)
    out = np.zeros((target_len, sigs.shape[1]), dtype=np.float32)
    if sigs.shape[0] > 1:
        x_old = np.linspace(0, 1, sigs.shape[0])
        x_new = np.linspace(0, 1, target_len)
        for idx in range(sigs.shape[1]):
            out[:, idx] = interp1d(x_old, sigs[:, idx], kind="linear", fill_value="extrapolate")(x_new)
    elif sigs.shape[0] == 1:
        out[:] = sigs[0, :]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def preprocess_for_pearson(raw: np.ndarray, roi_info: Dict[str, Any], apply_python_bandpass: bool) -> Tuple[Optional[np.ndarray], str]:
    reduced = orient_reduce_reorder(raw, roi_info)
    if reduced is None:
        return None, "roi_reduce_or_reorder_failed"
    if not np.isfinite(reduced).any():
        return None, "no_finite_values_after_roi_reduction"
    sigs = bandpass_filter_signals(reduced) if apply_python_bandpass else reduced
    sigs = standardize_timeseries(sigs)
    sigs = homogenize_length(sigs)
    return sigs, "ok"


def fisher_z(corr: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    clean = np.nan_to_num(corr.astype(np.float32), nan=0.0)
    clipped = np.clip(clean, -1.0 + eps, 1.0 - eps)
    z = np.arctanh(clipped)
    np.fill_diagonal(z, 0.0)
    return z.astype(np.float32)


def pearson_full_fisher(ts: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(ts, rowvar=False).astype(np.float32)
    return fisher_z(corr)


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


def reconstruction_smoke(desde_files: Dict[str, Path], historical_tensor: Path) -> pd.DataFrame:
    subjects, channels, tensor_info = load_historical_subject_ids(historical_tensor)
    if not subjects:
        return pd.DataFrame([{"status": f"historical_subject_ids_unavailable:{tensor_info.get('status')}"}])
    subject_to_idx = {sid: idx for idx, sid in enumerate(subjects)}
    preferred = ["002_S_0295", "002_S_0685", "002_S_2010", "301_S_6592", "006_S_0498", "002_S_0413", "002_S_0729"]
    selected = [sid for sid in preferred if sid in desde_files and sid in subject_to_idx]
    for sid in sorted(set(desde_files) & set(subject_to_idx)):
        if len(selected) >= 5:
            break
        if sid not in selected:
            selected.append(sid)
    selected = selected[:5]
    roi_info = build_roi_reduction_and_order()
    rows: List[Dict[str, Any]] = []
    with np.load(historical_tensor, allow_pickle=False) as zf:
        tensor = zf["global_tensor_data"]
        channel_idx = channels.index("Pearson_Full_FisherZ_Signed") if "Pearson_Full_FisherZ_Signed" in channels else 1
        for sid in selected:
            raw, _var, raw_shape, load_status = load_mat_signal(desde_files[sid])
            hist = np.asarray(tensor[subject_to_idx[sid], channel_idx, :, :], dtype=np.float32)
            for bandpass_on in [True, False]:
                row: Dict[str, Any] = {
                    "SubjectID": sid,
                    "raw_shape": raw_shape,
                    "load_status": load_status,
                    "python_bandpass": "ON_0p01_0p08" if bandpass_on else "OFF",
                    "historical_channel_index": channel_idx,
                    "historical_channel_name": channels[channel_idx] if channel_idx < len(channels) else "",
                }
                if raw is None:
                    row["status"] = "raw_not_loaded"
                    rows.append(row)
                    continue
                ts, pre_status = preprocess_for_pearson(raw, roi_info, apply_python_bandpass=bandpass_on)
                row["preprocess_status"] = pre_status
                if ts is None:
                    row["status"] = "preprocess_failed"
                    rows.append(row)
                    continue
                pearson = pearson_full_fisher(ts)
                pearson_scaled = robustscale_offdiag(pearson)
                mask = ~np.eye(hist.shape[0], dtype=bool)
                diff = pearson_scaled - hist
                off_diff = diff[mask]
                row.update(
                    {
                        "status": "ok",
                        "processed_ts_shape": str(tuple(int(x) for x in ts.shape)),
                        "pearson_scaled_vs_hist_corr_offdiag": pearson_flat(pearson_scaled, hist, offdiag_only=True),
                        "pearson_scaled_vs_hist_mae_offdiag": float(np.nanmean(np.abs(off_diff))),
                        "pearson_scaled_vs_hist_rmse_offdiag": float(np.sqrt(np.nanmean(off_diff ** 2))),
                        "pearson_scaled_vs_hist_max_abs_offdiag": float(np.nanmax(np.abs(off_diff))),
                        "hist_offdiag_std": float(np.nanstd(hist[mask])),
                        "computed_offdiag_std": float(np.nanstd(pearson_scaled[mask])),
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def decide_smoke_winner(smoke: pd.DataFrame) -> Dict[str, Any]:
    ok = smoke[smoke.get("status", pd.Series(dtype=str)).eq("ok")].copy() if not smoke.empty else pd.DataFrame()
    if ok.empty:
        return {"smoke_winner": "unknown", "smoke_reason": "no_ok_rows"}
    grouped = ok.groupby("python_bandpass").agg(
        median_corr=("pearson_scaled_vs_hist_corr_offdiag", "median"),
        median_mae=("pearson_scaled_vs_hist_mae_offdiag", "median"),
        n=("SubjectID", "nunique"),
    )
    out = {f"{idx}_{col}": float(row[col]) if col != "n" else int(row[col]) for idx, row in grouped.iterrows() for col in grouped.columns}
    if "ON_0p01_0p08" in grouped.index and "OFF" in grouped.index:
        on = grouped.loc["ON_0p01_0p08"]
        off = grouped.loc["OFF"]
        corr_margin = float(on["median_corr"] - off["median_corr"])
        mae_margin = float(off["median_mae"] - on["median_mae"])
        if corr_margin > 0.01 and mae_margin > 0:
            winner = "python_bandpass_ON"
        elif corr_margin < -0.01 and mae_margin < 0:
            winner = "python_bandpass_OFF"
        elif on["median_mae"] < off["median_mae"]:
            winner = "python_bandpass_ON_slight"
        else:
            winner = "python_bandpass_OFF_slight"
        out["smoke_winner"] = winner
        out["smoke_reason"] = f"corr_margin_ON_minus_OFF={corr_margin:.4g}; mae_margin_OFF_minus_ON={mae_margin:.4g}"
    else:
        out["smoke_winner"] = "unknown"
        out["smoke_reason"] = "missing_on_or_off_rows"
    return out


def summarize_stage(
    stats: pd.DataFrame,
    spectral: pd.DataFrame,
    overlap: pd.DataFrame,
    smoke: pd.DataFrame,
    provenance: pd.DataFrame,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["n_subjects"] = int(len(stats))
    out["n_around_10000"] = int((stats["heuristic_scale_label"] == "around_10000_global_scaled").sum()) if not stats.empty else 0
    out["n_finite_lt_0_95"] = int(stats["flag_finite_fraction_lt_0_95"].sum()) if "flag_finite_fraction_lt_0_95" in stats else 0
    ok_spec = spectral[spectral.get("spectral_status", pd.Series(dtype=str)).eq("ok")].copy() if not spectral.empty else pd.DataFrame()
    if not ok_spec.empty:
        out["median_spectral_below_0p01"] = float(ok_spec["median_energy_below_0p01"].median())
        out["median_spectral_passband_0p01_0p08"] = float(ok_spec["median_energy_0p01_0p08"].median())
        out["median_spectral_above_0p08"] = float(ok_spec["median_energy_above_0p08"].median())
        out["spectral_majority_label"] = str(ok_spec["heuristic_temporal_filter_label"].mode().iloc[0])
    else:
        out["spectral_majority_label"] = "unknown"
    if not overlap.empty:
        out["n_overlap_new_passband"] = int(len(overlap))
        out["overlap_majority_decision"] = str(overlap["comparison_decision"].mode().iloc[0])
    else:
        out["n_overlap_new_passband"] = 0
        out["overlap_majority_decision"] = "none"
    out.update(decide_smoke_winner(smoke))
    if not provenance.empty:
        candidates = provenance[provenance["is_candidate_provenance"].astype(bool)]
        out["n_provenance_keyword_hits"] = int(len(candidates))
        out["provenance_keyword_hit_preview"] = "|".join(candidates["relative_path"].astype(str).head(20).tolist())
    return out


def write_readme(
    path: Path,
    desde_root: Path,
    new_root: Path,
    historical_tensor: Path,
    stats: pd.DataFrame,
    spectral: pd.DataFrame,
    overlap: pd.DataFrame,
    smoke: pd.DataFrame,
    summary: Dict[str, Any],
) -> None:
    n_subjects = summary.get("n_subjects", 0)
    n_around = summary.get("n_around_10000", 0)
    n_finite_bad = summary.get("n_finite_lt_0_95", 0)
    is_114_bad = False
    if not stats.empty and "SubjectID" in stats.columns:
        hit114 = stats[stats["SubjectID"].astype(str).eq("114_S_6039")]
        is_114_bad = bool(not hit114.empty and float(hit114.iloc[0].get("finite_fraction", 1.0)) < 0.95)
    spectral_label = summary.get("spectral_majority_label", "unknown")
    smoke_winner = summary.get("smoke_winner", "unknown")
    overlap_decision = summary.get("overlap_majority_decision", "none")
    can_use = "NO"
    if spectral_label == "bandpassed_like" and "ON" not in str(smoke_winner) and overlap_decision in {"identical_or_near_identical", "same_stage_scaling_difference"}:
        can_use = "POSSIBLY_AFTER_MARTIN_CONFIRMATION"
    lines = [
        "# desde_cero Signal Stage And Filtering Audit",
        "",
        "Read-only audit. No source files were moved, copied, deleted, or rewritten. No final tensor extraction and no training were run.",
        "",
        "## Inputs",
        "",
        f"- desde_cero root: `{desde_root}` exists=`{desde_root.exists()}`",
        f"- new passband root: `{new_root}` exists=`{new_root.exists()}`",
        f"- historical tensor: `{historical_tensor}` exists=`{historical_tensor.exists()}`",
        "",
        "## Explicit Answers",
        "",
        f"- Is desde_cero likely the historical source folder? `YES/Likely`: it has `{n_subjects}` historical-style `.mat` subjects and previous audit showed complete `431/431` historical tensor coverage.",
        f"- Are signals around 10000? `{'YES' if n_around / max(n_subjects, 1) > 0.8 else 'NO/UNKNOWN'}` (`{n_around}/{n_subjects}` subjects labelled around_10000_global_scaled).",
        f"- `114_S_6039` finite signal issue? `{'YES' if is_114_bad else 'NO'}`.",
        f"- Subjects with finite_fraction < 0.95: `{n_finite_bad}`.",
        f"- Do they appear already temporally bandpassed? `{spectral_label}`; median energy below 0.01=`{summary.get('median_spectral_below_0p01', np.nan)}`, within 0.01-0.08=`{summary.get('median_spectral_passband_0p01_0p08', np.nan)}`, above 0.08=`{summary.get('median_spectral_above_0p08', np.nan)}`.",
        f"- Does Python bandpass ON or OFF better reproduce historical tensor channel 1? `{smoke_winner}` ({summary.get('smoke_reason', '')}).",
        f"- Are desde_cero and new-passband the same stage? `{overlap_decision}` over `{summary.get('n_overlap_new_passband', 0)}` overlapping subjects.",
        f"- Can we use desde_cero for final DPARSF-only v5? `{can_use}`. Current evidence is not enough to treat it as DPARSF-bandpass-only.",
        "- What must be confirmed with Martin: whether `ROISignalsAAL3` was generated before or after DPARSF temporal filtering, whether the old Python bandpass was part of the historical tensor recipe, why `301_S_6592` differs from the new passband folder, and how to handle `114_S_6039` plus missing new AD subjects.",
        "",
        "## Files To Review",
        "",
        "- `provenance_file_inventory.csv`",
        "- `signal_statistics_all.csv`",
        "- `spectral_energy_audit.csv`",
        "- `overlap_desde_cero_vs_new_passband.csv`",
        "- `historical_channel1_reconstruction_smoke.csv`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    desde_root = resolve(args.desde_cero_root)
    new_root = resolve(args.new_passband_root)
    historical_tensor = resolve(args.historical_tensor)
    output_dir = resolve(args.output_dir)
    prepare_output_dir(output_dir, args.overwrite)

    desde_files = discover_signal_files(desde_root, suffixes=(".mat",))
    new_files = discover_signal_files(new_root, suffixes=(".mat", ".txt"))
    provenance = provenance_inventory(desde_root, max_depth=3)
    stats = signal_statistics(desde_files)
    spectral = spectral_audit(desde_files, args.spectral_random_n, args.seed)
    overlap = compare_desde_vs_new(desde_files, new_files)
    smoke = reconstruction_smoke(desde_files, historical_tensor)
    summary = summarize_stage(stats, spectral, overlap, smoke, provenance)

    provenance.to_csv(output_dir / "provenance_file_inventory.csv", index=False)
    stats.to_csv(output_dir / "signal_statistics_all.csv", index=False)
    spectral.to_csv(output_dir / "spectral_energy_audit.csv", index=False)
    overlap.to_csv(output_dir / "overlap_desde_cero_vs_new_passband.csv", index=False)
    smoke.to_csv(output_dir / "historical_channel1_reconstruction_smoke.csv", index=False)
    pd.DataFrame([summary]).to_csv(output_dir / "stage_filtering_summary.csv", index=False)
    write_readme(output_dir / "README.md", desde_root, new_root, historical_tensor, stats, spectral, overlap, smoke, summary)

    print(f"Wrote stage/filtering audit to {output_dir}")
    print(f"desde_cero subjects: {len(desde_files)}")
    print(f"new passband subjects: {len(new_files)}")
    print(f"provenance keyword hits: {summary.get('n_provenance_keyword_hits', 0)}")
    print(f"spectral_majority_label: {summary.get('spectral_majority_label')}")
    print(f"smoke_winner: {summary.get('smoke_winner')} {summary.get('smoke_reason')}")
    print(f"overlap_majority_decision: {summary.get('overlap_majority_decision')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
