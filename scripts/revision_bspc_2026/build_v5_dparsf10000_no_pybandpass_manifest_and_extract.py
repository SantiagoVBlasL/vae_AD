#!/usr/bin/env python3
"""Build ADNI v5 DPARSF-10000 no-Python-bandpass manifest and gated extraction.

Modes:
- --manifest-only: build manifest/metadata/exclusion reports only.
- --smoke-n-subjects N: build manifest and extract a small smoke tensor.
- --run-full: explicitly enable full extraction for all v5_candidate subjects.

The extraction starts from ROI time series and never reuses historical tensors.
Python bandpass is intentionally disabled in every mode.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.io as scipy_io
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
from sklearn.feature_selection import mutual_info_regression

try:
    from dcor import distance_correlation
except Exception:  # pragma: no cover - environment dependent
    distance_correlation = None

try:
    from dyconnmap.graphs.threshold import threshold_omst_global_cost_efficiency
except Exception:  # pragma: no cover - environment dependent
    threshold_omst_global_cost_efficiency = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DESDE_CERO_ROOT = Path("/media/diego/Datos/desde_cero/ROISignalsAAL3")
DEFAULT_NEW_PASSBAND_ROOT = Path(
    "/media/diego/My_Book_Diego/vae_AD_data/adni_passband_20260510/"
    "ResultsAAL3/ROISignals_AAL3_FunImgARWSDCFN"
)
DEFAULT_V4_METADATA = (
    PROJECT_ROOT
    / "data"
    / "revision_bspc_2026"
    / "adni_expanded_v4_all_available"
    / "subject_metadata_adni_expanded_v4_all_available.csv"
)
DEFAULT_ORIGINAL_METADATA = PROJECT_ROOT / "data" / "SubjectsData_AAL3_procesado2.csv"
DEFAULT_ADNI_DOWNLOAD_NOW = PROJECT_ROOT / "data" / "adni_download_now.csv"
DEFAULT_OUTPUT_ROOT = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_dparsf10000_no_pybandpass"
)
DEFAULT_LOCAL_SYMLINK = (
    PROJECT_ROOT
    / "data"
    / "revision_bspc_2026"
    / "adni_expanded_v5_dparsf10000_no_pybandpass"
)

ROI_META_PATH = PROJECT_ROOT / "data" / "ROI_MNI_V7_vol.txt"
ROI_ORDER_CSV = PROJECT_ROOT / "data" / "aal3_131_manual_network_order.csv"

DATASET_NAME = "adni_expanded_v5_dparsf10000_no_pybandpass"
PREPROCESSING_SOURCE = "DPARSF_ROISignals_AAL3_10000"
PYTHON_BANDPASS_APPLIED = False
TR_SECONDS = 3.0
TARGET_LEN = 140
RAW_ROIS = 170
OUTPUT_ROIS = 131
N_NEIGHBORS_MI = 5
DFC_WIN_POINTS = 30
DFC_STEP = 5
GRANGER_LAG = 1
SMALL_ROI_VOXEL_THRESHOLD = 100
AAL3_MISSING_1BASED = [35, 36, 81, 82]

CHANNEL_NAMES = [
    "Pearson_OMST_GCE_Signed_Weighted",
    "Pearson_Full_FisherZ_Signed",
    "MI_KNN_Symmetric",
    "dFC_AbsDiffMean",
    "dFC_StdDev",
    "DistanceCorr",
    "Granger_F_lag1",
]

# Manual diagnosis overrides: applied after automated metadata merging.
# Source evidence is documented in:
#   results/revision_bspc_2026/adni_v5_dparsf_only_rebuild/pre_full_extraction_qc/unknown_subjects_resolution.csv
MANUAL_DIAGNOSIS_OVERRIDES: Dict[str, str] = {
    "035_S_6927": "AD",  # Confirmed from AD_fMRI_4_28_2026.csv and idaSearch_4_03_2026.csv (Age~60, Sex=F)
}

# Subjects to exclude from supervised training regardless of v5_candidate status.
# These remain in the manifest for unsupervised/reconstruction tasks.
MANUAL_EXCLUDE_FROM_SUPERVISED: set = {
    "128_S_2002",  # No diagnosis in any ADNI source; signal scale anomalous (96.9% near-zero, scale_label=unknown)
}

SUBJECT_RE = r"(?<!\d)(\d{3}_S_\d{4})(?!\d)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ADNI v5 manifest and optionally extract smoke/full tensors with Python bandpass OFF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--desde-cero-root", type=Path, default=DEFAULT_DESDE_CERO_ROOT)
    parser.add_argument("--new-passband-root", type=Path, default=DEFAULT_NEW_PASSBAND_ROOT)
    parser.add_argument("--v4-metadata", type=Path, default=DEFAULT_V4_METADATA)
    parser.add_argument("--original-metadata", type=Path, default=DEFAULT_ORIGINAL_METADATA)
    parser.add_argument("--adni-download-now", type=Path, default=DEFAULT_ADNI_DOWNLOAD_NOW)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--local-symlink", type=Path, default=DEFAULT_LOCAL_SYMLINK)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--smoke-n-subjects", type=int, default=0)
    parser.add_argument("--run-full", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip subjects whose individual tensor already exists and passes QC "
            "(correct shape, dtype, no NaNs, loadable). Corrupted or incomplete "
            "tensors are recalculated. Only valid with --run-full."
        ),
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalize_subject(value: Any) -> str:
    if pd.isna(value):
        return ""
    import re

    match = re.search(SUBJECT_RE, str(value).strip().upper())
    return match.group(1).upper() if match else ""


def safe_float(value: Any) -> float:
    try:
        if value == "":
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def timestamp_now() -> str:
    return pd.Timestamp.now().isoformat()


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


def choose_main_2d_entry(entries: Sequence[Tuple[str, Tuple[int, ...], str]]) -> Optional[Tuple[str, Tuple[int, ...], str]]:
    numeric = {"double", "single", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"}
    candidates = []
    for name, shape, klass in entries:
        if len(shape) == 2 and klass in numeric:
            score = (1 if ("signal" in name.lower() or "roi" in name.lower()) else 0, int(shape[0]) * int(shape[1]))
            candidates.append((*score, name, shape, klass))
    if not candidates:
        return None
    _score_name, _score_size, name, shape, klass = sorted(candidates, reverse=True)[0]
    return name, shape, klass


def load_mat_signal(path: Path) -> Tuple[Optional[np.ndarray], str, str, str]:
    try:
        entries = [(name, tuple(int(x) for x in shape), klass) for name, shape, klass in scipy_io.whosmat(path)]
        chosen = choose_main_2d_entry(entries)
        if chosen is None:
            return None, "", "", f"no_numeric_2d_var:{entries}"
        name, _shape, klass = chosen
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


def scale_label(vals: np.ndarray) -> str:
    if vals.size == 0:
        return "unknown"
    mean = float(np.nanmean(vals))
    median = float(np.nanmedian(vals))
    std = float(np.nanstd(vals))
    p01 = float(np.nanpercentile(vals, 1))
    p99 = float(np.nanpercentile(vals, 99))
    fraction_negative = float(np.mean(vals < 0))
    typical = np.nanmedian(np.abs([mean, median]))
    if np.isfinite(typical) and 1000 <= typical <= 50000 and fraction_negative < 0.01:
        return "around_10000_global_scaled"
    centered_tol = max(1e-6, 0.10 * std) if np.isfinite(std) else 1e-6
    if np.isfinite(mean) and abs(mean) <= centered_tol and fraction_negative > 0.01 and p01 < 0 < p99:
        return "zero_centered_or_regressed"
    return "unknown"


def signal_qc(path: Path) -> Dict[str, Any]:
    arr, var_name, shape, status = load_signal(path)
    row: Dict[str, Any] = {
        "signal_path": str(path),
        "shape": shape,
        "main_variable": var_name,
        "load_status": status,
        "finite_fraction": 0.0,
        "scale_label": "unknown",
    }
    if arr is None or arr.size == 0:
        return row
    vals = finite_values(arr)
    row["finite_fraction"] = float(vals.size / arr.size)
    row["scale_label"] = scale_label(vals)
    row["raw_mean"] = float(np.nanmean(vals)) if vals.size else np.nan
    row["raw_std"] = float(np.nanstd(vals)) if vals.size else np.nan
    row["raw_min"] = float(np.nanmin(vals)) if vals.size else np.nan
    row["raw_max"] = float(np.nanmax(vals)) if vals.size else np.nan
    return row


def first_existing_col(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower = {str(c).strip().lower(): str(c) for c in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        hit = lower.get(candidate.lower())
        if hit:
            return hit
    return None


def normalize_group(value: Any) -> str:
    text = str(value).strip().upper()
    if text in {"AD", "CN", "MCI"}:
        return text
    if "DEMENT" in text or "ALZ" in text:
        return "AD"
    if "CONTROL" in text or text == "NORMAL":
        return "CN"
    if "MCI" in text:
        return "MCI"
    return "" if text in {"", "NAN", "NONE"} else str(value).strip()


def metadata_table(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["SubjectID"])
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    sid_col = first_existing_col(df.columns, ["SubjectID", "Subject", "PTID", "subject_id"])
    if sid_col is None:
        return pd.DataFrame(columns=["SubjectID"])
    out = pd.DataFrame({"SubjectID": df[sid_col].map(normalize_subject)})
    col_map = {
        "ResearchGroup_Mapped": ["ResearchGroup_Mapped", "ResearchGroup", "DX", "Group"],
        "Age": ["Age", "AGE"],
        "Sex": ["Sex", "PTGENDER", "Gender"],
        "Manufacturer": ["Manufacturer", "MANUFACTURER"],
        "Site3": ["Site3", "SITE", "SITEID"],
        "ImageID": ["ImageID", "IMAGEUID", "Image Data ID"],
        "Visit": ["Visit", "VISCODE", "VisitCode"],
    }
    for target, candidates in col_map.items():
        col = first_existing_col(df.columns, candidates)
        out[target] = df[col].astype(str) if col else ""
    out["ResearchGroup_Mapped"] = out["ResearchGroup_Mapped"].map(normalize_group)
    out["metadata_source"] = label
    out = out[out["SubjectID"].astype(bool)].drop_duplicates("SubjectID", keep="first")
    return out


def combine_metadata(v4: Path, original: Path, adni_now: Path) -> pd.DataFrame:
    sources = [
        metadata_table(v4, "v4_metadata"),
        metadata_table(original, "original_metadata"),
        metadata_table(adni_now, "adni_download_now"),
    ]
    all_subjects = sorted(set().union(*(set(df["SubjectID"]) for df in sources if not df.empty)))
    rows: List[Dict[str, Any]] = []
    fields = ["ResearchGroup_Mapped", "Age", "Sex", "Manufacturer", "Site3", "ImageID", "Visit"]
    indexed = [df.set_index("SubjectID", drop=False) for df in sources]
    for sid in all_subjects:
        row: Dict[str, Any] = {"SubjectID": sid}
        metadata_sources = []
        for field in fields:
            value = ""
            for df in indexed:
                if sid in df.index:
                    candidate = str(df.loc[sid, field]) if field in df.columns else ""
                    if candidate and candidate.lower() not in {"nan", "none"}:
                        value = candidate
                        break
            row[field] = value
        for df in indexed:
            if sid in df.index:
                metadata_sources.append(str(df.loc[sid, "metadata_source"]))
        row["metadata_source"] = "|".join(metadata_sources)
        rows.append(row)
    return pd.DataFrame(rows)


def build_manifest(
    desde_root: Path,
    new_root: Path,
    v4_metadata: Path,
    original_metadata: Path,
    adni_download_now: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    desde_files = discover_signal_files(desde_root, suffixes=(".mat",))
    new_files = discover_signal_files(new_root, suffixes=(".mat", ".txt"))
    all_subjects = sorted(set(desde_files) | set(new_files))
    metadata = combine_metadata(v4_metadata, original_metadata, adni_download_now).set_index("SubjectID", drop=False)
    rows: List[Dict[str, Any]] = []
    excluded_rows: List[Dict[str, Any]] = []
    duplicate_rows: List[Dict[str, Any]] = []
    for sid in all_subjects:
        has_desde = sid in desde_files
        has_new = sid in new_files
        chosen = desde_files[sid] if has_desde else new_files[sid]
        source_root = desde_root if has_desde else new_root
        source_label = "desde_cero_historical_10000" if has_desde else "new_passband_20260510_10000"
        reason = "preferred_desde_cero_continuity" if has_desde and has_new else ("only_desde_cero" if has_desde else "only_new_passband_absent_in_desde_cero")
        if has_desde and has_new:
            duplicate_rows.append(
                {
                    "SubjectID": sid,
                    "desde_cero_path": str(desde_files[sid]),
                    "new_passband_path": str(new_files[sid]),
                    "chosen_path": str(chosen),
                    "chosen_source_label": source_label,
                    "resolution_reason": "prefer_desde_cero_for_historical_continuity",
                }
            )
        qc = signal_qc(chosen)
        exclusion_reasons: List[str] = []
        if not str(qc["load_status"]).startswith("ok"):
            exclusion_reasons.append(f"load_failed:{qc['load_status']}")
        if safe_float(qc["finite_fraction"]) < 0.95:
            exclusion_reasons.append("finite_fraction_lt_0.95")
        if sid == "114_S_6039" and safe_float(qc["finite_fraction"]) < 0.95:
            exclusion_reasons.append("114_S_6039_invalid_nonfinite_signal")
        v5_candidate = len(exclusion_reasons) == 0
        md = metadata.loc[sid].to_dict() if sid in metadata.index else {"SubjectID": sid}
        row = {
            "SubjectID": sid,
            "signal_path": str(chosen),
            "source_root": str(source_root),
            "source_label": source_label,
            "source_selection_reason": reason,
            "present_in_desde_cero": has_desde,
            "present_in_new_passband": has_new,
            "shape": qc["shape"],
            "finite_fraction": qc["finite_fraction"],
            "scale_label": qc["scale_label"],
            "load_status": qc["load_status"],
            "preprocessing_source": PREPROCESSING_SOURCE,
            "python_bandpass_applied": PYTHON_BANDPASS_APPLIED,
            "TR": TR_SECONDS,
            "target_len": TARGET_LEN,
            "roi_input_count": RAW_ROIS,
            "roi_output_count": OUTPUT_ROIS,
            "v5_candidate": v5_candidate,
            "exclusion_reason": "|".join(exclusion_reasons),
        }
        for field in ["ResearchGroup_Mapped", "Age", "Sex", "Manufacturer", "Site3", "ImageID", "Visit", "metadata_source"]:
            row[field] = md.get(field, "")
        # Apply manual diagnosis overrides (takes priority over automated metadata merge)
        if sid in MANUAL_DIAGNOSIS_OVERRIDES:
            row["ResearchGroup_Mapped"] = MANUAL_DIAGNOSIS_OVERRIDES[sid]
            row["metadata_source"] = (
                ("manual_override|" + row["metadata_source"]).rstrip("|")
                if row["metadata_source"]
                else "manual_override"
            )
        row["exclude_from_supervised"] = sid in MANUAL_EXCLUDE_FROM_SUPERVISED
        rows.append(row)
        if not v5_candidate:
            excluded_rows.append(row.copy())
    manifest = pd.DataFrame(rows)
    metadata_out = manifest[[
        "SubjectID",
        "ResearchGroup_Mapped",
        "Age",
        "Sex",
        "Manufacturer",
        "Site3",
        "ImageID",
        "Visit",
        "metadata_source",
        "source_label",
        "v5_candidate",
        "exclusion_reason",
        "exclude_from_supervised",
    ]].copy()
    excluded = pd.DataFrame(excluded_rows)
    duplicates = pd.DataFrame(duplicate_rows)
    return manifest, metadata_out, excluded, duplicates


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
    return {
        "missing_0based": missing_0based,
        "small_indices_166": small_indices_166,
        "new_order_indices": sorted_mapping["Index_131"].astype(int).tolist(),
        "roi_names_new_order": sorted_mapping[name_col].astype(str).tolist(),
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


def standardize_timeseries(sigs: np.ndarray) -> np.ndarray:
    sigs = np.nan_to_num(sigs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    mean = np.mean(sigs, axis=0, keepdims=True)
    std = np.std(sigs, axis=0, keepdims=True)
    std[std <= 1e-9] = 1.0
    return (sigs - mean) / std


def homogenize_length(sigs: np.ndarray, target_len: int = TARGET_LEN) -> np.ndarray:
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


def preprocess_timeseries_no_pybandpass(raw: np.ndarray, roi_info: Dict[str, Any]) -> Tuple[Optional[np.ndarray], str, Dict[str, Any]]:
    reduced = orient_reduce_reorder(raw, roi_info)
    if reduced is None:
        return None, "roi_reduce_or_reorder_failed", {}
    if not np.isfinite(reduced).any():
        return None, "no_finite_values_after_roi_reduction", {}
    raw_qc = {
        "reduced_shape": str(tuple(int(x) for x in reduced.shape)),
        "reduced_finite_fraction": float(np.isfinite(reduced).mean()),
        "reduced_mean": float(np.nanmean(reduced)),
        "reduced_std": float(np.nanstd(reduced)),
        "python_bandpass_applied": False,
    }
    standardized = standardize_timeseries(reduced)
    ts = homogenize_length(standardized)
    raw_qc.update(
        {
            "processed_shape": str(tuple(int(x) for x in ts.shape)),
            "processed_nan_count": int(np.isnan(ts).sum()),
            "processed_min": float(np.nanmin(ts)),
            "processed_max": float(np.nanmax(ts)),
        }
    )
    return ts, "ok", raw_qc


def fisher_z(corr: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    clean = np.nan_to_num(corr.astype(np.float32), nan=0.0)
    clipped = np.clip(clean, -1.0 + eps, 1.0 - eps)
    z = np.arctanh(clipped)
    np.fill_diagonal(z, 0.0)
    return z.astype(np.float32)


def pearson_full(ts: np.ndarray) -> np.ndarray:
    return fisher_z(np.corrcoef(ts, rowvar=False).astype(np.float32))


def pearson_omst(ts: np.ndarray) -> Tuple[np.ndarray, str]:
    z = pearson_full(ts)
    weights = np.abs(z).astype(np.float32)
    np.fill_diagonal(weights, 0.0)
    if threshold_omst_global_cost_efficiency is not None and not np.allclose(weights, 0):
        try:
            outputs = threshold_omst_global_cost_efficiency(weights, n_msts=None)
            if isinstance(outputs, tuple) and len(outputs) >= 2:
                mask = (np.asarray(outputs[1]).astype(np.float32) > 0).astype(np.float32)
                out = z * mask
                np.fill_diagonal(out, 0.0)
                return out.astype(np.float32), "dyconnmap_threshold_omst_global_cost_efficiency"
        except Exception as exc:
            status = f"dyconnmap_failed_fallback_mst:{exc}"
        else:
            status = "dyconnmap_unexpected_output_fallback_mst"
    else:
        status = "dyconnmap_unavailable_fallback_mst"
    try:
        import networkx as nx

        graph = nx.Graph()
        n = weights.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                w = float(weights[i, j])
                if w > 0:
                    graph.add_edge(i, j, weight=-w)
        mst = nx.minimum_spanning_tree(graph, weight="weight")
        mask = np.zeros_like(weights, dtype=np.float32)
        for i, j in mst.edges():
            mask[i, j] = mask[j, i] = 1.0
        out = z * mask
        np.fill_diagonal(out, 0.0)
        return out.astype(np.float32), status
    except Exception as exc:
        return z.astype(np.float32), f"{status}|fallback_full_pearson:{exc}"


def _mi_pair(ts: np.ndarray, i: int, j: int, n_neighbors: int) -> Tuple[int, int, float]:
    try:
        mij = mutual_info_regression(
            ts[:, i].reshape(-1, 1), ts[:, j], n_neighbors=n_neighbors, random_state=42, discrete_features=False
        )[0]
        mji = mutual_info_regression(
            ts[:, j].reshape(-1, 1), ts[:, i], n_neighbors=n_neighbors, random_state=42, discrete_features=False
        )[0]
        return i, j, float((mij + mji) / 2.0)
    except Exception:
        return i, j, 0.0


def mi_knn(ts: np.ndarray, n_neighbors: int = N_NEIGHBORS_MI, n_jobs: int = 1) -> np.ndarray:
    n_tp, n_rois = ts.shape
    out = np.zeros((n_rois, n_rois), dtype=np.float32)
    if n_tp <= n_neighbors:
        return out
    pairs = [(i, j) for i in range(n_rois) for j in range(i + 1, n_rois)]
    jobs = max(1, int(n_jobs))
    if jobs == 1:
        results = [_mi_pair(ts, i, j, n_neighbors) for i, j in pairs]
    else:
        results = Parallel(n_jobs=jobs, prefer="threads")(
            delayed(_mi_pair)(ts, i, j, n_neighbors) for i, j in pairs
        )
    for i, j, value in results:
        out[i, j] = out[j, i] = value
    return out


def dfc_absdiff_mean(ts: np.ndarray, win: int = DFC_WIN_POINTS, step: int = DFC_STEP) -> np.ndarray:
    n_tp, n_rois = ts.shape
    out = np.zeros((n_rois, n_rois), dtype=np.float32)
    if n_tp < win:
        return out
    prev = None
    acc = np.zeros((n_rois, n_rois), dtype=np.float64)
    n_diff = 0
    for start in range(0, n_tp - win + 1, step):
        corr = np.nan_to_num(np.corrcoef(ts[start : start + win], rowvar=False).astype(np.float32), nan=0.0)
        current = np.abs(corr)
        np.fill_diagonal(current, 0.0)
        if prev is not None:
            acc += np.abs(current - prev)
            n_diff += 1
        prev = current
    if n_diff:
        out = (acc / n_diff).astype(np.float32)
        np.fill_diagonal(out, 0.0)
    return out


def dfc_stddev(ts: np.ndarray, win: int = DFC_WIN_POINTS, step: int = DFC_STEP) -> np.ndarray:
    n_tp, n_rois = ts.shape
    mats = []
    if n_tp < win:
        return np.zeros((n_rois, n_rois), dtype=np.float32)
    for start in range(0, n_tp - win + 1, step):
        corr = np.nan_to_num(np.corrcoef(ts[start : start + win], rowvar=False).astype(np.float32), nan=0.0)
        np.fill_diagonal(corr, 0.0)
        mats.append(corr)
    if len(mats) < 2:
        return np.zeros((n_rois, n_rois), dtype=np.float32)
    out = np.std(np.stack(mats, axis=0), axis=0).astype(np.float32)
    np.fill_diagonal(out, 0.0)
    return out


def distance_corr(ts: np.ndarray) -> Tuple[np.ndarray, str]:
    n_rois = ts.shape[1]
    out = np.zeros((n_rois, n_rois), dtype=np.float32)
    if distance_correlation is not None:
        for i in range(n_rois):
            for j in range(i + 1, n_rois):
                try:
                    out[i, j] = out[j, i] = float(distance_correlation(ts[:, i], ts[:, j]))
                except Exception:
                    out[i, j] = out[j, i] = 0.0
        return out, "dcor.distance_correlation"
    # Conservative fallback: absolute Pearson, clearly reported in status.
    corr = np.abs(np.nan_to_num(np.corrcoef(ts, rowvar=False).astype(np.float32), nan=0.0))
    np.fill_diagonal(corr, 0.0)
    return corr.astype(np.float32), "fallback_abs_pearson_dcor_unavailable"


def _ols_rss(y: np.ndarray, x: np.ndarray) -> float:
    try:
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ beta
        return float(np.dot(resid, resid))
    except Exception:
        return np.nan


def _granger_f_lag1_fast(cause: np.ndarray, effect: np.ndarray) -> float:
    # Tests whether cause[t-1] improves prediction of effect[t] beyond effect[t-1].
    y = effect[1:]
    effect_lag = effect[:-1]
    cause_lag = cause[:-1]
    if np.std(y) < 1e-9 or np.std(effect_lag) < 1e-9 or np.std(cause_lag) < 1e-9:
        return 0.0
    ones = np.ones_like(y)
    restricted = np.column_stack([ones, effect_lag])
    unrestricted = np.column_stack([ones, effect_lag, cause_lag])
    rss_r = _ols_rss(y, restricted)
    rss_u = _ols_rss(y, unrestricted)
    df_den = max(len(y) - unrestricted.shape[1], 1)
    if not np.isfinite(rss_r) or not np.isfinite(rss_u) or rss_u <= 1e-12:
        return 0.0
    f_val = ((rss_r - rss_u) / 1.0) / (rss_u / df_den)
    return float(max(f_val, 0.0)) if np.isfinite(f_val) else 0.0


def granger_matrix(ts: np.ndarray, lag: int = GRANGER_LAG) -> np.ndarray:
    n_tp, n_rois = ts.shape
    out = np.zeros((n_rois, n_rois), dtype=np.float32)
    if lag != 1 or n_tp <= lag * 4 + 5:
        return out
    for i in range(n_rois):
        tsi = ts[:, i]
        for j in range(i + 1, n_rois):
            tsj = ts[:, j]
            f_ij = _granger_f_lag1_fast(tsi, tsj)
            f_ji = _granger_f_lag1_fast(tsj, tsi)
            out[i, j] = out[j, i] = (f_ij + f_ji) / 2.0
    np.fill_diagonal(out, 0.0)
    return out


def robustscale_offdiag(matrix: np.ndarray) -> np.ndarray:
    out = np.zeros_like(matrix, dtype=np.float32)
    if matrix.shape[0] <= 1:
        return matrix.astype(np.float32)
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    vals = np.nan_to_num(matrix[mask].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    med = np.median(vals)
    q25, q75 = np.percentile(vals, [25, 75])
    iqr = q75 - q25
    if not np.isfinite(iqr) or iqr <= 1e-12:
        out[mask] = vals.astype(np.float32)
    else:
        out[mask] = ((vals - med) / iqr).astype(np.float32)
    return out


def compute_channels(ts: np.ndarray, pairwise_n_jobs: int = 1) -> Tuple[np.ndarray, Dict[str, str], List[Dict[str, Any]]]:
    matrices: List[np.ndarray] = []
    statuses: Dict[str, str] = {}
    qc_rows: List[Dict[str, Any]] = []
    raw_results: Dict[str, np.ndarray] = {}
    omst, omst_status = pearson_omst(ts)
    raw_results["Pearson_OMST_GCE_Signed_Weighted"] = omst
    statuses["Pearson_OMST_GCE_Signed_Weighted"] = omst_status
    raw_results["Pearson_Full_FisherZ_Signed"] = pearson_full(ts)
    statuses["Pearson_Full_FisherZ_Signed"] = "ok"
    raw_results["MI_KNN_Symmetric"] = mi_knn(ts, n_jobs=pairwise_n_jobs)
    statuses["MI_KNN_Symmetric"] = "ok"
    raw_results["dFC_AbsDiffMean"] = dfc_absdiff_mean(ts)
    statuses["dFC_AbsDiffMean"] = "ok"
    raw_results["dFC_StdDev"] = dfc_stddev(ts)
    statuses["dFC_StdDev"] = "ok"
    dc, dc_status = distance_corr(ts)
    raw_results["DistanceCorr"] = dc
    statuses["DistanceCorr"] = dc_status
    raw_results["Granger_F_lag1"] = granger_matrix(ts)
    statuses["Granger_F_lag1"] = "ok"
    for ch in CHANNEL_NAMES:
        raw = raw_results[ch]
        scaled = robustscale_offdiag(raw)
        matrices.append(scaled.astype(np.float32))
        qc_rows.append(
            {
                "channel_name": ch,
                "calc_status": statuses[ch],
                "raw_min": float(np.nanmin(raw)),
                "raw_max": float(np.nanmax(raw)),
                "raw_nan_count": int(np.isnan(raw).sum()),
                "scaled_min": float(np.nanmin(scaled)),
                "scaled_max": float(np.nanmax(scaled)),
                "scaled_nan_count": int(np.isnan(scaled).sum()),
            }
        )
    return np.stack(matrices, axis=0).astype(np.float32), statuses, qc_rows


def extract_one_subject(
    row: Dict[str, Any],
    output_dir: Path,
    roi_info: Dict[str, Any],
    write_tensor: bool = True,
    pairwise_n_jobs: int = 1,
) -> Dict[str, Any]:
    sid = row["SubjectID"]
    path = Path(row["signal_path"])
    started = time.time()
    raw, _var, raw_shape, load_status = load_signal(path)
    result: Dict[str, Any] = {
        "SubjectID": sid,
        "signal_path": str(path),
        "raw_shape": raw_shape,
        "load_status": load_status,
        "status": "pending",
        "tensor_path": "",
        "elapsed_sec": np.nan,
    }
    if raw is None:
        result["status"] = "load_failed"
        result["elapsed_sec"] = time.time() - started
        return result
    ts, pre_status, pre_qc = preprocess_timeseries_no_pybandpass(raw, roi_info)
    result.update(pre_qc)
    result["preprocess_status"] = pre_status
    if ts is None:
        result["status"] = "preprocess_failed"
        result["elapsed_sec"] = time.time() - started
        return result
    tensor, statuses, qc_rows = compute_channels(ts, pairwise_n_jobs=pairwise_n_jobs)
    if tensor.shape != (len(CHANNEL_NAMES), OUTPUT_ROIS, OUTPUT_ROIS):
        result["status"] = f"bad_tensor_shape:{tensor.shape}"
        result["elapsed_sec"] = time.time() - started
        return result
    result["channel_statuses"] = json.dumps(statuses, sort_keys=True)
    result["tensor_nan_count"] = int(np.isnan(tensor).sum())
    result["tensor_min"] = float(np.nanmin(tensor))
    result["tensor_max"] = float(np.nanmax(tensor))
    if write_tensor:
        output_dir.mkdir(parents=True, exist_ok=True)
        tensor_path = output_dir / f"tensor_7ch_131rois_{sid}.npz"
        np.savez_compressed(
            tensor_path,
            tensor=tensor.astype(np.float32),
            SubjectID=sid,
            channel_names=np.asarray(CHANNEL_NAMES),
            python_bandpass_applied=np.asarray(False),
            preprocessing_source=np.asarray(PREPROCESSING_SOURCE),
            target_len_ts=np.asarray(TARGET_LEN),
            tr_seconds=np.asarray(TR_SECONDS),
        )
        result["tensor_path"] = str(tensor_path)
    result["status"] = "ok"
    result["elapsed_sec"] = time.time() - started
    result["_qc_rows"] = qc_rows
    return result


def select_smoke_subjects(manifest: pd.DataFrame, n: int) -> pd.DataFrame:
    candidates = manifest[manifest["v5_candidate"].astype(bool)].copy()
    candidates["group_sort"] = candidates["ResearchGroup_Mapped"].map({"AD": 0, "MCI": 1, "CN": 2}).fillna(3)
    selected_indices: List[int] = []
    ad = candidates[candidates["ResearchGroup_Mapped"].eq("AD")]
    if not ad.empty:
        selected_indices.append(ad.sort_values(["source_label", "SubjectID"]).index[0])
    for idx in candidates.sort_values(["SubjectID"]).index:
        if len(selected_indices) >= n:
            break
        if idx not in selected_indices:
            selected_indices.append(idx)
    return candidates.loc[selected_indices].copy()


def tensor_passes_qc(tensor_path: Path) -> Tuple[bool, str]:
    """Return (True, 'ok') if the individual tensor file is loadable and valid."""
    if not tensor_path.exists():
        return False, "not_found"
    try:
        with np.load(tensor_path, allow_pickle=False) as zf:
            if "tensor" not in zf:
                return False, "missing_tensor_key"
            t = zf["tensor"]
            if t.shape != (len(CHANNEL_NAMES), OUTPUT_ROIS, OUTPUT_ROIS):
                return False, f"bad_shape:{t.shape}"
            if t.dtype != np.float32:
                return False, f"bad_dtype:{t.dtype}"
            if np.isnan(t).any():
                return False, f"has_nans:{int(np.isnan(t).sum())}"
            for k in ("channel_names", "python_bandpass_applied", "preprocessing_source"):
                if k not in zf:
                    return False, f"missing_key:{k}"
        return True, "ok"
    except Exception as exc:
        return False, f"load_error:{exc}"


def run_extraction(manifest: pd.DataFrame, output_root: Path, n_jobs: int, smoke_n: int = 0, run_full: bool = False, resume: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[Path]]:
    if smoke_n <= 0 and not run_full:
        return pd.DataFrame(), pd.DataFrame(), None
    extraction_root = output_root / ("smoke_extraction" if smoke_n > 0 and not run_full else "subject_tensors")
    tensor_dir = extraction_root / "subject_tensors" if smoke_n > 0 and not run_full else extraction_root
    subset = select_smoke_subjects(manifest, smoke_n) if smoke_n > 0 and not run_full else manifest[manifest["v5_candidate"].astype(bool)].copy()
    roi_info = build_roi_reduction_and_order()
    records = subset.to_dict(orient="records")
    results: List[Dict[str, Any]] = []
    qc_rows: List[Dict[str, Any]] = []

    # --resume: skip subjects whose individual tensor already exists and passes QC.
    if resume and run_full and smoke_n <= 0:
        pending: List[Dict[str, Any]] = []
        for row in records:
            sid = row["SubjectID"]
            tensor_path = tensor_dir / f"tensor_7ch_131rois_{sid}.npz"
            passed, qc_msg = tensor_passes_qc(tensor_path)
            if passed:
                results.append({
                    "SubjectID": sid,
                    "signal_path": row.get("signal_path", ""),
                    "raw_shape": "",
                    "load_status": "skipped_resume",
                    "status": "ok",
                    "tensor_path": str(tensor_path),
                    "elapsed_sec": 0.0,
                    "resume_status": "skipped_existing_ok",
                })
            else:
                if qc_msg != "not_found":
                    print(f"[resume] {sid}: tensor failed QC ({qc_msg}), will recompute.")
                pending.append(row)
        print(f"[resume] {len(results)} subjects skipped (tensor OK), {len(pending)} to process.")
        records = pending

    if n_jobs <= 1 or len(records) <= 1:
        for row in records:
            res = extract_one_subject(row, tensor_dir, roi_info, write_tensor=True, pairwise_n_jobs=max(1, n_jobs))
            qc_rows.extend({"SubjectID": res["SubjectID"], **qc} for qc in res.pop("_qc_rows", []))
            results.append(res)
    else:
        # Multiprocessing is kept for full runs; smoke is usually small but follows the same path.
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = {pool.submit(extract_one_subject, row, tensor_dir, roi_info, True, 1): row["SubjectID"] for row in records}
            for future in as_completed(futures):
                res = future.result()
                qc_rows.extend({"SubjectID": res["SubjectID"], **qc} for qc in res.pop("_qc_rows", []))
                results.append(res)
    result_df = pd.DataFrame(results).sort_values("SubjectID") if results else pd.DataFrame()
    qc_df = pd.DataFrame(qc_rows)
    ok = result_df[result_df["status"].eq("ok")] if not result_df.empty else pd.DataFrame()
    global_path: Optional[Path] = None
    if not ok.empty:
        tensors = []
        subject_ids = []
        for _, row in ok.sort_values("SubjectID").iterrows():
            with np.load(row["tensor_path"], allow_pickle=False) as zf:
                tensors.append(zf["tensor"].astype(np.float32))
            subject_ids.append(row["SubjectID"])
        global_tensor = np.stack(tensors, axis=0).astype(np.float32)
        global_path = extraction_root / (
            "SMOKE_GLOBAL_TENSOR_ADNI_expanded_v5_dparsf10000_no_pybandpass.npz"
            if smoke_n > 0 and not run_full
            else "GLOBAL_TENSOR_ADNI_expanded_v5_dparsf10000_no_pybandpass.npz"
        )
        np.savez_compressed(
            global_path,
            global_tensor_data=global_tensor,
            subject_ids=np.asarray(subject_ids),
            channel_names=np.asarray(CHANNEL_NAMES),
            rois_count=np.asarray(OUTPUT_ROIS),
            target_len_ts=np.asarray(TARGET_LEN),
            tr_seconds=np.asarray(TR_SECONDS),
            python_bandpass_applied=np.asarray(False),
            preprocessing_source=np.asarray(PREPROCESSING_SOURCE),
            roi_order_name=np.asarray("aal3_manual_yeo17_order"),
            roi_names_in_order=np.asarray(roi_info["roi_names_new_order"]),
            network_labels_in_order=np.asarray(roi_info["network_labels_new_order"]),
        )
    result_df.to_csv(extraction_root / ("smoke_extraction_qc.csv" if smoke_n > 0 and not run_full else "full_extraction_qc.csv"), index=False)
    qc_df.to_csv(extraction_root / ("smoke_channel_qc.csv" if smoke_n > 0 and not run_full else "full_channel_qc.csv"), index=False)
    return result_df, qc_df, global_path


def ensure_output_root(path: Path, overwrite: bool, run_full: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    full_tensor = path / "GLOBAL_TENSOR_ADNI_expanded_v5_dparsf10000_no_pybandpass.npz"
    if run_full and full_tensor.exists() and not overwrite:
        raise RuntimeError(f"Full tensor already exists; pass --overwrite to replace: {full_tensor}")


def create_or_update_symlink(link: Path, target: Path, overwrite: bool) -> str:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        current = link.resolve()
        if current == target.resolve():
            return "already_correct"
        if not overwrite:
            return f"existing_symlink_points_elsewhere:{current}"
        link.unlink()
    elif link.exists():
        return "existing_non_symlink_not_modified"
    link.symlink_to(target, target_is_directory=True)
    return "created"


def write_manifest_outputs(
    output_root: Path,
    manifest: pd.DataFrame,
    metadata: pd.DataFrame,
    excluded: pd.DataFrame,
    duplicates: pd.DataFrame,
) -> None:
    manifest.to_csv(output_root / "subject_manifest_v5_dparsf10000_no_pybandpass.csv", index=False)
    metadata.to_csv(output_root / "subject_metadata_v5_dparsf10000_no_pybandpass.csv", index=False)
    excluded.to_csv(output_root / "excluded_subjects_v5.csv", index=False)
    duplicates.to_csv(output_root / "duplicate_resolution_v5.csv", index=False)


def write_readme(
    output_root: Path,
    args: argparse.Namespace,
    manifest: pd.DataFrame,
    excluded: pd.DataFrame,
    duplicates: pd.DataFrame,
    symlink_status: str,
    smoke_result: Optional[pd.DataFrame] = None,
    smoke_global_path: Optional[Path] = None,
) -> None:
    candidates = manifest[manifest["v5_candidate"].astype(bool)]
    counts = candidates["ResearchGroup_Mapped"].replace("", "Unknown").value_counts().to_dict()
    lines = [
        "# ADNI Expanded v5 DPARSF10000 No Python Bandpass",
        "",
        f"Generated: `{timestamp_now()}`",
        "",
        "## Method",
        "",
        f"- dataset_name: `{DATASET_NAME}`",
        f"- preprocessing_source: `{PREPROCESSING_SOURCE}`",
        "- Python bandpass: `OFF`",
        f"- TR: `{TR_SECONDS}`",
        f"- target_len: `{TARGET_LEN}`",
        f"- ROI reduction: `{RAW_ROIS}->{OUTPUT_ROIS}` using `data/ROI_MNI_V7_vol.txt`",
        "- ROI order: `aal3_manual_yeo17_order` using `data/aal3_131_manual_network_order.csv`",
        f"- channels: `{'|'.join(CHANNEL_NAMES)}`",
        "- no training was run",
        "",
        "## Manifest Summary",
        "",
        f"- total manifest subjects: `{len(manifest)}`",
        f"- v5_candidate subjects: `{len(candidates)}`",
        f"- excluded subjects: `{len(excluded)}`",
        f"- duplicates resolved: `{len(duplicates)}`",
        f"- diagnosis counts among candidates: `{json.dumps(counts, sort_keys=True)}`",
        f"- `114_S_6039` candidate: `{bool(candidates['SubjectID'].eq('114_S_6039').any())}`",
        f"- `035_S_6927` candidate: `{bool(candidates['SubjectID'].eq('035_S_6927').any())}`",
        f"- `094_S_6736` candidate: `{bool(candidates['SubjectID'].eq('094_S_6736').any())}`",
        f"- `301_S_6592` source: `{manifest.loc[manifest['SubjectID'].eq('301_S_6592'), 'source_label'].iloc[0] if manifest['SubjectID'].eq('301_S_6592').any() else 'missing'}`",
        f"- local symlink status: `{symlink_status}`",
        "",
        "## Files",
        "",
        "- `subject_manifest_v5_dparsf10000_no_pybandpass.csv`",
        "- `subject_metadata_v5_dparsf10000_no_pybandpass.csv`",
        "- `excluded_subjects_v5.csv`",
        "- `duplicate_resolution_v5.csv`",
    ]
    if smoke_result is not None:
        ok_n = int(smoke_result["status"].eq("ok").sum()) if not smoke_result.empty else 0
        lines.extend(
            [
                "",
                "## Smoke Extraction",
                "",
                f"- smoke subjects requested: `{args.smoke_n_subjects}`",
                f"- smoke subjects succeeded: `{ok_n}`",
                f"- smoke global tensor: `{smoke_global_path or ''}`",
                "- smoke extraction computed all 7 channels from ROI signals with Python bandpass OFF.",
            ]
        )
    if args.run_full:
        lines.extend(["", "## Full Extraction", "", "- full extraction was explicitly requested with `--run-full`."])
    else:
        lines.extend(["", "## Full Extraction", "", "- full extraction was NOT run. Use `--run-full` explicitly after reviewing smoke outputs."])
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_command_log(output_root: Path, args: argparse.Namespace) -> None:
    payload = {
        "timestamp": timestamp_now(),
        "argv": os.sys.argv,
        "parameters": {k: str(v) for k, v in vars(args).items()},
        "python_bandpass_applied": False,
        "no_training": True,
    }
    (output_root / "command_log.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    desde_root = resolve(args.desde_cero_root)
    new_root = resolve(args.new_passband_root)
    output_root = resolve(args.output_root)
    local_symlink = resolve(args.local_symlink)
    ensure_output_root(output_root, args.overwrite, args.run_full)

    manifest, metadata, excluded, duplicates = build_manifest(
        desde_root,
        new_root,
        resolve(args.v4_metadata),
        resolve(args.original_metadata),
        resolve(args.adni_download_now),
    )
    write_manifest_outputs(output_root, manifest, metadata, excluded, duplicates)
    symlink_status = create_or_update_symlink(local_symlink, output_root, args.overwrite)
    write_command_log(output_root, args)

    smoke_result: Optional[pd.DataFrame] = None
    smoke_global_path: Optional[Path] = None
    if args.smoke_n_subjects > 0:
        smoke_result, _smoke_qc, smoke_global_path = run_extraction(
            manifest,
            output_root,
            n_jobs=min(max(args.n_jobs, 1), max(args.smoke_n_subjects, 1)),
            smoke_n=args.smoke_n_subjects,
            run_full=False,
        )
    elif args.run_full:
        if args.resume and args.overwrite:
            print("WARNING: --resume and --overwrite are both set; --overwrite takes precedence (no skipping).")
        full_result, _full_qc, full_global_path = run_extraction(
            manifest,
            output_root,
            n_jobs=max(args.n_jobs, 1),
            smoke_n=0,
            run_full=True,
            resume=args.resume and not args.overwrite,
        )
        smoke_result = full_result
        smoke_global_path = full_global_path
    write_readme(output_root, args, manifest, excluded, duplicates, symlink_status, smoke_result, smoke_global_path)

    candidates = manifest[manifest["v5_candidate"].astype(bool)]
    print(f"Wrote v5 manifest to {output_root}")
    print(f"manifest_subjects={len(manifest)} v5_candidate={len(candidates)} excluded={len(excluded)} duplicates={len(duplicates)}")
    print(f"candidate_diagnosis_counts={candidates['ResearchGroup_Mapped'].replace('', 'Unknown').value_counts().to_dict()}")
    print(f"symlink_status={symlink_status} local_symlink={local_symlink}")
    if args.smoke_n_subjects > 0:
        print(f"smoke_global_tensor={smoke_global_path}")
        print(f"smoke_status_counts={smoke_result['status'].value_counts().to_dict() if smoke_result is not None and not smoke_result.empty else {}}")
    if not args.run_full:
        print("Full extraction not run. Pass --run-full explicitly after reviewing manifest/smoke outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
