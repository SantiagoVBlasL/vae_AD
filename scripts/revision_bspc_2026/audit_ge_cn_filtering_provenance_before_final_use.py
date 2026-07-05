#!/usr/bin/env python3
"""Audit GE-CN filtering provenance before any final v5.1 use.

Read-only audit:
- no training;
- no tensor modification;
- no final-use Python bandpass;
- diagnostic Python bandpass is computed only in memory to quantify sensitivity.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_and_import_local_ge_cn_roisignals_v5_1 import (  # noqa: E402
    canonical_manufacturer,
    clean_string,
    load_signal,
    normalize_subject,
    orient_time_by_roi,
)


DEFAULT_FINAL_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "final_cn_ge_inventory_before_martin_request"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "revision_bspc_2026" / "ge_cn_filtering_provenance_audit"
)
V5_1_DATA_ROOTS = [
    PROJECT_ROOT / "data" / "revision_bspc_2026" / "adni_expanded_v5_1_gecn9_no_pybandpass",
    Path("/media/diego/Datos/vae_AD_data/revision_bspc_2026/adni_expanded_v5_1_gecn9_no_pybandpass"),
]
REFERENCE_PASSBAND_ROOTS = [
    Path("/media/diego/My_Book_Diego/vae_AD_data/adni_passband_20260510/ResultsAAL3/ROISignals_AAL3_FunImgARWSDCFN"),
    PROJECT_ROOT / "data" / "OneDrive_1_10-5-2026" / "ResultsAAL3" / "ROISignals_AAL3_FunImgARWSDCFN",
]
CANDIDATE_GE_ROOTS = [
    Path("/media/diego/Datos/adni_expansion/MARTIN_20260429_PHILIPS10/OneDrive_2_29-4-2026/ResultsAAL3/ROISignals_AAL3_FunImgARWSDCF"),
    Path("/media/diego/Datos/adni_expansion/GE_batch7/ROISignals_AAL3_from_CovRegressed_GE_batch7"),
    Path("/media/diego/Datos/adni_expansion/GE_smoketest3/ROISignals_AAL3_from_CovRegressed_GE_smoketest3"),
    Path("/media/diego/Datos/adni_bridge_expansion/dparsf_single/GE/FunImgARWSDCovs"),
    Path("/media/diego/Datos/adni_bridge_expansion/dparsf_single/GE/Results/ROISignals_FunImgARWSDC"),
]

TR_SECONDS = 3.0
LOW_HZ = 0.01
HIGH_HZ = 0.08
EXPECTED_ROIS = 170
PROVENANCE_TERMS = [
    "Filter",
    "Band",
    "0.01",
    "0.08",
    "FunImgARWSDCF",
    "FunImgARWSDC",
    "CovRegressed",
    "DPARSF",
]
TEXT_SUFFIXES = {
    ".txt",
    ".log",
    ".m",
    ".matlabbatch",
    ".json",
    ".csv",
    ".tsv",
    ".ini",
    ".cfg",
    ".xml",
    ".html",
    ".md",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit GE-CN filtering provenance before final v5.1 use.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--final-dir", type=Path, default=DEFAULT_FINAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reference-limit", type=int, default=40)
    parser.add_argument("--max-provenance-file-mb", type=float, default=5.0)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def prepare_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def yes(value: Any) -> bool:
    return clean_string(value).lower() in {"true", "yes", "1"}


def to_float(value: Any, default: float = np.nan) -> float:
    text = clean_string(value)
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def to_int(value: Any, default: int = 0) -> int:
    num = to_float(value, np.nan)
    return default if not np.isfinite(num) else int(round(num))


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
    if "HISTORICAL_10000" in text:
        return "historical_10000"
    return "unknown"


def explicit_has_f(stage: str, path: Any) -> str:
    text = clean_string(path).upper()
    return "yes" if stage in {"ARWSDCFN", "ARWSDCF"} or "FUNIMGARWSDCF" in text else "no"


def source_root(path: str) -> str:
    p = clean_string(path)
    roots = [r for r in CANDIDATE_GE_ROOTS + REFERENCE_PASSBAND_ROOTS if p.startswith(str(r))]
    return str(max(roots, key=lambda x: len(str(x)))) if roots else ""


def select_subject_best_rows(final_dir: Path) -> pd.DataFrame:
    inventory = read_csv(final_dir / "cn_ge_final_inventory.csv")
    included = read_csv(final_dir / "cn_ge_already_included_v5_1_gecn9.csv")
    rejected = read_csv(final_dir / "cn_ge_qc_rejected_reason_summary_for_martin.csv")

    rows: List[Dict[str, Any]] = []
    for _, row in included.iterrows():
        sid = clean_string(row["SubjectID"])
        rows.append(
            {
                "SubjectID": sid,
                "was_included_in_v5_1_gecn9": True,
                "was_rejected_qc": False,
                "best_local_path": clean_string(row.get("best_path", "")),
                "stage_guess": normalize_stage(row.get("best_stage_guess", ""), row.get("best_path", "")),
                "n_rois": to_int(row.get("best_n_rois")),
                "n_timepoints": to_int(row.get("best_n_timepoints")),
                "finite_fraction": to_float(row.get("best_finite_fraction")),
                "scale_label": clean_string(row.get("best_scale_label", "")),
                "final_inventory_classification": clean_string(row.get("final_classification", "")),
            }
        )
    for _, row in rejected.iterrows():
        sid = clean_string(row["SubjectID"])
        rows.append(
            {
                "SubjectID": sid,
                "was_included_in_v5_1_gecn9": False,
                "was_rejected_qc": True,
                "best_local_path": clean_string(row.get("best_local_path", "")),
                "stage_guess": normalize_stage(row.get("stage_guess", ""), row.get("best_local_path", "")),
                "n_rois": to_int(row.get("n_rois")),
                "n_timepoints": to_int(row.get("n_timepoints")),
                "finite_fraction": to_float(row.get("finite_fraction")),
                "scale_label": clean_string(row.get("scale_label", "")),
                "final_inventory_classification": "already_local_but_rejected_qc",
            }
        )

    out = pd.DataFrame(rows).drop_duplicates("SubjectID", keep="first")
    if len(out) != 19:
        raise RuntimeError(f"Expected 19 local CN-GE subjects, found {len(out)}")
    out["source_root"] = out["best_local_path"].map(source_root)
    out["explicit_has_F_stage"] = [
        explicit_has_f(stage, path)
        for stage, path in zip(out["stage_guess"], out["best_local_path"])
    ]

    inv_real = inventory[inventory["path"].ne("NO_LOCAL_FILE")].copy()
    counts = inv_real.groupby("SubjectID").agg(
        local_candidate_file_count=("path", "nunique"),
        local_candidate_roots=("source_root", lambda s: "|".join(sorted(set(clean_string(x) for x in s if clean_string(x))))),
        local_candidate_stage_guesses=("stage_guess", lambda s: "|".join(sorted(set(clean_string(x) for x in s if clean_string(x))))),
    ).reset_index()
    out = out.merge(counts, on="SubjectID", how="left")
    return out.sort_values(["was_rejected_qc", "SubjectID"]).reset_index(drop=True)


def load_oriented_signal(path: Path) -> Tuple[Optional[np.ndarray], str, str, str]:
    arr, variable, shape, load_status = load_signal(path)
    if arr is None:
        return None, variable, shape, load_status
    oriented, _n_tp, _n_rois, orient_status = orient_time_by_roi(arr)
    return oriented, variable, shape, f"{load_status};{orient_status}"


def spectral_metrics(ts: Optional[np.ndarray], tr: float = TR_SECONDS) -> Dict[str, Any]:
    base = {
        "spectral_status": "not_computed",
        "spectral_valid_rois": 0,
        "energy_below_0p01": np.nan,
        "energy_0p01_0p08": np.nan,
        "energy_above_0p08": np.nan,
        "outside_band_energy": np.nan,
    }
    if ts is None or ts.ndim != 2 or ts.shape[0] < 8:
        base["spectral_status"] = "invalid_signal"
        return base
    arr = np.asarray(ts, dtype=np.float64)
    fs = 1.0 / tr
    fractions: List[Tuple[float, float, float]] = []
    for idx in range(arr.shape[1]):
        col = arr[:, idx]
        finite = np.isfinite(col)
        if finite.mean() < 0.95 or finite.sum() < 8:
            continue
        filled = col.copy()
        filled[~finite] = np.nanmean(filled[finite])
        centered = filled - np.mean(filled)
        if np.nanstd(centered) <= 1e-12:
            continue
        spectrum = np.abs(np.fft.rfft(centered)) ** 2
        freqs = np.fft.rfftfreq(centered.size, d=tr)
        non_dc = freqs > 0
        total = float(np.sum(spectrum[non_dc]))
        if total <= 0 or not np.isfinite(total):
            continue
        below = float(np.sum(spectrum[(freqs > 0) & (freqs < LOW_HZ)]) / total)
        band = float(np.sum(spectrum[(freqs >= LOW_HZ) & (freqs <= HIGH_HZ)]) / total)
        above = float(np.sum(spectrum[freqs > HIGH_HZ]) / total)
        fractions.append((below, band, above))
    if not fractions:
        base["spectral_status"] = "no_valid_roi_columns"
        return base
    arr_frac = np.asarray(fractions, dtype=np.float64)
    below, band, above = np.nanmedian(arr_frac, axis=0)
    base.update(
        {
            "spectral_status": "ok",
            "spectral_valid_rois": int(arr_frac.shape[0]),
            "energy_below_0p01": float(below),
            "energy_0p01_0p08": float(band),
            "energy_above_0p08": float(above),
            "outside_band_energy": float(below + above),
        }
    )
    return base


def discover_reference_files(limit: int) -> List[Path]:
    files: List[Path] = []
    by_subject: Dict[str, List[Path]] = {}
    for root in REFERENCE_PASSBAND_ROOTS:
        if not root.exists():
            continue
        candidates = sorted(root.glob("ROISignals_*.txt")) + sorted(root.glob("ROISignals_*.mat"))
        for path in candidates:
            sid = normalize_subject(path.name)
            if not sid:
                continue
            by_subject.setdefault(sid, []).append(path)
    for sid in sorted(by_subject):
        files.extend(by_subject[sid])
        if len(files) >= max(limit * 6, limit):
            break
    return files


def build_reference_spectral(limit: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    valid_subjects = set()
    for path in discover_reference_files(limit):
        sid = normalize_subject(path.name)
        if sid in valid_subjects:
            continue
        ts, _var, shape, load_status = load_oriented_signal(path)
        spec = spectral_metrics(ts)
        is_valid_reference = (
            sid
            and sid not in valid_subjects
            and ts is not None
            and ts.ndim == 2
            and int(ts.shape[1]) == EXPECTED_ROIS
            and clean_string(spec.get("spectral_status")) == "ok"
        )
        rows.append(
            {
                "reference_group": "known_passband_ARWSDCFN",
                "SubjectID": sid,
                "path": str(path),
                "shape": shape,
                "load_status": load_status,
                "n_timepoints": int(ts.shape[0]) if ts is not None and ts.ndim == 2 else 0,
                "n_rois": int(ts.shape[1]) if ts is not None and ts.ndim == 2 else 0,
                **spec,
            }
        )
        if is_valid_reference:
            valid_subjects.add(sid)
        if len(valid_subjects) >= limit:
            break
    return pd.DataFrame(rows)


def classify_spectrum(row: pd.Series, ref_stats: Dict[str, float]) -> str:
    if clean_string(row.get("spectral_status")) != "ok":
        return "uncertain"
    band = to_float(row.get("energy_0p01_0p08"))
    outside = to_float(row.get("outside_band_energy"))
    below = to_float(row.get("energy_below_0p01"))
    above = to_float(row.get("energy_above_0p08"))
    ref_band_p10 = ref_stats.get("band_p10", 0.70)
    ref_outside_p90 = ref_stats.get("outside_p90", 0.30)
    if band >= max(0.70, ref_band_p10 - 0.05) and outside <= min(0.30, ref_outside_p90 + 0.05):
        return "bandpassed_like"
    if band < 0.62 or outside > 0.38 or above > 0.30 or below > 0.20:
        return "unfiltered_like"
    return "uncertain"


def butter_bandpass_diagnostic(ts: np.ndarray, tr: float = TR_SECONDS) -> np.ndarray:
    fs = 1.0 / tr
    nyq = 0.5 * fs
    b, a = butter(2, [LOW_HZ / nyq, HIGH_HZ / nyq], btype="band")
    out = np.empty_like(ts, dtype=np.float64)
    for idx in range(ts.shape[1]):
        col = np.asarray(ts[:, idx], dtype=np.float64)
        finite = np.isfinite(col)
        if finite.mean() < 0.95 or finite.sum() < 8:
            out[:, idx] = np.nan
            continue
        filled = col.copy()
        filled[~finite] = np.nanmean(filled[finite])
        centered = filled - np.mean(filled)
        try:
            out[:, idx] = filtfilt(b, a, centered)
        except Exception:
            out[:, idx] = np.nan
    return out


def pearson_fisher_z(ts: np.ndarray) -> np.ndarray:
    arr = np.asarray(ts, dtype=np.float64)
    valid_cols = np.isfinite(arr).mean(axis=0) >= 0.95
    arr2 = arr[:, valid_cols]
    arr2 = np.where(np.isfinite(arr2), arr2, np.nanmean(arr2, axis=0))
    if arr2.shape[1] < 2:
        return np.full((arr.shape[1], arr.shape[1]), np.nan, dtype=np.float64)
    corr = np.corrcoef(arr2, rowvar=False)
    corr = np.clip(corr, -0.999999, 0.999999)
    z_small = np.arctanh(corr)
    np.fill_diagonal(z_small, 0.0)
    z = np.full((arr.shape[1], arr.shape[1]), np.nan, dtype=np.float64)
    idx = np.where(valid_cols)[0]
    z[np.ix_(idx, idx)] = z_small
    return z


def matrix_delta(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    if a.shape != b.shape or a.ndim != 2:
        return {"delta_status": "invalid_matrix", "corr_A_B": np.nan, "MAE_A_B": np.nan, "max_abs_diff_A_B": np.nan}
    mask = ~np.eye(a.shape[0], dtype=bool)
    av = a[mask]
    bv = b[mask]
    valid = np.isfinite(av) & np.isfinite(bv)
    if valid.sum() < 2:
        return {"delta_status": "insufficient_valid_edges", "corr_A_B": np.nan, "MAE_A_B": np.nan, "max_abs_diff_A_B": np.nan}
    diff = av[valid] - bv[valid]
    corr = float(np.corrcoef(av[valid], bv[valid])[0, 1])
    return {
        "delta_status": "ok",
        "corr_A_B": corr,
        "MAE_A_B": float(np.mean(np.abs(diff))),
        "max_abs_diff_A_B": float(np.max(np.abs(diff))),
        "valid_edges": int(valid.sum()),
    }


def build_spectral_and_delta(subjects: pd.DataFrame, ref_limit: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    reference = build_reference_spectral(ref_limit)
    ref_ok = reference[reference["spectral_status"].eq("ok") & reference["n_rois"].eq(EXPECTED_ROIS)].copy()
    if ref_ok.empty:
        ref_stats = {"band_p10": 0.70, "outside_p90": 0.30}
    else:
        ref_stats = {
            "band_p10": float(ref_ok["energy_0p01_0p08"].quantile(0.10)),
            "band_median": float(ref_ok["energy_0p01_0p08"].median()),
            "outside_p90": float(ref_ok["outside_band_energy"].quantile(0.90)),
            "outside_median": float(ref_ok["outside_band_energy"].median()),
            "n_reference": int(len(ref_ok)),
        }

    spectral_rows: List[Dict[str, Any]] = []
    delta_rows: List[Dict[str, Any]] = []
    for _, row in subjects.iterrows():
        path = Path(clean_string(row["best_local_path"]))
        ts, _var, shape, load_status = load_oriented_signal(path)
        spec = spectral_metrics(ts)
        spec_row = {
            "SubjectID": row["SubjectID"],
            "group": "ge_cn_local",
            "comparison_role": (
                "suspected_unfiltered_control"
                if clean_string(row["stage_guess"]) in {"CovRegressed", "ARWSDC"} or clean_string(row["explicit_has_F_stage"]) == "no"
                else "candidate_final_use"
            ),
            "path": str(path),
            "was_included_in_v5_1_gecn9": bool(row["was_included_in_v5_1_gecn9"]),
            "was_rejected_qc": bool(row["was_rejected_qc"]),
            "stage_guess": row["stage_guess"],
            "explicit_has_F_stage": row["explicit_has_F_stage"],
            "shape": shape,
            "load_status": load_status,
            "n_timepoints_loaded": int(ts.shape[0]) if ts is not None and ts.ndim == 2 else 0,
            "n_rois_loaded": int(ts.shape[1]) if ts is not None and ts.ndim == 2 else 0,
            **spec,
        }
        spec_row["spectrum_class"] = classify_spectrum(pd.Series(spec_row), ref_stats)
        spectral_rows.append(spec_row)

        delta_base = {
            "SubjectID": row["SubjectID"],
            "path": str(path),
            "stage_guess": row["stage_guess"],
            "n_rois": int(ts.shape[1]) if ts is not None and ts.ndim == 2 else 0,
            "n_timepoints": int(ts.shape[0]) if ts is not None and ts.ndim == 2 else 0,
            "diagnostic_python_bandpass_low_hz": LOW_HZ,
            "diagnostic_python_bandpass_high_hz": HIGH_HZ,
            "diagnostic_only_not_final_pipeline": True,
        }
        if ts is None or ts.ndim != 2 or ts.shape[1] < 2:
            delta_base.update({"delta_status": "invalid_signal", "corr_A_B": np.nan, "MAE_A_B": np.nan, "max_abs_diff_A_B": np.nan})
        else:
            direct = pearson_fisher_z(ts)
            filtered = butter_bandpass_diagnostic(ts)
            filtered_conn = pearson_fisher_z(filtered)
            delta_base.update(matrix_delta(direct, filtered_conn))
        delta_rows.append(delta_base)

    return pd.DataFrame(spectral_rows), pd.DataFrame(delta_rows), reference, ref_stats


def surrounding_dirs(path: Path) -> List[Path]:
    dirs: List[Path] = []
    current = path if path.is_dir() else path.parent
    for _ in range(5):
        if current.exists() and current not in dirs:
            dirs.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    return dirs


def iter_provenance_files(search_dirs: Sequence[Path], max_mb: float) -> Iterable[Path]:
    seen = set()
    max_bytes = int(max_mb * 1024 * 1024)
    for root in search_dirs:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__", ".Trash-1000"}]
            for filename in filenames:
                path = Path(dirpath) / filename
                key = str(path)
                if key in seen:
                    continue
                seen.add(key)
                lower = filename.lower()
                name_has_term = any(term.lower() in lower for term in PROVENANCE_TERMS)
                text_like = path.suffix.lower() in TEXT_SUFFIXES
                if not (name_has_term or text_like):
                    continue
                try:
                    if path.stat().st_size > max_bytes:
                        continue
                except OSError:
                    continue
                yield path


def scan_provenance_for_root(root: str, max_mb: float) -> Dict[str, Any]:
    root_path = Path(clean_string(root))
    if not root_path.exists():
        return {
            "provenance_supports_matlab_filtering": "unclear",
            "provenance_hit_count": 0,
            "provenance_evidence_files": "",
            "provenance_evidence_snippets": "",
        }
    dirs = surrounding_dirs(root_path)
    hits: List[Tuple[str, str]] = []
    filter_support = False
    negative_unfiltered = False
    pattern = re.compile("|".join(re.escape(term) for term in PROVENANCE_TERMS), flags=re.IGNORECASE)
    for path in iter_provenance_files(dirs, max_mb=max_mb):
        snippet = ""
        name_hit = pattern.search(path.name) is not None
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = ""
        m = pattern.search(text)
        if m:
            start = max(0, m.start() - 80)
            end = min(len(text), m.end() + 120)
            snippet = " ".join(text[start:end].split())
        elif name_hit:
            snippet = f"filename:{path.name}"
        else:
            continue
        upper = f"{path.name} {snippet}".upper()
        if ("0.01" in upper and "0.08" in upper) or "ARWSDCFN" in upper or "FUNIMGARWSDCF" in upper:
            filter_support = True
        if "COVREGRESSED" in upper or "FUNIMGARWSDCOVS" in upper:
            negative_unfiltered = True
        hits.append((str(path), snippet[:300]))
        if len(hits) >= 25:
            break
    if filter_support:
        support = "yes"
    elif negative_unfiltered:
        support = "no_intermediate_or_covregressed_evidence"
    else:
        support = "unclear"
    return {
        "provenance_supports_matlab_filtering": support,
        "provenance_hit_count": len(hits),
        "provenance_evidence_files": "|".join(path for path, _snippet in hits[:10]),
        "provenance_evidence_snippets": " || ".join(snippet for _path, snippet in hits[:10]),
    }


def build_provenance_summary(subjects: pd.DataFrame, max_mb: float) -> pd.DataFrame:
    by_root: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    for _, row in subjects.iterrows():
        root = clean_string(row["source_root"]) or str(Path(row["best_local_path"]).parent)
        if root not in by_root:
            by_root[root] = scan_provenance_for_root(root, max_mb=max_mb)
        rows.append({**row.to_dict(), **by_root[root]})
    return pd.DataFrame(rows)


def classify_delta(row: pd.Series, ref_delta_stats: Dict[str, float]) -> str:
    if clean_string(row.get("delta_status")) != "ok":
        return "uncertain"
    corr = to_float(row.get("corr_A_B"))
    mae = to_float(row.get("MAE_A_B"))
    ref_mae_p90 = ref_delta_stats.get("mae_p90", 0.08)
    if corr >= 0.95 and mae <= max(0.05, ref_mae_p90 * 1.25):
        return "small"
    if corr < 0.85 or mae > max(0.12, ref_mae_p90 * 2.0):
        return "large"
    return "moderate"


def reference_delta_stats(reference: pd.DataFrame) -> Dict[str, float]:
    rows: List[Dict[str, Any]] = []
    for _, ref in reference.iterrows():
        path = Path(clean_string(ref["path"]))
        ts, _var, _shape, _load = load_oriented_signal(path)
        if ts is None or ts.ndim != 2 or ts.shape[1] != EXPECTED_ROIS:
            continue
        direct = pearson_fisher_z(ts)
        filtered = butter_bandpass_diagnostic(ts)
        rows.append(matrix_delta(direct, pearson_fisher_z(filtered)))
        if len(rows) >= 20:
            break
    df = pd.DataFrame(rows)
    ok = df[df["delta_status"].eq("ok")] if not df.empty else pd.DataFrame()
    if ok.empty:
        return {"mae_p90": 0.08, "corr_p10": 0.95, "n_reference_delta": 0}
    return {
        "mae_p90": float(ok["MAE_A_B"].quantile(0.90)),
        "corr_p10": float(ok["corr_A_B"].quantile(0.10)),
        "n_reference_delta": int(len(ok)),
    }


def final_decision(
    row: pd.Series,
    delta_row: pd.Series,
    spectrum_class: str,
    delta_class: str,
) -> Tuple[str, str]:
    n_rois = to_int(row.get("n_rois"))
    finite = to_float(row.get("finite_fraction"))
    scale = clean_string(row.get("scale_label"))
    stage = clean_string(row.get("stage_guess"))
    provenance = clean_string(row.get("provenance_supports_matlab_filtering"))
    explicit_f = clean_string(row.get("explicit_has_F_stage")) == "yes"

    if n_rois != EXPECTED_ROIS:
        return "reprocess_required", f"wrong ROI count for AAL3 v5.1 use: n_rois={n_rois}"
    if not np.isfinite(finite) or finite < 0.95:
        return "reprocess_required", f"finite_fraction below threshold: {finite}"
    if scale != "around_10000_global_scaled":
        return "reprocess_required", f"scale_label not compatible with 10000-level output: {scale}"
    if stage in {"CovRegressed", "ARWSDC"}:
        return "reprocess_required", f"{stage}/non-F-stage is not acceptable for final no-Python-bandpass v5.1 without re-export"
    if spectrum_class == "unfiltered_like":
        return "reprocess_required", f"unfiltered-like spectrum despite stage={stage}; requires filtered DPARSF/MATLAB re-export"
    if delta_class == "large":
        return "reprocess_required", f"large diagnostic Python-bandpass delta despite stage={stage}; requires filtered DPARSF/MATLAB re-export"
    if provenance == "yes" and spectrum_class == "bandpassed_like" and delta_class == "small":
        return "final_use_ok", "provenance supports MATLAB/DPARSF filtering; spectrum and diagnostic delta are consistent"
    if spectrum_class == "bandpassed_like" and delta_class == "small":
        return "final_use_ok", "spectrum matches passband reference and diagnostic Python-bandpass delta is small"
    if explicit_f:
        return "quarantine_needs_martin_confirmation", f"path suggests F-stage but spectrum={spectrum_class}, diagnostic_delta={delta_class}, provenance={provenance}"
    return "reprocess_required", f"insufficient filtering provenance: stage={stage}, spectrum={spectrum_class}, diagnostic_delta={delta_class}, provenance={provenance}"


def build_decisions(
    provenance: pd.DataFrame,
    spectral: pd.DataFrame,
    delta: pd.DataFrame,
    ref_delta: Dict[str, float],
) -> pd.DataFrame:
    spec_by_sid = spectral.set_index("SubjectID")
    delta_by_sid = delta.set_index("SubjectID")
    rows: List[Dict[str, Any]] = []
    for _, row in provenance.iterrows():
        sid = row["SubjectID"]
        spec = spec_by_sid.loc[sid]
        drow = delta_by_sid.loc[sid]
        delta_class = classify_delta(drow, ref_delta)
        decision, reason = final_decision(row, drow, clean_string(spec.get("spectrum_class")), delta_class)
        rows.append(
            {
                "SubjectID": sid,
                "was_included_in_v5_1_gecn9": bool(row["was_included_in_v5_1_gecn9"]),
                "was_rejected_qc": bool(row["was_rejected_qc"]),
                "best_local_path": row["best_local_path"],
                "stage_guess": row["stage_guess"],
                "explicit_has_F_stage": row["explicit_has_F_stage"],
                "n_rois": row["n_rois"],
                "n_timepoints": row["n_timepoints"],
                "finite_fraction": row["finite_fraction"],
                "scale_label": row["scale_label"],
                "provenance_supports_matlab_filtering": row["provenance_supports_matlab_filtering"],
                "spectrum_class": spec.get("spectrum_class"),
                "energy_below_0p01": spec.get("energy_below_0p01"),
                "energy_0p01_0p08": spec.get("energy_0p01_0p08"),
                "energy_above_0p08": spec.get("energy_above_0p08"),
                "python_bandpass_delta_class": delta_class,
                "corr_direct_vs_diag_python_bandpass": drow.get("corr_A_B"),
                "MAE_direct_vs_diag_python_bandpass": drow.get("MAE_A_B"),
                "max_abs_diff_direct_vs_diag_python_bandpass": drow.get("max_abs_diff_A_B"),
                "final_use_decision": decision,
                "decision_reason": reason,
                "python_bandpass_final_recommended_path": "OFF",
            }
        )
    return pd.DataFrame(rows)


def write_readme(output_dir: Path, decisions: pd.DataFrame, ref_stats: Dict[str, float], ref_delta: Dict[str, float]) -> None:
    included = decisions[decisions["was_included_in_v5_1_gecn9"]]
    included_ok = included[included["final_use_decision"].eq("final_use_ok")]
    included_quarantine = included[~included["final_use_decision"].eq("final_use_ok")]
    reprocess = decisions[decisions["final_use_decision"].eq("reprocess_required")]
    quarantine = decisions[decisions["final_use_decision"].eq("quarantine_needs_martin_confirmation")]
    unfiltered_like = decisions[
        decisions["spectrum_class"].eq("unfiltered_like")
        | decisions["python_bandpass_delta_class"].eq("large")
        | decisions["stage_guess"].isin(["CovRegressed", "ARWSDC"])
    ]
    suspected_controls = decisions[decisions["stage_guess"].isin(["CovRegressed", "ARWSDC"])].copy()
    if suspected_controls.empty:
        control_band_median = np.nan
        control_outside_median = np.nan
    else:
        control_band_median = float(suspected_controls["energy_0p01_0p08"].astype(float).median())
        control_outside_median = float(
            (
                suspected_controls["energy_below_0p01"].astype(float)
                + suspected_controls["energy_above_0p08"].astype(float)
            ).median()
        )

    keep_subjects = ", ".join(included_ok["SubjectID"].tolist()) or "None"
    quarantine_subjects = ", ".join(pd.concat([included_quarantine, quarantine]).drop_duplicates("SubjectID")["SubjectID"].tolist()) or "None"
    reprocess_subjects = ", ".join(reprocess["SubjectID"].tolist()) or "None"
    lines = [
        "# GE-CN Filtering Provenance Audit Before Final v5.1 Use",
        "",
        "Read-only audit. No training and no tensor modification were performed. Diagnostic Python bandpass was used only in memory to quantify sensitivity.",
        "",
        "## Explicit Answers",
        "",
        f"1. Are the 9 included CN-GE safe for final v5.1 use? `{'YES' if len(included_ok) == len(included) else 'NO / NOT ALL'}`. Safe now: `{len(included_ok)}/{len(included)}`.",
        f"2. Which subjects should remain in `v5.1_gecn9`? `{keep_subjects}`.",
        f"3. Which subjects should be removed/quarantined? `{quarantine_subjects}`.",
        f"4. Which subjects require Martin re-export/reprocess? `{reprocess_subjects}`.",
        f"5. Did any local GE signals appear to be unfiltered MATLAB outputs? `{'YES' if not unfiltered_like.empty else 'NO CLEAR EVIDENCE'}`. Evidence includes non-F/CovRegressed stages, unfiltered-like spectra, or large diagnostic Python-bandpass deltas.",
        "6. Is Python bandpass OFF in final recommended path? `YES`. Python bandpass remains OFF for final v5.1; it was used here only as a diagnostic sensitivity test.",
        "7. What exactly should we tell Martin? Ask for DPARSF/MATLAB-filtered first-visit AAL3 ROISignals (170 ROIs, TR=3s, scale compatible with 10000-level outputs, passband 0.01-0.08 Hz) for every `reprocess_required` or quarantined GE-CN subject, and ask him to confirm whether the ARWSDCF batch was actually MATLAB-filtered.",
        "",
        "## Reference Threshold Context",
        "",
        f"- Reference passband spectral rows used: `{int(ref_stats.get('n_reference', 0))}`.",
        f"- Reference median band energy: `{ref_stats.get('band_median', np.nan):.4f}`.",
        f"- Reference p10 band energy: `{ref_stats.get('band_p10', np.nan):.4f}`.",
        f"- Reference p90 outside-band energy: `{ref_stats.get('outside_p90', np.nan):.4f}`.",
        f"- Reference delta rows used: `{int(ref_delta.get('n_reference_delta', 0))}`.",
        f"- Reference p90 diagnostic Python-bandpass MAE: `{ref_delta.get('mae_p90', np.nan):.4f}`.",
        f"- Suspected unfiltered/non-F local controls: `{len(suspected_controls)}`; median band energy `{control_band_median:.4f}`, median outside-band energy `{control_outside_median:.4f}`.",
        "",
        "## Decision Rule",
        "",
        "`final_use_ok` requires valid AAL3 shape/QC plus filtering provenance or strong passband-like spectrum with small diagnostic Python-bandpass delta. Wrong ROI count, CovRegressed/non-F-stage evidence, unfiltered-like spectrum, large delta, bad finite fraction, or bad scale forces `reprocess_required`. Tensor smoke/no-NaN status is not used as sufficient evidence.",
        "",
        "## Outputs",
        "",
        "- `ge_cn_filtering_provenance_summary.csv`",
        "- `ge_cn_spectral_audit.csv`",
        "- `ge_cn_python_bandpass_delta_audit.csv`",
        "- `ge_cn_final_use_decision.csv`",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    prepare_output_dir(args.output_dir)
    subjects = select_subject_best_rows(args.final_dir)
    provenance = build_provenance_summary(subjects, max_mb=args.max_provenance_file_mb)
    spectral, delta, reference, ref_stats = build_spectral_and_delta(provenance, args.reference_limit)
    ref_delta = reference_delta_stats(reference)
    decisions = build_decisions(provenance, spectral, delta, ref_delta)

    provenance.to_csv(args.output_dir / "ge_cn_filtering_provenance_summary.csv", index=False)
    spectral.to_csv(args.output_dir / "ge_cn_spectral_audit.csv", index=False)
    delta.to_csv(args.output_dir / "ge_cn_python_bandpass_delta_audit.csv", index=False)
    decisions.to_csv(args.output_dir / "ge_cn_final_use_decision.csv", index=False)
    reference.to_csv(args.output_dir / "reference_passband_spectral_audit.csv", index=False)
    command = {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "script": str(Path(__file__).resolve()),
        "final_dir": str(args.final_dir),
        "output_dir": str(args.output_dir),
        "reference_passband_roots": [str(x) for x in REFERENCE_PASSBAND_ROOTS],
        "candidate_ge_roots": [str(x) for x in CANDIDATE_GE_ROOTS],
        "tr_seconds": TR_SECONDS,
        "diagnostic_python_bandpass_low_hz": LOW_HZ,
        "diagnostic_python_bandpass_high_hz": HIGH_HZ,
        "python_bandpass_final_pipeline": False,
        "training_run": False,
        "tensor_modified": False,
        "reference_stats": ref_stats,
        "reference_delta_stats": ref_delta,
    }
    (args.output_dir / "command_log.json").write_text(json.dumps(command, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(args.output_dir, decisions, ref_stats, ref_delta)

    counts = decisions["final_use_decision"].value_counts().to_dict()
    included_counts = decisions[decisions["was_included_in_v5_1_gecn9"]]["final_use_decision"].value_counts().to_dict()
    print(f"Wrote GE-CN filtering provenance audit to {args.output_dir}")
    print(f"decision_counts={counts}")
    print(f"included_decision_counts={included_counts}")
    print("No training. No tensor modification. Python bandpass final path: OFF.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
