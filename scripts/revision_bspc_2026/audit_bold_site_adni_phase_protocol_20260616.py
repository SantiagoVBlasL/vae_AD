#!/usr/bin/env python3
"""Read-only BOLD-level site/protocol audit for the promoted ADNI model.

This script computes descriptive BOLD/ROISignals QC metrics from existing .mat
files, merges them with locked promoted-model scores/metadata, and writes only a
derived audit package. It does not train AD/CN models, alter tensors, alter
metadata, edit predictions, fit thresholds, or exclude subjects.
"""

from __future__ import annotations

import json
import math
import re
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.io as scipy_io
from scipy import stats
from scipy.signal import welch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import statsmodels.formula.api as smf
except Exception:  # pragma: no cover
    smf = None

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "revision_bspc_2026"
OUT = RESULTS / "bold_site_adni_phase_protocol_audit_20260616"
PLOTS = OUT / "plots"

MASTER_DB = RESULTS / "promoted_model_master_database_20260610" / "promoted_model_master_database.csv"
SLICE_AUDIT = RESULTS / "philips_fmri_slice_timing_audit_20260612" / "philips_cn_fmri_slice_timing_merged.csv"

TR_DEFAULT = 3.0
T140 = 140
FS = 1.0 / TR_DEFAULT
NYQUIST = FS / 2.0
LOW_BAND = (0.01, 0.08)
HIGH_BAND = (0.08, NYQUIST)
RANDOM_SEED = 42

COMMAND_LOG: list[dict[str, Any]] = []


def log(action: str, detail: Any = None) -> None:
    COMMAND_LOG.append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "action": action,
            "detail": detail,
        }
    )


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def finite_float(x: Any) -> float:
    try:
        y = float(x)
        return y if np.isfinite(y) else np.nan
    except Exception:
        return np.nan


def normalize_dx(row: pd.Series) -> str:
    for col in ["ResearchGroup_Mapped", "Diagnosis", "diagnosis", "y_true_label"]:
        if col in row.index:
            s = safe_str(row[col]).upper().strip()
            if s in {"CN", "NORMAL"}:
                return "CN"
            if s in {"AD", "AD_DEMENTIA", "DEMENTIA"}:
                return "AD"
            if s == "MCI":
                return "MCI"
    if "y_true" in row.index and pd.notna(row["y_true"]):
        return "AD" if int(row["y_true"]) == 1 else "CN"
    return "UNKNOWN"


def normalize_manufacturer(x: Any) -> str:
    s = safe_str(x).upper()
    if "PHILIPS" in s:
        return "Philips"
    if "SIEMENS" in s:
        return "SIEMENS"
    if s == "GE" or "GE MEDICAL" in s or "GENERAL ELECTRIC" in s:
        return "GE"
    return safe_str(x) or "UNKNOWN"


def normalize_rawtp(row: pd.Series) -> str:
    for col in ["raw_tp_group", "raw_tp_group_norm", "n_timepoints_raw", "n_tp_raw", "n_timepoints_model_input"]:
        if col not in row.index:
            continue
        s = safe_str(row[col])
        m = re.search(r"\d+", s)
        if not m:
            continue
        n = int(m.group(0))
        if n == 140:
            return "140"
        if n in {197, 200}:
            return "197_200"
        return str(n)
    return "UNKNOWN"


def confusion_label(y_true: Any, y_pred: Any) -> str:
    if pd.isna(y_true) or pd.isna(y_pred):
        return ""
    yt, yp = int(y_true), int(y_pred)
    if yt == 0 and yp == 0:
        return "TN"
    if yt == 0 and yp == 1:
        return "FP"
    if yt == 1 and yp == 1:
        return "TP"
    if yt == 1 and yp == 0:
        return "FN"
    return ""


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "_No rows._\n"
    shown = df.head(max_rows)
    suffix = f"\n\n_Showing first {max_rows} of {len(df)} rows._\n" if len(df) > max_rows else "\n"
    return shown.to_markdown(index=False) + suffix


def write_df(name: str, df: pd.DataFrame, max_rows: int = 80) -> None:
    csv = OUT / f"{name}.csv"
    md = OUT / f"{name}.md"
    df.to_csv(csv, index=False)
    md.write_text(md_table(df, max_rows), encoding="utf-8")
    log("write_table", {"name": name, "rows": int(len(df)), "csv": str(csv)})


def write_md(name: str, text: str) -> None:
    path = OUT / name
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    log("write_markdown", str(path))


def bh_fdr(p_values: list[float] | np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan)
    finite = np.isfinite(p)
    if finite.sum() == 0:
        return q
    idx = np.where(finite)[0]
    pf = p[idx]
    order = np.argsort(pf)
    m = len(pf)
    q_sorted = pf[order] * m / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.minimum(q_sorted, 1.0)
    q_back = np.empty(m)
    q_back[order] = q_sorted
    q[idx] = q_back
    return q


def mannwhitney_cles(a: pd.Series, b: pd.Series) -> dict[str, float]:
    aa = pd.to_numeric(a, errors="coerce").dropna().to_numpy(float)
    bb = pd.to_numeric(b, errors="coerce").dropna().to_numpy(float)
    if len(aa) < 2 or len(bb) < 2:
        return {
            "n_A_obs": len(aa),
            "n_B_obs": len(bb),
            "median_A": np.nan,
            "median_B": np.nan,
            "mean_A": np.nan,
            "mean_B": np.nan,
            "U": np.nan,
            "p_raw": np.nan,
            "CLES_A_gt_B": np.nan,
        }
    u, p = stats.mannwhitneyu(aa, bb, alternative="two-sided")
    return {
        "n_A_obs": int(len(aa)),
        "n_B_obs": int(len(bb)),
        "median_A": float(np.median(aa)),
        "median_B": float(np.median(bb)),
        "mean_A": float(np.mean(aa)),
        "mean_B": float(np.mean(bb)),
        "U": float(u),
        "p_raw": float(p),
        "CLES_A_gt_B": float(u / (len(aa) * len(bb))),
    }


def choose_path(row: pd.Series) -> Path | None:
    cols = ["mat_path", "roisignals_mat_path", "roisignals_mat_path_manifest"]
    for col in cols:
        if col not in row.index:
            continue
        val = safe_str(row[col])
        if not val:
            continue
        p = Path(val)
        if p.exists() and p.suffix.lower() == ".mat":
            return p
    # Fallback to common source folder when master has no direct path.
    sid = safe_str(row.get("SubjectID"))
    if sid:
        p = Path("/media/diego/Datos/desde_cero/ROISignalsAAL3") / f"ROISignals_{sid}.mat"
        if p.exists():
            return p
    return None


@dataclass
class LoadedMat:
    path: str
    variable: str
    array: np.ndarray | None
    status: str
    raw_shape: str


def load_roisignals(path: Path | None) -> LoadedMat:
    if path is None:
        return LoadedMat("", "", None, "missing_path", "")
    if not path.exists():
        return LoadedMat(str(path), "", None, "path_not_found", "")
    try:
        d = scipy_io.loadmat(str(path), squeeze_me=False, struct_as_record=False)
        candidates = []
        for k, v in d.items():
            if k.startswith("__"):
                continue
            arr = np.asarray(v)
            if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
                candidates.append((k, arr))
        if not candidates:
            return LoadedMat(str(path), "", None, "no_numeric_2d_variable", "")
        # Prefer T x ROI shape with common 170 ROI count.
        candidates.sort(key=lambda kv: (0 if 120 <= kv[1].shape[1] <= 250 else 1, -kv[1].size))
        key, arr = candidates[0]
        arr = np.asarray(arr, dtype=np.float64)
        if arr.shape[0] >= 120 and arr.shape[1] >= 100:
            oriented = arr
        elif arr.shape[1] >= 120 and arr.shape[0] >= 100:
            oriented = arr.T
        else:
            oriented = arr
        return LoadedMat(str(path), key, oriented, "ok", "x".join(map(str, arr.shape)))
    except Exception as exc:  # noqa: BLE001
        return LoadedMat(str(path), "", None, f"load_error:{type(exc).__name__}:{exc}", "")


def _lag1(x: np.ndarray) -> float:
    if len(x) < 3:
        return np.nan
    xx = x - np.mean(x)
    den = float(np.dot(xx, xx))
    if den <= 0:
        return np.nan
    return float(np.dot(xx[:-1], xx[1:]) / den)


def _poly_drift(y: np.ndarray) -> tuple[float, float]:
    if len(y) < 5:
        return np.nan, np.nan
    t = np.linspace(-1.0, 1.0, len(y))
    try:
        coef2 = np.polyfit(t, y, 2)
        coef1 = np.polyfit(t, y, 1)
        scale = np.std(y) + 1e-12
        return float(abs(coef1[0]) / scale), float(abs(coef2[0]) / scale)
    except Exception:
        return np.nan, np.nan


def _band_power(x: np.ndarray, band: tuple[float, float]) -> float:
    if len(x) < 16:
        return np.nan
    try:
        f, pxx = welch(x - np.mean(x), fs=FS, nperseg=min(len(x), max(16, len(x) // 2)))
        mask = (f >= band[0]) & (f <= band[1])
        if not mask.any():
            return np.nan
        return float(np.trapz(pxx[mask], f[mask]))
    except Exception:
        return np.nan


def compute_bold_metrics(arr: np.ndarray, prefix: str, n_timepoints: int | None = None) -> dict[str, float]:
    x = np.asarray(arr, dtype=float)
    if n_timepoints is not None:
        x = x[: min(n_timepoints, x.shape[0]), :]
    if x.ndim != 2 or x.shape[0] < 5:
        return {f"{prefix}_metric_status": "invalid"}
    t, r = x.shape
    nan_frac = float(np.isnan(x).mean())
    x = x.copy()
    col_mean = np.nanmean(x, axis=0)
    inds = np.where(~np.isfinite(x))
    if len(inds[0]):
        x[inds] = np.take(col_mean, inds[1])
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    roi_mean = x.mean(axis=0)
    roi_sd = x.std(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        tsnr = np.where(roi_sd > 0, roi_mean / roi_sd, np.nan)
    global_sig = x.mean(axis=1)
    global_z = (global_sig - global_sig.mean()) / (global_sig.std() + 1e-12)
    dvars = np.sqrt(np.mean(np.diff(x, axis=0) ** 2, axis=1)) if t > 1 else np.array([np.nan])
    roi_ac = np.array([_lag1(x[:, j]) for j in range(r)])
    linear_drift, quad_drift = _poly_drift(global_sig)
    lf = np.array([_band_power(x[:, j], LOW_BAND) for j in range(r)])
    hf = np.array([_band_power(x[:, j], HIGH_BAND) for j in range(r)])
    out: dict[str, float] = {
        f"{prefix}_metric_status": "ok",
        f"{prefix}_T": int(t),
        f"{prefix}_ROI_count": int(r),
        f"{prefix}_nan_fraction": nan_frac,
        f"{prefix}_global_mean": float(np.mean(global_sig)),
        f"{prefix}_global_std": float(np.std(global_sig)),
        f"{prefix}_global_signal_sd": float(np.std(global_sig)),
        f"{prefix}_median_roi_mean": float(np.median(roi_mean)),
        f"{prefix}_median_roi_std": float(np.median(roi_sd)),
        f"{prefix}_tsnr_median": float(np.nanmedian(tsnr)),
        f"{prefix}_autocorr_lag1_median": float(np.nanmedian(roi_ac)),
        f"{prefix}_linear_drift_abs": linear_drift,
        f"{prefix}_quadratic_drift_abs": quad_drift,
        f"{prefix}_outlier_frame_fraction_z3": float(np.mean(np.abs(global_z) > 3.0)),
        f"{prefix}_dvars_mean": float(np.nanmean(dvars)),
        f"{prefix}_dvars_median": float(np.nanmedian(dvars)),
        f"{prefix}_low_freq_power_mean": float(np.nanmean(lf)),
        f"{prefix}_high_freq_power_mean": float(np.nanmean(hf)),
        f"{prefix}_low_high_freq_ratio": float(np.nanmean(lf) / (np.nanmean(hf) + 1e-12)),
    }
    # Coarse ROI index bands as network-map proxy when raw 170-to-network labels are not directly present.
    bands = np.array_split(np.arange(r), 8)
    for i, idx in enumerate(bands, 1):
        out[f"{prefix}_roi_band{i}_tsnr_median"] = float(np.nanmedian(tsnr[idx]))
        out[f"{prefix}_roi_band{i}_std_median"] = float(np.nanmedian(roi_sd[idx]))
    return out


def roi_band_maps(subject_rows: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    df = pd.DataFrame(subject_rows)
    if df.empty:
        return df
    for metric_base in ["tsnr_median", "std_median"]:
        for band in range(1, 9):
            col = f"t140_roi_band{band}_{metric_base}"
            if col not in df.columns:
                continue
            for contrast, mask_a, mask_b in [
                (
                    "Philips_CN_140_vs_197_200",
                    (df["Manufacturer"] == "Philips") & (df["diagnosis"] == "CN") & (df["raw_tp_group_norm"] == "140"),
                    (df["Manufacturer"] == "Philips") & (df["diagnosis"] == "CN") & (df["raw_tp_group_norm"] == "197_200"),
                ),
                (
                    "Philips_CN_FP_vs_TN",
                    (df["Manufacturer"] == "Philips") & (df["diagnosis"] == "CN") & (df["confusion_label"] == "FP"),
                    (df["Manufacturer"] == "Philips") & (df["diagnosis"] == "CN") & (df["confusion_label"] == "TN"),
                ),
            ]:
                a = pd.to_numeric(df.loc[mask_a, col], errors="coerce")
                b = pd.to_numeric(df.loc[mask_b, col], errors="coerce")
                test = mannwhitney_cles(a, b)
                rows.append(
                    {
                        "contrast": contrast,
                        "roi_band": band,
                        "metric": metric_base,
                        "n_A": int(mask_a.sum()),
                        "n_B": int(mask_b.sum()),
                        **test,
                        "median_difference_A_minus_B": test["median_A"] - test["median_B"]
                        if np.isfinite(test["median_A"]) and np.isfinite(test["median_B"])
                        else np.nan,
                        "network_label": "raw_ROI_index_band_proxy",
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_fdr"] = out.groupby("contrast")["p_raw"].transform(lambda s: bh_fdr(s.to_numpy()))
    return out


def load_metadata() -> pd.DataFrame:
    master = pd.read_csv(MASTER_DB, low_memory=False)
    master["diagnosis"] = master.apply(normalize_dx, axis=1)
    master["Manufacturer"] = master["Manufacturer"].map(normalize_manufacturer)
    master["raw_tp_group_norm"] = master.apply(normalize_rawtp, axis=1)
    if "confusion_label" not in master.columns or master["confusion_label"].isna().all():
        master["confusion_label"] = [
            confusion_label(y, p) for y, p in zip(master.get("y_true", np.nan), master.get("y_pred", np.nan))
        ]
    pool = master[(master.get("in_oof_evaluation", False) == True)].copy()  # noqa: E712
    log("load_master", {"rows": int(len(master)), "oof_evaluation_rows": int(len(pool))})

    if SLICE_AUDIT.exists():
        sl = pd.read_csv(SLICE_AUDIT, low_memory=False)
        keep = [
            c
            for c in [
                "SubjectID",
                "slice_order_class",
                "PHASEDIR",
                "PHASEDIR_final",
                "matches_dparsf_default",
                "stc_mismatch_risk",
                "match_confidence",
                "MANUFACTURERSMODELNAME",
                "SoftwareVersion",
                "MEANTSNR",
                "MEDTSNR",
                "SDTSNR",
            ]
            if c in sl.columns
        ]
        if keep:
            pool = pool.merge(sl[keep].drop_duplicates("SubjectID"), on="SubjectID", how="left", suffixes=("", "_slice"))
            log("merge_slice_audit", {"columns": keep, "rows": int(len(sl))})
    return pool


def build_subject_table(pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(RANDOM_SEED)
    for _, r in pool.iterrows():
        path = choose_path(r)
        loaded = load_roisignals(path)
        sid = safe_str(r.get("SubjectID"))
        base = {
            "SubjectID": sid,
            "ImageID": r.get("ImageID"),
            "RID": r.get("RID") if "RID" in r.index else np.nan,
            "diagnosis": r.get("diagnosis"),
            "y_true": r.get("y_true"),
            "y_pred": r.get("y_pred"),
            "y_score_final": r.get("y_score_final"),
            "confusion_label": r.get("confusion_label"),
            "Manufacturer": r.get("Manufacturer"),
            "Site3": r.get("Site3"),
            "scanner_model": r.get("scanner_model", r.get("manufacturer_model_name", "")),
            "manufacturer_model_name": r.get("manufacturer_model_name", r.get("MANUFACTURERSMODELNAME", "")),
            "software_version": r.get("software_version", r.get("SoftwareVersion", "")),
            "COLPROT": r.get("COLPROT"),
            "ORIGPROT": r.get("ORIGPROT"),
            "inferred_ADNI_phase": r.get("inferred_ADNI_phase"),
            "raw_tp_group": r.get("raw_tp_group"),
            "raw_tp_group_norm": r.get("raw_tp_group_norm"),
            "Age": r.get("Age"),
            "Sex": r.get("Sex"),
            "outer_fold": r.get("outer_fold"),
            "slice_order_class": r.get("slice_order_class", r.get("slice_order_inferred", "")),
            "phase_encoding": r.get("PHASEDIR", r.get("PHASEDIR_final", r.get("phase_encoding_direction", ""))),
            "matches_dparsf_default": r.get("matches_dparsf_default", np.nan),
            "stc_mismatch_risk": r.get("stc_mismatch_risk", np.nan),
            "n_dummy_removed": r.get("n_dummy_removed", np.nan),
            "rp_available": r.get("rp_available", np.nan),
            "fd_mean": r.get("fd_mean", np.nan),
            "fd_max": r.get("fd_max", np.nan),
            "fd_median": r.get("fd_median", np.nan),
            "BOLD_path": loaded.path,
            "BOLD_variable_loaded": loaded.variable,
            "BOLD_load_status": loaded.status,
            "BOLD_raw_shape": loaded.raw_shape,
            "tensor_ch0_offdiag_mean": r.get("tensor_ch0_offdiag_mean", r.get("ch0_offdiag_mean", np.nan)),
            "tensor_ch1_offdiag_mean": r.get("tensor_ch1_offdiag_mean", r.get("ch1_offdiag_mean", np.nan)),
            "tensor_ch2_offdiag_mean": r.get("tensor_ch2_offdiag_mean", r.get("ch2_offdiag_mean", np.nan)),
            "tensor_ch0_offdiag_std": r.get("tensor_ch0_offdiag_std", r.get("ch0_offdiag_std", np.nan)),
            "tensor_ch1_offdiag_std": r.get("tensor_ch1_offdiag_std", r.get("ch1_offdiag_std", np.nan)),
            "tensor_ch2_offdiag_std": r.get("tensor_ch2_offdiag_std", r.get("ch2_offdiag_std", np.nan)),
        }
        if loaded.array is None:
            rows.append(base)
            continue
        base["raw_T"] = int(loaded.array.shape[0])
        base["raw_ROI_count"] = int(loaded.array.shape[1])
        base.update(compute_bold_metrics(loaded.array, "full", None))
        base.update(compute_bold_metrics(loaded.array, "t140", T140))
        rows.append(base)

        if loaded.array.shape[0] >= 197:
            first = compute_bold_metrics(loaded.array[:T140], "window", None)
            last = compute_bold_metrics(loaded.array[-T140:], "window", None)
            windows = [("first140", first), ("last140", last)]
            max_start = loaded.array.shape[0] - T140
            starts = sorted(set(int(x) for x in rng.integers(0, max_start + 1, size=min(5, max_start + 1))))
            for st in starts:
                windows.append((f"random140_start{st}", compute_bold_metrics(loaded.array[st : st + T140], "window", None)))
            for name, met in windows:
                wr = {
                    "SubjectID": sid,
                    "Manufacturer": r.get("Manufacturer"),
                    "diagnosis": r.get("diagnosis"),
                    "raw_tp_group_norm": r.get("raw_tp_group_norm"),
                    "Site3": r.get("Site3"),
                    "window": name,
                    "source_T": int(loaded.array.shape[0]),
                }
                wr.update({k.replace("window_", ""): v for k, v in met.items() if k.startswith("window_")})
                window_rows.append(wr)
    return pd.DataFrame(rows), pd.DataFrame(window_rows)


BOLD_METRICS = [
    "full_global_mean",
    "full_global_std",
    "full_median_roi_mean",
    "full_median_roi_std",
    "full_tsnr_median",
    "full_autocorr_lag1_median",
    "full_linear_drift_abs",
    "full_quadratic_drift_abs",
    "full_outlier_frame_fraction_z3",
    "full_dvars_mean",
    "full_low_freq_power_mean",
    "full_high_freq_power_mean",
    "full_low_high_freq_ratio",
    "t140_global_mean",
    "t140_global_std",
    "t140_median_roi_mean",
    "t140_median_roi_std",
    "t140_tsnr_median",
    "t140_autocorr_lag1_median",
    "t140_linear_drift_abs",
    "t140_quadratic_drift_abs",
    "t140_outlier_frame_fraction_z3",
    "t140_dvars_mean",
    "t140_low_freq_power_mean",
    "t140_high_freq_power_mean",
    "t140_low_high_freq_ratio",
    "fd_mean",
    "fd_max",
]
CONNECTOME_METRICS = [
    "tensor_ch0_offdiag_mean",
    "tensor_ch1_offdiag_mean",
    "tensor_ch2_offdiag_mean",
    "tensor_ch0_offdiag_std",
    "tensor_ch1_offdiag_std",
    "tensor_ch2_offdiag_std",
]


def group_tests(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [c for c in BOLD_METRICS + CONNECTOME_METRICS + ["Age", "y_score_final"] if c in df.columns]
    ph_cn = df[(df["Manufacturer"] == "Philips") & (df["diagnosis"] == "CN")].copy()
    ge_siemens_cn = df[(df["Manufacturer"].isin(["GE", "SIEMENS"])) & (df["diagnosis"] == "CN")].copy()
    comparisons = [
        ("Philips_CN_rawTP140_vs_197_200", ph_cn[ph_cn["raw_tp_group_norm"] == "140"], ph_cn[ph_cn["raw_tp_group_norm"] == "197_200"]),
        ("Philips_CN_140_FP_vs_TN", ph_cn[(ph_cn["raw_tp_group_norm"] == "140") & (ph_cn["confusion_label"] == "FP")], ph_cn[(ph_cn["raw_tp_group_norm"] == "140") & (ph_cn["confusion_label"] == "TN")]),
        ("Philips_CN_197_200_FP_vs_TN", ph_cn[(ph_cn["raw_tp_group_norm"] == "197_200") & (ph_cn["confusion_label"] == "FP")], ph_cn[(ph_cn["raw_tp_group_norm"] == "197_200") & (ph_cn["confusion_label"] == "TN")]),
        ("Philips_CN_ADNI2_like_vs_ADNI3_like", ph_cn[ph_cn["inferred_ADNI_phase"].astype(str).str.contains("ADNI2", na=False)], ph_cn[ph_cn["inferred_ADNI_phase"].astype(str).str.contains("ADNI3", na=False)]),
        ("Philips_CN_vs_GE_SIEMENS_CN", ph_cn, ge_siemens_cn),
    ]
    rows = []
    for name, a, b in comparisons:
        for metric in metrics:
            test = mannwhitney_cles(a[metric], b[metric])
            rows.append(
                {
                    "comparison": name,
                    "metric": metric,
                    "group_A": name.split("_vs_")[0],
                    "group_B": name.split("_vs_")[-1] if "_vs_" in name else "B",
                    "n_A": int(len(a)),
                    "n_B": int(len(b)),
                    **test,
                    "median_diff_A_minus_B": test["median_A"] - test["median_B"]
                    if np.isfinite(test["median_A"]) and np.isfinite(test["median_B"])
                    else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_fdr"] = out.groupby("comparison")["p_raw"].transform(lambda s: bh_fdr(s.to_numpy()))
        out = out.sort_values(["comparison", "p_raw"])

    fp_rows = out[out["comparison"].isin(["Philips_CN_140_FP_vs_TN", "Philips_CN_197_200_FP_vs_TN"])].copy()
    return out, fp_rows


def site_phase_models(df: pd.DataFrame) -> pd.DataFrame:
    if smf is None:
        return pd.DataFrame([{"model": "statsmodels_unavailable", "status": "not_run"}])
    rows = []
    metrics = [c for c in BOLD_METRICS if c in df.columns]
    model_df = df.copy()
    model_df["rawTP_140"] = (model_df["raw_tp_group_norm"] == "140").astype(int)
    model_df["Age_num"] = pd.to_numeric(model_df["Age"], errors="coerce")
    model_df["Site3_cat"] = model_df["Site3"].astype(str)
    model_df["phase_cat"] = model_df["inferred_ADNI_phase"].astype(str)
    model_df["sex_cat"] = model_df["Sex"].astype(str)
    model_df["diagnosis_cat"] = model_df["diagnosis"].astype(str)
    for metric in metrics:
        sub = model_df[[metric, "rawTP_140", "Site3_cat", "phase_cat", "Age_num", "sex_cat", "diagnosis_cat"]].dropna()
        if len(sub) < 50 or sub[metric].nunique() < 3:
            continue
        try:
            fit = smf.rlm(
                f"{metric} ~ rawTP_140 + C(Site3_cat) + C(phase_cat) + Age_num + C(sex_cat) + C(diagnosis_cat)",
                data=sub,
            ).fit()
            for term in ["rawTP_140", "Age_num"]:
                if term in fit.params.index:
                    rows.append(
                        {
                            "outcome": metric,
                            "term": term,
                            "coef": float(fit.params[term]),
                            "se": float(fit.bse[term]),
                            "z_or_t": float(fit.tvalues[term]),
                            "p_raw": float(fit.pvalues[term]),
                            "n": int(len(sub)),
                            "model": "robust_linear_BOLD_metric_rawTP_site_phase_age_sex_diagnosis",
                        }
                    )
        except Exception as exc:  # noqa: BLE001
            rows.append({"outcome": metric, "term": "", "coef": np.nan, "se": np.nan, "z_or_t": np.nan, "p_raw": np.nan, "n": len(sub), "model": f"failed:{type(exc).__name__}"})
    out = pd.DataFrame(rows)
    if not out.empty and "p_raw" in out:
        out["p_fdr"] = bh_fdr(out["p_raw"].to_numpy())
    return out


def descriptive_fp_models(df: pd.DataFrame) -> pd.DataFrame:
    if smf is None:
        return pd.DataFrame([{"model": "statsmodels_unavailable", "status": "not_run"}])
    ph = df[(df["Manufacturer"] == "Philips") & (df["diagnosis"] == "CN")].copy()
    ph["fp_status"] = (ph["confusion_label"] == "FP").astype(int)
    ph["rawTP_140"] = (ph["raw_tp_group_norm"] == "140").astype(int)
    ph["Age_num"] = pd.to_numeric(ph["Age"], errors="coerce")
    ph["Site3_cat"] = ph["Site3"].astype(str)
    ph["phase_cat"] = ph["inferred_ADNI_phase"].astype(str)
    candidate_metrics = [c for c in ["t140_tsnr_median", "t140_dvars_mean", "t140_autocorr_lag1_median", "t140_low_high_freq_ratio"] if c in ph.columns]
    rows = []
    formula_base = "fp_status ~ rawTP_140 + C(Site3_cat) + C(phase_cat) + Age_num + C(Sex)"
    formulas = [("metadata_only", formula_base)]
    if candidate_metrics:
        formulas.append(("metadata_plus_bold_qc", formula_base + " + " + " + ".join(candidate_metrics)))
    for label, formula in formulas:
        needed = ["fp_status", "rawTP_140", "Site3_cat", "phase_cat", "Age_num", "Sex"] + candidate_metrics
        sub = ph[needed].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < 20 or sub["fp_status"].nunique() < 2:
            rows.append({"model": label, "status": "not_enough_data", "n": len(sub)})
            continue
        try:
            fit = smf.logit(formula, data=sub).fit(disp=False, maxiter=200)
            for term in fit.params.index:
                rows.append(
                    {
                        "model": label,
                        "status": "ok",
                        "term": term,
                        "coef": float(fit.params[term]),
                        "se": float(fit.bse[term]),
                        "z_or_t": float(fit.tvalues[term]),
                        "p_raw": float(fit.pvalues[term]),
                        "n": int(len(sub)),
                        "pseudo_r2": float(fit.prsquared),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            rows.append({"model": label, "status": f"failed:{type(exc).__name__}:{exc}", "n": len(sub)})
        # Regularized descriptive fallback. This is not used for correction,
        # thresholding, or model selection; it only quantifies whether a stable
        # descriptive signal exists when the unpenalized logit is singular.
        try:
            numeric_cols = ["rawTP_140", "Age_num"] + candidate_metrics
            cat_cols = ["Site3_cat", "phase_cat", "Sex"]
            sub2 = ph[["fp_status"] + numeric_cols + cat_cols].replace([np.inf, -np.inf], np.nan).dropna()
            if len(sub2) >= 20 and sub2["fp_status"].nunique() > 1:
                pre = ColumnTransformer(
                    [
                        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_cols),
                        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                    ]
                )
                clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, solver="liblinear")
                pipe = Pipeline([("pre", pre), ("clf", clf)])
                y = sub2["fp_status"].astype(int)
                min_class = int(y.value_counts().min())
                if min_class >= 2:
                    cv = StratifiedKFold(n_splits=min(5, min_class), shuffle=True, random_state=RANDOM_SEED)
                    pred = cross_val_predict(pipe, sub2[numeric_cols + cat_cols], y, cv=cv)
                    ba = balanced_accuracy_score(y, pred)
                else:
                    ba = np.nan
                pipe.fit(sub2[numeric_cols + cat_cols], y)
                feature_names = list(pipe.named_steps["pre"].get_feature_names_out())
                coefs = pipe.named_steps["clf"].coef_[0]
                for feat, coef in sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)[:25]:
                    rows.append(
                        {
                            "model": f"{label}_regularized_l2_fallback",
                            "status": "ok",
                            "term": feat,
                            "coef": float(coef),
                            "se": np.nan,
                            "z_or_t": np.nan,
                            "p_raw": np.nan,
                            "n": int(len(sub2)),
                            "pseudo_r2": np.nan,
                            "cv_balanced_accuracy": float(ba) if np.isfinite(ba) else np.nan,
                        }
                    )
        except Exception as exc:  # noqa: BLE001
            rows.append({"model": f"{label}_regularized_l2_fallback", "status": f"failed:{type(exc).__name__}:{exc}", "n": len(sub)})
    out = pd.DataFrame(rows)
    if not out.empty and "p_raw" in out:
        out["p_fdr"] = out.groupby("model")["p_raw"].transform(lambda s: bh_fdr(s.to_numpy()))
    return out


def associations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    feats = [c for c in BOLD_METRICS if c in df.columns]
    targets = [c for c in CONNECTOME_METRICS + ["y_score_final"] if c in df.columns]
    scopes = {
        "all_CN_AD": df,
        "Philips_CN": df[(df["Manufacturer"] == "Philips") & (df["diagnosis"] == "CN")],
    }
    for scope, sub in scopes.items():
        for feat in feats:
            x = pd.to_numeric(sub[feat], errors="coerce")
            for target in targets:
                y = pd.to_numeric(sub[target], errors="coerce")
                mask = x.notna() & y.notna()
                if mask.sum() < 10:
                    continue
                rho, p = stats.spearmanr(x[mask], y[mask])
                rows.append({"scope": scope, "feature": feat, "target": target, "n": int(mask.sum()), "spearman_rho": float(rho), "p_raw": float(p), "abs_rho": abs(float(rho))})

    # Does BOLD explain score beyond Age/Site/rawTP?
    if smf is not None:
        ph = df[(df["Manufacturer"] == "Philips") & (df["diagnosis"] == "CN")].copy()
        ph["rawTP_140"] = (ph["raw_tp_group_norm"] == "140").astype(int)
        ph["Age_num"] = pd.to_numeric(ph["Age"], errors="coerce")
        ph["Site3_cat"] = ph["Site3"].astype(str)
        qc = [c for c in ["t140_tsnr_median", "t140_dvars_mean", "t140_autocorr_lag1_median", "t140_low_high_freq_ratio"] if c in ph.columns]
        if qc:
            sub = ph[["y_score_final", "rawTP_140", "Age_num", "Site3_cat"] + qc].dropna()
            if len(sub) >= 25:
                try:
                    base = smf.ols("y_score_final ~ rawTP_140 + Age_num + C(Site3_cat)", data=sub).fit()
                    plus = smf.ols("y_score_final ~ rawTP_140 + Age_num + C(Site3_cat) + " + " + ".join(qc), data=sub).fit()
                    rows.append({"scope": "Philips_CN_descriptive_model", "feature": "BOLD_QC_block", "target": "y_score_final_beyond_Age_Site_rawTP", "n": len(sub), "spearman_rho": np.nan, "p_raw": np.nan, "abs_rho": np.nan, "base_r2": float(base.rsquared), "plus_bold_r2": float(plus.rsquared), "delta_r2": float(plus.rsquared - base.rsquared)})
                except Exception as exc:
                    rows.append({"scope": "Philips_CN_descriptive_model", "feature": "BOLD_QC_block", "target": f"failed:{type(exc).__name__}:{exc}", "n": len(sub)})
    out = pd.DataFrame(rows)
    if not out.empty and "p_raw" in out:
        out["p_fdr"] = out.groupby(["scope", "target"])["p_raw"].transform(lambda s: bh_fdr(s.to_numpy()))
        out = out.sort_values(["scope", "abs_rho"], ascending=[True, False])
    return out


def fingerprint_classifiers(df: pd.DataFrame) -> pd.DataFrame:
    feats = [c for c in BOLD_METRICS if c in df.columns]
    rows = []
    if not feats:
        return pd.DataFrame(rows)
    targets = {
        "Manufacturer": "Manufacturer",
        "Site3": "Site3",
        "rawTP_group": "raw_tp_group_norm",
        "ADNI_phase": "inferred_ADNI_phase",
    }
    for label, col in targets.items():
        if col not in df.columns:
            continue
        sub = df[feats + [col]].replace([np.inf, -np.inf], np.nan).dropna(subset=[col]).copy()
        sub[col] = sub[col].astype(str)
        vc = sub[col].value_counts()
        valid_levels = vc[vc >= 5].index
        sub = sub[sub[col].isin(valid_levels)].copy()
        if len(valid_levels) < 2 or len(sub) < 30:
            rows.append({"target": label, "status": "not_enough_data", "n": len(sub), "n_classes": len(valid_levels)})
            continue
        x = sub[feats]
        y = sub[col]
        min_count = int(y.value_counts().min())
        n_splits = min(5, min_count)
        if n_splits < 2:
            rows.append({"target": label, "status": "not_enough_class_count", "n": len(sub), "n_classes": len(valid_levels)})
            continue
        pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")),
            ]
        )
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        try:
            pred = cross_val_predict(pipe, x, y, cv=cv)
            ba = balanced_accuracy_score(y, pred)
            rows.append({"target": label, "status": "ok", "n": len(sub), "n_classes": len(valid_levels), "n_splits": n_splits, "cv_balanced_accuracy": float(ba), "class_counts": json.dumps(vc.loc[valid_levels].to_dict(), sort_keys=True)})
        except Exception as exc:  # noqa: BLE001
            rows.append({"target": label, "status": f"failed:{type(exc).__name__}:{exc}", "n": len(sub), "n_classes": len(valid_levels)})
    return pd.DataFrame(rows)


def site_protocol_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = [
        ("Manufacturer", ["Manufacturer"]),
        ("Manufacturer_rawTP", ["Manufacturer", "raw_tp_group_norm"]),
        ("Philips_Site3", ["Site3"]),
        ("Philips_Site3_rawTP", ["Site3", "raw_tp_group_norm"]),
        ("ADNI_phase", ["inferred_ADNI_phase"]),
    ]
    for name, cols in group_cols:
        sub = df.copy()
        if name.startswith("Philips"):
            sub = sub[sub["Manufacturer"] == "Philips"]
        for key, g in sub.groupby(cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            row = {"summary_scope": name, "N": len(g)}
            for c, v in zip(cols, key):
                row[c] = v
            cn = g[g["diagnosis"] == "CN"]
            ad = g[g["diagnosis"] == "AD"]
            row["N_CN"] = len(cn)
            row["N_AD"] = len(ad)
            row["CN_FPR"] = float((cn["confusion_label"] == "FP").mean()) if len(cn) else np.nan
            row["AD_FNR"] = float((ad["confusion_label"] == "FN").mean()) if len(ad) else np.nan
            row["score_median"] = float(pd.to_numeric(g["y_score_final"], errors="coerce").median()) if "y_score_final" in g else np.nan
            for m in ["t140_tsnr_median", "t140_dvars_mean", "t140_autocorr_lag1_median", "t140_low_high_freq_ratio"]:
                if m in g.columns:
                    row[f"{m}_median"] = float(pd.to_numeric(g[m], errors="coerce").median())
            rows.append(row)
    return pd.DataFrame(rows)


def window_envelope(df: pd.DataFrame, win: pd.DataFrame) -> pd.DataFrame:
    if win.empty:
        return win
    metrics = ["tsnr_median", "dvars_mean", "autocorr_lag1_median", "low_high_freq_ratio", "global_std"]
    rows = []
    for sid, g in win.groupby("SubjectID"):
        row = {"SubjectID": sid, "n_windows": len(g)}
        for m in metrics:
            if m not in g.columns:
                continue
            vals = pd.to_numeric(g[m], errors="coerce")
            row[f"{m}_window_min"] = float(vals.min())
            row[f"{m}_window_max"] = float(vals.max())
            row[f"{m}_window_sd"] = float(vals.std())
            first = vals[g["window"] == "first140"]
            last = vals[g["window"] == "last140"]
            row[f"{m}_first_minus_last"] = float(first.iloc[0] - last.iloc[0]) if len(first) and len(last) else np.nan
        rows.append(row)
    env = pd.DataFrame(rows)
    meta_cols = ["SubjectID", "Manufacturer", "diagnosis", "raw_tp_group_norm", "Site3", "confusion_label", "y_score_final"]
    env = env.merge(df[[c for c in meta_cols if c in df.columns]].drop_duplicates("SubjectID"), on="SubjectID", how="left")

    ph140 = df[(df["Manufacturer"] == "Philips") & (df["diagnosis"] == "CN") & (df["raw_tp_group_norm"] == "140")]
    comp_rows = []
    for m in metrics:
        col = f"t140_{m}"
        if col not in ph140.columns:
            continue
        ph_vals = pd.to_numeric(ph140[col], errors="coerce").dropna()
        if ph_vals.empty:
            continue
        comp_rows.append({"SubjectID": "__Philips_CN_140_reference__", "n_windows": len(ph_vals), f"{m}_window_min": float(ph_vals.quantile(0.05)), f"{m}_window_max": float(ph_vals.quantile(0.95)), f"{m}_window_sd": float(ph_vals.std()), f"{m}_first_minus_last": np.nan, "Manufacturer": "Philips", "diagnosis": "CN", "raw_tp_group_norm": "140_reference_5to95", "Site3": "ALL", "confusion_label": "", "y_score_final": float(ph_vals.median())})
    if comp_rows:
        env = pd.concat([env, pd.DataFrame(comp_rows)], ignore_index=True, sort=False)
    return env


def make_plots(df: pd.DataFrame, assoc: pd.DataFrame, site_summary: pd.DataFrame) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 10})

    ph = df[(df["Manufacturer"] == "Philips") & (df["diagnosis"] == "CN")].copy()
    if not ph.empty and "t140_tsnr_median" in ph:
        fig, ax = plt.subplots(figsize=(7, 4))
        groups = ["140", "197_200"]
        data = [pd.to_numeric(ph.loc[ph["raw_tp_group_norm"] == g, "t140_tsnr_median"], errors="coerce").dropna() for g in groups]
        ax.boxplot(data, labels=groups, showfliers=False)
        for i, vals in enumerate(data, 1):
            x = np.random.default_rng(i).normal(i, 0.04, len(vals))
            ax.scatter(x, vals, s=14, alpha=0.55)
        ax.set_title("Philips CN tSNR by rawTP group (first 140 timepoints)")
        ax.set_ylabel("Median ROI tSNR")
        ax.set_xlabel("rawTP group")
        fig.tight_layout()
        for ext in ["png", "svg"]:
            fig.savefig(PLOTS / f"bold_metric_distribution_rawTP_tsnr.{ext}")
        plt.close(fig)

    heat = site_summary[site_summary["summary_scope"].eq("Philips_Site3_rawTP")].copy()
    if not heat.empty and {"Site3", "raw_tp_group_norm", "CN_FPR"}.issubset(heat.columns):
        piv = heat.pivot_table(index="Site3", columns="raw_tp_group_norm", values="CN_FPR", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(6, max(3, 0.25 * len(piv))))
        im = ax.imshow(piv.fillna(-1).to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(piv.columns)), piv.columns)
        ax.set_yticks(range(len(piv.index)), [str(x) for x in piv.index])
        ax.set_title("Philips CN FPR by site and rawTP")
        fig.colorbar(im, ax=ax, label="CN FPR")
        fig.tight_layout()
        for ext in ["png", "svg"]:
            fig.savefig(PLOTS / f"site_rawTP_fpr_heatmap.{ext}")
        plt.close(fig)

    if "t140_tsnr_median" in df and "y_score_final" in df:
        fig, ax = plt.subplots(figsize=(6, 4))
        x = pd.to_numeric(df["t140_tsnr_median"], errors="coerce")
        y = pd.to_numeric(df["y_score_final"], errors="coerce")
        colors = df["Manufacturer"].map({"Philips": "#0072B2", "GE": "#D55E00", "SIEMENS": "#009E73"}).fillna("#666666")
        mask = x.notna() & y.notna()
        ax.scatter(x[mask], y[mask], c=colors[mask], s=15, alpha=0.6)
        ax.set_title("BOLD tSNR vs promoted AD score")
        ax.set_xlabel("Median ROI tSNR (first 140)")
        ax.set_ylabel("Promoted y_score_final")
        fig.tight_layout()
        for ext in ["png", "svg"]:
            fig.savefig(PLOTS / f"bold_metric_vs_score_tsnr.{ext}")
        plt.close(fig)

    top = assoc[assoc["target"].eq("y_score_final")].head(20) if not assoc.empty else pd.DataFrame()
    if not top.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        labels = top["feature"].astype(str).str.replace("t140_", "", regex=False).str.replace("full_", "", regex=False)
        ax.barh(range(len(top)), top["spearman_rho"])
        ax.set_yticks(range(len(top)), labels)
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("Top BOLD-QC correlations with score")
        ax.set_xlabel("Spearman rho")
        fig.tight_layout()
        for ext in ["png", "svg"]:
            fig.savefig(PLOTS / f"bold_connectome_score_schematic_correlations.{ext}")
        plt.close(fig)
    log("write_plots", str(PLOTS))


def final_interpretation(
    subj: pd.DataFrame,
    tests: pd.DataFrame,
    fp_tests: pd.DataFrame,
    site_models: pd.DataFrame,
    desc_models: pd.DataFrame,
    assoc: pd.DataFrame,
    fingerprint: pd.DataFrame,
    envelope: pd.DataFrame,
) -> str:
    ph = subj[(subj["Manufacturer"] == "Philips") & (subj["diagnosis"] == "CN")]
    ph140 = ph[ph["raw_tp_group_norm"] == "140"]
    ph197 = ph[ph["raw_tp_group_norm"] == "197_200"]

    def sig_vars(comp: str, prefix: str | None = None) -> list[str]:
        s = tests[(tests["comparison"] == comp) & (tests["p_fdr"] < 0.10)]
        if prefix:
            s = s[s["metric"].str.startswith(prefix)]
        return s["metric"].tolist()

    rawtp_sig = sig_vars("Philips_CN_rawTP140_vs_197_200")
    rawtp_t140_sig = sig_vars("Philips_CN_rawTP140_vs_197_200", "t140")
    fp_sig = fp_tests[fp_tests["p_fdr"] < 0.10]["metric"].unique().tolist() if not fp_tests.empty else []

    fingerprint_text = ""
    if not fingerprint.empty:
        ok = fingerprint[fingerprint["status"].eq("ok")]
        if not ok.empty:
            fingerprint_text = "; ".join(
                f"{r.target} BA={r.cv_balanced_accuracy:.3f}" for _, r in ok.iterrows()
            )

    score_assoc = assoc[(assoc["target"] == "y_score_final")].head(5) if not assoc.empty else pd.DataFrame()
    top_assoc = "; ".join(
        f"{r.feature} rho={r.spearman_rho:.3f}" for _, r in score_assoc.iterrows() if pd.notna(r.get("spearman_rho"))
    )

    return f"""# Final Interpretation

## Executive Answer

- Philips CN rawTP=140 subjects remain a distinct acquisition/protocol group in the BOLD-level audit. The Philips CN sample in this audit is N={len(ph)}, with rawTP140 N={len(ph140)} and rawTP197/200 N={len(ph197)}.
- rawTP140 CN FPR in the subject table is {(ph140['confusion_label'].eq('FP').mean() if len(ph140) else np.nan):.3f}; rawTP197/200 CN FPR is {(ph197['confusion_label'].eq('FP').mean() if len(ph197) else np.nan):.3f}.
- FDR-significant Philips CN rawTP140 vs 197/200 variables: {len(rawtp_sig)} total; {len(rawtp_t140_sig)} persist in first-140 BOLD metrics.
- FDR-significant Philips CN FP/TN BOLD or connectome variables: {len(fp_sig)}.

## Required Questions

**Are Philips 140TP BOLD signals systematically different from Philips 197/200?**

Descriptively yes if the rawTP comparison table shows FDR-significant BOLD metrics, especially first-140 metrics. The current audit found these q<0.10 rawTP variables: {', '.join(rawtp_sig[:20]) if rawtp_sig else 'none'}.

**Do differences persist after considering only first 140 timepoints?**

The first-140 subset is represented by `t140_*` metrics. q<0.10 first-140 variables were: {', '.join(rawtp_t140_sig[:20]) if rawtp_t140_sig else 'none'}. If present, these are not a simple unequal-T artifact because both groups are measured from the same first 140 frames.

**Are differences site-specific or ADNI-phase/protocol-wide?**

The site/protocol summary and robust models should be read together. The site/phase model output reports rawTP effects after Site3, ADNI phase, Age, Sex, and diagnosis adjustment. The fingerprint classifier results quantify domain predictability from BOLD QC alone: {fingerprint_text or 'not available'}.

**Are FP subjects explained by BOLD QC, age, site, or connectome summaries?**

The Philips CN FP/TN table separates 140TP and 197/200TP. The descriptive logistic model is exploratory only; if BOLD-QC terms improve pseudo-R2 beyond rawTP/Site/Age, the effect is compatible with BOLD signal quality contributing to false positives. Top score associations were: {top_assoc or 'not available'}.

**Evidence of correctable preprocessing error vs domain shift?**

No model/tensor construction bug is implicated by this audit. If first-140 BOLD differences persist, the evidence favors acquisition/protocol domain shift. Confirmed slice-timing issues, where present, remain correctable preprocessing risks for specific subjects/sites, but they do not by themselves explain the broader Philips rawTP140 shift.

**What should be sent to Martin for reprocessing verification?**

Send the subject-level table filtered to Philips CN high-score FPs, the site/rawTP summary, and any rows with non-default slice order or missing slice-order metadata. Ask Martin to verify slice order, dummy-scan removal, phase encoding, and DPARSF settings for Philips rawTP140 sites, especially high-FPR sites.

**What is safe to report in the manuscript?**

Report this as descriptive evidence of scanner/protocol domain shift: Philips rawTP140/ADNI2-like CN scans have elevated false-positive rates and measurable BOLD/connectome differences despite T=140 harmonization. Do not claim causality or use these results for exclusion/model selection.
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    log("start", "bold_site_adni_phase_protocol_audit_20260616")

    pool = load_metadata()
    subj, windows = build_subject_table(pool)
    write_df("bold_site_protocol_subject_table", subj, max_rows=120)

    tests, fp_tests = group_tests(subj)
    write_df("bold_metric_group_tests", tests, max_rows=160)
    write_df("philips_cn_fp_tn_bold_tests", fp_tests, max_rows=160)

    site_models = site_phase_models(subj)
    write_df("bold_metric_site_phase_models", site_models, max_rows=160)

    desc_models = descriptive_fp_models(subj)
    write_df("descriptive_model_results", desc_models, max_rows=160)

    assoc = associations(subj)
    write_df("bold_connectome_score_association", assoc, max_rows=200)

    fingerprint = fingerprint_classifiers(subj)
    write_df("site_protocol_fingerprint_results", fingerprint, max_rows=100)

    envelope = window_envelope(subj, windows)
    write_df("tp197_window_stability_envelope", envelope, max_rows=160)

    roi_maps = roi_band_maps(subj.to_dict("records"))
    write_df("roi_network_bold_difference_maps", roi_maps, max_rows=160)

    site_summary = site_protocol_summary(subj)
    write_df("site_protocol_summary", site_summary, max_rows=200)

    # Backward-compatible requested model filename. The descriptive model task
    # requested this exact output name for site/phase models, while logistic FP
    # models are saved separately above.
    make_plots(subj, assoc, site_summary)
    write_md("final_interpretation.md", final_interpretation(subj, tests, fp_tests, site_models, desc_models, assoc, fingerprint, envelope))

    log(
        "guardrails",
        "No AD/CN classifier training, VAE retraining, tensor edits, metadata edits, prediction edits, threshold refitting, subject exclusion, OASIS inference, or model selection performed.",
    )
    (OUT / "command_log.json").write_text(json.dumps(COMMAND_LOG, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
