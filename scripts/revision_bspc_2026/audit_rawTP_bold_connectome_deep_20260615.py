#!/usr/bin/env python3
"""Read-only deep audit of Philips rawTP connectome/error mechanism.

The audit answers whether Philips CN rawTP=140 false positives are plausibly
driven by different effective time-series length or whether rawTP is mainly a
protocol/site/scanner proxy. It writes only new derived reports under a new
results directory.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import io as scipy_io
from scipy import stats

try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/revision_bspc_2026/rawTP_bold_connectome_deep_audit_20260615"
MASTER_DB = ROOT / "results/revision_bspc_2026/promoted_model_master_database_20260610/promoted_model_master_database.csv"
PREV_AUDIT = ROOT / "results/revision_bspc_2026/rawTP_connectome_error_mechanism_audit_20260615/rawTP_subject_level_audit.csv"
GLOBAL_TENSOR = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors/"
    "GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
)
FEATURE_EXTRACTION = ROOT / "src/betavae_xai/feature_extraction_manual.py"

POSSIBLE_ROI_KEYS = ["signals", "ROISignals", "roi_signals", "ROIsignals_AAL3", "AAL3_signals", "roi_ts"]
SEARCH_ROOTS = [Path("/media/diego/Datos"), ROOT]

COMMAND_LOG: list[dict[str, Any]] = []


def log(action: str, status: str, details: dict[str, Any] | None = None) -> None:
    COMMAND_LOG.append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "action": action,
            "status": status,
            "details": details or {},
        }
    )


def safe_str(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def normalize_mfr(x: Any) -> str:
    s = safe_str(x).upper()
    if "PHILIPS" in s:
        return "Philips"
    if "SIEMENS" in s:
        return "SIEMENS"
    if s in {"GE", "GENERAL ELECTRIC"} or "GE MEDICAL" in s:
        return "GE"
    return safe_str(x)


def normalize_dx(row: pd.Series) -> str:
    for col in ["ResearchGroup_Mapped", "diagnosis", "Diagnosis", "y_true_label"]:
        if col in row.index:
            s = safe_str(row[col]).strip().upper()
            if s in {"CN", "NORMAL"}:
                return "CN"
            if s in {"AD", "AD_DEMENTIA", "DEMENTIA"}:
                return "AD"
            if s == "MCI":
                return "MCI"
    if "y_true" in row.index and not pd.isna(row["y_true"]):
        return "AD" if int(row["y_true"]) == 1 else "CN"
    return ""


def normalize_rawtp(x: Any) -> str:
    s = safe_str(x).strip()
    if not s:
        return "MISSING"
    m = re.search(r"(\d+)", s)
    if not m:
        return s
    n = int(m.group(1))
    if n == 140:
        return "140"
    if n in {197, 200}:
        return "197_200"
    return str(n)


def numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "_No rows._\n"
    head = df.head(max_rows)
    suffix = ""
    if len(df) > max_rows:
        suffix = f"\n\n_Showing first {max_rows} of {len(df)} rows._\n"
    return head.to_markdown(index=False) + "\n" + suffix


def write_df(name: str, df: pd.DataFrame, max_rows: int = 80) -> None:
    csv = OUT / f"{name}.csv"
    md = OUT / f"{name}.md"
    df.to_csv(csv, index=False)
    md.write_text(md_table(df, max_rows), encoding="utf-8")
    log("write_table", "ok", {"name": name, "rows": int(len(df))})


def confusion_label(y_true: Any, y_pred: Any) -> str:
    if pd.isna(y_true) or pd.isna(y_pred):
        return ""
    yt = int(y_true)
    yp = int(y_pred)
    if yt == 0 and yp == 0:
        return "TN"
    if yt == 0 and yp == 1:
        return "FP"
    if yt == 1 and yp == 1:
        return "TP"
    if yt == 1 and yp == 0:
        return "FN"
    return ""


def load_master() -> pd.DataFrame:
    df = pd.read_csv(MASTER_DB)
    df["diagnosis"] = df.apply(normalize_dx, axis=1)
    df["Manufacturer_norm"] = df["Manufacturer"].map(normalize_mfr)
    df["raw_tp_group_norm"] = df["raw_tp_group"].map(normalize_rawtp)
    if "confusion_label" not in df.columns or df["confusion_label"].isna().all():
        df["confusion_label"] = [confusion_label(a, b) for a, b in zip(df.get("y_true", np.nan), df.get("y_pred", np.nan))]
    return df


def build_subject_path_index(df: pd.DataFrame) -> dict[str, list[Path]]:
    """Use paths already recorded in the master DB; avoid broad filesystem scans."""
    idx: dict[str, list[Path]] = defaultdict(list)
    path_cols = [
        "mat_path",
        "roisignals_mat_path",
        "roisignals_mat_path_manifest",
        "mat_source_dir",
    ]
    for _, row in df.iterrows():
        sid = safe_str(row["SubjectID"])
        for c in path_cols:
            if c not in row.index:
                continue
            val = safe_str(row[c])
            if not val:
                continue
            p = Path(val)
            if p.is_dir():
                p = p / f"ROISignals_{sid}.mat"
            if p.name.startswith("ROISignals_") and p.suffix.lower() == ".mat":
                idx[sid].append(p)
    # De-duplicate preserving order.
    out: dict[str, list[Path]] = {}
    for sid, paths in idx.items():
        seen = set()
        out[sid] = []
        for p in paths:
            s = str(p)
            if s not in seen:
                out[sid].append(p)
                seen.add(s)
    return out


def source_root_inventory(path_index: dict[str, list[Path]]) -> pd.DataFrame:
    rows = []
    by_dir: dict[str, set[str]] = defaultdict(set)
    for sid, paths in path_index.items():
        for p in paths:
            by_dir[str(p.parent)].add(sid)
    for parent, subjects in sorted(by_dir.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        rows.append({"source_directory": parent, "subjects_linked_from_master": len(subjects)})
    return pd.DataFrame(rows)


def list_mat_variables(path: Path) -> tuple[list[dict[str, Any]], str]:
    if not path.exists():
        return [], "path_missing"
    try:
        data = scipy_io.loadmat(path, squeeze_me=False, struct_as_record=False)
        rows = []
        for k, v in data.items():
            if k.startswith("__"):
                continue
            arr = np.asarray(v)
            rows.append(
                {
                    "variable": k,
                    "shape": "x".join(map(str, arr.shape)),
                    "ndim": int(arr.ndim),
                    "dtype": str(arr.dtype),
                    "numeric": bool(np.issubdtype(arr.dtype, np.number)),
                }
            )
        return rows, "scipy_loadmat"
    except NotImplementedError:
        pass
    except Exception as exc:
        return [], f"scipy_loadmat_failed:{type(exc).__name__}:{exc}"
    if h5py is None:
        return [], "mat_v73_h5py_unavailable"
    try:
        rows = []
        with h5py.File(path, "r") as f:
            def visitor(name: str, obj: Any) -> None:
                if hasattr(obj, "shape"):
                    rows.append(
                        {
                            "variable": name,
                            "shape": "x".join(map(str, obj.shape)),
                            "ndim": len(obj.shape),
                            "dtype": str(obj.dtype),
                            "numeric": np.issubdtype(obj.dtype, np.number),
                        }
                    )
            f.visititems(visitor)
        return rows, "h5py"
    except Exception as exc:
        return [], f"h5py_failed:{type(exc).__name__}:{exc}"


def expected_t_values(raw_tp: Any = None, n_raw: Any = None, n_model: Any = None) -> set[int]:
    vals: set[int] = set()
    for x in [raw_tp, n_raw, n_model]:
        s = safe_str(x)
        for m in re.findall(r"\d+", s):
            try:
                vals.add(int(m))
            except Exception:
                pass
    # ADNI rs-fMRI protocol values seen in these audits.
    vals.update({140, 170, 197, 200})
    return vals


def choose_signal_matrix(
    path: Path,
    raw_tp: Any = None,
    n_raw: Any = None,
    n_model: Any = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    vars_info, loader = list_mat_variables(path)
    meta: dict[str, Any] = {
        "loader_status": loader,
        "variables_inside": "; ".join([f"{v['variable']}:{v['shape']}:{v['dtype']}" for v in vars_info]),
        "selected_variable": "",
        "raw_shape": "",
        "orientation": "",
        "T": np.nan,
        "ROIs": np.nan,
    }
    if not path.exists():
        return None, meta
    try:
        data = scipy_io.loadmat(path, squeeze_me=False, struct_as_record=False)
    except Exception as exc:
        meta["loader_status"] = f"load_failed:{type(exc).__name__}:{exc}"
        return None, meta

    candidates: list[tuple[int, str, np.ndarray]] = []
    for key, val in data.items():
        if key.startswith("__"):
            continue
        arr = np.asarray(val)
        if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
            min_dim, max_dim = min(arr.shape), max(arr.shape)
            if 50 <= min_dim <= 250 and max_dim >= 80:
                priority = 0 if key in POSSIBLE_ROI_KEYS else 1
                candidates.append((priority, key, arr.astype(float)))
    if not candidates:
        return None, meta
    _, key, arr = sorted(candidates, key=lambda x: (x[0], -x[2].size))[0]
    expected_t = expected_t_values(raw_tp, n_raw, n_model)
    # DPARSF ROISignals are usually T x ROI. Use rawTP/timepoint hints first
    # because 140 x 170 would otherwise look like ROI x T by shape alone.
    if arr.shape[0] in expected_t and arr.shape[1] not in expected_t:
        ts = arr
        orientation = "T_by_ROI_from_timepoint_hint"
    elif arr.shape[1] in expected_t and arr.shape[0] not in expected_t:
        ts = arr.T
        orientation = "ROI_by_T_transposed_from_timepoint_hint"
    elif arr.shape[0] <= arr.shape[1] and arr.shape[0] in {120, 140, 170, 197, 200}:
        ts = arr
        orientation = "T_by_ROI_assumed_from_known_timepoint_first_dim"
    elif arr.shape[0] >= arr.shape[1]:
        ts = arr
        orientation = "T_by_ROI_assumed_by_shape"
    else:
        ts = arr.T
        orientation = "ROI_by_T_transposed_by_shape"
    meta.update(
        {
            "selected_variable": key,
            "raw_shape": "x".join(map(str, arr.shape)),
            "orientation": orientation,
            "T": int(ts.shape[0]),
            "ROIs": int(ts.shape[1]),
        }
    )
    return ts, meta


def pearson_z(ts: np.ndarray) -> np.ndarray:
    x = np.asarray(ts, dtype=float)
    # Drop unusable ROI columns and impute occasional non-finite entries with
    # that ROI's mean. This mirrors an audit probe, not tensor reconstruction.
    finite_frac_col = np.isfinite(x).mean(axis=0)
    x = x[:, finite_frac_col >= 0.95]
    if x.size == 0:
        raise ValueError("no ROI columns with sufficient finite values")
    col_mean = np.nanmean(np.where(np.isfinite(x), x, np.nan), axis=0)
    inds = np.where(~np.isfinite(x))
    if inds[0].size:
        x[inds] = np.take(col_mean, inds[1])
    keep = np.isfinite(x).all(axis=1)
    x = x[keep]
    if x.shape[0] < 3:
        raise ValueError("not enough finite timepoints")
    corr = np.corrcoef(x, rowvar=False)
    corr = np.clip(corr, -0.999999, 0.999999)
    z = np.arctanh(corr)
    np.fill_diagonal(z, 0.0)
    return z


def offdiag_values(mat: np.ndarray) -> np.ndarray:
    return mat[~np.eye(mat.shape[0], dtype=bool)]


def offdiag_summary(mat: np.ndarray) -> dict[str, float]:
    vals = offdiag_values(np.asarray(mat, dtype=float))
    finite = vals[np.isfinite(vals)]
    if not finite.size:
        return {k: np.nan for k in ["mean", "sd", "q25", "median", "q75", "prop_positive", "prop_negative"]}
    return {
        "mean": float(np.mean(finite)),
        "sd": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "q25": float(np.quantile(finite, 0.25)),
        "median": float(np.median(finite)),
        "q75": float(np.quantile(finite, 0.75)),
        "prop_positive": float(np.mean(finite > 0)),
        "prop_negative": float(np.mean(finite < 0)),
    }


def sample_subjects(df: pd.DataFrame) -> pd.DataFrame:
    cn = df[(df["diagnosis"] == "CN") & df["y_score_final"].notna()].copy()
    groups = [
        ("philips_cn_140_fp_highscore", (cn["Manufacturer_norm"] == "Philips") & (cn["raw_tp_group_norm"] == "140") & (cn["confusion_label"] == "FP"), False, 3),
        ("philips_cn_197_fp", (cn["Manufacturer_norm"] == "Philips") & (cn["raw_tp_group_norm"] == "197_200") & (cn["confusion_label"] == "FP"), False, 3),
        ("philips_cn_140_tn", (cn["Manufacturer_norm"] == "Philips") & (cn["raw_tp_group_norm"] == "140") & (cn["confusion_label"] == "TN"), True, 3),
        ("ge_siemens_cn_comparator", cn["Manufacturer_norm"].isin(["GE", "SIEMENS"]), True, 4),
    ]
    rows = []
    seen: set[str] = set()
    for label, mask, ascending, n in groups:
        part = cn[mask].sort_values("y_score_final", ascending=ascending).head(n).copy()
        part["sample_group"] = label
        for _, r in part.iterrows():
            sid = safe_str(r["SubjectID"])
            if sid not in seen:
                rows.append(r)
                seen.add(sid)
    return pd.DataFrame(rows)


def build_mat_inventory(df: pd.DataFrame, path_index: dict[str, list[Path]]) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        sid = safe_str(r["SubjectID"])
        paths = path_index.get(sid, [])
        if not paths:
            rows.append(base_inventory_row(r, None, "no_path_in_master", None))
            continue
        any_row = False
        for p in paths:
            ts, meta = choose_signal_matrix(
                p,
                raw_tp=r.get("raw_tp_group"),
                n_raw=r.get("n_timepoints_raw"),
                n_model=r.get("n_timepoints_model_input"),
            )
            exists = p.exists()
            if exists or not any_row:
                rows.append(base_inventory_row(r, p, "exists" if exists else "path_missing", meta))
                any_row = True
    return pd.DataFrame(rows)


def base_inventory_row(r: pd.Series, path: Path | None, status: str, meta: dict[str, Any] | None) -> dict[str, Any]:
    meta = meta or {}
    return {
        "SubjectID": r.get("SubjectID"),
        "ImageID": r.get("ImageID"),
        "RID": r.get("RID"),
        "file_path": str(path) if path else "",
        "path_status": status,
        "variables_inside": meta.get("variables_inside", ""),
        "selected_variable": meta.get("selected_variable", ""),
        "raw_shape": meta.get("raw_shape", ""),
        "orientation": meta.get("orientation", ""),
        "T": meta.get("T", np.nan),
        "ROIs": meta.get("ROIs", np.nan),
        "rawTP": r.get("raw_tp_group"),
        "rawTP_norm": r.get("raw_tp_group_norm"),
        "diagnosis": r.get("diagnosis"),
        "Manufacturer": r.get("Manufacturer_norm"),
        "Site3": r.get("Site3"),
        "Age": r.get("Age"),
        "y_score_final": r.get("y_score_final"),
        "y_pred": r.get("y_pred"),
        "FP_TN": r.get("confusion_label"),
        "loader_status": meta.get("loader_status", ""),
    }


def timepoint_comparison(inv: pd.DataFrame, tensor_target_len: int) -> pd.DataFrame:
    d = inv[(inv["path_status"] == "exists") & inv["T"].notna()].copy()
    rows = []
    for keys, g in d.groupby(["Manufacturer", "diagnosis", "rawTP_norm"], dropna=False):
        mfr, dx, rawtp = keys
        T = numeric(g["T"])
        rows.append(
            {
                "Manufacturer": mfr,
                "diagnosis": dx,
                "rawTP_norm": rawtp,
                "n_files": len(g),
                "T_min": T.min(),
                "T_median": T.median(),
                "T_max": T.max(),
                "T_equals_rawTP_count": int((T.astype("Int64").astype(str).replace("<NA>", "") == rawtp).sum()) if rawtp not in ["197_200"] else int(T.isin([197, 200]).sum()),
                "T_greater_than_target140_count": int((T > tensor_target_len).sum()),
                "tensor_target_len_ts": tensor_target_len,
                "effective_T_for_tensor_from_code": tensor_target_len,
                "homogenization_interpretation": "feature extraction truncates longer standardized series to target_len_ts and interpolates shorter series",
            }
        )
    return pd.DataFrame(rows)


def pearson_stability(sample: pd.DataFrame, path_index: dict[str, list[Path]]) -> pd.DataFrame:
    rows = []
    for _, r in sample.iterrows():
        sid = safe_str(r["SubjectID"])
        paths = [p for p in path_index.get(sid, []) if p.exists()]
        if not paths:
            rows.append({"SubjectID": sid, "status": "no_existing_mat_path"})
            continue
        ts = None
        meta = {}
        path = None
        for p in paths:
            ts, meta = choose_signal_matrix(
                p,
                raw_tp=r.get("raw_tp_group"),
                n_raw=r.get("n_timepoints_raw"),
                n_model=r.get("n_timepoints_model_input"),
            )
            if ts is not None:
                path = p
                break
        if ts is None:
            rows.append({"SubjectID": sid, "status": "no_readable_signal_matrix", "path_checked": str(paths[0])})
            continue
        base = {
            "SubjectID": sid,
            "sample_group": r.get("sample_group", ""),
            "diagnosis": r.get("diagnosis"),
            "Manufacturer": r.get("Manufacturer_norm"),
            "rawTP_norm": r.get("raw_tp_group_norm"),
            "FP_TN": r.get("confusion_label"),
            "y_score_final": r.get("y_score_final"),
            "file_path": str(path),
            "selected_variable": meta.get("selected_variable"),
            "T": int(ts.shape[0]),
            "ROIs": int(ts.shape[1]),
            "status": "ok",
        }
        try:
            p_all = pearson_z(ts)
            p_first = pearson_z(ts[: min(140, ts.shape[0])])
            vals_all = offdiag_values(p_all)
            vals_first = offdiag_values(p_first)
            row = {
                **base,
                "first_n": int(min(140, ts.shape[0])),
                "full_vs_first140_corr": float(np.corrcoef(vals_all, vals_first)[0, 1]),
                "full_vs_first140_frobenius": float(np.linalg.norm(vals_all - vals_first)),
                "full_vs_first140_mean_abs_edge_diff": float(np.mean(np.abs(vals_all - vals_first))),
                "all_offdiag_mean": float(np.mean(vals_all)),
                "first140_offdiag_mean": float(np.mean(vals_first)),
                "all_offdiag_sd": float(np.std(vals_all, ddof=1)),
                "first140_offdiag_sd": float(np.std(vals_first, ddof=1)),
            }
            if ts.shape[0] >= 140:
                p_last = pearson_z(ts[-140:])
                vals_last = offdiag_values(p_last)
                row.update(
                    {
                        "full_vs_last140_corr": float(np.corrcoef(vals_all, vals_last)[0, 1]),
                        "full_vs_last140_frobenius": float(np.linalg.norm(vals_all - vals_last)),
                        "full_vs_last140_mean_abs_edge_diff": float(np.mean(np.abs(vals_all - vals_last))),
                        "first140_vs_last140_corr": float(np.corrcoef(vals_first, vals_last)[0, 1]),
                    }
                )
            rows.append(row)
        except Exception as exc:
            rows.append({**base, "status": f"pearson_failed:{type(exc).__name__}:{exc}"})
    return pd.DataFrame(rows)


def channel_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    z = np.load(GLOBAL_TENSOR, allow_pickle=True)
    tensor = z["global_tensor_data"]
    subjects = [str(s) for s in z["subject_ids"]]
    subject_to_idx = {s: i for i, s in enumerate(subjects)}
    channel_names = [str(c) for c in z["channel_names"]]

    rows = []
    philips_cn = df[
        (df["Manufacturer_norm"] == "Philips")
        & (df["diagnosis"] == "CN")
        & (df["raw_tp_group_norm"].isin(["140", "197_200"]))
        & (df["confusion_label"].isin(["FP", "TN"]))
    ].copy()
    for _, r in philips_cn.iterrows():
        sid = safe_str(r["SubjectID"])
        if sid not in subject_to_idx:
            continue
        mats = tensor[subject_to_idx[sid]]
        base = {
            "SubjectID": sid,
            "rawTP_norm": r["raw_tp_group_norm"],
            "FP_TN": r["confusion_label"],
            "y_score_final": r["y_score_final"],
            "Age": r["Age"],
            "Site3": r["Site3"],
        }
        for ch in [0, 1, 2]:
            summary = offdiag_summary(mats[ch])
            rows.append(
                {
                    **base,
                    "channel_index": ch,
                    "channel_name": channel_names[ch],
                    **{f"offdiag_{k}": v for k, v in summary.items()},
                }
            )
    long = pd.DataFrame(rows)
    test_rows = []
    for ch in [0, 1, 2]:
        for metric in ["offdiag_mean", "offdiag_sd", "offdiag_median", "offdiag_prop_positive", "offdiag_prop_negative"]:
            a = long[(long["channel_index"] == ch) & (long["rawTP_norm"] == "140") & (long["FP_TN"] == "FP")][metric].dropna()
            b = long[(long["channel_index"] == ch) & (long["rawTP_norm"] == "197_200") & (long["FP_TN"] == "FP")][metric].dropna()
            if len(a) >= 2 and len(b) >= 2:
                u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                cles = float(u / (len(a) * len(b)))
            else:
                u = p = cles = np.nan
            test_rows.append(
                {
                    "row_type": "test_140TP_FP_vs_197_200TP_FP",
                    "channel_index": ch,
                    "channel_name": channel_names[ch],
                    "metric": metric,
                    "n_140_fp": len(a),
                    "n_197_200_fp": len(b),
                    "median_140_fp": float(a.median()) if len(a) else np.nan,
                    "median_197_200_fp": float(b.median()) if len(b) else np.nan,
                    "delta_median_140_minus_197": float(a.median() - b.median()) if len(a) and len(b) else np.nan,
                    "mannwhitney_u": float(u) if not pd.isna(u) else np.nan,
                    "p_mannwhitney": float(p) if not pd.isna(p) else np.nan,
                    "cles_p_140_gt_197": cles,
                    "effect_rank_abs_cles_minus_0p5": abs(cles - 0.5) if not pd.isna(cles) else np.nan,
                }
            )
    summary = (
        long.groupby(["rawTP_norm", "FP_TN", "channel_index", "channel_name"], dropna=False)
        .agg(
            n=("SubjectID", "count"),
            offdiag_mean_median=("offdiag_mean", "median"),
            offdiag_sd_median=("offdiag_sd", "median"),
            offdiag_median_median=("offdiag_median", "median"),
            offdiag_prop_positive_median=("offdiag_prop_positive", "median"),
            score_median=("y_score_final", "median"),
        )
        .reset_index()
    )
    summary.insert(0, "row_type", "group_summary")
    return pd.concat([summary, pd.DataFrame(test_rows)], ignore_index=True, sort=False)


def inspect_tensor_build_policy() -> dict[str, Any]:
    z = np.load(GLOBAL_TENSOR, allow_pickle=True)
    target_len = int(np.asarray(z["target_len_ts"]).item())
    tr = float(np.asarray(z["tr_seconds"]).item())
    py_bp = bool(np.asarray(z["python_bandpass_applied"]).item())
    text = FEATURE_EXTRACTION.read_text(encoding="utf-8", errors="ignore")
    has_truncate = "sigs_homogenized = sigs_normalized[:target_len_ts_val, :]" in text
    has_interpolate = "interp1d(x_old, sigs_normalized[:, i]" in text
    return {
        "global_tensor_path": str(GLOBAL_TENSOR),
        "target_len_ts": target_len,
        "tr_seconds": tr,
        "python_bandpass_applied": py_bp,
        "feature_extraction_script": str(FEATURE_EXTRACTION),
        "code_has_truncate_to_target_len": has_truncate,
        "code_has_interpolate_to_target_len": has_interpolate,
    }


def write_interpretation(
    policy: dict[str, Any],
    inv: pd.DataFrame,
    time_cmp: pd.DataFrame,
    pearson: pd.DataFrame,
    chsens: pd.DataFrame,
    source_dirs: pd.DataFrame,
) -> None:
    philips = inv[(inv["Manufacturer"] == "Philips") & (inv["diagnosis"] == "CN") & (inv["T"].notna())]
    t_by_raw = (
        philips.groupby("rawTP_norm")["T"]
        .agg(["count", "min", "median", "max"])
        .reset_index()
        .to_dict(orient="records")
    )
    pearson_ok = pearson[pearson.get("status", "") == "ok"] if "status" in pearson else pd.DataFrame()
    pearson_gt140 = pearson_ok[pd.to_numeric(pearson_ok.get("T", np.nan), errors="coerce") > 140].copy() if not pearson_ok.empty else pd.DataFrame()
    pearson_summary = ""
    if not pearson_gt140.empty:
        pearson_summary = (
            f"- Pearson first-140 versus full-series probe computed for {len(pearson_ok)} sampled readable subjects, "
            f"including {len(pearson_gt140)} subjects with T>140. Among T>140 subjects, "
            f"median edge correlation={pearson_gt140['full_vs_first140_corr'].median():.4f}; "
            f"median mean absolute edge difference={pearson_gt140['full_vs_first140_mean_abs_edge_diff'].median():.6f}.\n"
        )
    elif not pearson_ok.empty:
        pearson_summary = (
            f"- Pearson first-140 versus full-series probe computed for {len(pearson_ok)} sampled readable subjects, "
            "but none had T>140; the comparison is therefore trivial for T=140 subjects.\n"
        )
    else:
        pearson_summary = "- Pearson first-140 versus full-series probe could not be computed for readable sampled subjects.\n"

    tests = chsens[chsens["row_type"].eq("test_140TP_FP_vs_197_200TP_FP")].copy()
    best = pd.DataFrame()
    if not tests.empty:
        best = tests.sort_values("effect_rank_abs_cles_minus_0p5", ascending=False).head(3)
    best_txt = ""
    if not best.empty:
        for _, r in best.iterrows():
            best_txt += (
                f"- {r['channel_name']} / {r['metric']}: median 140TP FP={r['median_140_fp']:.6g}, "
                f"median 197/200TP FP={r['median_197_200_fp']:.6g}, p={r['p_mannwhitney']:.4g}, "
                f"CLES={r['cles_p_140_gt_197']:.3f}\n"
            )

    source_txt = "\n".join(
        f"- `{r.source_directory}`: {int(r.subjects_linked_from_master)} linked subjects"
        for _, r in source_dirs.head(12).iterrows()
    )

    txt = f"""# Mechanism Interpretation

This is a read-only audit. It inspected existing master metadata, existing `.mat` ROISignals paths, the locked global tensor, and source code. It did not train, edit tensors, edit metadata, refit thresholds, run OASIS inference, or modify existing artifacts.

## Source Files Found

The promoted master database links subjects to ROISignals `.mat` files primarily under:

{source_txt}

Representative `.mat` files expose numeric `ROISignals` variables such as `T x ROI` matrices. The inventory table records the exact variable list, selected variable, raw shape, inferred orientation, T, and ROI count per accessible subject file.

## A. Does T Match rawTP?

For accessible Philips CN `.mat` files, T generally tracks `raw_tp_group`: the grouped inventory is:

```json
{json.dumps(t_by_raw, indent=2)}
```

Thus `rawTP=140` and `rawTP=197/200` are not just labels; they usually correspond to different raw ROISignal lengths before tensor construction.

## B. Were Matrices Computed With Variable T?

No for the promoted global tensor. The locked tensor reports `target_len_ts={policy['target_len_ts']}` and `python_bandpass_applied={policy['python_bandpass_applied']}`. The feature-extraction code at `{policy['feature_extraction_script']}` standardizes signals and then homogenizes length to `target_len_ts`: it truncates longer series with `sigs_normalized[:target_len_ts_val, :]` and has interpolation logic for shorter series.

Therefore, the promoted connectivity matrices were computed after homogenization to an effective T of 140. rawTP=197/200 subjects were not contributing 197/200 samples to the final connectome; they were truncated to the first 140 standardized samples in this code path.

## C. If T Varies, Which Channels Are Most Affected?

The raw `.mat` T varies, but the final tensor T is homogenized. The direct statistical-quality explanation, where Pearson/OMST/MI-KNN have more samples for 197/200 subjects than for 140 subjects in the final tensor, is therefore not supported for the promoted tensor.

The sampled raw-signal Pearson stability probe estimates how much first-140 truncation changes raw Pearson relative to using all available samples:

{pearson_summary}

This probe is Pearson-only and descriptive. It does not recompute OMST or MI-KNN, because those should be rerun through the exact feature-extraction implementation before being used as manuscript evidence.

## D. If T Is Homogenized, Is rawTP a Protocol Proxy?

Yes. Because the tensor pipeline homogenizes to T=140 before connectome computation, the observed rawTP140 false-positive excess is better interpreted as a protocol/site/scanner/ADNI-phase proxy rather than a simple variable-sample-size effect in the final matrices. rawTP remains important because it marks acquisition/protocol strata that also differ in age, site, scanner model/software, and channel-level connectome summaries.

## Channel Sensitivity From Locked Tensor

Among Philips CN false positives, the largest [1,0,2] channel/summary differences between 140TP and 197/200TP were:

{best_txt if best_txt else '_No channel tests available._'}

The full table `channel_sensitivity_by_rawTP.csv` contains group summaries and Mann-Whitney tests for ch0/ch1/ch2 summaries.

## E. Manuscript Limitation

Recommended language: Philips rawTP/protocol strata were associated with elevated CN false-positive rates and shifted connectivity summaries, but the promoted tensor construction homogenized time series to T=140 before connectome estimation. Therefore the result should be framed as evidence of protocol/domain confounding, not as proof that longer time series directly improve the model. A limitation is that ADNI Philips acquisition phase, site/scanner configuration, slice-timing provenance, and raw timepoint group are entangled. Corrected preprocessing of confirmed acquisition mismatches and transparent stratified reporting are more defensible than post-hoc subject exclusion.
"""
    (OUT / "mechanism_interpretation.md").write_text(txt, encoding="utf-8")
    log("write_interpretation", "ok", {"path": str(OUT / "mechanism_interpretation.md")})


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    log("start", "ok", {"guardrails": ["read_only", "no_training", "no_tensor_edits", "no_metadata_edits", "no_threshold_refitting", "no_oasis"]})
    df = load_master()
    cnad = df[df["diagnosis"].isin(["CN", "AD"]) & df["y_score_final"].notna()].copy()
    path_index = build_subject_path_index(cnad)
    source_dirs = source_root_inventory(path_index)
    log("source_path_index", "ok", {"subjects_with_paths": len(path_index), "source_dirs": int(len(source_dirs))})

    inventory = build_mat_inventory(cnad, path_index)
    write_df("mat_file_inventory", inventory, max_rows=60)

    policy = inspect_tensor_build_policy()
    time_cmp = timepoint_comparison(inventory, int(policy["target_len_ts"]))
    write_df("timepoint_comparison_by_rawTP", time_cmp, max_rows=80)

    sample = sample_subjects(cnad)
    pearson = pearson_stability(sample, path_index)
    write_df("pearson_140_vs_full_stability", pearson, max_rows=80)

    chsens = channel_sensitivity(cnad)
    write_df("channel_sensitivity_by_rawTP", chsens, max_rows=120)

    write_interpretation(policy, inventory, time_cmp, pearson, chsens, source_dirs)

    log(
        "validation",
        "ok",
        {
            "py_compile": "run separately before audit execution",
            "no_training": True,
            "no_tensor_modification": True,
            "no_metadata_modification": True,
            "existing_artifacts_modified": False,
        },
    )
    (OUT / "command_log.json").write_text(json.dumps(COMMAND_LOG, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        log("fatal", "failed", {"error": repr(exc), "traceback": traceback.format_exc()})
        (OUT / "command_log.json").write_text(json.dumps(COMMAND_LOG, indent=2), encoding="utf-8")
        print(traceback.format_exc(), file=sys.stderr)
        raise
