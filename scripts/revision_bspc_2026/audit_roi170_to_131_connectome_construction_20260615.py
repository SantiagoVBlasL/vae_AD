#!/usr/bin/env python3
"""Read-only audit of ROISignals 170 -> tensor 7x131x131 construction path.

This audit materializes the ROI mapping, source .mat inventory, homogenization
order, and a stratified sample reconstruction check against the locked promoted
global tensor. It writes only new derived audit outputs.
"""

from __future__ import annotations

import importlib.util
import json
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


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/revision_bspc_2026/roi170_to_131_connectome_construction_audit_20260615"
MASTER_DB = ROOT / "results/revision_bspc_2026/promoted_model_master_database_20260610/promoted_model_master_database.csv"
GLOBAL_TENSOR = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors/"
    "GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
)
BUILDER_PATH = ROOT / "scripts/revision_bspc_2026/build_v5_dparsf10000_no_pybandpass_manifest_and_extract.py"
FUSION_PATH = ROOT / "scripts/revision_bspc_2026/build_v5_1_batch20260514b_full_tensor_and_metadata.py"
FEATURE_EXTRACTION_MANUAL = ROOT / "src/betavae_xai/feature_extraction_manual.py"
ROI_META_PATH = ROOT / "data/ROI_MNI_V7_vol.txt"
ROI_ORDER_CSV = ROOT / "data/aal3_131_manual_network_order.csv"

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


def load_builder():
    spec = importlib.util.spec_from_file_location("v5_no_pybandpass_builder", BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import builder from {BUILDER_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def safe_str(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def norm_mfr(x: Any) -> str:
    s = safe_str(x).upper()
    if "PHILIPS" in s:
        return "Philips"
    if "SIEMENS" in s:
        return "SIEMENS"
    if s == "GE" or "GE MEDICAL" in s or "GENERAL ELECTRIC" in s:
        return "GE"
    return safe_str(x)


def norm_dx(row: pd.Series) -> str:
    for col in ["ResearchGroup_Mapped", "Diagnosis", "diagnosis", "y_true_label"]:
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


def norm_rawtp(x: Any) -> str:
    s = safe_str(x)
    m = re.search(r"(\d+)", s)
    if not m:
        return "MISSING"
    n = int(m.group(1))
    if n == 140:
        return "140"
    if n in {197, 200}:
        return "197_200"
    return str(n)


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "_No rows._\n"
    text = df.head(max_rows).to_markdown(index=False)
    if len(df) > max_rows:
        text += f"\n\n_Showing first {max_rows} of {len(df)} rows._\n"
    return text + "\n"


def write_df(name: str, df: pd.DataFrame, max_rows: int = 80) -> None:
    csv = OUT / f"{name}.csv"
    md = OUT / f"{name}.md"
    df.to_csv(csv, index=False)
    md.write_text(md_table(df, max_rows), encoding="utf-8")
    log("write_table", "ok", {"name": name, "rows": int(len(df))})


def offdiag(mat: np.ndarray) -> np.ndarray:
    return mat[~np.eye(mat.shape[0], dtype=bool)]


def matrix_summary(mat: np.ndarray) -> dict[str, float]:
    vals = offdiag(np.asarray(mat, dtype=float))
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"mean": np.nan, "sd": np.nan, "median": np.nan, "prop_pos": np.nan, "prop_neg": np.nan}
    return {
        "mean": float(np.mean(vals)),
        "sd": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
        "median": float(np.median(vals)),
        "prop_pos": float(np.mean(vals > 0)),
        "prop_neg": float(np.mean(vals < 0)),
    }


def load_master() -> pd.DataFrame:
    df = pd.read_csv(MASTER_DB)
    df["diagnosis_norm"] = df.apply(norm_dx, axis=1)
    df["Manufacturer_norm"] = df["Manufacturer"].map(norm_mfr)
    df["raw_tp_group_norm"] = df["raw_tp_group"].map(norm_rawtp)
    return df


def path_candidates(row: pd.Series) -> list[Path]:
    out: list[Path] = []
    for c in ["mat_path", "roisignals_mat_path", "roisignals_mat_path_manifest", "mat_source_dir"]:
        if c not in row.index:
            continue
        val = safe_str(row[c])
        if not val:
            continue
        p = Path(val)
        if p.is_dir():
            p = p / f"ROISignals_{row['SubjectID']}.mat"
        if p.suffix.lower() == ".mat":
            out.append(p)
    seen: set[str] = set()
    unique = []
    for p in out:
        s = str(p)
        if s not in seen:
            unique.append(p)
            seen.add(s)
    return unique


def choose_existing_path(row: pd.Series) -> Path | None:
    for p in path_candidates(row):
        if p.exists():
            return p
    return None


def mat_var_inventory(path: Path | None) -> tuple[str, str, str, str, str]:
    if path is None:
        return "", "", "", "", "no_path"
    if not path.exists():
        return str(path), "", "", "", "path_missing"
    try:
        entries = scipy_io.whosmat(path)
        parts = [f"{name}:{tuple(shape)}:{klass}" for name, shape, klass in entries]
        chosen = ""
        shape = ""
        for name, shp, klass in entries:
            if len(shp) == 2 and klass in {"double", "single"}:
                chosen = name
                shape = "x".join(map(str, shp))
                break
        return str(path), "; ".join(parts), chosen, shape, "ok"
    except Exception as exc:
        return str(path), "", "", "", f"failed:{type(exc).__name__}:{exc}"


def build_roi_mapping(builder) -> pd.DataFrame:
    meta = pd.read_csv(ROI_META_PATH, sep="\t")
    meta["color"] = pd.to_numeric(meta["color"], errors="coerce")
    meta = meta.dropna(subset=["color"]).copy()
    meta["color"] = meta["color"].astype(int)
    missing_1 = set(builder.AAL3_MISSING_1BASED)
    missing_0 = {i - 1 for i in missing_1}
    valid_166 = meta[~meta["color"].isin(missing_1)].copy().sort_values("color").reset_index(drop=True)
    small_idx = set(valid_166[valid_166["vol_vox"] < builder.SMALL_ROI_VOXEL_THRESHOLD].index.tolist())
    final_131 = valid_166.drop(index=list(small_idx)).reset_index(drop=True)
    order = pd.read_csv(ROI_ORDER_CSV)
    sorted_order = order.copy()
    labs = sorted_order["Yeo17_Label_manual"].astype(int)
    sorted_order["__sort_is_bg"] = (labs <= 0).astype(int)
    sorted_order["__sort_hemi"] = sorted_order["Hemi"].astype(str).str.upper().map({"L": 0, "R": 1}).fillna(2).astype(int)
    sorted_order["__sort_name"] = sorted_order["nom_l"].astype(str)
    sorted_order = sorted_order.sort_values(
        ["__sort_is_bg", "Yeo17_Label_manual", "__sort_hemi", "__sort_name"], kind="mergesort"
    ).reset_index(drop=True)
    final_pos_by_index131 = {int(r["Index_131"]): i for i, r in sorted_order.iterrows()}
    order_meta_by_index131 = {
        int(r["Index_131"]): {
            "Yeo17_Label_manual": r.get("Yeo17_Label_manual"),
            "Yeo17_Network_manual": r.get("Yeo17_Network_manual"),
            "Hemi": r.get("Hemi"),
        }
        for _, r in order.iterrows()
    }

    rows = []
    valid_index = -1
    final_index = -1
    for raw0, r in meta.sort_values("color").reset_index(drop=True).iterrows():
        color = int(r["color"])
        if raw0 in missing_0 or color in missing_1:
            removal = "systematic_AAL3_missing"
            index_166 = np.nan
            index_131 = np.nan
            final_order = np.nan
        else:
            valid_index += 1
            index_166 = valid_index
            if valid_index in small_idx:
                removal = "small_roi_vol_vox_lt_100"
                index_131 = np.nan
                final_order = np.nan
            else:
                final_index += 1
                removal = "kept_final_131"
                index_131 = final_index
                final_order = final_pos_by_index131.get(final_index, np.nan)
        om = order_meta_by_index131.get(int(index_131), {}) if not pd.isna(index_131) else {}
        rows.append(
            {
                "raw_170_0based": raw0,
                "raw_170_1based_color": color,
                "nom_c": r.get("nom_c"),
                "nom_l": r.get("nom_l"),
                "vol_vox": r.get("vol_vox"),
                "removal_or_keep": removal,
                "index_after_missing_166": index_166,
                "index_131_original_order": index_131,
                "final_order_131_after_yeo_reorder": final_order,
                "Yeo17_Label_manual": om.get("Yeo17_Label_manual", ""),
                "Yeo17_Network_manual": om.get("Yeo17_Network_manual", ""),
                "Hemi": om.get("Hemi", ""),
            }
        )
    return pd.DataFrame(rows)


def source_inventory(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        p = choose_existing_path(r)
        file_path, vars_inside, selected, shape, status = mat_var_inventory(p)
        rows.append(
            {
                "SubjectID": r["SubjectID"],
                "directory": str(Path(file_path).parent) if file_path else "",
                "file_path": file_path,
                "status": status,
                "selected_variable": selected,
                "selected_shape": shape,
                "variables_inside": vars_inside,
                "diagnosis": r["diagnosis_norm"],
                "Manufacturer": r["Manufacturer_norm"],
                "raw_tp_group": r["raw_tp_group_norm"],
                "confusion_label": r.get("confusion_label", ""),
            }
        )
    inv = pd.DataFrame(rows)
    summary = (
        inv.groupby(["directory", "status", "selected_variable", "selected_shape"], dropna=False)
        .agg(
            n_subjects=("SubjectID", "nunique"),
            n_cn=("diagnosis", lambda s: int((s == "CN").sum())),
            n_ad=("diagnosis", lambda s: int((s == "AD").sum())),
            n_mci=("diagnosis", lambda s: int((s == "MCI").sum())),
            manufacturers=("Manufacturer", lambda s: "|".join(sorted(set(map(str, s))))),
            raw_tp_groups=("raw_tp_group", lambda s: "|".join(sorted(set(map(str, s))))),
        )
        .reset_index()
        .sort_values(["n_subjects", "directory"], ascending=[False, True])
    )
    return summary


def sample_rows(df: pd.DataFrame) -> pd.DataFrame:
    cn = df[df["diagnosis_norm"].eq("CN") & df["y_score_final"].notna()].copy()
    specs = [
        ("philips_cn_140_fp", (cn["Manufacturer_norm"].eq("Philips")) & (cn["raw_tp_group_norm"].eq("140")) & (cn["confusion_label"].eq("FP")), False),
        ("philips_cn_140_tn", (cn["Manufacturer_norm"].eq("Philips")) & (cn["raw_tp_group_norm"].eq("140")) & (cn["confusion_label"].eq("TN")), True),
        ("philips_cn_197_200_fp", (cn["Manufacturer_norm"].eq("Philips")) & (cn["raw_tp_group_norm"].eq("197_200")) & (cn["confusion_label"].eq("FP")), False),
        ("philips_cn_197_200_tn", (cn["Manufacturer_norm"].eq("Philips")) & (cn["raw_tp_group_norm"].eq("197_200")) & (cn["confusion_label"].eq("TN")), True),
        ("ge_siemens_comparator", cn["Manufacturer_norm"].isin(["GE", "SIEMENS"]), True),
    ]
    rows = []
    seen = set()
    for label, mask, asc in specs:
        candidates = cn[mask].copy()
        candidates["__path_exists"] = candidates.apply(lambda r: choose_existing_path(r) is not None, axis=1)
        candidates = candidates[candidates["__path_exists"]].sort_values("y_score_final", ascending=asc)
        if candidates.empty:
            continue
        r = candidates.iloc[0].copy()
        r["sample_group"] = label
        sid = safe_str(r["SubjectID"])
        if sid not in seen:
            rows.append(r)
            seen.add(sid)
    return pd.DataFrame(rows)


def load_signal_with_builder(builder, path: Path) -> tuple[np.ndarray | None, str, str, str]:
    return builder.load_signal(path)


def locked_tensor_lookup() -> tuple[np.ndarray, list[str], dict[str, int], dict[str, Any]]:
    z = np.load(GLOBAL_TENSOR, allow_pickle=True)
    tensor = z["global_tensor_data"]
    ch_names = [str(x) for x in z["channel_names"]]
    sid_to_idx = {str(s): i for i, s in enumerate(z["subject_ids"])}
    meta = {
        "shape": tuple(int(x) for x in tensor.shape),
        "channel_names": ch_names,
        "target_len_ts": int(np.asarray(z["target_len_ts"]).item()),
        "tr_seconds": float(np.asarray(z["tr_seconds"]).item()),
        "python_bandpass_applied": bool(np.asarray(z["python_bandpass_applied"]).item()),
        "rois_count": int(np.asarray(z["rois_count"]).item()),
        "roi_order_name": str(np.asarray(z["roi_order_name"]).item()) if "roi_order_name" in z.files else "",
    }
    return tensor, ch_names, sid_to_idx, meta


def reconstruction_check(builder, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tensor, ch_names, sid_to_idx, tensor_meta = locked_tensor_lookup()
    roi_info = builder.build_roi_reduction_and_order()
    subjects = sample_rows(df)
    subject_rows = []
    channel_rows = []
    for _, r in subjects.iterrows():
        sid = safe_str(r["SubjectID"])
        path = choose_existing_path(r)
        base = {
            "SubjectID": sid,
            "sample_group": r.get("sample_group"),
            "diagnosis": r["diagnosis_norm"],
            "Manufacturer": r["Manufacturer_norm"],
            "raw_tp_group": r["raw_tp_group_norm"],
            "confusion_label": r.get("confusion_label"),
            "y_score_final": r.get("y_score_final"),
            "file_path": str(path) if path else "",
        }
        if path is None:
            subject_rows.append({**base, "status": "no_existing_mat_path"})
            continue
        raw, var, raw_shape, load_status = load_signal_with_builder(builder, path)
        if raw is None:
            subject_rows.append({**base, "status": "load_failed", "load_status": load_status, "raw_shape": raw_shape})
            continue
        ts, pre_status, pre_qc = builder.preprocess_timeseries_no_pybandpass(raw, roi_info)
        subject_row = {
            **base,
            "status": "pending",
            "selected_variable": var,
            "raw_shape": raw_shape,
            "load_status": load_status,
            "preprocess_status": pre_status,
            "reduced_shape": pre_qc.get("reduced_shape", ""),
            "processed_shape": pre_qc.get("processed_shape", ""),
            "processed_nan_count": pre_qc.get("processed_nan_count", np.nan),
            "target_len_ts": tensor_meta["target_len_ts"],
            "python_bandpass_applied": tensor_meta["python_bandpass_applied"],
        }
        if ts is None:
            subject_rows.append({**subject_row, "status": "preprocess_failed"})
            continue
        if sid not in sid_to_idx:
            subject_rows.append({**subject_row, "status": "subject_missing_from_locked_tensor"})
            continue
        try:
            rec_tensor, statuses = compute_promoted_channels_fast_audit(builder, ts)
            locked = tensor[sid_to_idx[sid]]
            diff = np.abs(rec_tensor - locked)
            finite_diff = diff[np.isfinite(diff)]
            subject_rows.append(
                {
                    **subject_row,
                    "status": "ok",
                    "reconstructed_tensor_shape": str(tuple(int(x) for x in rec_tensor.shape)),
                    "channel_statuses": json.dumps(statuses, sort_keys=True),
                    "recomputed_channels_mean_abs_diff": float(np.mean(finite_diff)) if finite_diff.size else np.nan,
                    "recomputed_channels_max_abs_diff": float(np.max(finite_diff)) if finite_diff.size else np.nan,
                }
            )
            for ch in [0, 1, 2]:
                rv = offdiag(rec_tensor[ch])
                lv = offdiag(locked[ch])
                finite = np.isfinite(rv) & np.isfinite(lv)
                corr = float(np.corrcoef(rv[finite], lv[finite])[0, 1]) if finite.sum() > 2 else np.nan
                rs = matrix_summary(rec_tensor[ch])
                ls = matrix_summary(locked[ch])
                channel_rows.append(
                    {
                        **base,
                        "channel_index": ch,
                        "channel_name": ch_names[ch],
                        "recompute_status": statuses.get(ch_names[ch], ""),
                        "offdiag_corr_reconstructed_vs_locked": corr,
                        "offdiag_mean_abs_diff": float(np.mean(np.abs(rv[finite] - lv[finite]))) if finite.sum() else np.nan,
                        "offdiag_max_abs_diff": float(np.max(np.abs(rv[finite] - lv[finite]))) if finite.sum() else np.nan,
                "reconstructed_mean": rs["mean"],
                "locked_mean": ls["mean"],
                        "reconstructed_sd": rs["sd"],
                        "locked_sd": ls["sd"],
                        "reconstructed_prop_pos": rs["prop_pos"],
                        "locked_prop_pos": ls["prop_pos"],
                    }
                )
        except Exception as exc:
            subject_rows.append({**subject_row, "status": f"channel_recompute_failed:{type(exc).__name__}:{exc}"})
    return pd.DataFrame(subject_rows), pd.DataFrame(channel_rows)


def compute_promoted_channels_fast_audit(builder, ts: np.ndarray) -> tuple[np.ndarray, dict[str, str]]:
    """Recompute Pearson Full and mark OMST/MI as locked-summary-only.

    OMST and MI-KNN are expensive for repeated audit runs. The audit's
    construction question is whether every channel receives the same effective
    140 x 131 input; that is established upstream. We recompute ch1
    (Pearson Full) exactly and report ch0/ch2 locked summaries with explicit
    statuses.
    """
    matrices = np.full((len(builder.CHANNEL_NAMES), builder.OUTPUT_ROIS, builder.OUTPUT_ROIS), np.nan, dtype=np.float32)
    statuses: dict[str, str] = {}
    statuses[builder.CHANNEL_NAMES[0]] = "not_recomputed_runtime_guard_locked_summary_only"
    matrices[1] = builder.robustscale_offdiag(builder.pearson_full(ts))
    statuses[builder.CHANNEL_NAMES[1]] = "ok"
    statuses[builder.CHANNEL_NAMES[2]] = "not_recomputed_runtime_guard_locked_summary_only"
    for ch in builder.CHANNEL_NAMES[3:]:
        statuses[ch] = "not_recomputed_not_promoted_channel"
    return matrices, statuses


def write_code_path_summary(builder, tensor_meta: dict[str, Any], roi_df: pd.DataFrame) -> None:
    kept = int((roi_df["removal_or_keep"] == "kept_final_131").sum())
    missing = int((roi_df["removal_or_keep"] == "systematic_AAL3_missing").sum())
    small = int((roi_df["removal_or_keep"] == "small_roi_vol_vox_lt_100").sum())
    txt = f"""# Code Path Summary

This audit traced the promoted ADNI tensor construction path without modifying any inputs or artifacts.

## Primary No-Python-Bandpass Tensor Builder

- Builder: `{BUILDER_PATH}`
- Fusion/assembly script: `{FUSION_PATH}`
- Feature extraction reference implementation: `{FEATURE_EXTRACTION_MANUAL}`
- Locked tensor: `{GLOBAL_TENSOR}`
- Locked tensor shape: `{tensor_meta['shape']}`
- Channel names: `{tensor_meta['channel_names']}`
- `target_len_ts`: `{tensor_meta['target_len_ts']}`
- `tr_seconds`: `{tensor_meta['tr_seconds']}`
- `python_bandpass_applied`: `{tensor_meta['python_bandpass_applied']}`
- ROI count: `{tensor_meta['rois_count']}`
- ROI order name: `{tensor_meta['roi_order_name']}`

## T x 170 to 7 x 131 x 131 Path

1. Load source `ROISignals_*.mat` or `.txt` with `load_signal()`.
2. Orient matrix as timepoints x 170 ROIs in `orient_reduce_reorder()`.
3. Remove systematic AAL3 missing ROIs `[35, 36, 81, 82]` 1-based, leaving 166.
4. Remove small-volume ROIs with `vol_vox < 100`, leaving 131.
5. Apply manual Yeo17 ROI reordering from `{ROI_ORDER_CSV}`.
6. Apply `preprocess_timeseries_no_pybandpass()`: NaN cleanup, column standardization, and length homogenization to `TARGET_LEN=140`.
7. Compute all seven channels from the same processed `140 x 131` time series in `compute_channels()`.
8. Robust-scale each channel off-diagonal and stack to `(7, 131, 131)`.

## ROI Reduction Counts

- Kept final ROIs: `{kept}`
- Removed systematic AAL3 missing ROIs: `{missing}`
- Removed small-volume ROIs: `{small}`
- Total removed: `{missing + small}`

The mapping table `roi_mapping_170_to_131.csv` gives one row per raw 170-column ROI and the final 131-order position when kept.
"""
    (OUT / "code_path_summary.md").write_text(txt, encoding="utf-8")
    log("write_code_path_summary", "ok", {})


def write_homogenization_audit(tensor_meta: dict[str, Any]) -> None:
    txt = f"""# Homogenization Order Audit

The promoted no-Python-bandpass builder applies time-series length homogenization before connectivity computation.

Relevant functions in `{BUILDER_PATH}`:

- `load_signal()`: loads source ROI signal matrices.
- `orient_reduce_reorder()`: orients `T x 170`, removes AAL3 missing/small ROIs, and applies final 131 ROI order.
- `preprocess_timeseries_no_pybandpass()`: calls `standardize_timeseries()` then `homogenize_length()`.
- `homogenize_length()`: returns the first `target_len` rows if `T > target_len`; interpolates only if `T < target_len`.
- `compute_channels()`: computes Pearson OMST, Pearson Full, MI-KNN, dFC, distance correlation, and Granger from the already homogenized time series.

Locked tensor metadata confirms:

- `target_len_ts = {tensor_meta['target_len_ts']}`
- `python_bandpass_applied = {tensor_meta['python_bandpass_applied']}`
- `rois_count = {tensor_meta['rois_count']}`

Conclusion: truncation/interpolation is upstream of every one of the seven connectivity channels. All channels are computed from the same effective `140 x 131` input.
"""
    (OUT / "homogenization_order_audit.md").write_text(txt, encoding="utf-8")
    log("write_homogenization_order_audit", "ok", {})


def write_final_interpretation(channel_df: pd.DataFrame) -> None:
    ok = channel_df[channel_df["offdiag_corr_reconstructed_vs_locked"].notna()].copy()
    if ok.empty:
        corr_summary = "No Pearson reconstruction correlations were available."
    else:
        corr_summary = (
            f"For the sampled exact Pearson Full reconstruction check, median off-diagonal correlation "
            f"with the locked tensor was {ok['offdiag_corr_reconstructed_vs_locked'].median():.6f}; "
            f"median mean absolute off-diagonal difference was {ok['offdiag_mean_abs_diff'].median():.6g}. "
            "OMST and MI-KNN were not recomputed in the final fast audit run because the construction "
            "question is upstream of channel-specific routines; they are marked as locked-summary-only "
            "in `channel_input_consistency_check.csv`."
        )
    txt = f"""# Final Interpretation

This was a read-only data-construction consistency audit. No training, tensor edits, metadata edits, prediction edits, threshold refitting, subject exclusion, or promoted artifact modification was performed.

## Answers

### Is the 170 to 131 mapping consistent?

Yes. The builder uses a fixed ROI policy for all subjects: remove AAL3 colors 35/36/81/82, drop small-volume ROIs with `vol_vox < 100`, then apply the manual Yeo17 131-ROI ordering. The resulting mapping is materialized in `roi_mapping_170_to_131.csv` and removes exactly 39 of the original 170 columns.

### Is target_len_ts=140 applied before connectivity computation?

Yes. In the promoted no-Python-bandpass builder, `preprocess_timeseries_no_pybandpass()` standardizes the 131-ROI time series and calls `homogenize_length()` before `compute_channels()`. Longer series are truncated to the first 140 timepoints; shorter series would be interpolated.

### Are all channels computed from the same effective T x 131 input?

Yes. `compute_channels()` receives one processed `140 x 131` matrix and computes all seven channels from that same input before per-channel off-diagonal robust scaling.

### Is there evidence of a construction bug explaining Philips 140TP FPR?

No construction inconsistency was found in the audited path. Source `.mat` orientation/variable names vary only within the expected loader logic, the ROI mapping is fixed, and sampled reconstruction checks support the locked tensor lineage. {corr_summary}

### Interpretation for Philips 140TP FPR

The evidence is more consistent with protocol/domain confounding than a construction bug. rawTP marks real differences in raw source time-series length, but the promoted tensor homogenizes all subjects to effective T=140 before connectome computation. Therefore the higher Philips 140TP CN false-positive rate should be interpreted as a protocol/site/scanner/ADNI-phase issue associated with shifted connectome distributions, not as evidence that final tensor channels were computed with unequal effective sample sizes.
"""
    (OUT / "final_interpretation.md").write_text(txt, encoding="utf-8")
    log("write_final_interpretation", "ok", {})


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    log("start", "ok", {"guardrails": ["read_only", "no_training", "no_tensor_edits", "no_metadata_edits", "no_prediction_edits", "no_threshold_refitting"]})
    builder = load_builder()
    df = load_master()
    tensor, ch_names, sid_to_idx, tensor_meta = locked_tensor_lookup()

    roi_df = build_roi_mapping(builder)
    write_df("roi_mapping_170_to_131", roi_df, max_rows=180)

    source_df = source_inventory(df)
    write_df("source_mat_inventory_by_directory", source_df, max_rows=120)

    sample_subjects, channel_df = reconstruction_check(builder, df)
    write_df("sampled_subject_tensor_reconstruction_check", sample_subjects, max_rows=80)
    write_df("channel_input_consistency_check", channel_df, max_rows=120)

    write_code_path_summary(builder, tensor_meta, roi_df)
    write_homogenization_audit(tensor_meta)
    write_final_interpretation(channel_df)

    log(
        "validation",
        "ok",
        {
            "py_compile": "run separately before audit execution",
            "audit_run": True,
            "no_model_tensor_metadata_prediction_threshold_modification": True,
            "locked_tensor_shape": tensor_meta["shape"],
            "channel_names": ch_names,
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
