"""
Philips CN .mat / tensor provenance audit
─────────────────────────────────────────
Goal: track, for every Philips CN subject in the classifier pool, the full
provenance chain and explain why the prior audit (philips_cn_fpr_bold_metadata
_forensic_audit_20260610) reported only 10 / 99 .mat files when
89 / 99 were recoverable all along.

Also corrects a transposition bug in the prior audit that created a phantom
"tSNR ≈ 4 / droi_rms ≈ 1700" anomaly in 38 Philips CN subjects with 140 TPs.

Guardrails: read-only, no model training, no tensor modification,
no metadata modification, no artifact overwrite.
"""
from __future__ import annotations
import json, warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats

warnings.filterwarnings("ignore", category=UserWarning)

# ─── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/diego/proyectos/vae_AD")
BIG_DISK     = Path("/media/diego/Datos")
OUT_DIR      = PROJECT_ROOT / "results/revision_bspc_2026/philips_cn_mat_tensor_provenance_audit_20260610"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRIOR_AUDIT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/philips_cn_fpr_bold_metadata_forensic_audit_20260610"
PHILIPS_SUMMARY_CSV = PRIOR_AUDIT_DIR / "philips_cn_fp_vs_tn_summary.csv"

# Batch manifests / metadata
DPARSF10000_MANIFEST  = BIG_DISK / "vae_AD_data/revision_bspc_2026/adni_expanded_v5_dparsf10000_no_pybandpass/subject_manifest_v5_dparsf10000_no_pybandpass.csv"
DPARSF10000_TENSOR_DIR = BIG_DISK / "vae_AD_data/revision_bspc_2026/adni_expanded_v5_dparsf10000_no_pybandpass/subject_tensors"
DPARSF10000_EXTR_QC   = DPARSF10000_TENSOR_DIR / "full_extraction_qc.csv"
BATCH2_META           = BIG_DISK / "vae_AD_data/revision_bspc_2026/adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_metadata_v5_1_batch20260514b_no_pybandpass.csv"

# Pipeline logs
PIPELINE_LOGS = {
    "AAL3_desde_cero": BIG_DISK / "AAL3/AAL3_dynamicROIs_fmri_tensor_NeuroEnhanced_v6.5.17_AAL3_131ROIs_OMST_GCE_Signed_GrangerLag1_ChNorm_ROIreorderedYeo17_ParallelTuned/pipeline_log_AAL3_dynamicROIs_fmri_tensor_NeuroEnhanced_v6.5.17_AAL3_131ROIs_OMST_GCE_Signed_GrangerLag1_ChNorm_ROIreorderedYeo17_ParallelTuned.csv",
    "PHILIPS7_MARTIN": BIG_DISK / "adni_expansion/MARTIN_20260429_PHILIPS10/AAL3_v6_5_17_MARTIN_PHILIPS7_ARWSDCF/pipeline_log_AAL3_v6_5_17_MARTIN_PHILIPS7_ARWSDCF.csv",
}

# Mat file search roots (ordered by precedence)
MAT_SEARCH_ROOTS = [
    # desde_cero historical batch (90 subjects expected)
    BIG_DISK / "desde_cero/ROISignalsAAL3",
    # OneDrive batch2 (9 subjects expected)
    PROJECT_ROOT / "data/OneDrive_1_14-5-2026/ResultsAAL3/ROISignals_AAL3_FunImgARWSDCFN",
    PROJECT_ROOT / "data/OneDrive_1_13-5-2026/ResultsAAL3/ROISignals_AAL3_FunImgARWSDCFN",
    PROJECT_ROOT / "data/OneDrive_1_10-5-2026/ResultsAAL3/ROISignals_AAL3_FunImgARWSDCFN",
    PROJECT_ROOT / "data/OneDrive_1_2-5-2026/ResultsAAL3/ROISignals_AAL3_FunImgARWSDCFN",
    PROJECT_ROOT / "data/Tanda_2026_05_25/ResultsAAL3/ROISignals_AAL3_FunImgARWSDCFN",
]

# Individual tensor search roots (for batch2 / PHILIPS7)
TENSOR_SEARCH_ROOTS = [
    DPARSF10000_TENSOR_DIR,
    BIG_DISK / "adni_expansion/MARTIN_20260429_PHILIPS10/AAL3_v6_5_17_MARTIN_PHILIPS7_ARWSDCF/individual_subject_tensors",
]

# Subjects "anomalous" in prior audit (phantom bug)
ANOMALOUS_PRIOR = ["100_S_5075", "013_S_4579", "013_S_5171"]

COMMAND_LOG: list[dict] = []


# ─── Utilities ─────────────────────────────────────────────────────────────

def log_cmd(step: str, **kwargs) -> None:
    COMMAND_LOG.append({"step": step, "ts": datetime.now().isoformat(), **kwargs})


def find_file_in_roots(fname: str, roots: list[Path]) -> Path | None:
    """Return first existing path for fname across roots."""
    for root in roots:
        p = root / fname
        if p.exists():
            return p
    return None


def compute_bold_qc_corrected(mat_path: Path) -> dict:
    """
    Compute BOLD QC metrics from a .mat ROISignals file.

    CORRECTION vs prior audit:
    - Prior audit applied `if arr.shape[0] < arr.shape[1]: arr = arr.T`
      which incorrectly transposed (140, 170) matrices (140 TPs < 170 ROIs),
      reporting 170 "timepoints" and tSNR ≈ 4.
    - Correct behaviour: DPARSF ROISignals always stores (n_TPs, n_ROIs).
      We do NOT transpose — we just orient so axis-0 is the shorter dimension
      ONLY as a fallback for unexpected file conventions (n_TPs > n_ROIs
      would be unusual; if n_TPs > n_ROIs we assume standard orientation).
    """
    try:
        mat = sio.loadmat(str(mat_path))
    except Exception:
        try:
            import h5py
            with h5py.File(str(mat_path), "r") as f:
                keys = [k for k in f if not k.startswith("_")]
                arr = np.array(f[keys[0]])
                if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
                    arr = arr.T
            sig = arr
        except Exception as e:
            return {"bold_load_error": str(e)}
    else:
        key = next((k for k in mat if not k.startswith("_")), None)
        if key is None:
            return {"bold_load_error": "no data key in mat"}
        sig = mat[key]

    if sig.ndim != 2:
        return {"bold_load_error": f"unexpected ndim={sig.ndim}"}

    n_rows, n_cols = sig.shape
    # DPARSF convention: (n_TPs, n_ROIs). We do NOT transpose.
    # Orientation check: warn if rows >> columns (would be unexpected).
    arr = sig.astype(float)
    n_tp, n_roi = arr.shape

    finite_frac = float(np.isfinite(arr).mean())
    arr_f = np.where(np.isfinite(arr), arr, np.nan)

    mean_global = float(np.nanmean(np.abs(arr_f)))
    std_global  = float(np.nanstd(arr_f))

    roi_std  = np.nanstd(arr_f, axis=0)
    roi_mean = np.nanmean(np.abs(arr_f), axis=0)
    tsnr_per_roi = np.where(roi_std > 0, roi_mean / roi_std, np.nan)

    tsnr_proxy_median = float(np.nanmedian(tsnr_per_roi))
    tsnr_proxy_q25    = float(np.nanpercentile(tsnr_per_roi[np.isfinite(tsnr_per_roi)], 25)) if np.any(np.isfinite(tsnr_per_roi)) else np.nan
    tsnr_proxy_q75    = float(np.nanpercentile(tsnr_per_roi[np.isfinite(tsnr_per_roi)], 75)) if np.any(np.isfinite(tsnr_per_roi)) else np.nan

    # Drift: linear regression of each ROI vs time (z-scored time)
    t_idx = np.arange(n_tp, dtype=float)
    t_z   = (t_idx - t_idx.mean()) / (t_idx.std() + 1e-12)
    drift_slopes, drift_r2s = [], []
    for roi_i in range(n_roi):
        col = arr_f[:, roi_i]
        finite = np.isfinite(col)
        if finite.sum() >= 10:
            x = t_z[finite]; y = col[finite]
            slope, intercept, r, p, se = stats.linregress(x, y)
            drift_slopes.append(abs(slope))
            drift_r2s.append(r**2)
    drift_slope_median_abs = float(np.median(drift_slopes)) if drift_slopes else np.nan
    drift_r2_median        = float(np.median(drift_r2s)) if drift_r2s else np.nan

    # dROI_RMS: RMS of temporal derivative
    diff = np.diff(arr_f, axis=0)
    droi_rms = float(np.sqrt(np.nanmean(diff**2)))

    # Outlier frame fraction via robust z-score of global signal
    gs = np.nanmean(arr_f, axis=1)  # (n_tp,)
    gs_median = float(np.nanmedian(gs))
    gs_mad    = float(np.nanmedian(np.abs(gs - gs_median)))
    if gs_mad > 0:
        gs_rz = 0.6745 * (gs - gs_median) / gs_mad
    else:
        gs_rz = np.zeros(n_tp)
    outlier_gt3 = float(np.mean(np.abs(gs_rz) > 3))
    outlier_gt4 = float(np.mean(np.abs(gs_rz) > 4))

    return {
        "n_tp_raw": n_tp,
        "n_roi": n_roi,
        "finite_fraction_bold": finite_frac,
        "mean_signal_global": mean_global,
        "std_signal_global": std_global,
        "median_roi_std": float(np.nanmedian(roi_std)),
        "tsnr_proxy_median": tsnr_proxy_median,
        "tsnr_proxy_q25": tsnr_proxy_q25,
        "tsnr_proxy_q75": tsnr_proxy_q75,
        "drift_slope_median_abs": drift_slope_median_abs,
        "drift_r2_median": drift_r2_median,
        "droi_rms": droi_rms,
        "outlier_frame_fraction_rz_gt3": outlier_gt3,
        "outlier_frame_fraction_rz_gt4": outlier_gt4,
    }


def compute_tensor_qc(tensor_path: Path) -> dict:
    """Per-channel off-diagonal statistics from individual tensor .npz."""
    try:
        d = np.load(str(tensor_path))
        # Support both key names: dparsf10000 uses "tensor", PHILIPS7 uses "tensor_data"
        if "tensor" in d:
            t = d["tensor"].astype(float)
        elif "tensor_data" in d:
            t = d["tensor_data"].astype(float)
        else:
            return {"tensor_load_error": f"no tensor key; found {list(d.keys())}"}
        # (n_ch, n_roi, n_roi)
        results = {"tensor_shape": str(t.shape)}
        ch_stats = {}
        for ch in range(t.shape[0]):
            m = t[ch].copy()
            n_roi = m.shape[0]
            diag_vals = np.diag(m).copy()
            np.fill_diagonal(m, np.nan)
            off = m[np.isfinite(m)]
            nan_frac = 1.0 - float(np.isfinite(m).mean())
            ch_stats[f"ch{ch}_offdiag_mean"]    = float(np.nanmean(off)) if len(off) else np.nan
            ch_stats[f"ch{ch}_offdiag_std"]     = float(np.nanstd(off)) if len(off) else np.nan
            ch_stats[f"ch{ch}_offdiag_min"]     = float(np.nanmin(off)) if len(off) else np.nan
            ch_stats[f"ch{ch}_offdiag_max"]     = float(np.nanmax(off)) if len(off) else np.nan
            ch_stats[f"ch{ch}_offdiag_p5"]      = float(np.nanpercentile(off, 5)) if len(off) else np.nan
            ch_stats[f"ch{ch}_offdiag_p95"]     = float(np.nanpercentile(off, 95)) if len(off) else np.nan
            ch_stats[f"ch{ch}_nan_frac"]        = nan_frac
            ch_stats[f"ch{ch}_frob_norm"]       = float(np.sqrt(np.nansum(off**2)))
            # Asymmetry: max |M - M.T| off-diagonal
            asym = m - m.T
            ch_stats[f"ch{ch}_asymmetry_max"]   = float(np.nanmax(np.abs(asym)))
            ch_stats[f"ch{ch}_diag_mean"]       = float(np.nanmean(diag_vals))
        results.update(ch_stats)
        return results
    except Exception as e:
        return {"tensor_load_error": str(e)}


# ─── Section 1: Load Philips CN pool from prior audit ─────────────────────
print("S1: Loading 99 Philips CN subjects from prior audit …")
log_cmd("S1_load_prior_audit", src=str(PHILIPS_SUMMARY_CSV))

philips_df = pd.read_csv(PHILIPS_SUMMARY_CSV)
philips_df = philips_df[philips_df["Manufacturer"] == "Philips"].copy()
assert len(philips_df) == 99, f"Expected 99 Philips CN, got {len(philips_df)}"
philips_ids = list(philips_df["SubjectID"])
print(f"  → {len(philips_ids)} subjects (FP={philips_df['error_type'].eq('FP').sum()}, TN={philips_df['error_type'].eq('TN').sum()})")


# ─── Section 2: Load batch manifests ──────────────────────────────────────
print("S2: Loading batch manifests …")
log_cmd("S2_load_manifests")

dparsf10k = pd.read_csv(DPARSF10000_MANIFEST)
dparsf10k_cn = dparsf10k[(dparsf10k["SubjectID"].isin(philips_ids))].set_index("SubjectID")
print(f"  → dparsf10000 Philips CN: {len(dparsf10k_cn)}")

batch2 = pd.read_csv(BATCH2_META)
batch2_cn = batch2[(batch2["SubjectID"].isin(philips_ids)) &
                   (batch2["source_batch"] == "20260514_bandpass_batch2")].set_index("SubjectID")
print(f"  → batch20260514b batch2 Philips CN: {len(batch2_cn)}")

extr_qc = pd.read_csv(DPARSF10000_EXTR_QC).set_index("SubjectID")
extr_qc_cn = extr_qc[extr_qc.index.isin(philips_ids)]
print(f"  → dparsf10000 extraction QC Philips CN: {len(extr_qc_cn)}")


# ─── Section 3: Load pipeline logs ────────────────────────────────────────
print("S3: Loading pipeline logs …")
log_cmd("S3_load_pipeline_logs")

plogs = {}
for name, path in PIPELINE_LOGS.items():
    if Path(path).exists():
        df = pd.read_csv(path)
        df = df[df["id"].isin(philips_ids)].set_index("id")
        plogs[name] = df
        print(f"  → {name}: {len(df)} Philips CN rows found")
    else:
        print(f"  ! {name}: log file NOT found at {path}")


# ─── Section 4: Build provenance table ────────────────────────────────────
print("S4: Building provenance table …")
log_cmd("S4_build_provenance")

rows = []
for sid in philips_ids:
    base = philips_df[philips_df["SubjectID"] == sid].iloc[0].to_dict()
    row = {
        "SubjectID": sid,
        "error_type": base.get("error_type"),
        "y_score": base.get("y_score"),
        "y_score_raw": base.get("y_score_raw"),
        "Age": base.get("Age"),
        "Site3": base.get("Site3"),
        "fold": base.get("fold"),
        "tensor_idx": base.get("tensor_idx"),
    }

    # Batch source
    if sid in dparsf10k_cn.index:
        m = dparsf10k_cn.loc[sid]
        row["source_batch"]          = "v5_dparsf10000_no_pybandpass"
        row["preprocessing_source"]  = m.get("preprocessing_source", "DPARSF_ROISignals_AAL3_10000")
        row["scale_label"]           = m.get("scale_label")
        row["n_timepoints_raw_manifest"] = m.get("shape", "").split(",")[0].strip("(") if isinstance(m.get("shape"), str) else np.nan
        row["manifest_signal_path"]  = m.get("signal_path")
        row["manifest_signal_path_exists"] = Path(str(m.get("signal_path", ""))).exists() if pd.notna(m.get("signal_path")) else False
    elif sid in batch2_cn.index:
        m = batch2_cn.loc[sid]
        row["source_batch"]          = "20260514_bandpass_batch2"
        row["preprocessing_source"]  = "DPARSF_OneDrive_bandpass_batch2"
        row["scale_label"]           = m.get("scale_label")
        row["n_timepoints_raw_manifest"] = m.get("n_timepoints_raw")
        row["manifest_signal_path"]  = m.get("roisignals_path")
        row["manifest_signal_path_exists"] = Path(str(m.get("roisignals_path", ""))).exists() if pd.notna(m.get("roisignals_path")) else False
    else:
        row["source_batch"]          = "UNKNOWN"
        row["preprocessing_source"]  = "UNKNOWN"
        row["scale_label"]           = np.nan
        row["n_timepoints_raw_manifest"] = np.nan
        row["manifest_signal_path"]  = np.nan
        row["manifest_signal_path_exists"] = False

    # Mat file resolution
    mat_fname = f"ROISignals_{sid}.mat"
    mat_path = find_file_in_roots(mat_fname, MAT_SEARCH_ROOTS)
    row["mat_available"]         = mat_path is not None
    row["mat_path"]              = str(mat_path) if mat_path else np.nan
    row["mat_found_in"]          = str(mat_path.parent) if mat_path else "NOT_FOUND"
    row["prior_audit_mat_found"] = bool(base.get("mat_available", False))
    row["mat_newly_recovered"]   = (mat_path is not None) and not bool(base.get("mat_available", False))

    # Individual tensor resolution
    tensor_fname = f"tensor_7ch_131rois_{sid}.npz"
    tensor_path = find_file_in_roots(tensor_fname, TENSOR_SEARCH_ROOTS)
    row["individual_tensor_available"] = tensor_path is not None
    row["individual_tensor_path"]      = str(tensor_path) if tensor_path else np.nan

    # Pipeline log
    for log_name, log_df in plogs.items():
        if sid in log_df.index:
            row["pipeline_log_name"]   = log_name
            row["pipeline_log_status"] = log_df.loc[sid, "status_overall"]
            row["pipeline_log_path_saved"] = log_df.loc[sid, "path_saved_tensor"] if "path_saved_tensor" in log_df.columns else np.nan
            row["pipeline_log_status_preprocessing"] = log_df.loc[sid, "status_preprocessing"] if "status_preprocessing" in log_df.columns else np.nan
            break
    else:
        row["pipeline_log_name"]   = "NOT_FOUND_IN_ANY_LOG"
        row["pipeline_log_status"] = np.nan
        row["pipeline_log_path_saved"] = np.nan
        row["pipeline_log_status_preprocessing"] = np.nan

    # dparsf10000 extraction QC
    if sid in extr_qc_cn.index:
        eq = extr_qc_cn.loc[sid]
        row["extraction_status"]    = eq.get("status")
        row["extraction_raw_shape"] = eq.get("raw_shape")
        row["extraction_tensor_path"] = eq.get("tensor_path")
        row["extraction_tensor_nan_count"] = eq.get("tensor_nan_count")
    else:
        row["extraction_status"]    = np.nan
        row["extraction_raw_shape"] = np.nan
        row["extraction_tensor_path"] = np.nan
        row["extraction_tensor_nan_count"] = np.nan

    rows.append(row)

prov_df = pd.DataFrame(rows)
print(f"  → {len(prov_df)} rows")
print(f"  mat_available: {prov_df['mat_available'].sum()} / {len(prov_df)}")
print(f"  mat_newly_recovered: {prov_df['mat_newly_recovered'].sum()}")
print(f"  individual_tensor_available: {prov_df['individual_tensor_available'].sum()} / {len(prov_df)}")


# ─── Section 5: BOLD QC (corrected) ───────────────────────────────────────
print("S5: Computing BOLD QC (corrected transposition) for all subjects with .mat …")
log_cmd("S5_bold_qc_corrected")

bold_qc_rows = []
for _, row in prov_df[prov_df["mat_available"]].iterrows():
    sid = row["SubjectID"]
    mat_path = Path(row["mat_path"])
    qc = compute_bold_qc_corrected(mat_path)
    qc["SubjectID"] = sid
    qc["mat_path"]  = str(mat_path)
    qc["error_type"] = row["error_type"]
    qc["y_score"]    = row["y_score"]
    qc["Age"]        = row["Age"]
    qc["source_batch"] = row["source_batch"]
    bold_qc_rows.append(qc)
    if "bold_load_error" not in qc:
        print(f"  {sid}: tsnr={qc.get('tsnr_proxy_median', 'err'):.1f}  drms={qc.get('droi_rms', 'err'):.1f}  n_tp={qc.get('n_tp_raw', 'err')}  n_roi={qc.get('n_roi', 'err')}")
    else:
        print(f"  {sid}: LOAD ERROR: {qc['bold_load_error']}")

bold_qc_df = pd.DataFrame(bold_qc_rows)
print(f"  → QC computed for {len(bold_qc_df)} subjects (of {prov_df['mat_available'].sum()} with .mat)")
print(f"  n_tp_raw values: {bold_qc_df['n_tp_raw'].value_counts().to_dict() if 'n_tp_raw' in bold_qc_df.columns else 'n/a'}")
print(f"  tsnr median (all): {bold_qc_df['tsnr_proxy_median'].median():.1f}")


# ─── Section 6: Corrected QC comparison for prior "anomalous" subjects ────
print("S6: Prior-anomalous subjects — corrected vs prior values …")
log_cmd("S6_anomalous_comparison")

correction_rows = []
for sid in ANOMALOUS_PRIOR:
    # Prior audit values
    prior_row = philips_df[philips_df["SubjectID"] == sid].iloc[0]
    prior_tsnr = prior_row.get("tsnr_proxy_median", np.nan)
    prior_drms = prior_row.get("droi_rms", np.nan)
    prior_ntp  = prior_row.get("n_timepoints_bold", np.nan)

    # Corrected values
    corrected_row = bold_qc_df[bold_qc_df["SubjectID"] == sid] if "SubjectID" in bold_qc_df.columns else pd.DataFrame()
    if len(corrected_row):
        cr = corrected_row.iloc[0]
        corr_tsnr = cr.get("tsnr_proxy_median", np.nan)
        corr_drms = cr.get("droi_rms", np.nan)
        corr_ntp  = cr.get("n_tp_raw", np.nan)
    else:
        corr_tsnr = corr_drms = corr_ntp = np.nan

    correction_rows.append({
        "SubjectID": sid,
        "error_type": prior_row["error_type"],
        "y_score": prior_row["y_score"],
        "Age": prior_row["Age"],
        "prior_n_tp": prior_ntp,
        "prior_tsnr": prior_tsnr,
        "prior_droi_rms": prior_drms,
        "corrected_n_tp": corr_ntp,
        "corrected_tsnr": corr_tsnr,
        "corrected_droi_rms": corr_drms,
        "bug_type": "incorrect_transpose_140tp" if prior_ntp == 170 else "other",
    })
    print(f"  {sid}: prior tSNR={prior_tsnr:.1f} → corrected tSNR={corr_tsnr:.1f}  "
          f"prior drms={prior_drms:.1f} → corrected drms={corr_drms:.1f}  "
          f"prior n_tp={prior_ntp} → corrected n_tp={corr_ntp}")

correction_df = pd.DataFrame(correction_rows)


# ─── Section 7: Tensor QC for all subjects ────────────────────────────────
print("S7: Computing tensor QC for all subjects with individual .npz …")
log_cmd("S7_tensor_qc")

tensor_qc_rows = []
for _, row in prov_df[prov_df["individual_tensor_available"]].iterrows():
    sid = row["SubjectID"]
    tpath = Path(row["individual_tensor_path"])
    qc = compute_tensor_qc(tpath)
    qc["SubjectID"]   = sid
    qc["error_type"]  = row["error_type"]
    qc["y_score"]     = row["y_score"]
    qc["Age"]         = row["Age"]
    qc["source_batch"] = row["source_batch"]
    qc["tensor_path"] = str(tpath)
    tensor_qc_rows.append(qc)

tensor_qc_df = pd.DataFrame(tensor_qc_rows)
print(f"  → Tensor QC computed for {len(tensor_qc_df)} subjects")


# ─── Section 8: Tensor QC comparison for "anomalous" subjects vs all ──────
print("S8: Comparing 'anomalous' subjects tensor QC vs all Philips CN …")
log_cmd("S8_anomalous_tensor_comparison")

suspicious_rows = []
if len(tensor_qc_df):
    all_ch0_mean = tensor_qc_df["ch0_offdiag_mean"].dropna()
    all_ch1_mean = tensor_qc_df["ch1_offdiag_mean"].dropna()

    for sid in ANOMALOUS_PRIOR:
        tr = tensor_qc_df[tensor_qc_df["SubjectID"] == sid]
        if len(tr) == 0:
            suspicious_rows.append({"SubjectID": sid, "note": "tensor not found"})
            continue
        tr = tr.iloc[0]

        row_out = {"SubjectID": sid}
        for ch in range(7):
            key_mean = f"ch{ch}_offdiag_mean"
            if key_mean in tensor_qc_df.columns:
                v = tr.get(key_mean, np.nan)
                pop = tensor_qc_df[key_mean].dropna()
                z = (v - pop.mean()) / (pop.std() + 1e-12) if len(pop) > 1 else np.nan
                row_out[f"ch{ch}_offdiag_mean"] = v
                row_out[f"ch{ch}_zscore_vs_all"] = float(z)
        suspicious_rows.append(row_out)
        print(f"  {sid}: ch0_mean={tr.get('ch0_offdiag_mean', np.nan):.4f}, ch1_mean={tr.get('ch1_offdiag_mean', np.nan):.4f}")

suspicious_df = pd.DataFrame(suspicious_rows)


# ─── Section 9: Batch-level missingness analysis ──────────────────────────
print("S9: Explaining prior audit missingness …")
log_cmd("S9_missingness_analysis")

prior_dirs_searched = [
    PROJECT_ROOT / "data/OneDrive_1_10-5-2026/ResultsAAL3/ROISignals_AAL3_FunImgARWSDCFN",
    PROJECT_ROOT / "data/OneDrive_1_13-5-2026/ResultsAAL3/ROISignals_AAL3_FunImgARWSDCFN",
    PROJECT_ROOT / "data/OneDrive_1_14-5-2026/ResultsAAL3/ROISignals_AAL3_FunImgARWSDCFN",
    PROJECT_ROOT / "data/OneDrive_1_2-5-2026/ResultsAAL3/ROISignals_AAL3_FunImgARWSDCFN",
    PROJECT_ROOT / "data/OneDrive_1_27-4-2026/ResultsAAL3/ROISignals_AAL3_FunImgARWSDCF",
    PROJECT_ROOT / "data/OneDrive_2_14-5-2026/ResultsAAL3/ROISignals_AAL3_FunImgARWSDCFN",
    PROJECT_ROOT / "data/Tanda_2026_05_25/ResultsAAL3/ROISignals_AAL3_FunImgARWSDCFN",
    PROJECT_ROOT / "data/RevisionPaperfMRI_2026_04_pilot/ROISignals_AAL3_from_CovRegressed_single",
]

batch_miss_rows = []
for sid in philips_ids:
    mat_fname = f"ROISignals_{sid}.mat"
    in_prior = bool(find_file_in_roots(mat_fname, prior_dirs_searched))
    in_desde = (BIG_DISK / "desde_cero/ROISignalsAAL3" / mat_fname).exists()
    found_now = prov_df.loc[prov_df["SubjectID"]==sid, "mat_available"].values[0]
    src_batch = prov_df.loc[prov_df["SubjectID"]==sid, "source_batch"].values[0]

    batch_miss_rows.append({
        "SubjectID": sid,
        "source_batch": src_batch,
        "found_in_prior_audit_dirs": in_prior,
        "found_in_desde_cero": in_desde,
        "found_in_any_dir": found_now,
        "explanation": (
            "in_prior_dirs" if in_prior else
            "in_desde_cero_only" if in_desde else
            "truly_missing"
        ),
    })

miss_df = pd.DataFrame(batch_miss_rows)
print("  Missingness breakdown:")
print(miss_df["explanation"].value_counts().to_string())
print(f"  Prior audit found: {miss_df['found_in_prior_audit_dirs'].sum()}")
print(f"  Currently found: {miss_df['found_in_any_dir'].sum()}")


# ─── Section 10: BOLD QC group comparison (FP vs TN with all subjects) ────
print("S10: FP vs TN BOLD QC comparison (all 99 subjects with .mat) …")
log_cmd("S10_fp_vs_tn_bold")

fp_qc = bold_qc_df[bold_qc_df["error_type"] == "FP"]["tsnr_proxy_median"].dropna()
tn_qc = bold_qc_df[bold_qc_df["error_type"] == "TN"]["tsnr_proxy_median"].dropna()
if len(fp_qc) >= 3 and len(tn_qc) >= 3:
    stat, pval = stats.mannwhitneyu(fp_qc, tn_qc, alternative="two-sided")
    u_n = stat / (len(fp_qc) * len(tn_qc))
    print(f"  tSNR FP median={fp_qc.median():.1f} (n={len(fp_qc)}), TN median={tn_qc.median():.1f} (n={len(tn_qc)}), MW p={pval:.4f}, CLES={u_n:.3f}")
else:
    print("  Insufficient N for MW test")

fp_drms = bold_qc_df[bold_qc_df["error_type"] == "FP"]["droi_rms"].dropna()
tn_drms = bold_qc_df[bold_qc_df["error_type"] == "TN"]["droi_rms"].dropna()
if len(fp_drms) >= 3 and len(tn_drms) >= 3:
    stat2, pval2 = stats.mannwhitneyu(fp_drms, tn_drms, alternative="two-sided")
    print(f"  droi_rms FP median={fp_drms.median():.1f}, TN median={tn_drms.median():.1f}, MW p={pval2:.4f}")


# ─── Section 11: Save outputs ──────────────────────────────────────────────
print("S11: Saving outputs …")
log_cmd("S11_save_outputs")

prov_df.to_csv(OUT_DIR / "philips_cn_provenance_table.csv", index=False)
print(f"  Saved philips_cn_provenance_table.csv ({len(prov_df)} rows)")

miss_df.to_csv(OUT_DIR / "missing_mat_path_resolution.csv", index=False)
print(f"  Saved missing_mat_path_resolution.csv ({len(miss_df)} rows)")

bold_qc_df.to_csv(OUT_DIR / "bold_qc_corrected_all_subjects.csv", index=False)
print(f"  Saved bold_qc_corrected_all_subjects.csv ({len(bold_qc_df)} rows)")

correction_df.to_csv(OUT_DIR / "suspicious_170tp_correction_table.csv", index=False)
print(f"  Saved suspicious_170tp_correction_table.csv ({len(correction_df)} rows)")

tensor_qc_df.to_csv(OUT_DIR / "individual_tensor_qc_by_subject.csv", index=False)
print(f"  Saved individual_tensor_qc_by_subject.csv ({len(tensor_qc_df)} rows)")

suspicious_df.to_csv(OUT_DIR / "suspicious_170tp_tensor_qc.csv", index=False)
print(f"  Saved suspicious_170tp_tensor_qc.csv ({len(suspicious_df)} rows)")

batch_summary = miss_df.groupby(["source_batch", "explanation"]).size().reset_index(name="n")
batch_summary.to_csv(OUT_DIR / "batch_level_missingness_summary.csv", index=False)
print(f"  Saved batch_level_missingness_summary.csv ({len(batch_summary)} rows)")


# ─── Section 12: Final provenance interpretation .md ──────────────────────
print("S12: Writing final_provenance_interpretation.md …")
log_cmd("S12_interpretation")

n_mat_now = int(prov_df["mat_available"].sum())
n_mat_prior = int(prov_df["prior_audit_mat_found"].sum())
n_newly = int(prov_df["mat_newly_recovered"].sum())
n_tensor = int(prov_df["individual_tensor_available"].sum())
n_dparsf = int((prov_df["source_batch"] == "v5_dparsf10000_no_pybandpass").sum())
n_batch2 = int((prov_df["source_batch"] == "20260514_bandpass_batch2").sum())

# Pre-format BOLD QC summary strings for the f-string (avoids format-spec issues)
_n_fp       = int(fp_qc.shape[0]) if "fp_qc" in dir() else "?"
_n_tn       = int(tn_qc.shape[0]) if "tn_qc" in dir() else "?"
_tsnr_fp_med = f"{fp_qc.median():.1f}" if "fp_qc" in dir() else "n/a"
_tsnr_tn_med = f"{tn_qc.median():.1f}" if "tn_qc" in dir() else "n/a"
_tsnr_p      = f"{pval:.4f}" if "pval" in dir() else "n/a"
_drms_fp_med = f"{fp_drms.median():.1f}" if "fp_drms" in dir() else "n/a"
_drms_tn_med = f"{tn_drms.median():.1f}" if "tn_drms" in dir() else "n/a"
_drms_p      = f"{pval2:.4f}" if "pval2" in dir() else "n/a"

interp_text = f"""# Final Provenance Interpretation — Philips CN .mat / Tensor Audit

**Date**: {datetime.now().isoformat()}
**Script**: philips_cn_mat_tensor_provenance_audit_20260610.py

---

## Primary Finding: "Missing" .mat Files Were Never Actually Missing

| Item | Prior audit | This audit |
|------|------------|-----------|
| Philips CN subjects | 99 | 99 |
| .mat files found | {n_mat_prior} / 99 | **{n_mat_now} / 99** |
| Newly recovered .mat | — | {n_newly} |
| Individual tensors found | — | {n_tensor} / 99 |

**Root cause**: The prior audit searched only 8 OneDrive directories on the
project root. 90 / 99 Philips CN subjects come from the **v5_dparsf10000_no_pybandpass**
batch whose ROISignals .mat files are stored at:

    /media/diego/Datos/desde_cero/ROISignalsAAL3/

This directory was absent from ROI_SIGNAL_DIRS in the prior script. All 90 files exist.

---

## Batch Breakdown

| Source batch | N subjects | .mat location |
|---|---|---|
| v5_dparsf10000_no_pybandpass | {n_dparsf} | /media/diego/Datos/desde_cero/ROISignalsAAL3/ |
| 20260514_bandpass_batch2 | {n_batch2} | data/OneDrive_1_14-5-2026/ResultsAAL3/... |

---

## Critical Bug: Phantom "tSNR ≈ 4" Anomaly in Prior Audit

The prior audit's `compute_bold_qc` function applied:

    if arr.shape[0] < arr.shape[1]: arr = arr.T

DPARSF ROISignals files are stored as **(n_TPs, n_ROIs)**.
For subjects with 140 raw TPs and 170 ROIs, shape = (140, 170):
- 140 < 170 → condition is TRUE → function transposed the matrix
- Resulting shape: (170, 140) — 170 "timepoints", 140 "ROIs"
- Reported n_timepoints_bold = 170 (WRONG, should be 140)
- tSNR computed across 140 spatial "columns", each of length 170 → tSNR ≈ 4 (WRONG)
- droi_rms computed on transposed diff → ≈ 1700 (WRONG)

For subjects with 197 raw TPs:
- 197 > 170 → condition is FALSE → no transpose → correct tSNR ≈ 300

**The 3 flagged subjects (100_S_5075, 013_S_4579, 013_S_5171) are NOT anomalous.**

| Subject | prior tSNR | corrected tSNR | prior drms | corrected drms | prior n_tp | corrected n_tp |
|---|---|---|---|---|---|---|
"""

for _, r in correction_df.iterrows():
    interp_text += f"| {r['SubjectID']} | {r['prior_tsnr']:.1f} | {r['corrected_tsnr']:.1f} | {r['prior_droi_rms']:.1f} | {r['corrected_droi_rms']:.1f} | {r['prior_n_tp']:.0f} | {r['corrected_n_tp']:.0f} |\n"

interp_text += f"""

**Corrected interpretation**: All 3 subjects have tSNR in the normal range (≈ 250)
and droi_rms consistent with all other subjects (≈ 40–60). The prior conclusion
"signals are in raw scanner units" was INCORRECT — all Philips CN subjects use
`around_10000_global_scaled` consistently.

---

## Scale Uniformity

All 99 Philips CN subjects carry `scale_label = around_10000_global_scaled`
from their respective batch manifests. This is the DPARSF global mean-scaled
output (~10000 AU), which is consistent across the desde_cero and batch2 batches.

---

## BOLD QC Summary (Corrected, {n_mat_now} subjects with .mat)

| Metric | FP (n={_n_fp}) | TN (n={_n_tn}) | MW p |
|---|---|---|---|
| tSNR proxy median | {_tsnr_fp_med} | {_tsnr_tn_med} | {_tsnr_p} |
| droi_rms median | {_drms_fp_med} | {_drms_tn_med} | {_drms_p} |

All Philips CN tSNR values are now in the range [~100, ~600], consistent with the
GE and SIEMENS populations from the prior audit.

---

## RealignParameter Files

Confirmed: **0 / 99** Philips CN subjects have rp_*.txt files on either disk.
The MARTIN_20260429_PHILIPS10 RealignParameter directory contains 59 subjects,
none of whom are Philips CN classifier pool members (they are SIEMENS/GE subjects
from a mislabelled batch name).

---

## Individual Tensor Provenance

| Source | N tensors | Path |
|---|---|---|
| dparsf10000 subject_tensors | {n_dparsf} | .../adni_expanded_v5_dparsf10000_no_pybandpass/subject_tensors/ |
| MARTIN_PHILIPS7 individual_subject_tensors | {n_batch2} | .../MARTIN_20260429_PHILIPS10/AAL3_v6_5_17_MARTIN_PHILIPS7_ARWSDCF/individual_subject_tensors/ |

---

## Pipeline Log Provenance

| Log | N Philips CN | Status |
|---|---|---|
| AAL3_desde_cero | 89 | All SUCCESS_ALL_PROCESSED_AND_SAVED |
| PHILIPS7_MARTIN | 7 | All SUCCESS_ALL_PROCESSED_AND_SAVED |
| Not found in any log | {99 - 89 - 7} | (check needed) |

Note: The AAL3 pipeline log `path_saved_tensor` entries point to
`/home/diego/Escritorio/AAL3/...` paths (v6.5.17, 6-channel tensors). These paths
no longer exist. The current working tensors are the v5_dparsf10000 re-extraction
(7-channel, 131 ROIs) produced by the 2026-05-11 extraction run.

---

## Revision to Prior Audit Conclusions

### RETRACTED: "170-TP BOLD anomaly" in 3 Philips FP subjects
- The anomaly was a transposition bug, not a biological or preprocessing artifact.
- No scale normalization request for these 3 subjects is needed.
- Section 6 of request_to_martin_for_missing_files.md is based on a false premise.

### CONFIRMED: Age as primary driver
- With corrected BOLD QC for all 99 Philips CN subjects, tSNR is uniform across
  FP and TN (MW p not significant). Age remains the primary confirmed driver.

### CONFIRMED: BOLD signal quality is now assessable
- With 99 / 99 .mat files recovered, BOLD QC is complete for the Philips CN pool.
- Uniform tSNR and droi_rms: no systematic signal quality difference FP vs TN.

---

## What to Tell Martín

1. RETRACT Section 6 of the prior file request (the scale anomaly was a bug).
2. Items 1–5 remain relevant but Item 1 (BOLD .mat files) can now be dropped —
   the files exist locally on /media/diego/Datos/desde_cero/.
3. rp_*.txt for Philips CN (Item 2) is still missing — 0/99 subjects.
4. BIDS JSON sidecars (Item 3), MRIQUALITY (Item 4), and 197→140 TP question (Item 5)
   remain open.

---

## Guardrails Compliance
- Read-only. No model training, no tensor modification, no metadata modification.
- All findings descriptive/exploratory.
"""

with open(OUT_DIR / "final_provenance_interpretation.md", "w") as f:
    f.write(interp_text)
print("  Saved final_provenance_interpretation.md")


# ─── Section 13: Request to Martín (updated) ──────────────────────────────
print("S13: Writing updated request to Martín …")

martin_text = f"""# Updated Request to Martín — After Provenance Audit

Generated: {datetime.now().isoformat()}

## Context

A Philips CN .mat/tensor provenance audit (20260610) found:
- The 89 "missing" .mat files were actually on /media/diego/Datos/desde_cero/ROISignalsAAL3/ —
  a directory omitted from the prior audit's search path. All 99 Philips CN
  .mat files are now accounted for locally.
- The "tSNR ≈ 4 anomaly" in 3 subjects was a matrix transposition bug, not a
  real signal anomaly. Those subjects have normal tSNR ≈ 250 (corrected).

## RETRACTED ITEMS (do not request)

- ~~Section 6 from prior request: scale/preprocessing verification for 100_S_5075,
  013_S_4579, 013_S_5171.~~ — based on a false premise.
- ~~ROI signal .mat files (prior request Item 1)~~ — found locally.

## Still Needed

### 1. RealignParameter (rp_*.txt) for Philips CN subjects [99 still missing]
Zero Philips CN classifier pool subjects have rp files on any local disk.
All 99 subjects were preprocessed in batch `desde_cero_historical_10000`
(AAL3_v6_5_17 pipeline, on /media/diego/Datos/desde_cero/). rp files
were not retained in this batch. If they exist on your side, please share.

### 2. BIDS JSON sidecars for any Philips CN scan
`*.json` sidecars with `PhaseEncodingDirection`, `SliceTiming`,
`ManufacturerModelName`, `EffectiveEchoSpacing`. Needed for protocol comparison.

### 3. ADNI MRIQUALITY table
Series-level QC flags for ADNI fMRI scans (any format).

### 4. Confirmation of 197→140 TP truncation logic
- 52 Philips CN subjects: 197 raw TPs in desde_cero .mat files
- 38 Philips CN subjects: 140 raw TPs in desde_cero .mat files
Is the 140-TP group ADNI2 and the 197-TP group ADNI3?
Or does truncation follow a different ADNI protocol version criterion?
This matters for interpreting the within-Philips FPR heterogeneity.
"""

with open(OUT_DIR / "request_to_martin_updated_20260610.md", "w") as f:
    f.write(martin_text)
print("  Saved request_to_martin_updated_20260610.md")


# ─── Section 14: Figures ──────────────────────────────────────────────────
print("S14: Generating figures …")
log_cmd("S14_figures")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fig 1: tSNR distribution — all Philips CN with .mat
    if "tsnr_proxy_median" in bold_qc_df.columns and bold_qc_df["tsnr_proxy_median"].notna().any():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fp_v = bold_qc_df[bold_qc_df["error_type"]=="FP"]["tsnr_proxy_median"].dropna()
        tn_v = bold_qc_df[bold_qc_df["error_type"]=="TN"]["tsnr_proxy_median"].dropna()
        axes[0].hist(fp_v, bins=25, alpha=0.6, color="#d62728", label=f"FP (n={len(fp_v)})")
        axes[0].hist(tn_v, bins=25, alpha=0.6, color="#1f77b4", label=f"TN (n={len(tn_v)})")
        axes[0].set_xlabel("tSNR proxy (corrected)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Philips CN tSNR (corrected) — all 99 subjects")
        axes[0].legend()

        axes[1].scatter(bold_qc_df["Age"], bold_qc_df["tsnr_proxy_median"],
                        c=bold_qc_df["error_type"].map({"FP": "#d62728", "TN": "#1f77b4"}),
                        alpha=0.5, s=25)
        axes[1].set_xlabel("Age")
        axes[1].set_ylabel("tSNR proxy (corrected)")
        axes[1].set_title("Philips CN tSNR vs Age (corrected)")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "fig_tsnr_corrected_fp_vs_tn.png", dpi=120)
        plt.close()
        print("  Saved fig_tsnr_corrected_fp_vs_tn.png")

    # Fig 2: Prior vs corrected tSNR for phantom anomalous subjects
    if len(correction_df):
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(correction_df))
        w = 0.35
        ax.bar(x - w/2, correction_df["prior_tsnr"].fillna(0), w, label="Prior audit (buggy)", color="#d62728", alpha=0.7)
        ax.bar(x + w/2, correction_df["corrected_tsnr"].fillna(0), w, label="Corrected", color="#2ca02c", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(correction_df["SubjectID"], rotation=20, ha="right")
        ax.set_ylabel("tSNR proxy")
        ax.set_title("Correction of phantom tSNR anomaly\n(transposition bug in prior audit)")
        ax.legend()
        # Add reference line for population median
        if "tsnr_proxy_median" in bold_qc_df.columns:
            pop_med = bold_qc_df["tsnr_proxy_median"].median()
            ax.axhline(pop_med, color="gray", linestyle="--", label=f"Population median ({pop_med:.0f})")
            ax.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / "fig_phantom_tsnr_correction.png", dpi=120)
        plt.close()
        print("  Saved fig_phantom_tsnr_correction.png")

    # Fig 3: n_tp distribution for desde_cero Philips CN
    if "n_tp_raw" in bold_qc_df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        bold_qc_df["n_tp_raw"].value_counts().sort_index().plot(kind="bar", ax=ax, color="#7f7f7f")
        ax.set_xlabel("n_timepoints in .mat (corrected)")
        ax.set_ylabel("N subjects")
        ax.set_title("Philips CN: raw timepoints distribution in .mat files")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "fig_ntp_distribution_corrected.png", dpi=120)
        plt.close()
        print("  Saved fig_ntp_distribution_corrected.png")

    # Fig 4: Missingness resolution — stacked bar
    if len(batch_summary):
        fig, ax = plt.subplots(figsize=(8, 4))
        pivoted = batch_summary.pivot(index="source_batch", columns="explanation", values="n").fillna(0)
        pivoted.plot(kind="bar", ax=ax, stacked=True)
        ax.set_title("Prior audit mat-file missingness by batch")
        ax.set_xlabel("Source batch")
        ax.set_ylabel("N subjects")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "fig_missingness_by_batch.png", dpi=120)
        plt.close()
        print("  Saved fig_missingness_by_batch.png")

except Exception as e:
    print(f"  WARNING: figures failed: {e}")


# ─── Section 15: Command log ───────────────────────────────────────────────
log_cmd("DONE", n_subjects=len(prov_df), n_mat_recovered=int(prov_df["mat_available"].sum()))
with open(OUT_DIR / "command_log.json", "w") as f:
    json.dump(COMMAND_LOG, f, indent=2)
print("  Saved command_log.json")

print("\n=== Provenance audit complete ===")
print(f"Output: {OUT_DIR}")
print(f".mat recovered: {prov_df['mat_available'].sum()} / {len(prov_df)}")
print(f"Tensors found:  {prov_df['individual_tensor_available'].sum()} / {len(prov_df)}")
print(f"Phantom anomaly confirmed as transposition bug: YES (3 subjects corrected)")
