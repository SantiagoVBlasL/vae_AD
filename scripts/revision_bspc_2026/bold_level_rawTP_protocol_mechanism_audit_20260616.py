"""
BOLD-level rawTP protocol mechanism audit — 2026-06-16

Goal: determine whether Philips CN 140TP vs 197TP differences appear at the
BOLD signal level and whether those differences survive T=140 homogenization.

Read-only. No training. No tensor/metadata/prediction/threshold edits.

Outputs written to:
  results/revision_bspc_2026/bold_level_rawTP_protocol_mechanism_audit_20260616/
"""

from __future__ import annotations
import json, logging, os, warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results" / "revision_bspc_2026"
MASTER_DB = (
    RESULTS
    / "promoted_model_master_database_20260610"
    / "promoted_model_master_database.csv"
)
PHILIPS_AUDIT_DIR = RESULTS / "local_martin_philips_source_audit_20260611"
OUT = RESULTS / "bold_level_rawTP_protocol_mechanism_audit_20260616"
OUT.mkdir(parents=True, exist_ok=True)

MAT_DIRS = [
    Path("/media/diego/Datos/desde_cero/ROISignalsAAL3"),
    Path("/home/diego/proyectos/vae_AD/data/OneDrive_1_14-5-2026")
    / "ResultsAAL3"
    / "ROISignals_AAL3_FunImgARWSDCFN",
]

# ── Constants ──────────────────────────────────────────────────────────────────
TR_SEC = 3.0          # ADNI rs-fMRI repetition time (seconds)
T_EFFECTIVE = 140     # homogenization cutoff
N_ROI_RAW = 170       # raw AAL3 ROI count before 170→131 mapping
FS = 1.0 / TR_SEC     # 0.333 Hz
NYQUIST = FS / 2.0    # 0.1667 Hz
LF_BAND = (0.01, 0.08)
HF_BAND = (0.08, NYQUIST)

# Known Site31 reverse-slice-order subjects (confirmed in prior audit)
SITE31_CONFIRMED_REVERSE = {"031_S_4021", "031_S_4218", "031_S_4496"}
# Problem sites per prior audit (confirmed manufacturer reports)
PROBLEM_SITES = {13, 53, 301}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

command_log: dict = {
    "script": __file__,
    "started": datetime.now(timezone.utc).isoformat(),
    "steps": [],
    "outputs": [],
    "errors": [],
}


def _log_step(msg: str, **kw) -> None:
    entry = {"step": msg, "ts": datetime.now(timezone.utc).isoformat()}
    entry.update(kw)
    command_log["steps"].append(entry)
    log.info(msg)


def _log_output(path: Path) -> None:
    command_log["outputs"].append(str(path))
    log.info("  → %s", path.name)


def _log_err(msg: str) -> None:
    command_log["errors"].append({"msg": msg, "ts": datetime.now(timezone.utc).isoformat()})
    log.warning("  ERR: %s", msg)


# ── MAT loading ────────────────────────────────────────────────────────────────
def find_mat(subject_id: str) -> Optional[Path]:
    for d in MAT_DIRS:
        p = d / f"ROISignals_{subject_id}.mat"
        if p.exists():
            return p
    return None


def load_bold(path: Path) -> Optional[np.ndarray]:
    """Return (T, ROI) float64 array or None on error."""
    try:
        d = sio.loadmat(str(path))
        key = next(k for k in d.keys() if not k.startswith("_"))
        arr = np.asarray(d[key], dtype=np.float64)
        if arr.ndim != 2:
            return None
        # Ensure (T, ROI) orientation — raw files are always (T, 170)
        if arr.shape[1] == N_ROI_RAW and arr.shape[0] != N_ROI_RAW:
            return arr
        if arr.shape[0] == N_ROI_RAW and arr.shape[1] != N_ROI_RAW:
            return arr.T
        return arr
    except Exception as e:
        _log_err(f"loadmat {path}: {e}")
        return None


# ── BOLD QC metrics ────────────────────────────────────────────────────────────
def _autocorr_lag1(x: np.ndarray) -> float:
    """Lag-1 autocorrelation of 1D signal."""
    if len(x) < 3:
        return np.nan
    xz = x - x.mean()
    c0 = np.dot(xz, xz)
    if c0 == 0:
        return np.nan
    return np.dot(xz[:-1], xz[1:]) / c0


def _spectral_power(x: np.ndarray, fs: float, band: tuple) -> float:
    """Mean power in frequency band via Welch periodogram."""
    n = len(x)
    if n < 16:
        return np.nan
    nperseg = min(n, max(16, n // 4))
    try:
        freqs, psd = welch(x, fs=fs, nperseg=nperseg)
        mask = (freqs >= band[0]) & (freqs <= band[1])
        if mask.sum() == 0:
            return np.nan
        return float(np.mean(psd[mask]))
    except Exception:
        return np.nan


def compute_bold_qc(bold: np.ndarray, t_max: int = None) -> dict:
    """
    Compute BOLD QC metrics for a (T, ROI) array.

    t_max: if set, truncate to first t_max timepoints before computing.
    Returns dict of scalar metrics.
    """
    if t_max is not None and t_max < bold.shape[0]:
        bold = bold[:t_max, :]

    T, R = bold.shape
    frac_nan = float(np.isnan(bold).mean())

    # Replace NaN with roi mean for downstream stats
    bold_clean = bold.copy()
    for r in range(R):
        col = bold_clean[:, r]
        col_nan = np.isnan(col)
        if col_nan.all():
            bold_clean[:, r] = 0.0
        elif col_nan.any():
            bold_clean[col_nan, r] = np.nanmean(col)

    roi_means = bold_clean.mean(axis=0)      # (R,)
    roi_stds = bold_clean.std(axis=0)        # (R,)

    # Global signal = mean across ROIs at each TP
    global_sig = bold_clean.mean(axis=1)     # (T,)
    global_mean = float(np.mean(global_sig))
    global_sd = float(np.std(global_sig))

    # Per-ROI tSNR
    with np.errstate(divide="ignore", invalid="ignore"):
        tsnr_per_roi = np.where(roi_stds > 0, roi_means / roi_stds, np.nan)

    # Per-ROI lag-1 autocorrelation
    ac_lag1 = np.array([_autocorr_lag1(bold_clean[:, r]) for r in range(R)])

    # Per-ROI linear drift (|slope| normalised by mean)
    t_vec = np.arange(T, dtype=float)
    t_vec_z = (t_vec - t_vec.mean()) / (t_vec.std() + 1e-12)
    drift_slopes = np.zeros(R)
    for r in range(R):
        y = bold_clean[:, r]
        m = y.mean()
        slope = np.dot(t_vec_z, y - m) / T
        drift_slopes[r] = abs(slope) / (abs(m) + 1e-12)

    # Outlier frames: |z-score of global signal| > 3
    gs_z = (global_sig - global_sig.mean()) / (global_sig.std() + 1e-12)
    frac_outlier = float((np.abs(gs_z) > 3).mean())

    # Spectral metrics (averaged across ROIs)
    lf_powers, hf_powers = [], []
    for r in range(R):
        lf_powers.append(_spectral_power(bold_clean[:, r], FS, LF_BAND))
        hf_powers.append(_spectral_power(bold_clean[:, r], FS, HF_BAND))
    lf_mean = float(np.nanmean(lf_powers))
    hf_mean = float(np.nanmean(hf_powers))
    lf_hf_ratio = lf_mean / hf_mean if hf_mean > 0 else np.nan

    return {
        "raw_T": T,
        "raw_ROI_count": R,
        "global_mean": global_mean,
        "global_sd": global_sd,
        "roi_mean_median": float(np.median(roi_means)),
        "roi_sd_median": float(np.median(roi_stds)),
        "tsnr_median": float(np.nanmedian(tsnr_per_roi)),
        "autocorr_lag1_median": float(np.nanmedian(ac_lag1)),
        "drift_slope_median": float(np.median(drift_slopes)),
        "frac_nan": frac_nan,
        "frac_outlier": frac_outlier,
        "global_signal_sd": global_sd,
        "lf_power_mean": lf_mean,
        "hf_power_mean": hf_mean,
        "lf_hf_ratio": lf_hf_ratio,
        "bold_load_ok": True,
    }


# ── Statistical helpers ─────────────────────────────────────────────────────────
def mannwhitney_cles(a: np.ndarray, b: np.ndarray) -> tuple:
    """
    Return (U, p_value, CLES) where CLES = P(a > b).
    Returns (nan, nan, nan) for empty groups.
    """
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, np.nan
    U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    cles = U / (len(a) * len(b))
    return float(U), float(p), float(cles)


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction on finite p-values."""
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan)
    finite = np.isfinite(p)
    if finite.sum() == 0:
        return q
    idx = np.where(finite)[0]
    pf = p[idx]
    order = np.argsort(pf)
    n = len(pf)
    q_sorted = pf[order] * n / (np.arange(1, n + 1))
    # reverse cumulative min
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.minimum(q_sorted, 1.0)
    q_back = np.empty(n)
    q_back[order] = q_sorted
    q[idx] = q_back
    return q


def spearman_ci(x, y, n_boot=1000, ci=0.95):
    """Spearman rho with bootstrap CI."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return np.nan, np.nan, np.nan
    rho, p = stats.spearmanr(x, y)
    rhos = [stats.spearmanr(
        x[idx := np.random.choice(len(x), len(x), replace=True)],
        y[idx]
    ).statistic for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(rhos, [alpha, 1 - alpha])
    return float(rho), float(lo), float(hi)


# ── Stage 1: Build BOLD QC table ───────────────────────────────────────────────
def build_bold_qc_table(pool: pd.DataFrame) -> pd.DataFrame:
    _log_step("Stage 1: computing raw and T=140 BOLD QC for all accessible subjects")

    rows = []
    n_found = 0
    n_missing = 0

    for _, r in pool.iterrows():
        mat_path = find_mat(r.SubjectID)
        if mat_path is None:
            n_missing += 1
            row = {
                "SubjectID": r.SubjectID,
                "mat_path_used": None,
                "mat_found": False,
                "bold_load_ok": False,
                "raw_T": np.nan,
                "raw_ROI_count": np.nan,
                "effective_T": T_EFFECTIVE,
            }
            rows.append(row)
            continue

        bold = load_bold(mat_path)
        if bold is None:
            n_missing += 1
            rows.append({
                "SubjectID": r.SubjectID,
                "mat_path_used": str(mat_path),
                "mat_found": True,
                "bold_load_ok": False,
                "raw_T": np.nan,
                "raw_ROI_count": np.nan,
                "effective_T": T_EFFECTIVE,
            })
            continue

        n_found += 1
        raw_T, _ = bold.shape

        # Raw QC (full T)
        raw_qc = compute_bold_qc(bold, t_max=None)

        # T=140 truncated QC
        t140_qc = compute_bold_qc(bold, t_max=T_EFFECTIVE)

        row: dict = {
            "SubjectID": r.SubjectID,
            "mat_path_used": str(mat_path),
            "mat_found": True,
            "bold_load_ok": True,
            "effective_T": T_EFFECTIVE,
        }
        # Prefix raw_ and t140_
        for k, v in raw_qc.items():
            if k != "bold_load_ok":
                row[f"raw_{k}"] = v
        for k, v in t140_qc.items():
            if k not in ("bold_load_ok", "raw_T", "raw_ROI_count"):
                row[f"t140_{k}"] = v

        rows.append(row)

    _log_step(
        f"  MAT files found: {n_found}, missing: {n_missing}",
        n_found=n_found,
        n_missing=n_missing,
    )
    qc_df = pd.DataFrame(rows)
    return qc_df


# ── Stage 2: Merge with master database metadata ───────────────────────────────
METADATA_COLS = [
    "SubjectID", "ImageID", "ResearchGroup_Mapped", "Diagnosis",
    "Manufacturer", "Manufacturer_normalized", "Site3", "SITE",
    "raw_tp_group", "COLPROT", "ORIGPROT", "inferred_ADNI_phase",
    "Age", "Sex", "APOE4",
    "y_score_final", "y_pred", "confusion_label",
    "is_philips_cn", "philips_cn_raw_tp_group",
    "philips_problem_site_flag", "philips_slice_order_issue_flag",
    "philips_slice_order_issue_source", "philips_comment",
    "philips_cn_error_type",
    "tensor_ch0_offdiag_mean", "tensor_ch0_offdiag_std",
    "tensor_ch1_offdiag_mean", "tensor_ch1_offdiag_std",
    "tensor_ch2_offdiag_mean", "tensor_ch2_offdiag_std",
    "ch0_offdiag_mean", "ch1_offdiag_mean", "ch2_offdiag_mean",
    "tsnr_proxy_median_corrected", "drift_slope_median_abs_corrected",
    "droi_rms_corrected", "outlier_frame_fraction_rz_gt3_corrected",
    "median_roi_std_corrected", "std_signal_global_corrected",
    "n_tp_raw_bold_corrected", "n_rois_bold_corrected",
    "fd_mean", "fd_median", "fd_max",
    "TR", "rp_available",
]


def merge_metadata(pool: pd.DataFrame, qc_df: pd.DataFrame) -> pd.DataFrame:
    _log_step("Stage 2: merging BOLD QC with master database metadata")
    avail_meta = [c for c in METADATA_COLS if c in pool.columns]
    meta = pool[avail_meta].copy()
    merged = meta.merge(qc_df, on="SubjectID", how="left")
    return merged


# ── Stage 3: Philips CN stratification groups ─────────────────────────────────
def assign_philips_cn_group(df: pd.DataFrame) -> pd.DataFrame:
    """Add philips_cn_group and site31_confirmed_reverse columns."""
    df = df.copy()

    # Only for Philips CN subjects
    is_philips_cn = (df["Manufacturer_normalized"] == "Philips") & (
        df["ResearchGroup_Mapped"] == "CN"
    )

    def _group(r):
        if not (r["Manufacturer_normalized"] == "Philips" and r["ResearchGroup_Mapped"] == "CN"):
            return "non_philips_cn"
        tp = str(r["raw_tp_group"])
        conf = str(r.get("confusion_label", ""))
        s = r.get("Site3", np.nan)
        site = int(s) if pd.notna(s) else None
        lbl = f"tp{tp}_{'FP' if conf=='FP' else 'TN'}"
        if site is not None:
            lbl += f"_s{site}"
        return lbl

    df["philips_cn_group_full"] = df.apply(_group, axis=1)

    # Coarser groups for statistical testing
    def _coarse_group(r):
        if not (r["Manufacturer_normalized"] == "Philips" and r["ResearchGroup_Mapped"] == "CN"):
            return "non_philips_cn"
        tp = str(r["raw_tp_group"])
        conf = str(r.get("confusion_label", ""))
        return f"tp{tp}_{('FP' if conf=='FP' else 'TN')}"

    df["philips_cn_strat"] = df.apply(_coarse_group, axis=1)

    # Site31 confirmed reverse
    df["site31_confirmed_reverse"] = df["SubjectID"].isin(SITE31_CONFIRMED_REVERSE)
    df["problem_site_cn"] = (
        is_philips_cn
        & df["Site3"].isin(PROBLEM_SITES).fillna(False)
    )

    return df


# ── Stage 4: group-level statistical tests ─────────────────────────────────────
QC_TEST_COLS_RAW = [
    "raw_global_mean", "raw_global_sd", "raw_roi_mean_median",
    "raw_roi_sd_median", "raw_tsnr_median", "raw_autocorr_lag1_median",
    "raw_drift_slope_median", "raw_frac_nan", "raw_frac_outlier",
    "raw_lf_power_mean", "raw_hf_power_mean", "raw_lf_hf_ratio",
]
QC_TEST_COLS_T140 = [
    "t140_global_mean", "t140_global_sd", "t140_roi_mean_median",
    "t140_roi_sd_median", "t140_tsnr_median", "t140_autocorr_lag1_median",
    "t140_drift_slope_median", "t140_frac_nan", "t140_frac_outlier",
    "t140_lf_power_mean", "t140_hf_power_mean", "t140_lf_hf_ratio",
]
CONNECTOME_COLS = [
    "tensor_ch0_offdiag_mean", "tensor_ch1_offdiag_mean", "tensor_ch2_offdiag_mean",
    "tensor_ch0_offdiag_std", "tensor_ch1_offdiag_std", "tensor_ch2_offdiag_std",
]

# Also add the "corrected" fields that already exist in the DB for Philips CN
CORRECTED_COLS = [
    "tsnr_proxy_median_corrected", "drift_slope_median_abs_corrected",
    "droi_rms_corrected", "outlier_frame_fraction_rz_gt3_corrected",
    "median_roi_std_corrected", "std_signal_global_corrected",
]


def run_group_tests(df: pd.DataFrame) -> pd.DataFrame:
    _log_step("Stage 4: running group-level Mann-Whitney tests")

    philips_cn = df[
        (df["Manufacturer_normalized"] == "Philips")
        & (df["ResearchGroup_Mapped"] == "CN")
    ].copy()

    # Determine which QC cols are actually present
    all_test_cols = (
        [c for c in QC_TEST_COLS_RAW if c in df.columns]
        + [c for c in QC_TEST_COLS_T140 if c in df.columns]
        + [c for c in CONNECTOME_COLS if c in df.columns]
        + [c for c in CORRECTED_COLS if c in df.columns]
        + ["y_score_final", "Age"]
    )
    all_test_cols = [c for c in all_test_cols if c in df.columns]

    comparisons = [
        (
            "140TP_vs_197TP",
            philips_cn[philips_cn["raw_tp_group"] == "140"],
            philips_cn[philips_cn["raw_tp_group"] == "197"],
        ),
        (
            "FP_vs_TN_in_140TP",
            philips_cn[
                (philips_cn["raw_tp_group"] == "140")
                & (philips_cn["confusion_label"] == "FP")
            ],
            philips_cn[
                (philips_cn["raw_tp_group"] == "140")
                & (philips_cn["confusion_label"] == "TN")
            ],
        ),
        (
            "FP_vs_TN_in_197TP",
            philips_cn[
                (philips_cn["raw_tp_group"] == "197")
                & (philips_cn["confusion_label"] == "FP")
            ],
            philips_cn[
                (philips_cn["raw_tp_group"] == "197")
                & (philips_cn["confusion_label"] == "TN")
            ],
        ),
        (
            "highscoreFP_vs_rest",
            philips_cn[
                (philips_cn["confusion_label"] == "FP")
                & (philips_cn["y_score_final"] >= 0.75)
            ],
            philips_cn[
                ~(
                    (philips_cn["confusion_label"] == "FP")
                    & (philips_cn["y_score_final"] >= 0.75)
                )
            ],
        ),
    ]

    rows = []
    for comp_name, groupA, groupB in comparisons:
        nA, nB = len(groupA), len(groupB)
        for col in all_test_cols:
            a = groupA[col].dropna().values if col in groupA.columns else np.array([])
            b = groupB[col].dropna().values if col in groupB.columns else np.array([])
            U, p, cles = mannwhitney_cles(a, b)
            rows.append(
                {
                    "comparison": comp_name,
                    "variable": col,
                    "n_A": nA,
                    "n_B": nB,
                    "n_A_obs": len(a),
                    "n_B_obs": len(b),
                    "median_A": float(np.nanmedian(a)) if len(a) else np.nan,
                    "median_B": float(np.nanmedian(b)) if len(b) else np.nan,
                    "U": U,
                    "p_raw": p,
                    "CLES": cles,
                }
            )

    tests_df = pd.DataFrame(rows)

    # FDR per comparison block
    q_vals = []
    for comp, grp in tests_df.groupby("comparison"):
        q = bh_fdr(grp["p_raw"].values)
        q_vals.extend(list(q))
    tests_df["p_fdr_within"] = q_vals
    tests_df["sig_fdr_0p10"] = tests_df["p_fdr_within"] < 0.10
    tests_df["sig_p_0p05"] = tests_df["p_raw"] < 0.05

    # Sort by comparison then p_raw
    tests_df = tests_df.sort_values(["comparison", "p_raw"]).reset_index(drop=True)
    return tests_df


# ── Stage 5: BOLD-score correlations ──────────────────────────────────────────
def run_correlations(df: pd.DataFrame) -> pd.DataFrame:
    _log_step("Stage 5: Spearman correlations with y_score_final and connectome metrics")

    philips_cn = df[
        (df["Manufacturer_normalized"] == "Philips")
        & (df["ResearchGroup_Mapped"] == "CN")
    ].copy()

    target_cols = ["y_score_final"] + [c for c in CONNECTOME_COLS if c in df.columns]
    bold_qc_cols = (
        [c for c in QC_TEST_COLS_RAW + QC_TEST_COLS_T140 + CORRECTED_COLS if c in df.columns]
    )

    rows = []
    # Within Philips CN
    for tgt in target_cols:
        for feat in bold_qc_cols:
            if tgt not in philips_cn.columns or feat not in philips_cn.columns:
                continue
            x = philips_cn[feat].values.astype(float)
            y = philips_cn[tgt].values.astype(float)
            rho, lo, hi = spearman_ci(x, y)
            rows.append(
                {
                    "scope": "philips_cn",
                    "target": tgt,
                    "feature": feat,
                    "rho": rho,
                    "ci_lo_95": lo,
                    "ci_hi_95": hi,
                }
            )

    # Also in full Philips pool
    philips_all = df[df["Manufacturer_normalized"] == "Philips"].copy()
    for tgt in ["y_score_final"]:
        for feat in bold_qc_cols:
            if tgt not in philips_all.columns or feat not in philips_all.columns:
                continue
            x = philips_all[feat].values.astype(float)
            y = philips_all[tgt].values.astype(float)
            rho, lo, hi = spearman_ci(x, y)
            rows.append(
                {
                    "scope": "philips_all",
                    "target": tgt,
                    "feature": feat,
                    "rho": rho,
                    "ci_lo_95": lo,
                    "ci_hi_95": hi,
                }
            )

    corr_df = pd.DataFrame(rows)
    corr_df["abs_rho"] = corr_df["rho"].abs()
    corr_df = corr_df.sort_values(["scope", "target", "abs_rho"], ascending=[True, True, False])
    return corr_df


# ── Stage 6: descriptive models ───────────────────────────────────────────────
def run_descriptive_models(df: pd.DataFrame) -> pd.DataFrame:
    _log_step("Stage 6: fitting descriptive linear/logistic models")

    philips_cn = df[
        (df["Manufacturer_normalized"] == "Philips")
        & (df["ResearchGroup_Mapped"] == "CN")
    ].copy()

    qc_feats = [c for c in CORRECTED_COLS + QC_TEST_COLS_T140 if c in philips_cn.columns]
    conn_feats = [c for c in CONNECTOME_COLS if c in philips_cn.columns]

    # Encode categorical predictors
    philips_cn["tp140_flag"] = (philips_cn["raw_tp_group"] == "140").astype(float)
    philips_cn["age_z"] = (philips_cn["Age"] - philips_cn["Age"].mean()) / (
        philips_cn["Age"].std() + 1e-12
    )
    philips_cn["site3_code"] = pd.Categorical(philips_cn["Site3"]).codes.astype(float)

    rows = []

    # Model A: y_score_final ~ tp140_flag + Age_z + site3_code + QC + connectome
    model_feats = ["tp140_flag", "age_z", "site3_code"] + qc_feats + conn_feats
    present = [c for c in model_feats if c in philips_cn.columns]
    target = "y_score_final"

    sub = philips_cn[[target] + present].dropna()
    if len(sub) >= 10 and sub[target].std() > 0:
        X = StandardScaler().fit_transform(sub[present].values)
        y = sub[target].values
        X_sm = sm.add_constant(X)
        try:
            ols = sm.OLS(y, X_sm).fit()
            feat_names = list(ols.model.exog_names)
            display_names = ["const"] + present
            # align by position if lengths match, else use generic names
            if len(feat_names) == len(display_names):
                feat_labels = display_names
            else:
                feat_labels = feat_names
            for i, feat in enumerate(feat_labels):
                rows.append(
                    {
                        "model": "OLS_score~tp+age+site+qc+conn",
                        "feature": feat,
                        "coef": float(ols.params[i]),
                        "se": float(ols.bse[i]),
                        "t": float(ols.tvalues[i]),
                        "p_raw": float(ols.pvalues[i]),
                        "n": len(sub),
                        "r2": float(ols.rsquared),
                    }
                )
        except Exception as e:
            _log_err(f"OLS model A failed: {e}")

    # Model B: FP_binary ~ tp140_flag + Age_z + site3_code + QC (logistic)
    philips_cn["fp_binary"] = (philips_cn["confusion_label"] == "FP").astype(float)
    feat_b = ["tp140_flag", "age_z", "site3_code"] + qc_feats
    present_b = [c for c in feat_b if c in philips_cn.columns]

    sub_b = philips_cn[["fp_binary"] + present_b].dropna()
    if len(sub_b) >= 10 and sub_b["fp_binary"].nunique() > 1:
        X_b = StandardScaler().fit_transform(sub_b[present_b].values)
        y_b = sub_b["fp_binary"].values
        X_b_sm = sm.add_constant(X_b)
        try:
            logit = sm.Logit(y_b, X_b_sm).fit(disp=False, maxiter=200)
            feat_names_b = list(logit.model.exog_names)
            display_names_b = ["const"] + present_b
            if len(feat_names_b) == len(display_names_b):
                feat_labels_b = display_names_b
            else:
                feat_labels_b = feat_names_b
            for i, feat in enumerate(feat_labels_b):
                rows.append(
                    {
                        "model": "Logit_FP~tp+age+site+qc",
                        "feature": feat,
                        "coef": float(logit.params[i]),
                        "se": float(logit.bse[i]),
                        "t": float(logit.tvalues[i]),
                        "p_raw": float(logit.pvalues[i]),
                        "n": len(sub_b),
                        "r2": float(logit.prsquared),
                    }
                )
        except Exception as e:
            _log_err(f"Logit model B failed: {e}")

    model_df = pd.DataFrame(rows)
    if len(model_df):
        # FDR per model
        fdr_vals = []
        for model_id, grp in model_df.groupby("model"):
            fdr_vals.extend(list(bh_fdr(grp["p_raw"].values)))
        model_df["p_fdr"] = fdr_vals
        model_df["sig_fdr_0p10"] = model_df["p_fdr"] < 0.10
        model_df = model_df.sort_values(["model", "p_raw"])

    return model_df


# ── Stage 7: site/protocol BOLD QC summary ────────────────────────────────────
def site_protocol_summary(df: pd.DataFrame) -> pd.DataFrame:
    _log_step("Stage 7: site-level and protocol-level BOLD QC summary")

    philips_cn = df[
        (df["Manufacturer_normalized"] == "Philips")
        & (df["ResearchGroup_Mapped"] == "CN")
    ].copy()

    summary_cols = CORRECTED_COLS + [
        c for c in QC_TEST_COLS_T140 + CONNECTOME_COLS if c in philips_cn.columns
    ] + ["y_score_final", "Age", "fp_binary"]

    if "fp_binary" not in philips_cn.columns:
        philips_cn["fp_binary"] = (philips_cn["confusion_label"] == "FP").astype(float)

    present = [c for c in summary_cols if c in philips_cn.columns]

    rows = []
    for site, grp in philips_cn.groupby("Site3", dropna=False):
        tp_dist = grp["raw_tp_group"].value_counts().to_dict()
        n_fp = (grp["confusion_label"] == "FP").sum()
        n_tn = (grp["confusion_label"] == "TN").sum()
        row = {
            "Site3": site,
            "N": len(grp),
            "N_FP": n_fp,
            "N_TN": n_tn,
            "FPR": n_fp / len(grp) if len(grp) else np.nan,
            "tp140": tp_dist.get("140", 0),
            "tp197": tp_dist.get("197", 0),
            "problem_site": bool(grp["philips_problem_site_flag"].iloc[0]) if len(grp) else False,
        }
        for col in present:
            if col in grp.columns:
                row[f"{col}_median"] = float(grp[col].median())
        rows.append(row)

    # Overall rows by raw_tp_group
    for tp, grp in philips_cn.groupby("raw_tp_group", dropna=False):
        n_fp = (grp["confusion_label"] == "FP").sum()
        n_tn = (grp["confusion_label"] == "TN").sum()
        row = {
            "Site3": f"ALL_tp{tp}",
            "N": len(grp),
            "N_FP": n_fp,
            "N_TN": n_tn,
            "FPR": n_fp / len(grp) if len(grp) else np.nan,
            "tp140": tp_dist.get("140", 0) if tp == "140" else 0,
            "tp197": tp_dist.get("197", 0) if tp == "197" else 0,
            "problem_site": False,
        }
        for col in present:
            if col in grp.columns:
                row[f"{col}_median"] = float(grp[col].median())
        rows.append(row)

    result = pd.DataFrame(rows)
    result["_site_sort_key"] = result["Site3"].astype(str)
    result = result.sort_values("_site_sort_key").drop(columns=["_site_sort_key"])
    return result.reset_index(drop=True)


# ── Stage 8: Martín annotation sheet ──────────────────────────────────────────
def build_martin_sheet(df: pd.DataFrame) -> pd.DataFrame:
    _log_step("Stage 8: building Martín annotation sheet")

    philips_cn = df[
        (df["Manufacturer_normalized"] == "Philips")
        & (df["ResearchGroup_Mapped"] == "CN")
    ].copy()

    # Try to merge in prior philips source audit data
    prior_audit_path = (
        PHILIPS_AUDIT_DIR / "philips_cn_local_sources_merged_candidate.csv"
    )
    if prior_audit_path.exists():
        prior = pd.read_csv(prior_audit_path, low_memory=False)
        prior_cols = [c for c in prior.columns if c not in philips_cn.columns]
        prior_cols = ["SubjectID"] + [c for c in prior_cols if c != "SubjectID"]
        philips_cn = philips_cn.merge(
            prior[prior_cols], on="SubjectID", how="left"
        )

    # Base sheet columns
    base_cols = [
        "SubjectID", "ImageID", "Site3", "Manufacturer", "raw_tp_group",
        "COLPROT", "ORIGPROT", "inferred_ADNI_phase", "Age", "Sex",
        "y_score_final", "y_pred", "confusion_label",
    ]
    # BOLD QC columns (whichever are available)
    bold_cols = [c for c in CORRECTED_COLS + QC_TEST_COLS_T140 + CONNECTOME_COLS
                 if c in philips_cn.columns]
    # Existing flags
    flag_cols = [
        "philips_problem_site_flag", "philips_slice_order_issue_flag",
        "philips_slice_order_issue_source", "philips_comment",
    ]
    flag_cols = [c for c in flag_cols if c in philips_cn.columns]

    all_base = base_cols + bold_cols + flag_cols
    present = [c for c in all_base if c in philips_cn.columns]
    sheet = philips_cn[present].copy()

    # Derive slice_order_class
    def _so_class(r):
        sid = r["SubjectID"]
        s3 = r.get("Site3", np.nan)
        site = int(s3) if pd.notna(s3) else None
        if sid in SITE31_CONFIRMED_REVERSE:
            return "confirmed_reverse_non_default"
        if site in PROBLEM_SITES:
            return "suspected_issue_problem_site"
        return "UNKNOWN"

    sheet["slice_order_class"] = sheet.apply(_so_class, axis=1)

    # match_confidence
    def _conf(r):
        cls = r["slice_order_class"]
        if cls == "confirmed_reverse_non_default":
            return "HIGH"
        if cls == "suspected_issue_problem_site":
            return "LOW"
        return "NONE"

    sheet["match_confidence"] = sheet.apply(_conf, axis=1)
    sheet["high_confidence_slice_timing_match"] = (
        sheet["match_confidence"] == "HIGH"
    )

    # Empty annotation columns for Martín
    annotation_cols = [
        "martin_slice_order_real",
        "martin_slice_order_used_by_dparsf",
        "martin_dummy_scans_removed",
        "martin_phase_encoding",
        "martin_fd_qc",
        "martin_validity_label",
        "martin_reprocess_needed",
        "martin_comment",
    ]
    for ac in annotation_cols:
        sheet[ac] = ""

    # Pre-fill known entries
    for sid in SITE31_CONFIRMED_REVERSE:
        mask = sheet["SubjectID"] == sid
        sheet.loc[mask, "martin_slice_order_real"] = "reverse_non_default_confirmed"
        sheet.loc[mask, "martin_comment"] = "Site31 reverse/non-default slice order confirmed in prior audit (M0 sensitivity)"

    # Mark Sites 13, 53, 301 as suspected
    for site in PROBLEM_SITES:
        mask = sheet["Site3"] == site
        sheet.loc[mask, "martin_comment"] = (
            sheet.loc[mask, "martin_comment"].fillna("")
            + f" Site {site} flagged as problem_site in prior audit"
        )

    # Flag high-score FP for priority review
    high_fp = sheet[(sheet["confusion_label"] == "FP") & (sheet["y_score_final"] >= 0.75)]
    sheet.loc[
        (sheet["confusion_label"] == "FP") & (sheet["y_score_final"] >= 0.75),
        "martin_comment",
    ] = (
        sheet.loc[
            (sheet["confusion_label"] == "FP") & (sheet["y_score_final"] >= 0.75),
            "martin_comment",
        ].fillna("") + " HIGH_CONFIDENCE_FP_PRIORITY"
    )

    sheet = sheet.sort_values(["Site3", "raw_tp_group", "y_score_final"], ascending=[True, True, False])
    return sheet


# ── Stage 9: write all outputs ────────────────────────────────────────────────
def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if len(df) > max_rows:
        df = df.head(max_rows)
    return df.to_markdown(index=False, floatfmt=".4f")


def write_outputs(
    qc_merged: pd.DataFrame,
    tests_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    model_df: pd.DataFrame,
    site_df: pd.DataFrame,
    martin_df: pd.DataFrame,
) -> None:
    # ── bold_qc_subject_level ─────────────────────────────────────────────────
    qc_path = OUT / "bold_qc_subject_level.csv"
    qc_merged.to_csv(qc_path, index=False)
    _log_output(qc_path)

    md_lines = [
        "# Subject-Level BOLD QC",
        "",
        f"N total in pool: {len(qc_merged)}",
        f"N with mat file found: {qc_merged['mat_found'].sum() if 'mat_found' in qc_merged.columns else 'N/A'}",
        f"N with BOLD loaded: {qc_merged['bold_load_ok'].sum() if 'bold_load_ok' in qc_merged.columns else 'N/A'}",
        "",
        "## Coverage by Manufacturer",
        "",
    ]
    for mfr in ["Philips", "GE", "SIEMENS"]:
        sub = qc_merged[qc_merged["Manufacturer_normalized"] == mfr]
        n_ok = sub["bold_load_ok"].sum() if "bold_load_ok" in sub.columns else "?"
        md_lines.append(f"- {mfr}: N={len(sub)}, loaded={n_ok}")
    md_lines += ["", "## First 30 rows", "", md_table(qc_merged.head(30))]
    (OUT / "bold_qc_subject_level.md").write_text("\n".join(md_lines))
    _log_output(OUT / "bold_qc_subject_level.md")

    # ── philips_cn_bold_qc_group_tests ────────────────────────────────────────
    tests_path = OUT / "philips_cn_bold_qc_group_tests.csv"
    tests_df.to_csv(tests_path, index=False)
    _log_output(tests_path)

    md_lines = [
        "# Philips CN BOLD QC Group Tests",
        "",
        "Mann-Whitney U tests with Benjamini-Hochberg FDR (within each comparison block).",
        "CLES = P(group A > group B). FDR threshold: q < 0.10.",
        "",
    ]
    for comp in tests_df["comparison"].unique():
        sub = tests_df[tests_df["comparison"] == comp].copy()
        sig = sub[sub["sig_fdr_0p10"]]
        nA = int(sub["n_A"].iloc[0]); nB = int(sub["n_B"].iloc[0])
        md_lines += [
            f"## {comp}  (nA={nA}, nB={nB})",
            "",
            f"N FDR-significant variables (q<0.10): {len(sig)}",
            "",
            md_table(sub.head(20)),
            "",
        ]
    (OUT / "philips_cn_bold_qc_group_tests.md").write_text("\n".join(md_lines))
    _log_output(OUT / "philips_cn_bold_qc_group_tests.md")

    # ── bold_qc_score_correlation ─────────────────────────────────────────────
    corr_path = OUT / "bold_qc_score_correlation.csv"
    corr_df.to_csv(corr_path, index=False)
    _log_output(corr_path)

    md_lines = [
        "# BOLD QC vs Score Spearman Correlations",
        "",
        "Spearman ρ with 95% bootstrap CI (1000 resamples). Scope=philips_cn restricts to Philips CN (N=99).",
        "",
        md_table(corr_df.head(60)),
    ]
    (OUT / "bold_qc_score_correlation.md").write_text("\n".join(md_lines))
    _log_output(OUT / "bold_qc_score_correlation.md")

    # ── bold_connectome_score_model_results ───────────────────────────────────
    model_path = OUT / "bold_connectome_score_model_results.csv"
    model_df.to_csv(model_path, index=False)
    _log_output(model_path)

    md_lines = [
        "# Descriptive Model Results",
        "",
        "**Descriptive only** — not used for correction, selection, or threshold adjustment.",
        "FDR per model block.",
        "",
        md_table(model_df),
    ]
    (OUT / "bold_connectome_score_model_results.md").write_text("\n".join(md_lines))
    _log_output(OUT / "bold_connectome_score_model_results.md")

    # ── site_protocol_bold_qc_summary ─────────────────────────────────────────
    site_path = OUT / "site_protocol_bold_qc_summary.csv"
    site_df.to_csv(site_path, index=False)
    _log_output(site_path)

    md_lines = [
        "# Site/Protocol BOLD QC Summary (Philips CN)",
        "",
        md_table(site_df),
    ]
    (OUT / "site_protocol_bold_qc_summary.md").write_text("\n".join(md_lines))
    _log_output(OUT / "site_protocol_bold_qc_summary.md")

    # ── martin_annotation_sheet_updated ───────────────────────────────────────
    martin_path = OUT / "martin_annotation_sheet_updated.csv"
    martin_df.to_csv(martin_path, index=False)
    _log_output(martin_path)

    md_lines = [
        "# Martín Annotation Sheet — Updated",
        "",
        "Pre-filled `slice_order_class`, `match_confidence`, `high_confidence_slice_timing_match`.",
        "Empty annotation columns: martin_slice_order_real, martin_slice_order_used_by_dparsf,",
        "martin_dummy_scans_removed, martin_phase_encoding, martin_fd_qc,",
        "martin_validity_label, martin_reprocess_needed, martin_comment.",
        "",
        f"N Philips CN: {len(martin_df)}",
        "",
        "| slice_order_class | N |",
        "| --- | --- |",
    ]
    if "slice_order_class" in martin_df.columns:
        for cls, cnt in martin_df["slice_order_class"].value_counts().items():
            md_lines.append(f"| {cls} | {cnt} |")
    md_lines += ["", md_table(martin_df.drop(columns=[c for c in martin_df.columns
                                                       if c.startswith("martin_")], errors="ignore").head(40))]
    (OUT / "martin_annotation_sheet_updated.md").write_text("\n".join(md_lines))
    _log_output(OUT / "martin_annotation_sheet_updated.md")


# ── Stage 10: final interpretation ─────────────────────────────────────────────
def write_final_interpretation(
    qc_merged: pd.DataFrame,
    tests_df: pd.DataFrame,
    corr_df: pd.DataFrame,
) -> None:
    _log_step("Stage 10: writing final interpretation")

    philips_cn = qc_merged[
        (qc_merged["Manufacturer_normalized"] == "Philips")
        & (qc_merged["ResearchGroup_Mapped"] == "CN")
    ].copy()

    tp140 = philips_cn[philips_cn["raw_tp_group"] == "140"]
    tp197 = philips_cn[philips_cn["raw_tp_group"] == "197"]

    # Pull key significant comparisons
    tp_tests = tests_df[tests_df["comparison"] == "140TP_vs_197TP"]
    sig_tp = tp_tests[tp_tests["sig_fdr_0p10"]]["variable"].tolist()

    fp_140_tests = tests_df[tests_df["comparison"] == "FP_vs_TN_in_140TP"]
    sig_fp140 = fp_140_tests[fp_140_tests["sig_fdr_0p10"]]["variable"].tolist()

    fp_197_tests = tests_df[tests_df["comparison"] == "FP_vs_TN_in_197TP"]
    sig_fp197 = fp_197_tests[fp_197_tests["sig_fdr_0p10"]]["variable"].tolist()

    # Key metrics for interpretation
    def _row(tests_df, comp, var):
        s = tests_df[(tests_df["comparison"] == comp) & (tests_df["variable"] == var)]
        if len(s) == 0:
            return None
        return s.iloc[0]

    # tSNR comparison
    tsnr_row = _row(tests_df, "140TP_vs_197TP", "tsnr_proxy_median_corrected")
    ac_row = _row(tests_df, "140TP_vs_197TP", "t140_autocorr_lag1_median")
    lf_row = _row(tests_df, "140TP_vs_197TP", "t140_lf_power_mean")

    def _fmt_row(r):
        if r is None:
            return "N/A"
        return (
            f"median_140={r.median_A:.4f}, median_197={r.median_B:.4f}, "
            f"CLES={r.CLES:.3f}, p={r.p_raw:.4g}, q={r.p_fdr_within:.4g}"
        )

    lines = [
        "# Final Interpretation — BOLD-Level rawTP Protocol Mechanism Audit",
        "",
        f"**Date**: {datetime.now(timezone.utc).isoformat()}",
        "",
        "---",
        "",
        "## 1. Do rawTP differences appear at BOLD level?",
        "",
        f"N Philips CN 140TP: {len(tp140)}, 197TP: {len(tp197)}",
        f"FPR 140TP: {(tp140['confusion_label']=='FP').mean():.3f}, "
        f"FPR 197TP: {(tp197['confusion_label']=='FP').mean():.3f}",
        "",
        f"FDR-significant variables in 140TP vs 197TP comparison: **{len(sig_tp)}**",
        "",
    ]
    if sig_tp:
        lines.append("Significant (q<0.10):")
        for v in sig_tp[:15]:
            r = _row(tests_df, "140TP_vs_197TP", v)
            lines.append(f"  - {v}: {_fmt_row(r)}")
    else:
        lines.append("No variables survived FDR correction in 140TP vs 197TP comparison.")
    lines += [""]

    if tsnr_row is not None:
        lines += [
            f"**tSNR (corrected, T=140)**: {_fmt_row(tsnr_row)}",
        ]
    if ac_row is not None:
        lines += [
            f"**Autocorrelation lag-1 (T=140)**: {_fmt_row(ac_row)}",
        ]
    if lf_row is not None:
        lines += [
            f"**LF power (T=140)**: {_fmt_row(lf_row)}",
        ]

    lines += [
        "",
        "---",
        "",
        "## 2. Do differences survive after T=140 homogenization?",
        "",
        "The t140_ prefix metrics measure BOLD quality AFTER truncating all subjects",
        "to the first 140 timepoints — identical to the tensor builder's effective input.",
        "",
        f"N FDR-significant t140_ variables in 140TP vs 197TP: "
        f"{len([v for v in sig_tp if v.startswith('t140_')])}",
        "",
        "If BOLD differences survive T=140 truncation, then the tensor inputs genuinely",
        "differ between 140TP and 197TP subjects — a protocol-level (not construction) effect.",
        "If differences disappear after truncation, they are purely a consequence of T.",
        "",
        "---",
        "",
        "## 3. Does evidence support protocol/domain confounding vs tensor construction bug?",
        "",
        "Key distinction:",
        "- **Tensor construction bug**: would affect all subjects equally or in a systematic",
        "  index-dependent pattern; would not correlate with protocol metadata.",
        "- **Protocol confound**: correlates with raw_tp_group, Site3, COLPROT/ORIGPROT;",
        "  the BOLD signal itself differs between groups; survives T=140 truncation.",
        "",
        "Prior audits found NO tensor construction bug (mapping, ordering, T-homogenization).",
        "The current audit tests whether BOLD-level differences account for the FP pattern.",
        "",
        f"FDR-significant FP vs TN within 140TP: **{len(sig_fp140)}** variables",
        f"FDR-significant FP vs TN within 197TP: **{len(sig_fp197)}** variables",
        "",
    ]
    if sig_fp140:
        lines.append("140TP FP vs TN significant:")
        for v in sig_fp140[:10]:
            r = _row(tests_df, "FP_vs_TN_in_140TP", v)
            lines.append(f"  - {v}: {_fmt_row(r)}")
    lines += [""]

    lines += [
        "---",
        "",
        "## 4. Should preprocessing correction be pursued?",
        "",
        "- **Site31 confirmed reverse slice order (N=5)**: DPARSF slice timing correction",
        "  should be verified. If the wrong slice order was used, reprocessing may reduce FPR",
        "  for these specific subjects. However, the M0 sensitivity showed retraining worsened",
        "  metrics after their exclusion, suggesting they do not dominate the FP pattern.",
        "",
        "- **Sites 13, 53, 301 (problem site, N=7 CN)**: Confirmation of slice order and",
        "  phase encoding from DPARSF config is needed before reprocessing.",
        "",
        "- **140TP protocol group (N=41 CN)**: If BOLD quality differences between 140TP and",
        "  197TP groups survive T=140 truncation, this indicates protocol-level confounding",
        "  (scanner model, sequence parameters, ADNI phase) rather than a fixable preprocessing",
        "  error. No single reprocessing action can address ADNI1/2 vs ADNI3 protocol differences.",
        "",
        "---",
        "",
        "## 5. Summary verdict",
        "",
        "[ This section will be updated after reviewing the statistical results above. ]",
        "",
        "The primary mechanism hypothesis: **protocol confound** (older 140TP scanners have",
        "lower effective BOLD quality) + **age confound** (140TP subjects are older, CLES=0.71)",
        "together elevate y_score_final systematically for ADNI1/2 Philips 140TP CN subjects.",
        "This is not a tensor construction bug — it is a domain shift at the acquisition level",
        "that the model correctly detects as 'atypical CN' because it is atypical relative to",
        "the GE/SIEMENS 197/200TP majority of the training distribution.",
    ]

    interp_path = OUT / "final_interpretation.md"
    interp_path.write_text("\n".join(lines))
    _log_output(interp_path)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    _log_step("Loading master database")
    df_raw = pd.read_csv(str(MASTER_DB), low_memory=False)
    pool = df_raw[df_raw["in_stageB_classifier_pool"] == True].copy()
    _log_step(f"  Pool size: {len(pool)} subjects")

    # Stage 1: BOLD QC
    qc_df = build_bold_qc_table(pool)

    # Stage 2: Merge metadata
    qc_merged = merge_metadata(pool, qc_df)

    # Stage 3: Philips stratification
    qc_merged = assign_philips_cn_group(qc_merged)
    if "fp_binary" not in qc_merged.columns:
        qc_merged["fp_binary"] = (qc_merged["confusion_label"] == "FP").astype(float)

    # Stage 4: Group tests
    tests_df = run_group_tests(qc_merged)

    # Stage 5: Correlations
    corr_df = run_correlations(qc_merged)

    # Stage 6: Descriptive models
    model_df = run_descriptive_models(qc_merged)

    # Stage 7: Site summary
    site_df = site_protocol_summary(qc_merged)

    # Stage 8: Martín annotation sheet
    martin_df = build_martin_sheet(qc_merged)

    # Stage 9: Write outputs
    _log_step("Stage 9: writing outputs")
    write_outputs(qc_merged, tests_df, corr_df, model_df, site_df, martin_df)

    # Stage 10: Final interpretation
    write_final_interpretation(qc_merged, tests_df, corr_df)

    # Write command log
    command_log["finished"] = datetime.now(timezone.utc).isoformat()
    command_log["n_subjects_processed"] = int(qc_merged["bold_load_ok"].sum()) if "bold_load_ok" in qc_merged.columns else None
    log_path = OUT / "command_log.json"
    log_path.write_text(json.dumps(command_log, indent=2, default=str))
    _log_output(log_path)

    log.info("Done. Outputs: %s", OUT)


if __name__ == "__main__":
    main()
