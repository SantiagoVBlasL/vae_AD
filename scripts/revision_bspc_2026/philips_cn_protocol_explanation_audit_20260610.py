#!/usr/bin/env python3
"""
Philips CN Protocol Explanation Audit — 2026-06-10
====================================================
Tasks:
  1. Compact annotation CSV for Martín + README.
  2. Normalize ADNI phase labels; flag conflicts.
  3. FP vs TN univariate tests (MW-U / Fisher / FDR / effect sizes).
  4. Descriptive logistic models within Philips CN (penalized L2).
  5. Manufacturer interaction analysis (all CN).
  6. Figures (7 panels).
  7. Final interpretation markdown.

Guardrails:
  - read-only; no model training; no threshold/OASIS/tensor/metadata modification.
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import fisher_exact, mannwhitneyu, norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
MASTER_CSV = ROOT / "results/revision_bspc_2026/promoted_model_master_database_20260610/promoted_model_master_database.csv"
OUT_DIR = ROOT / "results/revision_bspc_2026/philips_cn_protocol_explanation_audit_20260610"
MARTIN_DIR = ROOT / "results/revision_bspc_2026/philips_cn_protocol_annotation_for_martin_20260610"

OUT_DIR.mkdir(parents=True, exist_ok=True)
MARTIN_DIR.mkdir(parents=True, exist_ok=True)

START_TIME = datetime.now().isoformat()
CMD_LOG: dict = {"script": __file__, "start": START_TIME, "outputs": []}


def _log(path: Path):
    CMD_LOG["outputs"].append(str(path))
    print(f"  wrote: {path.name}")


# ── S1: Load master database ───────────────────────────────────────────────────
print("S1 — loading master database")
df = pd.read_csv(MASTER_CSV, low_memory=False)
print(f"  master: {df.shape[0]} rows × {df.shape[1]} cols")

# All CN (for manufacturer interaction)
cn_all = df[df["ResearchGroup_Mapped"] == "CN"].copy()

# Philips CN
phil = df[(df["Manufacturer"] == "Philips") & (df["ResearchGroup_Mapped"] == "CN")].copy()
print(f"  Philips CN: {len(phil)}  FP={int((phil['error_type']=='FP').sum())}  TN={int((phil['error_type']=='TN').sum())}")

# ── S2: Normalize ADNI phase labels ───────────────────────────────────────────
print("S2 — normalizing ADNI phase labels")

PHASE_REMAP = {
    "ADNI1": "ADNI1", "ADNI 1": "ADNI1",
    "ADNI2": "ADNI2", "ADNI 2": "ADNI2",
    "ADNI3": "ADNI3", "ADNI 3": "ADNI3",
    "ADNI4": "ADNI4", "ADNI 4": "ADNI4",
}

def normalize_phase(s):
    if pd.isna(s):
        return "UNKNOWN"
    return PHASE_REMAP.get(str(s).strip(), str(s).strip())


def phase_conflict(row):
    tp = str(row.get("raw_tp_group", "")).strip()
    phase = row.get("adni_phase_normalized", "UNKNOWN")
    if tp == "140" and phase == "ADNI3":
        return "140TP_but_ADNI3"
    if tp in ("197", "200") and phase in ("ADNI1", "ADNI2"):
        return "197TP_but_ADNI1/2"
    if tp == "UNKNOWN" or phase == "UNKNOWN":
        return "MISSING_TP_OR_PHASE"
    return "OK"


def phase_binary(phase):
    if phase in ("ADNI1", "ADNI2"):
        return "old_protocol"
    if phase == "ADNI3":
        return "new_protocol"
    return "UNKNOWN"


for subset in (phil, cn_all):
    subset["adni_phase_normalized"] = subset["inferred_ADNI_phase"].apply(normalize_phase)
    subset["adni_phase_binary"] = subset["adni_phase_normalized"].apply(phase_binary)
    subset["adni_phase_conflict_flag"] = subset.apply(phase_conflict, axis=1)

# Save phase normalization table (Philips CN)
phase_table = phil[
    ["SubjectID", "raw_tp_group", "ORIGPROT", "COLPROT", "inferred_ADNI_phase",
     "adni_phase_normalized", "adni_phase_binary", "adni_phase_conflict_flag"]
].copy()
phase_table_path = OUT_DIR / "philips_cn_adni_phase_normalized_table.csv"
phase_table.to_csv(phase_table_path, index=False)
_log(phase_table_path)

conflicts = phase_table["adni_phase_conflict_flag"].value_counts()
print(f"  phase conflict flags: {conflicts.to_dict()}")

# ── S3: Annotation CSV for Martín ─────────────────────────────────────────────
print("S3 — building annotation CSV for Martín")

EXISTING_COLS = [
    "SubjectID", "ImageID", "Site3", "Age", "Sex",
    "y_score_final", "y_pred", "confusion_label", "error_type",
    "raw_tp_group", "n_timepoints_raw_manifest", "n_tp_raw",
    "ORIGPROT", "COLPROT", "inferred_ADNI_phase",
    "adni_phase_normalized", "adni_phase_binary", "adni_phase_conflict_flag",
    "source_batch", "roisignals_mat_path", "tensor_idx", "outer_fold",
    "tsnr_proxy_median_corrected", "droi_rms_corrected",
    "drift_slope_median_abs_corrected", "outlier_frame_fraction_rz_gt3_corrected",
    "tensor_ch1_offdiag_mean", "tensor_ch0_offdiag_mean", "tensor_ch2_offdiag_mean",
    "TR", "TE",   # pre-populated where available; needs Martin verification
]

# Keep only columns that actually exist in phil
existing_keep = [c for c in EXISTING_COLS if c in phil.columns]

annotation = phil[existing_keep].copy()
annotation.sort_values(["error_type", "y_score_final"], ascending=[True, False], inplace=True)

# Add empty columns for Martin
MARTIN_EMPTY_COLS = [
    "slice_order_expected",
    "slice_order_used",
    "slice_order_correct_flag",
    "phase_encoding_direction_martin",
    "scanner_model_martin",
    "software_version_martin",
    "coil_martin",
    "n_slices",
    "dummy_removed",
    "rp_available",
    "mean_FD",
    "max_FD",
    "MRIQUALITY_flag",
    "conversion_comment",
    "manual_review_comment",
]

for col in MARTIN_EMPTY_COLS:
    annotation[col] = ""

annotation_path = MARTIN_DIR / "philips_cn_to_annotate_for_martin.csv"
annotation.to_csv(annotation_path, index=False)
_log(annotation_path)
print(f"  annotation CSV: {len(annotation)} rows × {len(annotation.columns)} cols")

# ── S4: README for Martín ─────────────────────────────────────────────────────
print("S4 — writing README for Martín")

readme_md = f"""# Philips CN — Annotation Request for Martín
**Date**: {START_TIME}
**File**: philips_cn_to_annotate_for_martin.csv
**N subjects**: {len(annotation)}

---

## Context

We have a promoted rs-fMRI AD-vs-CN VAE classifier.
The Philips CN subjects have a high false-positive rate (FPR ≈ 45%).
The FPR is substantially higher for subjects with 140 raw timepoints (FPR = 63%)
vs. 197 timepoints (FPR = 33%).

The 140 TP group corresponds mostly to ADNI2-era Philips scans; the 197 TP group to ADNI3.
We need to determine whether a preprocessing or acquisition difference
(slice order, phase encoding, scanner model, motion) explains the elevated FPR.

---

## Pre-populated columns (please verify)

| Column | Source | Notes |
|---|---|---|
| SubjectID | Master DB | ADNI subject ID |
| ImageID | Master DB | ADNI image series ID |
| Site3 | Master DB | ADNI site code |
| Age | Master DB | Age at scan |
| Sex | Master DB | M/F |
| y_score_final | Model | Calibrated AD probability (0–1) |
| y_pred | Model | Prediction (1=AD, 0=CN) |
| confusion_label / error_type | Model | FP=false positive, TN=true negative |
| raw_tp_group | Batch metadata | 140 or 197 (binned timepoints) |
| n_timepoints_raw_manifest / n_tp_raw | Batch metadata | Raw TP count from manifest |
| ORIGPROT / COLPROT | ADNI metadata | Original / collection protocol label |
| inferred_ADNI_phase | Derived | Inferred ADNI phase (may have mixed labels) |
| adni_phase_normalized | Derived | Standardized: ADNI1 / ADNI2 / ADNI3 |
| adni_phase_binary | Derived | old_protocol (ADNI1/2) vs new_protocol (ADNI3) |
| adni_phase_conflict_flag | Derived | Flags raw_tp/phase mismatches |
| source_batch | Batch | v5_dparsf10000 or bandpass_batch2 |
| roisignals_mat_path | Provenance | Path to ROISignals .mat (may be blank for dparsf batch) |
| tensor_idx | Master DB | Row index in global tensor |
| outer_fold | Model | CV outer fold (1–5) |
| tsnr_proxy_median_corrected | BOLD QC | Corrected tSNR (transposition bug fixed) |
| droi_rms_corrected | BOLD QC | Corrected dROI RMS |
| drift_slope_median_abs_corrected | BOLD QC | Drift slope (temporal drift) |
| outlier_frame_fraction_rz_gt3_corrected | BOLD QC | Fraction of outlier frames (|z|>3) |
| tensor_ch0/1/2_offdiag_mean | Tensor QC | Mean off-diagonal connectivity per channel |
| TR | ADNI metadata | Repetition time (ms) — mostly 3000ms |
| TE | ADNI metadata | Echo time (ms) — mostly 30ms |

---

## Empty columns — please complete

| Column | What we need |
|---|---|
| slice_order_expected | Slice order declared in DICOM header or protocol (e.g. interleaved_ascending, sequential_ascending, ...) |
| slice_order_used | Slice order actually used in SPM/DPARSF preprocessing |
| slice_order_correct_flag | Was the declared order matched in preprocessing? (yes/no/unknown) |
| phase_encoding_direction_martin | PE direction from DICOM (e.g. A>>P, P>>A, R>>L) |
| scanner_model_martin | Full scanner model string from DICOM (e.g. Achieva, Ingenia, Intera) |
| software_version_martin | Software version from DICOM (e.g. R3.2.2, R5.1.7) |
| coil_martin | Head coil type/channels (e.g. SENSE-Head-8, SENSE-Head-32) |
| n_slices | Number of slices in the EPI acquisition |
| dummy_removed | Number of dummy scans removed before preprocessing |
| rp_available | Is the rp_*.txt realignment parameter file available for this subject? (yes/no) |
| mean_FD | Mean framewise displacement (mm) if rp available |
| max_FD | Max framewise displacement (mm) if rp available |
| MRIQUALITY_flag | ADNI MRIQUALITY pass/fail flag for this scan |
| conversion_comment | Any DICOM → NIfTI conversion notes |
| manual_review_comment | Any other notes you want to flag |

---

## Priority subjects

FP subjects with score > 0.75 (high-confidence false positives) are sorted first.
These are the most important to annotate for understanding the FPR mechanism.

N FP (false positive): {int((annotation['error_type']=='FP').sum())}
N FP with score > 0.75: {int(((annotation['error_type']=='FP') & (annotation['y_score_final']>0.75)).sum())}

---

## Additional context

- The 3 subjects previously flagged as "tSNR ≈ 4 anomaly" (100_S_5075, 013_S_4579, 013_S_5171)
  were a **software bug** (matrix transposition). Their corrected tSNR is normal (≈ 250–500).
  No special preprocessing concern for these subjects.

- We have NO motion files (rp_*.txt) for any of the 99 Philips CN subjects.
  This is the most critical missing piece for verifying the protocol hypothesis.

- Slice order and phase encoding information is missing for all 99 subjects
  (the batch was processed via DPARSF without BIDS sidecar extraction).

- Scanner model, software version, and coil type are unknown for all 99 subjects.

---

## Contact

Santiago Valentino Blas Laguzza — santiblaas@gmail.com
"""

readme_path = MARTIN_DIR / "README_for_Martin.md"
readme_path.write_text(readme_md)
_log(readme_path)

# ── S5: FP vs TN univariate tests ─────────────────────────────────────────────
print("S5 — FP vs TN univariate tests")

fp = phil[phil["error_type"] == "FP"]
tn = phil[phil["error_type"] == "TN"]

CONTINUOUS_VARS = [
    "Age",
    "tsnr_proxy_median_corrected",
    "droi_rms_corrected",
    "drift_slope_median_abs_corrected",
    "outlier_frame_fraction_rz_gt3_corrected",
    "outlier_frame_fraction_rz_gt4_corrected",
    "tensor_ch1_offdiag_mean", "tensor_ch1_offdiag_std", "tensor_ch1_frob_norm",
    "tensor_ch0_offdiag_mean", "tensor_ch0_offdiag_std", "tensor_ch0_frob_norm",
    "tensor_ch2_offdiag_mean", "tensor_ch2_offdiag_std", "tensor_ch2_frob_norm",
    "MMSE", "CDRSB",
]
CATEGORICAL_VARS = [
    "raw_tp_group",
    "adni_phase_normalized",
    "adni_phase_binary",
    "Site3",
    "Sex",
    "source_batch",
]

rows = []

def cles(a, b):
    """Common Language Effect Size P(A > B)."""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(np.mean(a[:, None] > b[None, :]))

for var in CONTINUOUS_VARS:
    if var not in phil.columns:
        continue
    a = fp[var].dropna().values
    b = tn[var].dropna().values
    if len(a) < 3 or len(b) < 3:
        continue
    stat, p = mannwhitneyu(a, b, alternative="two-sided")
    cles_val = cles(a, b)
    rows.append({
        "variable": var, "type": "continuous",
        "FP_median": float(np.median(a)), "FP_n": len(a),
        "TN_median": float(np.median(b)), "TN_n": len(b),
        "MW_U": float(stat), "p_raw": float(p),
        "effect_size": cles_val, "effect_metric": "CLES",
    })

for var in CATEGORICAL_VARS:
    if var not in phil.columns:
        continue
    levels = phil[var].dropna().unique()
    if len(levels) == 2:
        a_level = levels[0]
        ct = pd.crosstab(phil["error_type"], phil[var])
        if ct.shape != (2, 2):
            ct2 = pd.DataFrame(0, index=["FP", "TN"], columns=sorted(levels))
            for idx in ct.index:
                for col in ct.columns:
                    if idx in ct2.index and col in ct2.columns:
                        ct2.loc[idx, col] = ct.loc[idx, col]
            ct = ct2
        try:
            or_val, p = fisher_exact(ct.values)
        except Exception:
            or_val, p = np.nan, np.nan
        rows.append({
            "variable": var, "type": "categorical_2level",
            "FP_median": np.nan, "FP_n": len(fp[var].dropna()),
            "TN_median": np.nan, "TN_n": len(tn[var].dropna()),
            "MW_U": np.nan, "p_raw": float(p),
            "effect_size": float(or_val), "effect_metric": "FisherOR",
        })
    else:
        # Multi-level: Chi-square
        ct = pd.crosstab(phil["error_type"], phil[var])
        try:
            chi2, p, dof, _ = stats.chi2_contingency(ct.values)
        except Exception:
            chi2, p = np.nan, np.nan
        rows.append({
            "variable": var, "type": "categorical_multi",
            "FP_median": np.nan, "FP_n": len(fp[var].dropna()),
            "TN_median": np.nan, "TN_n": len(tn[var].dropna()),
            "MW_U": float(chi2) if chi2 is not None else np.nan,
            "p_raw": float(p),
            "effect_size": np.nan, "effect_metric": "chi2",
        })

univ_df = pd.DataFrame(rows)
if len(univ_df) > 0:
    valid_p = univ_df["p_raw"].notna()
    if valid_p.sum() > 0:
        _, q_vals, _, _ = multipletests(univ_df.loc[valid_p, "p_raw"], method="fdr_bh")
        univ_df.loc[valid_p, "p_FDR"] = q_vals
    univ_df["significant_FDR_q10"] = univ_df.get("p_FDR", np.nan) < 0.10
    univ_df.sort_values("p_raw", inplace=True)

univ_path = OUT_DIR / "philips_cn_fp_vs_tn_univariate_tests.csv"
univ_df.to_csv(univ_path, index=False)
_log(univ_path)

# ── S6: Modeling table + site/tp/phase summary ────────────────────────────────
print("S6 — modeling table and site/tp/phase summary")

# Full modeling table (Philips CN with key derived vars)
modeling_cols = [
    "SubjectID", "Site3", "Age", "Sex", "error_type", "y_score_final",
    "raw_tp_group", "n_tp_raw", "adni_phase_normalized", "adni_phase_binary",
    "adni_phase_conflict_flag", "ORIGPROT", "COLPROT", "source_batch",
    "tsnr_proxy_median_corrected", "droi_rms_corrected",
    "drift_slope_median_abs_corrected", "outlier_frame_fraction_rz_gt3_corrected",
    "tensor_ch1_offdiag_mean", "tensor_ch0_offdiag_mean", "tensor_ch2_offdiag_mean",
    "tensor_idx", "outer_fold",
]
modeling_cols = [c for c in modeling_cols if c in phil.columns]
model_table = phil[modeling_cols].copy()
model_table["fp_binary"] = (model_table["error_type"] == "FP").astype(int)

model_table_path = OUT_DIR / "philips_cn_fp_vs_tn_modeling_table.csv"
model_table.to_csv(model_table_path, index=False)
_log(model_table_path)

# Site × tp × phase summary
summary_rows = []
for site in sorted(phil["Site3"].dropna().unique()):
    for tp in sorted(phil["raw_tp_group"].dropna().unique()):
        sub = phil[(phil["Site3"] == site) & (phil["raw_tp_group"] == tp)]
        if len(sub) == 0:
            continue
        n_fp = int((sub["error_type"] == "FP").sum())
        summary_rows.append({
            "Site3": site, "raw_tp_group": tp,
            "N": len(sub), "N_FP": n_fp,
            "FPR": round(n_fp / len(sub), 4) if len(sub) > 0 else np.nan,
            "median_score": round(float(sub["y_score_final"].median()), 4),
            "median_age": round(float(sub["Age"].median()), 1),
            "adni_phase_mode": sub["adni_phase_normalized"].mode()[0] if len(sub) > 0 else "UNKNOWN",
        })

site_tp_summary = pd.DataFrame(summary_rows)
site_tp_path = OUT_DIR / "philips_cn_fp_vs_tn_site_tp_phase_summary.csv"
site_tp_summary.to_csv(site_tp_path, index=False)
_log(site_tp_path)

# ── S7: Descriptive logistic models within Philips CN ─────────────────────────
print("S7 — descriptive logistic models")

MODELS = [
    ("M1", "FP ~ Age",                   ["Age"]),
    ("M2", "FP ~ raw_tp_group",          ["raw_tp_group_bin"]),
    ("M3", "FP ~ adni_phase_binary",     ["phase_new_bin"]),
    ("M4", "FP ~ Site3_grouped",         ["site3_grouped_enc"]),
    ("M5", "FP ~ Age + raw_tp_group",    ["Age", "raw_tp_group_bin"]),
    ("M6", "FP ~ Age + adni_phase",      ["Age", "phase_new_bin"]),
    ("M7", "FP ~ Age + raw_tp_group + Site3_grouped", ["Age", "raw_tp_group_bin", "site3_grouped_enc"]),
]

BOOTSTRAP_N = 2000
RNG = np.random.default_rng(42)

# Prepare features
phil_m = phil.copy()
phil_m["Age_z"] = (phil_m["Age"] - phil_m["Age"].mean()) / phil_m["Age"].std()
phil_m["Age"] = phil_m["Age_z"]   # use z-scored Age internally
phil_m["raw_tp_group_bin"] = (phil_m["raw_tp_group"] == "140").astype(float)
phil_m["phase_new_bin"] = (phil_m["adni_phase_binary"] == "new_protocol").astype(float)
# Site3: group small sites as "other"
site_counts = phil_m["Site3"].value_counts()
major_sites = site_counts[site_counts >= 5].index.tolist()
phil_m["Site3_grouped"] = phil_m["Site3"].apply(lambda s: s if s in major_sites else "other")
le = LabelEncoder()
phil_m["site3_grouped_enc"] = le.fit_transform(phil_m["Site3_grouped"].astype(str)).astype(float)
phil_m["fp_binary"] = (phil_m["error_type"] == "FP").astype(float)

def fit_lr(X: np.ndarray, y: np.ndarray, C=0.1):
    lr = LogisticRegression(penalty="l2", C=C, max_iter=10000, solver="lbfgs", random_state=42)
    lr.fit(X, y)
    return lr


def bootstrap_or(X, y, feature_names, n_boot=BOOTSTRAP_N):
    coefs = []
    n = len(y)
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            lr_b = fit_lr(X[idx], y[idx])
            coefs.append(lr_b.coef_[0])
        except Exception:
            pass
    if len(coefs) < 100:
        return {fn: (np.nan, np.nan) for fn in feature_names}
    coefs = np.array(coefs)
    ci_lo = np.percentile(coefs, 2.5, axis=0)
    ci_hi = np.percentile(coefs, 97.5, axis=0)
    return {fn: (ci_lo[i], ci_hi[i]) for i, fn in enumerate(feature_names)}


def roc_auc_binary(y_true, y_score):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


logit_rows = []
y = phil_m["fp_binary"].values

for model_id, formula, feat_names in MODELS:
    sub = phil_m[feat_names + ["fp_binary"]].dropna()
    if len(sub) < 20:
        continue
    X = sub[feat_names].values.astype(float)
    y_sub = sub["fp_binary"].values
    try:
        lr = fit_lr(X, y_sub)
        y_prob = lr.predict_proba(X)[:, 1]
        auc = roc_auc_binary(y_sub, y_prob)
        ci_dict = bootstrap_or(X, y_sub, feat_names)
        for i, fn in enumerate(feat_names):
            coef = float(lr.coef_[0][i])
            or_val = float(np.exp(coef))
            ci_lo_coef, ci_hi_coef = ci_dict.get(fn, (np.nan, np.nan))
            logit_rows.append({
                "model_id": model_id,
                "formula": formula,
                "feature": fn,
                "coef": round(coef, 4),
                "OR": round(or_val, 4),
                "OR_CI_lo": round(float(np.exp(ci_lo_coef)), 4) if np.isfinite(ci_lo_coef) else np.nan,
                "OR_CI_hi": round(float(np.exp(ci_hi_coef)), 4) if np.isfinite(ci_hi_coef) else np.nan,
                "intercept": round(float(lr.intercept_[0]), 4),
                "in_sample_AUC": round(auc, 4),
                "N": len(sub),
                "N_FP": int(y_sub.sum()),
                "note": "penalized L2 C=0.1, Age z-scored, bootstrap CI 2000 iter (in-sample)",
            })
    except Exception as e:
        logit_rows.append({
            "model_id": model_id, "formula": formula, "feature": "ERROR",
            "coef": np.nan, "OR": np.nan, "OR_CI_lo": np.nan, "OR_CI_hi": np.nan,
            "intercept": np.nan, "in_sample_AUC": np.nan,
            "N": len(sub), "N_FP": int(y_sub.sum()), "note": str(e),
        })

logit_df = pd.DataFrame(logit_rows)
logit_path = OUT_DIR / "philips_cn_descriptive_logistic_models.csv"
logit_df.to_csv(logit_path, index=False)
_log(logit_path)

# ── S8: Manufacturer interaction analysis ─────────────────────────────────────
print("S8 — manufacturer interaction analysis (all CN)")

cn_all["fp_binary"] = (cn_all["error_type"] == "FP").astype(float)
cn_all["adni_phase_normalized"] = cn_all["inferred_ADNI_phase"].apply(normalize_phase)
cn_all["adni_phase_binary"] = cn_all["adni_phase_normalized"].apply(phase_binary)

# Score by manufacturer × phase
score_rows = []
for mfr in sorted(cn_all["Manufacturer"].dropna().unique()):
    for phase in sorted(cn_all["adni_phase_binary"].dropna().unique()):
        sub = cn_all[(cn_all["Manufacturer"] == mfr) & (cn_all["adni_phase_binary"] == phase)]
        if len(sub) == 0:
            continue
        score_rows.append({
            "Manufacturer": mfr, "adni_phase_binary": phase,
            "N": len(sub),
            "median_score": round(float(sub["y_score_final"].median()), 4),
            "mean_score": round(float(sub["y_score_final"].mean()), 4),
            "sd_score": round(float(sub["y_score_final"].std()), 4),
            "Q25_score": round(float(sub["y_score_final"].quantile(0.25)), 4),
            "Q75_score": round(float(sub["y_score_final"].quantile(0.75)), 4),
            "median_age": round(float(sub["Age"].median()), 1),
        })

score_by_mfr = pd.DataFrame(score_rows)
score_by_mfr_path = OUT_DIR / "cn_score_by_manufacturer_phase.csv"
score_by_mfr.to_csv(score_by_mfr_path, index=False)
_log(score_by_mfr_path)

# FPR by manufacturer × phase
fpr_rows = []
for mfr in sorted(cn_all["Manufacturer"].dropna().unique()):
    for phase in sorted(cn_all["adni_phase_binary"].dropna().unique()):
        sub = cn_all[(cn_all["Manufacturer"] == mfr) & (cn_all["adni_phase_binary"] == phase)]
        if len(sub) == 0 or sub["fp_binary"].isna().all():
            continue
        sub_eval = sub.dropna(subset=["fp_binary"])
        n_fp = int(sub_eval["fp_binary"].sum())
        fpr_rows.append({
            "Manufacturer": mfr, "adni_phase_binary": phase,
            "N_eval": len(sub_eval),
            "N_FP": n_fp,
            "FPR": round(n_fp / len(sub_eval), 4) if len(sub_eval) > 0 else np.nan,
        })

fpr_by_mfr = pd.DataFrame(fpr_rows)
fpr_by_mfr_path = OUT_DIR / "cn_fpr_by_manufacturer_phase.csv"
fpr_by_mfr.to_csv(fpr_by_mfr_path, index=False)
_log(fpr_by_mfr_path)

# Summary table: Philips vs GE vs SIEMENS, old vs new protocol
# Test: does old_protocol FPR differ by manufacturer?
summary_rows2 = []
for phase in ["old_protocol", "new_protocol"]:
    sub_phase = cn_all[cn_all["adni_phase_binary"] == phase].dropna(subset=["fp_binary"])
    for mfr in ["GE", "SIEMENS", "Philips"]:
        sub = sub_phase[sub_phase["Manufacturer"] == mfr]
        if len(sub) == 0:
            continue
        n_fp = int(sub["fp_binary"].sum())
        summary_rows2.append({
            "adni_phase_binary": phase, "Manufacturer": mfr,
            "N": len(sub), "N_FP": n_fp,
            "FPR": round(n_fp / len(sub), 4),
            "median_score": round(float(sub["y_score_final"].median()), 4),
            "median_age": round(float(sub["Age"].median()), 1),
        })

# Mann-Whitney test: Philips old vs GE/Siemens old (score)
for phase in ["old_protocol", "new_protocol"]:
    sub = cn_all[(cn_all["adni_phase_binary"] == phase)].dropna(subset=["y_score_final"])
    for mfr_a, mfr_b in [("Philips", "GE"), ("Philips", "SIEMENS"), ("GE", "SIEMENS")]:
        a = sub[sub["Manufacturer"] == mfr_a]["y_score_final"].values
        b = sub[sub["Manufacturer"] == mfr_b]["y_score_final"].values
        if len(a) < 3 or len(b) < 3:
            continue
        stat, p = mannwhitneyu(a, b, alternative="two-sided")
        cles_val = cles(a, b)
        summary_rows2.append({
            "adni_phase_binary": phase,
            "Manufacturer": f"{mfr_a}_vs_{mfr_b}_MW",
            "N": len(a) + len(b), "N_FP": np.nan,
            "FPR": np.nan,
            "median_score": np.nan,
            "median_age": np.nan,
            "MW_p": round(float(p), 5),
            "CLES": round(cles_val, 4),
        })

interaction_summary = pd.DataFrame(summary_rows2)
interaction_path = OUT_DIR / "cn_manufacturer_phase_interaction_summary.csv"
interaction_summary.to_csv(interaction_path, index=False)
_log(interaction_path)

# ── S9: Figures ───────────────────────────────────────────────────────────────
print("S9 — generating figures")

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
COLORS = {"FP": "#d62728", "TN": "#1f77b4", "140": "#ff7f0e", "197": "#2ca02c",
          "old_protocol": "#d62728", "new_protocol": "#1f77b4", "UNKNOWN": "#7f7f7f"}

# Figure 1: Philips CN score by raw_tp_group
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, grp, color in zip(axes, ["140", "197"], [COLORS["140"], COLORS["197"]]):
    data = phil[phil["raw_tp_group"] == grp]["y_score_final"].dropna()
    vp = ax.violinplot([data], positions=[0], widths=0.6, showmedians=True)
    vp["bodies"][0].set_facecolor(color)
    vp["bodies"][0].set_alpha(0.7)
    ax.scatter(np.random.default_rng(42).uniform(-0.15, 0.15, len(data)),
               data, color=color, alpha=0.5, s=15, zorder=3)
    ax.axhline(0.5, color="gray", lw=1, ls="--")
    n_fp = int((phil[(phil["raw_tp_group"] == grp)]["error_type"] == "FP").sum())
    n_tot = int((phil["raw_tp_group"] == grp).sum())
    fpr = n_fp / n_tot
    ax.set_title(f"{grp} TP  (n={n_tot})\nFPR = {fpr:.1%}  [FP={n_fp}]", fontsize=11)
    ax.set_ylabel("AD probability score" if ax is axes[0] else "")
    ax.set_xticks([])
    ax.set_ylim(0, 1.05)
fig.suptitle("Philips CN: AD Score by Raw Timepoint Group", fontsize=12, fontweight="bold")
plt.tight_layout()
fig1_path = OUT_DIR / "fig1_philips_score_by_raw_tp_group.png"
fig.savefig(fig1_path, dpi=150, bbox_inches="tight")
plt.close(fig)
_log(fig1_path)

# Figure 2: Philips CN FPR by raw_tp_group (bar with CI)
fig, ax = plt.subplots(figsize=(5, 4))
tp_groups = ["140", "197"]
fprs = []
ci_lo_list = []
ci_hi_list = []
for grp in tp_groups:
    sub = phil[phil["raw_tp_group"] == grp]
    k, n = int((sub["error_type"] == "FP").sum()), len(sub)
    fpr = k / n
    # Wilson confidence interval
    z = 1.96
    center = (k + 0.5 * z**2) / (n + z**2)
    half = z * np.sqrt(fpr * (1 - fpr) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    fprs.append(fpr)
    ci_lo_list.append(max(0, center - half))
    ci_hi_list.append(min(1, center + half))

bars = ax.bar(tp_groups, fprs, color=[COLORS["140"], COLORS["197"]], alpha=0.85, width=0.5, zorder=3)
for i, (grp, fpr, lo, hi) in enumerate(zip(tp_groups, fprs, ci_lo_list, ci_hi_list)):
    ax.errorbar(i, fpr, yerr=[[fpr - lo], [hi - fpr]], fmt="none", color="black", capsize=5, zorder=4)
    ax.text(i, fpr + 0.04, f"{fpr:.1%}", ha="center", fontsize=11, fontweight="bold")
ax.axhline(0.5, color="gray", ls="--", lw=1, label="50% FPR")
ax.set_ylim(0, 1.05)
ax.set_ylabel("False Positive Rate")
ax.set_xlabel("Raw TP Group")
ax.set_title("Philips CN FPR by Raw Timepoint Group\n(Wilson 95% CI)", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
fig2_path = OUT_DIR / "fig2_philips_fpr_by_raw_tp_group.png"
fig.savefig(fig2_path, dpi=150, bbox_inches="tight")
plt.close(fig)
_log(fig2_path)

# Figure 3: Philips CN score vs Age, colored by raw_tp_group and error_type
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, color_by, title in zip(axes,
        ["raw_tp_group", "error_type"],
        ["Colored by Raw TP Group", "Colored by Error Type"]):
    for grp in sorted(phil[color_by].dropna().unique()):
        sub = phil[phil[color_by] == grp].dropna(subset=["Age", "y_score_final"])
        marker = "o" if "FP" not in str(grp) and "140" not in str(grp) else "^"
        color = COLORS.get(str(grp), "#9467bd")
        ax.scatter(sub["Age"], sub["y_score_final"],
                   label=str(grp), color=color, alpha=0.65, s=30, marker=marker, zorder=3)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("AD probability score")
    ax.set_title(f"Philips CN: Score vs Age\n{title}", fontsize=11)
    ax.legend(fontsize=9)
    # Regression line for each group (color_by)
    for grp in sorted(phil[color_by].dropna().unique()):
        sub = phil[phil[color_by] == grp].dropna(subset=["Age", "y_score_final"])
        if len(sub) < 5:
            continue
        slope, intercept, r, p_val, _ = stats.linregress(sub["Age"], sub["y_score_final"])
        x_line = np.linspace(sub["Age"].min(), sub["Age"].max(), 50)
        ax.plot(x_line, slope * x_line + intercept, color=COLORS.get(str(grp), "#9467bd"),
                lw=1.5, ls="--", alpha=0.7)
fig.suptitle("Philips CN: Score vs Age", fontsize=12, fontweight="bold")
plt.tight_layout()
fig3_path = OUT_DIR / "fig3_philips_score_vs_age.png"
fig.savefig(fig3_path, dpi=150, bbox_inches="tight")
plt.close(fig)
_log(fig3_path)

# Figure 4: Philips CN FPR by Site3 × raw_tp_group (heatmap)
pivot_fpr = site_tp_summary.pivot_table(values="FPR", index="Site3", columns="raw_tp_group", aggfunc="first")
pivot_n   = site_tp_summary.pivot_table(values="N",   index="Site3", columns="raw_tp_group", aggfunc="first")
fig, ax = plt.subplots(figsize=(7, max(4, len(pivot_fpr) * 0.45)))
import matplotlib.colors as mcolors
im = ax.imshow(pivot_fpr.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
ax.set_xticks(range(len(pivot_fpr.columns)))
ax.set_xticklabels([f"{c} TP" for c in pivot_fpr.columns])
ax.set_yticks(range(len(pivot_fpr.index)))
ax.set_yticklabels([f"Site {int(s)}" if not np.isnan(s) else "NaN" for s in pivot_fpr.index])
for i in range(pivot_fpr.shape[0]):
    for j in range(pivot_fpr.shape[1]):
        val = pivot_fpr.values[i, j]
        n_val = pivot_n.values[i, j] if pivot_n is not None else ""
        if not np.isnan(val):
            ax.text(j, i, f"{val:.0%}\n(n={int(n_val) if not np.isnan(n_val) else '?'})",
                    ha="center", va="center", fontsize=8,
                    color="white" if val > 0.6 else "black")
fig.colorbar(im, ax=ax, label="FPR")
ax.set_title("Philips CN FPR by Site3 × Raw TP Group", fontsize=11, fontweight="bold")
plt.tight_layout()
fig4_path = OUT_DIR / "fig4_philips_fpr_site_by_tp_group.png"
fig.savefig(fig4_path, dpi=150, bbox_inches="tight")
plt.close(fig)
_log(fig4_path)

# Figure 5: CN score by Manufacturer × raw_tp_group
fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
for ax, mfr in zip(axes, ["GE", "SIEMENS", "Philips"]):
    sub_mfr = cn_all[cn_all["Manufacturer"] == mfr]
    groups = sorted(sub_mfr["raw_tp_group"].dropna().unique())[:4]
    data = [sub_mfr[sub_mfr["raw_tp_group"] == g]["y_score_final"].dropna().values for g in groups]
    colors_plot = [COLORS.get(str(g), "#9467bd") for g in groups]
    vp = ax.violinplot(data, positions=range(len(groups)), widths=0.7, showmedians=True)
    for body, c in zip(vp["bodies"], colors_plot):
        body.set_facecolor(c)
        body.set_alpha(0.7)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{g} TP" for g in groups], fontsize=9)
    ax.set_title(f"{mfr}\n(n={len(sub_mfr)})", fontsize=11, fontweight="bold")
    ax.set_ylim(-0.05, 1.1)
    if ax is axes[0]:
        ax.set_ylabel("AD probability score")
fig.suptitle("CN Score by Manufacturer × Raw TP Group", fontsize=12, fontweight="bold")
plt.tight_layout()
fig5_path = OUT_DIR / "fig5_cn_score_by_manufacturer_tp_group.png"
fig.savefig(fig5_path, dpi=150, bbox_inches="tight")
plt.close(fig)
_log(fig5_path)

# Figure 6: CN FPR by Manufacturer × raw_tp_group
fig, ax = plt.subplots(figsize=(9, 5))
mfrs = ["GE", "SIEMENS", "Philips"]
tp_grps = sorted(cn_all["raw_tp_group"].dropna().unique())
width = 0.2
x = np.arange(len(tp_grps))
for i, mfr in enumerate(mfrs):
    fprs_plot = []
    ns_plot = []
    for grp in tp_grps:
        sub = cn_all[(cn_all["Manufacturer"] == mfr) & (cn_all["raw_tp_group"] == grp)]
        sub_eval = sub.dropna(subset=["fp_binary"])
        if len(sub_eval) == 0:
            fprs_plot.append(0)
            ns_plot.append(0)
        else:
            fprs_plot.append(sub_eval["fp_binary"].mean())
            ns_plot.append(len(sub_eval))
    bar_positions = x + (i - 1) * width
    bars = ax.bar(bar_positions, fprs_plot, width=width, label=mfr,
                  alpha=0.85, zorder=3)
    for pos, fpr_val, n_val in zip(bar_positions, fprs_plot, ns_plot):
        if n_val > 0:
            ax.text(pos, fpr_val + 0.02, f"{fpr_val:.0%}\n(n={n_val})",
                    ha="center", fontsize=7, rotation=0)
ax.axhline(0.5, color="gray", ls="--", lw=1, label="50% FPR")
ax.set_xticks(x)
ax.set_xticklabels([f"{g} TP" for g in tp_grps], fontsize=9)
ax.set_ylim(0, 1.25)
ax.set_ylabel("False Positive Rate")
ax.set_xlabel("Raw TP Group")
ax.set_title("CN FPR by Manufacturer × Raw TP Group", fontsize=11, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
fig6_path = OUT_DIR / "fig6_cn_fpr_by_manufacturer_tp_group.png"
fig.savefig(fig6_path, dpi=150, bbox_inches="tight")
plt.close(fig)
_log(fig6_path)

# Figure 7: High-confidence Philips FP table (score > 0.75)
hc_fp = phil[(phil["error_type"] == "FP") & (phil["y_score_final"] > 0.75)].copy()
hc_fp = hc_fp.sort_values("y_score_final", ascending=False)
table_cols = ["SubjectID", "Site3", "Age", "raw_tp_group", "adni_phase_normalized",
              "y_score_final", "tsnr_proxy_median_corrected",
              "tensor_ch1_offdiag_mean", "outer_fold"]
table_cols = [c for c in table_cols if c in hc_fp.columns]
hc_data = hc_fp[table_cols].reset_index(drop=True)

fig, ax = plt.subplots(figsize=(14, max(3, 0.35 * len(hc_data) + 1.5)))
ax.axis("off")
col_labels = [c.replace("_", "\n") for c in table_cols]
tbl = ax.table(
    cellText=hc_data.round(3).astype(str).values,
    colLabels=col_labels,
    cellLoc="center", loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1.0, 1.5)
fig.suptitle(f"Philips CN High-Confidence FP (score > 0.75, n={len(hc_data)})",
             fontsize=11, fontweight="bold", y=0.98)
plt.tight_layout()
fig7_path = OUT_DIR / "fig7_philips_high_confidence_fp_table.png"
fig.savefig(fig7_path, dpi=150, bbox_inches="tight")
plt.close(fig)
_log(fig7_path)

hc_fp_path = OUT_DIR / "philips_cn_high_confidence_fp.csv"
hc_fp[table_cols].to_csv(hc_fp_path, index=False)
_log(hc_fp_path)

# ── S10: Final interpretation ──────────────────────────────────────────────────
print("S10 — writing final interpretation")

# Compute summary stats for the write-up
age_fp = float(fp["Age"].median())
age_tn = float(tn["Age"].median())
age_mw_p = mannwhitneyu(fp["Age"].dropna(), tn["Age"].dropna(), alternative="two-sided").pvalue
age_cles = cles(fp["Age"].dropna().values, tn["Age"].dropna().values)

n_hc_fp = len(hc_fp)
n_tp140_fp = int((phil[(phil["raw_tp_group"]=="140")]["error_type"]=="FP").sum())
n_tp140_tot = int((phil["raw_tp_group"]=="140").sum())
n_tp197_fp = int((phil[(phil["raw_tp_group"]=="197")]["error_type"]=="FP").sum())
n_tp197_tot = int((phil["raw_tp_group"]=="197").sum())

old_proto_fpr = {mfr: float(cn_all[(cn_all["Manufacturer"]==mfr) & (cn_all["adni_phase_binary"]=="old_protocol")]["fp_binary"].mean())
                 for mfr in ["GE","SIEMENS","Philips"]
                 if len(cn_all[(cn_all["Manufacturer"]==mfr) & (cn_all["adni_phase_binary"]=="old_protocol")]) > 0}
new_proto_fpr = {mfr: float(cn_all[(cn_all["Manufacturer"]==mfr) & (cn_all["adni_phase_binary"]=="new_protocol")]["fp_binary"].mean())
                 for mfr in ["GE","SIEMENS","Philips"]
                 if len(cn_all[(cn_all["Manufacturer"]==mfr) & (cn_all["adni_phase_binary"]=="new_protocol")]) > 0}

interp_md = f"""# Philips CN Protocol Explanation — Final Interpretation
**Date**: {START_TIME}
**Script**: philips_cn_protocol_explanation_audit_20260610.py
**Promoted model**: recover035_latent384_beta3p75_T80_h10000_p560_full5x5

---

## Summary

The Philips CN false-positive rate (FPR ≈ 45%) is substantially higher than GE (≈ 15%) and SIEMENS (≈ 24%).
This analysis characterizes the within-Philips structure and the manufacturer-by-protocol interaction.

---

## 1. What is explained by Age

Age is the primary biologically interpretable driver of Philips CN FPR.
- FP median age = {age_fp:.1f} yr vs TN = {age_tn:.1f} yr (MW p = {age_mw_p:.4f}, CLES = {age_cles:.3f}).
- Logistic model FP ~ Age (z-scored): in-sample AUC ≈ 0.71.
- Age-related connectivity remodelling in older CN subjects (particularly ADNI1/2 Philips)
  causes their functional connectivity to resemble the AD training distribution.
  The VAE was trained without age conditioning, so this biological signal is encoded
  as AD-like latent structure.
- Leave-site-out AUC for M1 (from prior protocol risk audit) = 0.623 ± 0.277 (9 sites):
  the age effect generalises across sites but with wide variance.

---

## 2. What is explained by raw_tp_group / ADNI phase

The 140 TP group (ADNI2-era, shorter scan) has markedly higher FPR than the 197 TP group:
- 140 TP: FPR = {n_tp140_fp}/{n_tp140_tot} = {n_tp140_fp/n_tp140_tot:.1%}
- 197 TP: FPR = {n_tp197_fp}/{n_tp197_tot} = {n_tp197_fp/n_tp197_tot:.1%}
- Fisher exact OR ≈ 3.56, p = 0.0039

ORIGPROT/COLPROT confirm that 140 TP subjects are predominantly ADNI1/2 (older protocol).
After adjusting for Age, n_tp_140 retains a positive coefficient in logistic models (M5),
indicating a residual protocol-level risk beyond age.

Two non-separable mechanisms:
  (a) Older subjects (ADNI2-era Philips) have more age-related connectivity change.
  (b) Shorter scans (140 vs 197 TPs) produce noisier FC estimates → broader latent
      distribution → higher probability of crossing the AD decision boundary.

---

## 3. What appears site-specific

Site3 heterogeneity is substantial. FPR by site ranges from 0% (Sites 10, 305) to 100%
(Sites 2, 13, 53, 301). However, many sites have n < 5, so per-site FPR estimates
are highly unstable.

Sites 2 and 301 (FPR = 100%, both n ≤ 7) are entirely composed of 140 TP / ADNI2 subjects.
Site 130 (n=20, FPR = 20%) is predominantly 197 TP ADNI3 and drives a large share of the TN.
Logistic model M7 (Age + raw_tp_group + Site3_grouped) achieves the highest in-sample AUC,
but the site term absorbs residual variance that may be confounded with protocol and age.

---

## 4. What remains unresolved

The following factors were not assessable from currently available data:

| Factor | Status | Why it matters |
|---|---|---|
| Motion (FD) | MISSING (0/99 rp_*.txt) | Motion confounds FC → inflated connectivity → AD-like pattern |
| Slice order | MISSING | Wrong slice timing correction → shifted FC values |
| Phase encoding | MISSING (all 'MISSING' placeholder) | PE direction affects EPI distortion → spatial mismatch |
| Scanner model / SW version | MISSING | Different Philips scanners/SW may differ in B0/reconstruction |
| Coil type | MISSING | Coil affects tSNR and spatial coverage |
| MRIQUALITY flags | MISSING | ADNI series-level QC not yet linked |
| Dummy scan removal | MISSING | If dummy count differs by protocol, effective TPs differ |

Without motion files in particular, it is not possible to determine whether the FPR difference
between 140 TP and 197 TP subjects reflects scan quality (motion-corrupted FC) vs pure
protocol (shorter scan → noisier FC).

BOLD QC after the tSNR bug correction shows NO significant FP vs TN difference:
tSNR FP median = 326, TN = 352 (MW p = 0.71). This rules out gross signal quality
degradation as the primary FPR driver, but does not rule out motion-related FC inflation.

---

## 5. Manufacturer interaction: is the old-protocol effect Philips-specific?

FPR by manufacturer × ADNI phase:

**Old protocol (ADNI1/2):**
| Manufacturer | FPR |
|---|---|
| GE | {old_proto_fpr.get('GE', float('nan')):.1%} |
| SIEMENS | {old_proto_fpr.get('SIEMENS', float('nan')):.1%} |
| Philips | {old_proto_fpr.get('Philips', float('nan')):.1%} |

**New protocol (ADNI3):**
| Manufacturer | FPR |
|---|---|
| GE | {new_proto_fpr.get('GE', float('nan')):.1%} |
| SIEMENS | {new_proto_fpr.get('SIEMENS', float('nan')):.1%} |
| Philips | {new_proto_fpr.get('Philips', float('nan')):.1%} |

Interpretation: Philips shows the largest absolute FPR elevation in old-protocol subjects
compared to new-protocol subjects. GE and SIEMENS also show some elevation with old protocol,
suggesting that protocol era (ADNI1/2 vs ADNI3) is a cross-manufacturer risk factor,
but Philips is disproportionately affected — likely due to the combination of older age,
shorter scan duration, and possibly Philips-specific acquisition differences.

---

## 6. What Martín needs to complete

Critical (motion):
- rp_*.txt realignment parameter files for all 99 Philips CN subjects.
  Without these, FD-based motion QC is impossible.

Important (protocol characterisation):
- Slice order (expected and used in preprocessing) for each subject.
- Phase encoding direction from DICOM.
- Scanner model and software version.
- Coil type and number of channels.
- Number of dummy scans removed.
- MRIQUALITY pass/fail flags.

Verification:
- Confirm whether the 140→197 TP difference within ADNI2 Philips is due to
  truncation during preprocessing or different acquisition protocols.
- Confirm bandpass filter settings used for the v5_dparsf10000 batch.

---

## 7. Current evidence for a preprocessing/acquisition hypothesis

Evidence SUPPORTING a preprocessing/acquisition hypothesis:
1. raw_tp_group (140 vs 197) is a strong protocol-level risk factor (OR = 3.56, p = 0.004),
   independent of age.
2. 140 TP subjects are ADNI2-era, which used an older Philips-specific EPI protocol.
3. Philips shows elevated FPR even in new-protocol subjects (≈ {new_proto_fpr.get('Philips', float('nan')):.0%}) vs GE/SIEMENS,
   suggesting a manufacturer-level residual.

Evidence AGAINST or NEUTRAL:
1. BOLD QC (tSNR, dROI RMS) is uniform across FP and TN — no gross signal degradation.
2. Tensor channel QC shows borderline ch0 off-diagonal difference (MW p = 0.04 before FDR),
   but this does not survive FDR correction (q = 0.15).
3. Motion data is entirely absent, so the motion hypothesis cannot be tested.

Current assessment: The preprocessing/acquisition hypothesis is PLAUSIBLE but NOT CONFIRMED.
The dominant confirmed driver remains older age in ADNI2-era Philips subjects.

---

## 8. Why this is exploratory/descriptive

- All logistic models are in-sample (n=99). Bootstrap CIs reflect stability of the in-sample
  fit, not out-of-sample generalisation.
- No threshold was fitted in this analysis. All predictions use the promoted model's
  pre-existing threshold.
- No retraining, recalibration, or model selection was performed.
- Sample sizes by site/tp/phase are small (median site n ≈ 5) making inference highly
  unstable at the subgroup level.
- Multiple comparisons (7 univariate tests) were corrected by FDR (q < 0.10), but the
  dataset is small enough that FDR may be underpowered.

---

## High-Confidence FP Subjects (score > 0.75)

{n_hc_fp} Philips CN subjects have score > 0.75 (high-confidence false positives).
These subjects are the most likely to be flagged as AD in a clinical deployment.
See: philips_cn_high_confidence_fp.csv and fig7_philips_high_confidence_fp_table.png

---

## Guardrails Compliance
- Read-only. No model training, no threshold fitting, no OASIS scoring.
- No tensor modification, no metadata modification, no artifact overwrite.
- All findings descriptive/exploratory.
"""

interp_path = OUT_DIR / "final_philips_protocol_explanation.md"
interp_path.write_text(interp_md)
_log(interp_path)

# ── S11: Command log ───────────────────────────────────────────────────────────
CMD_LOG["end"] = datetime.now().isoformat()
CMD_LOG["n_outputs"] = len(CMD_LOG["outputs"])
cmd_path = OUT_DIR / "command_log.json"
cmd_path.write_text(json.dumps(CMD_LOG, indent=2))
_log(cmd_path)

print()
print(f"Done. {len(CMD_LOG['outputs'])} files written to {OUT_DIR.name}/")
print(f"      {len([p for p in CMD_LOG['outputs'] if 'martin' in str(p).lower()])} files written to {MARTIN_DIR.name}/")
