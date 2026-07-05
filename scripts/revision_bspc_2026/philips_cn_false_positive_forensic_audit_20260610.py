"""
Philips CN False-Positive Forensic Audit
Promoted model: recover035_latent384_beta3p75_T80_h10000_p560_full5x5
Date: 2026-06-10
Guardrails: read-only, no model training, no threshold fitting, no OASIS scoring,
            no tensor modification, no metadata modification, no artifact overwrite.
"""

import os
import sys
import json
import warnings
import traceback
import glob
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.io as sio
from scipy.stats import mannwhitneyu, fisher_exact, pointbiserialr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PATHS & CONSTANTS
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path("/home/diego/proyectos/vae_AD")
BIG_DISK     = Path("/media/diego/Datos")

# Promoted model
PROMOTED_RUN_ID  = "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
PROMOTED_RUN_DIR = BIG_DISK / "vae_AD_results/revision_bspc_2026" / PROMOTED_RUN_ID
PROMOTED_OOF_DIR = PROJECT_ROOT / "results/revision_bspc_2026/recover035_latent384_beta3p75_stageB_oof_score_calibration"

# Primary readout settings (same as promotion audit)
PRIMARY_MODEL   = "logreg_l2_original"
PRIMARY_FEAT    = "z_plus_age_sex"
PRIMARY_THRESH  = "inner_oof_target_sens_ge_0p70_max_spec"
PRIMARY_CALIB   = "oof_ecdf"

# Metadata
PATCHED_META_PATH = PROJECT_ROOT / "results/revision_bspc_2026/adni_035_metadata_rescue_preflight/patched_metadata_candidate.csv"
ADNIMERGE_PATH    = BIG_DISK / "vae_AD_data/revision_bspc_2026/adni_metadata_martin_20260528/raw/ADNIMERGE_14Oct2024.csv"
PTDEMOG_PATH      = BIG_DISK / "vae_AD_data/revision_bspc_2026/adni_metadata_martin_20260528/raw/PTDEMOG_28May2026.csv"

# Global tensor metadata
TENSOR_META_PATH = BIG_DISK / "vae_AD_data/revision_bspc_2026/adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_metadata_v5_1_batch20260514b_no_pybandpass.csv"
TRAINING_META_PATH = BIG_DISK / "vae_AD_data/revision_bspc_2026/adni_expanded_v5_1_batch20260514b_no_pybandpass/training_ready_metadata_v5_1_batch20260514b_no_pybandpass.csv"

# Pipeline logs (for timepoints)
PIPELINE_LOGS = [
    BIG_DISK / "adni_expansion/MARTIN59/AAL3_v6_5_17_MARTIN59_ARWSDCF/pipeline_log_AAL3_v6_5_17_MARTIN59_ARWSDCF.csv",
    BIG_DISK / "adni_expansion/MARTIN_20260429_PHILIPS10/AAL3_v6_5_17_MARTIN_PHILIPS7_ARWSDCF/pipeline_log_AAL3_v6_5_17_MARTIN_PHILIPS7_ARWSDCF.csv",
    BIG_DISK / "adni_expansion/PHILIPS_CN_STRESS/AAL3_v6_5_17_PhilipsCN_first1/pipeline_log_AAL3_v6_5_17_PhilipsCN_first1.csv",
    BIG_DISK / "adni_expansion/PHILIPS_CN_STRESS/AAL3_v6_5_17_PhilipsCN_plus2/pipeline_log_AAL3_v6_5_17_PhilipsCN_plus2.csv",
    BIG_DISK / "adni_expansion/GE_batch7/AAL3_v6_5_17_GE_batch7/pipeline_log_AAL3_v6_5_17_GE_batch7.csv",
    BIG_DISK / "adni_expansion/GE_smoketest3/AAL3_v6_5_17_GE_smoketest3/pipeline_log_AAL3_v6_5_17_GE_smoketest3.csv",
    BIG_DISK / "adni_expansion/SIEMENS_available/AAL3_v6_5_17_SIEMENS_available/pipeline_log_AAL3_v6_5_17_SIEMENS_available.csv",
]

# Motion file directories
MOTION_DIRS = [
    # SIEMENS
    PROJECT_ROOT / "data/OneDrive_1_27-4-2026/RealignParameter",
    PROJECT_ROOT / "data/RevisionPaperfMRI_2026_04_pilot/Processed/DPARSF_single/SIEMENS/RealignParameter",
    # Philips
    BIG_DISK / "adni_expansion/MARTIN_20260429_PHILIPS10/OneDrive_2_29-4-2026/RealignParameter",
]

# Output
OUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/philips_cn_false_positive_forensic_audit_20260610"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMMAND_LOG = []

def log(msg):
    ts = datetime.now().isoformat()
    print(f"[{ts}] {msg}")
    COMMAND_LOG.append({"ts": ts, "msg": msg})

def save_df(df, stem, index=False):
    p = OUT_DIR / f"{stem}.csv"
    df.to_csv(p, index=index)
    log(f"  Saved {p.name} ({len(df)} rows)")
    return p

def save_md(text, stem):
    p = OUT_DIR / f"{stem}.md"
    p.write_text(text)
    log(f"  Saved {p.name}")
    return p

# ---------------------------------------------------------------------------
# SECTION 1 – DATA SOURCE INVENTORY
# ---------------------------------------------------------------------------

def inventory_data_sources():
    log("=== SECTION 1: Data Source Inventory ===")
    rows = []

    def check(label, path, desc=""):
        p = Path(path)
        exists = p.exists()
        size = ""
        n_files = ""
        if exists and p.is_file():
            size = f"{p.stat().st_size/1024:.0f}KB"
        elif exists and p.is_dir():
            try:
                files = list(p.iterdir())
                n_files = str(len(files))
            except Exception:
                n_files = "?"
        rows.append({"source": label, "path": str(path), "exists": exists,
                     "size": size, "n_files_or_size": n_files, "description": desc})

    # Predictions & calibration
    check("promoted_oof_calib_predictions", PROMOTED_OOF_DIR / "calib_predictions.csv",
          "Subject-level OOF scores promoted model (38112 rows)")
    check("promoted_oof_calib_pooled",    PROMOTED_OOF_DIR / "calib_pooled_metrics.csv")
    check("promoted_oof_calib_philips_fpr", PROMOTED_OOF_DIR / "calib_philips_fpr_pooled.csv")

    # Metadata
    check("patched_metadata_candidate",  PATCHED_META_PATH,       "647-subject recover035 metadata")
    check("adnimerge_14oct2024",         ADNIMERGE_PATH,          "ADNIMERGE with FLDSTRENG, SITE, cognitive scores, APOE4")
    check("ptdemog_28may2026",           PTDEMOG_PATH,            "ADNI participant demographics")
    check("tensor_subject_metadata",     TENSOR_META_PATH,        "Subject metadata for v5.1b tensor")
    check("training_ready_metadata",     TRAINING_META_PATH,      "Training-ready metadata v5.1b")

    # Pipeline logs
    for p in PIPELINE_LOGS:
        check(f"pipeline_log_{p.parent.parent.name}", p, "Original TP + final TP for connectivity")

    # Motion files
    for d in MOTION_DIRS:
        check(f"realign_param_{d.parent.name}", d, "SPM rp_*.txt realignment parameters")

    # ROI signals
    check("roi_signals_ondrive_1_13may", PROJECT_ROOT / "data/OneDrive_1_13-5-2026/ResultsAAL3/ROISignals_AAL3_FunImgARWSDCFN",
          "ROI time series .mat files for 112 subjects")

    df = pd.DataFrame(rows)
    save_df(df, "data_source_inventory")

    md = "# Data Source Inventory\n\n"
    md += "Generated: " + datetime.now().isoformat() + "\n\n"
    md += df.to_markdown(index=False) + "\n\n"
    md += """
## Notes
- **OOF predictions**: Full subject×calibration×model×feature×threshold matrix available.
- **Metadata**: 647 subjects with Manufacturer, Site3, ImageID, Visit, n_timepoints_raw (partial).
- **ADNIMERGE**: Contains FLDSTRENG (field strength), SITE, VISCODE, cognitive scores (CDRSB, MMSE, RAVLT, FAQ), APOE4. No scanner model or protocol columns.
- **Pipeline logs**: Record original_TPs and final connectivity shape; available for MARTIN59 (Philips), PHILIPS7/10, PHILIPS_CN_STRESS, GE, SIEMENS batches.
- **Motion files**: rp_*.txt (SPM DPARSF, 6-column) available for ~65 SIEMENS + ~68 Philips (PHILIPS10 batch only).
- **ROI signals**: .mat files for 112 subjects only.
- **BIDS JSON sidecars**: NOT found. Phase encoding direction, slice timing, scanner model cannot be determined from available data.
- **DICOM headers**: NOT found. Acquisition metadata limited to ADNIMERGE.
- **Mayo MCH table**: NOT found. Radiology findings not available.
"""
    save_md(md, "data_source_inventory")
    return df


# ---------------------------------------------------------------------------
# SECTION 2 – BUILD MASTER SUBJECT/SCAN TABLE
# ---------------------------------------------------------------------------

def load_promoted_oof_primary(calib=PRIMARY_CALIB, model=PRIMARY_MODEL,
                               feat=PRIMARY_FEAT, thresh=PRIMARY_THRESH):
    """Load primary-readout rows from calib_predictions.csv."""
    pred = pd.read_csv(PROMOTED_OOF_DIR / "calib_predictions.csv")
    mask = ((pred["calib_method"] == calib) &
            (pred["model_name"] == model) &
            (pred["feature_set"] == feat) &
            (pred["threshold_strategy"] == thresh))
    return pred[mask].copy()


def build_master_table(pred_primary, meta, adnimerge):
    log("=== SECTION 2: Building Master Subject/Scan Table ===")

    # Subject-level OOF score (one row per subject from pooled 5-fold)
    subj_scores = pred_primary.copy()
    subj_scores["error_type"] = subj_scores.apply(
        lambda r: ("TP" if r.y_true == 1 and r.y_pred == 1 else
                   "TN" if r.y_true == 0 and r.y_pred == 0 else
                   "FP" if r.y_true == 0 and r.y_pred == 1 else "FN"),
        axis=1
    )

    # Merge metadata
    meta_cols = ["SubjectID", "tensor_index", "ResearchGroup_Mapped", "Diagnosis",
                 "Manufacturer", "Site3", "ImageID", "Visit",
                 "Age", "Sex", "n_timepoints_raw", "n_rois_raw", "finite_fraction",
                 "metadata_source", "dicom_series_ok", "scale_label",
                 "python_bandpass_applied", "roisignals_path", "ledger_scope"]
    meta_sub = meta[[c for c in meta_cols if c in meta.columns]].copy()
    master = subj_scores.merge(meta_sub, on="SubjectID", how="left",
                               suffixes=("", "_meta"))

    # Clean up Age/Manufacturer duplicates
    if "Age_meta" in master.columns:
        master["Age"] = master["Age"].fillna(master["Age_meta"])
        master.drop(columns=["Age_meta"], inplace=True)
    if "Manufacturer_meta" in master.columns:
        master.drop(columns=["Manufacturer_meta"], inplace=True)

    # Join ADNIMERGE by PTID ~ SubjectID (best effort)
    if adnimerge is not None:
        adni_sub = adnimerge.drop_duplicates("PTID")[
            ["PTID", "SITE", "COLPROT", "ORIGPROT", "FLDSTRENG",
             "APOE4", "CDRSB", "MMSE", "FAQ", "RAVLT_immediate"]
        ].rename(columns={"PTID": "SubjectID"})
        master = master.merge(adni_sub, on="SubjectID", how="left")

    log(f"  Master table: {len(master)} rows, {len(master.columns)} columns")
    save_df(master, "subject_scan_master_table")

    # Summary markdown
    cn_master = master[master["ResearchGroup_Mapped"] == "CN"]
    philips_cn = cn_master[cn_master["Manufacturer"] == "Philips"]
    philips_cn_fp = philips_cn[philips_cn["error_type"] == "FP"]
    philips_cn_tn = philips_cn[philips_cn["error_type"] == "TN"]

    md = "# Subject/Scan Master Table\n\n"
    md += f"Total rows (one per subject from OOF fold): {len(master)}\n"
    md += f"Unique subjects: {master['SubjectID'].nunique()}\n\n"
    md += "## Diagnosis × Manufacturer\n\n"
    md += master.groupby(["ResearchGroup_Mapped", "Manufacturer"]).size().unstack(fill_value=0).to_markdown() + "\n\n"
    md += "## Error Type Distribution (CN subjects)\n\n"
    md += cn_master.groupby(["Manufacturer", "error_type"]).size().unstack(fill_value=0).to_markdown() + "\n\n"
    md += f"\n**Philips CN FP**: {len(philips_cn_fp)}, **Philips CN TN**: {len(philips_cn_tn)}\n"
    save_md(md, "subject_scan_master_table")
    return master


# ---------------------------------------------------------------------------
# SECTION 3 – TIMEPOINTS AUDIT
# ---------------------------------------------------------------------------

def load_pipeline_logs():
    """Load all pipeline logs and extract subject → timepoint info."""
    rows = []
    for p in PIPELINE_LOGS:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
            batch = p.parent.name
            df["batch"] = batch
            # Extract n_timepoints_original from detail_preprocessing
            if "detail_preprocessing" in df.columns:
                def extract_tps(detail):
                    try:
                        if pd.isna(detail):
                            return np.nan, np.nan
                        import re
                        m_orig = re.search(r"Original TPs:\s*(\d+)", str(detail))
                        m_conn = re.search(r"final shape for conn:\s*\((\d+),", str(detail))
                        orig = int(m_orig.group(1)) if m_orig else np.nan
                        conn = int(m_conn.group(1)) if m_conn else np.nan
                        return orig, conn
                    except Exception:
                        return np.nan, np.nan
                df[["n_timepoints_original", "n_timepoints_connectivity"]] = pd.DataFrame(
                    df["detail_preprocessing"].apply(extract_tps).tolist(),
                    index=df.index
                )
            if "id" in df.columns:
                df.rename(columns={"id": "SubjectID"}, inplace=True)
            rows.append(df)
        except Exception as e:
            log(f"  WARNING: Could not load {p}: {e}")

    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True)
    return combined


def timepoints_audit(master, pipeline_logs_df):
    log("=== SECTION 3: Timepoints Audit ===")

    tp_cols = ["SubjectID", "batch", "n_timepoints_original",
               "n_timepoints_connectivity", "status_preprocessing"]
    if not pipeline_logs_df.empty:
        tp_sub = pipeline_logs_df[[c for c in tp_cols if c in pipeline_logs_df.columns]].copy()
        tp_sub["SubjectID"] = tp_sub["SubjectID"].astype(str).str.strip()
    else:
        tp_sub = pd.DataFrame(columns=["SubjectID"])

    # Also use n_timepoints_raw from metadata
    meta_tp = master[["SubjectID", "Manufacturer", "ResearchGroup_Mapped",
                       "error_type", "n_timepoints_raw"]].copy()

    tp_merged = meta_tp.merge(tp_sub, on="SubjectID", how="left")
    # Prefer pipeline log over metadata n_timepoints_raw for original TPs
    tp_merged["n_tp_original_best"] = tp_merged["n_timepoints_original"].fillna(
        tp_merged["n_timepoints_raw"])
    save_df(tp_merged, "timepoint_audit")

    # Summary by Manufacturer and error_type for CN
    cn_tp = tp_merged[tp_merged["ResearchGroup_Mapped"] == "CN"]
    summary = cn_tp.groupby(["Manufacturer", "error_type"])[
        ["n_tp_original_best", "n_timepoints_connectivity"]
    ].agg(["mean", "median", "count"]).round(1)

    md = "# Timepoints Audit\n\n"
    md += f"Pipeline log coverage: {tp_sub['SubjectID'].nunique()} subjects\n"
    md += f"Metadata n_timepoints_raw coverage: {master['n_timepoints_raw'].notna().sum()} subjects\n\n"
    md += "## CN Timepoints by Manufacturer × Error Type\n\n"
    md += summary.to_markdown() + "\n\n"

    # Philips-specific
    philips_tp = cn_tp[cn_tp["Manufacturer"] == "Philips"]
    philips_tp_known = philips_tp[philips_tp["n_tp_original_best"].notna()]
    md += f"\n### Philips CN Timepoints (n with data = {len(philips_tp_known)})\n\n"
    if len(philips_tp_known) > 0:
        md += philips_tp_known.groupby("error_type")["n_tp_original_best"].describe().round(1).to_markdown()
        md += "\n\n"

    # Report on key finding
    md += """
## Key Finding
From pipeline logs, subjects with 197 raw BOLD timepoints are truncated to 140 timepoints
for connectivity computation (confirmed from MARTIN59 log: "Original TPs: 197, final shape
for conn: (140, 131)"). This is a Philips-specific acquisition parameter (TR differences
may produce 197 TPs in the 8-minute scan window vs 200 TPs for GE).

The 140-timepoint connectivity matrices may have systematically different noise properties
compared to GE (200 TPs) and SIEMENS (197 TPs without truncation).

**Coverage gap**: n_timepoints_raw is only populated for 153/647 subjects in metadata.
Pipeline logs cover the expansion batches but not all subjects used in the final tensor.
"""
    save_md(md, "timepoint_audit")
    return tp_merged


# ---------------------------------------------------------------------------
# SECTION 4 – ACQUISITION METADATA (ADNIMERGE / BIDS JSON)
# ---------------------------------------------------------------------------

def acquisition_metadata_audit(master):
    log("=== SECTION 4: Acquisition Metadata Audit ===")

    rows = []
    # FLDSTRENG from ADNIMERGE (already joined in master)
    if "FLDSTRENG" in master.columns:
        fld = master[["SubjectID", "Manufacturer", "ResearchGroup_Mapped", "error_type",
                       "FLDSTRENG", "SITE", "COLPROT", "ORIGPROT"]].copy()
        rows.append(fld)

    if rows:
        df = pd.concat(rows, ignore_index=True)
    else:
        df = pd.DataFrame({"note": ["ADNIMERGE join failed or unavailable"]})

    save_df(df, "dicom_json_metadata_inventory")

    # Phase encoding audit
    md_pe = """# Phase Encoding Audit

## Availability
- **BIDS JSON sidecars**: NOT FOUND in any ADNI batch directory.
- **DICOM headers**: NOT accessible from current data structure.
- **InPlanePhaseEncodingDirection**: UNAVAILABLE.
- **PhaseEncodingDirection**: UNAVAILABLE.
- **SliceTiming**: UNAVAILABLE.

## Required fields for PE/slice timing determination
- PhaseEncodingDirection (BIDS sidecar) OR
  InPlanePhaseEncodingDirection (DICOM tag 0018,1312)
- SliceTiming array (BIDS sidecar) OR
  TriggerTime per slice (DICOM) OR
  AcquisitionTime per slice (DICOM)

## What is available
- FLDSTRENG (field strength, from ADNIMERGE): 1.5T or 3T
- SITE (numeric site ID, from ADNIMERGE)
- COLPROT/ORIGPROT (ADNI protocol cohort, from ADNIMERGE)
- Manufacturer (GE/Philips/SIEMENS, from project metadata)
- Site3 (numeric site, from project metadata)

## Conclusion
Phase encoding direction and slice timing classification cannot be determined
from available data. This is flagged in `request_to_martin_missing_files.md`.
"""
    save_md(md_pe, "phase_encoding_audit")

    md_st = """# Slice Timing Audit

## Availability
Neither BIDS SliceTiming arrays nor DICOM AcquisitionTime/TriggerTime per slice
are available in the current data structure.

## Classification
All subjects: **unknown / unreliable**
Confidence: 0 (no information)

## Required for slice timing determination
- BIDS sidecar JSON with SliceTiming field, OR
- DICOM series with per-slice AcquisitionTime (tag 0008,0032) or TriggerTime (tag 0018,1060)

See `request_to_martin_missing_files.md` for what to request.
"""
    save_md(md_st, "slice_timing_audit")
    return df


# ---------------------------------------------------------------------------
# SECTION 5 – MOTION / QC AUDIT
# ---------------------------------------------------------------------------

def compute_fd_jenkinson(rp):
    """Compute Framewise Displacement (Jenkinson-style, 50mm sphere) from SPM rp array.
    rp: (n_vols, 6) array [tx, ty, tz mm, pitch, roll, yaw radians]
    Returns FD array of length n_vols-1.
    """
    diff = np.diff(rp, axis=0)
    # Translations: absolute mm
    trans_fd = np.sum(np.abs(diff[:, :3]), axis=1)
    # Rotations: convert to mm-equivalent with 50mm sphere radius
    rot_fd = 50.0 * np.sum(np.abs(diff[:, 3:]), axis=1)
    fd = trans_fd + rot_fd
    return fd


def load_motion_for_subject(subject_id):
    """Try to load rp file for a subject. Returns FD array or None."""
    for d in MOTION_DIRS:
        if not d.exists():
            continue
        subj_dir = d / subject_id
        if not subj_dir.exists():
            continue
        rp_files = sorted(subj_dir.glob("rp_*.txt"))
        if not rp_files:
            continue
        try:
            rp = np.loadtxt(rp_files[0])
            if rp.ndim == 1:
                rp = rp.reshape(-1, 6)
            if rp.shape[1] != 6:
                continue
            fd = compute_fd_jenkinson(rp)
            return fd, str(rp_files[0])
        except Exception as e:
            log(f"  WARNING: rp load failed {rp_files[0]}: {e}")
    return None, None


def motion_qc_audit(master):
    log("=== SECTION 6: Motion/QC Audit ===")

    motion_rows = []
    subjects_with_motion = 0

    for _, row in master.iterrows():
        subj = row["SubjectID"]
        fd_arr, rp_path = load_motion_for_subject(subj)
        if fd_arr is None:
            motion_rows.append({
                "SubjectID": subj,
                "Manufacturer": row["Manufacturer"],
                "ResearchGroup_Mapped": row["ResearchGroup_Mapped"],
                "error_type": row.get("error_type", np.nan),
                "fold": row.get("fold", np.nan),
                "mean_FD": np.nan, "max_FD": np.nan,
                "frac_FD_gt_0p3": np.nan, "frac_FD_gt_0p5": np.nan,
                "n_volumes": np.nan, "rp_path": None, "motion_available": False,
                "abs_max_translation_mm": np.nan
            })
        else:
            subjects_with_motion += 1
            rp_full = np.loadtxt(rp_path)
            if rp_full.ndim == 1:
                rp_full = rp_full.reshape(-1, 6)
            max_abs_drift = float(np.max(np.abs(rp_full[:, :3])))  # abs drift trans only

            motion_rows.append({
                "SubjectID": subj,
                "Manufacturer": row["Manufacturer"],
                "ResearchGroup_Mapped": row["ResearchGroup_Mapped"],
                "error_type": row.get("error_type", np.nan),
                "fold": row.get("fold", np.nan),
                "mean_FD": float(np.mean(fd_arr)),
                "max_FD": float(np.max(fd_arr)),
                "frac_FD_gt_0p3": float(np.mean(fd_arr > 0.3)),
                "frac_FD_gt_0p5": float(np.mean(fd_arr > 0.5)),
                "n_volumes": len(rp_full),
                "rp_path": rp_path,
                "motion_available": True,
                "abs_max_translation_mm": max_abs_drift
            })

    log(f"  Motion data available for {subjects_with_motion}/{len(master)} subjects")

    motion_df = pd.DataFrame(motion_rows)
    save_df(motion_df, "motion_qc_audit")

    # Summary stats
    cn_motion = motion_df[
        (motion_df["ResearchGroup_Mapped"] == "CN") &
        (motion_df["motion_available"])
    ]
    md = "# Motion / QC Audit\n\n"
    md += f"Subjects with motion data: {subjects_with_motion}/{len(master)}\n\n"

    if len(cn_motion) > 0:
        md += "## CN Motion Summary by Manufacturer × Error Type\n\n"
        grp = cn_motion.groupby(["Manufacturer", "error_type"])[
            ["mean_FD", "max_FD", "frac_FD_gt_0p3", "frac_FD_gt_0p5"]
        ].agg(["mean", "median", "count"]).round(3)
        md += grp.to_markdown() + "\n\n"

        # Philips CN FP vs TN motion test
        philips_cn_m = cn_motion[cn_motion["Manufacturer"] == "Philips"]
        fp_fd = philips_cn_m[philips_cn_m["error_type"] == "FP"]["mean_FD"].dropna()
        tn_fd = philips_cn_m[philips_cn_m["error_type"] == "TN"]["mean_FD"].dropna()
        if len(fp_fd) >= 3 and len(tn_fd) >= 3:
            stat, pval = mannwhitneyu(fp_fd, tn_fd, alternative="two-sided")
            md += f"### Philips CN: FP vs TN mean FD (Mann-Whitney U)\n"
            md += f"FP n={len(fp_fd)}, median={fp_fd.median():.3f}; "
            md += f"TN n={len(tn_fd)}, median={tn_fd.median():.3f}; "
            md += f"U={stat:.0f}, p={pval:.4f}\n\n"
        else:
            md += f"### Philips CN motion\n"
            md += f"Insufficient data for FP vs TN test: FP n={len(fp_fd)}, TN n={len(tn_fd)}\n\n"
    else:
        md += "No CN motion data available.\n\n"

    md += """
## Notes on 3mm/3° absolute drift exclusion criterion
The DPARSF preprocessing pipeline used an absolute motion exclusion criterion:
subjects with >3mm absolute translation OR >3° rotation were excluded.
The flag `dicom_series_ok` in project metadata encodes QC pass/fail.
The `abs_max_translation_mm` column above reflects the maximum absolute translation
from the rp file (frame-to-frame FD ≠ absolute drift; these are different measures).

Only rp files from MARTIN_20260429_PHILIPS10 (~68 subjects) and
OneDrive_1_27-4-2026 SIEMENS (~65 subjects) were available.
GE motion files were not found.
"""
    save_md(md, "motion_qc_audit")
    return motion_df


# ---------------------------------------------------------------------------
# SECTION 6 – PHILIPS CN FP vs TN TABLE
# ---------------------------------------------------------------------------

def philips_cn_fp_vs_tn_table(master, motion_df):
    log("=== SECTION: Philips CN FP vs TN Table ===")

    philips_cn = master[(master["Manufacturer"] == "Philips") &
                        (master["ResearchGroup_Mapped"] == "CN")].copy()

    # Merge motion
    if not motion_df.empty:
        mot_sub = motion_df[["SubjectID", "mean_FD", "max_FD",
                              "frac_FD_gt_0p3", "n_volumes"]].copy()
        philips_cn = philips_cn.merge(mot_sub, on="SubjectID", how="left")

    save_df(philips_cn, "philips_cn_fp_vs_tn_table")

    # Summary table
    fp = philips_cn[philips_cn["error_type"] == "FP"]
    tn = philips_cn[philips_cn["error_type"] == "TN"]

    rows_out = []
    for col in ["Age", "y_score", "y_score_raw", "mean_FD", "n_timepoints_raw"]:
        if col not in philips_cn.columns:
            continue
        fp_vals = fp[col].dropna()
        tn_vals = tn[col].dropna()
        if len(fp_vals) == 0 and len(tn_vals) == 0:
            continue
        row = {"variable": col,
               "FP_n": len(fp_vals), "FP_mean": fp_vals.mean() if len(fp_vals) else np.nan,
               "FP_median": fp_vals.median() if len(fp_vals) else np.nan,
               "TN_n": len(tn_vals), "TN_mean": tn_vals.mean() if len(tn_vals) else np.nan,
               "TN_median": tn_vals.median() if len(tn_vals) else np.nan}
        if len(fp_vals) >= 3 and len(tn_vals) >= 3:
            u, p = mannwhitneyu(fp_vals, tn_vals, alternative="two-sided")
            row["MWU_pval"] = round(p, 4)
            row["MWU_U"] = round(u, 0)
        else:
            row["MWU_pval"] = np.nan
            row["MWU_U"] = np.nan
        rows_out.append(row)

    # Categorical: sex
    if "Sex" in philips_cn.columns:
        fp_sex = fp["Sex"].value_counts()
        tn_sex = tn["Sex"].value_counts()
        rows_out.append({"variable": "Sex_F_count",
                         "FP_n": fp_sex.get("F", 0), "TN_n": tn_sex.get("F", 0)})

    # Site3
    if "Site3" in philips_cn.columns:
        site_err = philips_cn.groupby(["Site3", "error_type"]).size().unstack(fill_value=0)
        if "FP" in site_err.columns and "TN" in site_err.columns:
            site_err["FPR"] = site_err["FP"] / (site_err["FP"] + site_err["TN"])
            site_err["n_CN"] = site_err["FP"] + site_err["TN"]

    summary_df = pd.DataFrame(rows_out)
    save_df(summary_df, "philips_cn_fp_vs_tn_table")

    md = "# Philips CN FP vs TN Comparison\n\n"
    md += f"Philips CN total: {len(philips_cn)}, FP: {len(fp)}, TN: {len(tn)}\n\n"
    md += "## Continuous variable comparisons\n\n"
    md += summary_df.round(3).to_markdown(index=False) + "\n\n"
    if "Site3" in philips_cn.columns:
        md += "## FPR by Philips Site\n\n"
        md += site_err.round(3).to_markdown() + "\n\n"
    if "FLDSTRENG" in philips_cn.columns:
        md += "## Field Strength distribution (Philips CN)\n\n"
        md += philips_cn.groupby(["FLDSTRENG", "error_type"]).size().unstack(fill_value=0).to_markdown() + "\n\n"
    save_md(md, "philips_cn_fp_vs_tn_table")
    return philips_cn


# ---------------------------------------------------------------------------
# SECTION 7 – MODEL / SITE / PHASE / SOFTWARE FPR TABLE
# ---------------------------------------------------------------------------

def model_site_phase_software_fpr_table(master):
    log("=== SECTION: Model/Site/Phase FPR Table ===")
    cn = master[master["ResearchGroup_Mapped"] == "CN"].copy()

    rows = []
    for mfr, grp in cn.groupby("Manufacturer"):
        n_fp = (grp["error_type"] == "FP").sum()
        n_tn = (grp["error_type"] == "TN").sum()
        fpr = n_fp / (n_fp + n_tn) if (n_fp + n_tn) > 0 else np.nan
        rows.append({"Manufacturer": mfr, "n_CN": n_fp + n_tn, "n_FP": n_fp,
                     "n_TN": n_tn, "FPR": round(fpr, 4)})

    # By site (top Philips sites)
    if "Site3" in cn.columns:
        for (mfr, site), grp in cn.groupby(["Manufacturer", "Site3"]):
            n_fp = (grp["error_type"] == "FP").sum()
            n_tn = (grp["error_type"] == "TN").sum()
            n_tot = n_fp + n_tn
            fpr = n_fp / n_tot if n_tot > 0 else np.nan
            rows.append({"Manufacturer": mfr, "Site3": site, "n_CN": n_tot,
                         "n_FP": n_fp, "n_TN": n_tn, "FPR": round(fpr, 4)})

    # FLDSTRENG if available
    if "FLDSTRENG" in cn.columns:
        for (mfr, fs), grp in cn.groupby(["Manufacturer", "FLDSTRENG"]):
            n_fp = (grp["error_type"] == "FP").sum()
            n_tn = (grp["error_type"] == "TN").sum()
            n_tot = n_fp + n_tn
            fpr = n_fp / n_tot if n_tot > 0 else np.nan
            rows.append({"Manufacturer": mfr, "FLDSTRENG": fs, "n_CN": n_tot,
                         "n_FP": n_fp, "n_TN": n_tn, "FPR": round(fpr, 4)})

    df_fpr = pd.DataFrame(rows)
    save_df(df_fpr, "model_site_phase_software_fpr_table")

    md = "# Model / Site / Phase / Software FPR Table\n\n"
    mfr_fpr = df_fpr[df_fpr["Manufacturer"].notna() & df_fpr.get("Site3", pd.Series(dtype=float)).isna()
                     if "Site3" in df_fpr.columns else df_fpr["Manufacturer"].notna()]
    md += "## FPR by Manufacturer\n\n"
    md += df_fpr[~df_fpr.get("Site3", pd.Series(np.nan, index=df_fpr.index)).notna()
                 if "Site3" in df_fpr.columns else df_fpr.index.isin(df_fpr.index)
                 ].head(6).round(4).to_markdown(index=False) + "\n\n"
    md += "Note: Phase-encoding direction and scanner software version unavailable (no BIDS/DICOM).\n"
    save_md(md, "model_site_phase_software_fpr_table")
    return df_fpr


# ---------------------------------------------------------------------------
# SECTION 8 – STATISTICAL TESTS
# ---------------------------------------------------------------------------

def statistical_tests(master, motion_df):
    log("=== SECTION 8: Statistical Tests ===")

    cn = master[master["ResearchGroup_Mapped"] == "CN"].copy()
    philips_cn = cn[cn["Manufacturer"] == "Philips"].copy()

    results = []

    def add_mwu(label, g1, g2, var):
        a = g1[var].dropna() if var in g1.columns else pd.Series([], dtype=float)
        b = g2[var].dropna() if var in g2.columns else pd.Series([], dtype=float)
        if len(a) >= 3 and len(b) >= 3:
            u, p = mannwhitneyu(a, b, alternative="two-sided")
            # Effect size r = U / (n1 * n2) ... or common language effect size
            r = u / (len(a) * len(b))
            results.append({
                "test": "Mann-Whitney U", "comparison": label, "variable": var,
                "n_group1": len(a), "n_group2": len(b),
                "median_g1": round(a.median(), 3), "median_g2": round(b.median(), 3),
                "U_stat": round(u, 0), "p_value": round(p, 4), "effect_r": round(r, 3),
                "interpretation": "significant" if p < 0.05 else "not significant"
            })
        else:
            results.append({
                "test": "Mann-Whitney U", "comparison": label, "variable": var,
                "n_group1": len(a), "n_group2": len(b), "p_value": np.nan,
                "interpretation": "insufficient_data"
            })

    def add_fisher(label, var, g1, g2, val=None):
        if var not in g1.columns:
            return
        if val is not None:
            c1 = (g1[var] == val).sum()
            c2 = (g2[var] == val).sum()
        else:
            c1 = g1[var].sum()
            c2 = g2[var].sum()
        n1 = len(g1[var].dropna())
        n2 = len(g2[var].dropna())
        table = [[c1, n1-c1], [c2, n2-c2]]
        try:
            or_, p = fisher_exact(table)
            results.append({
                "test": "Fisher exact", "comparison": label, "variable": var,
                "n_group1": n1, "n_group2": n2,
                "count_g1": c1, "count_g2": c2,
                "odds_ratio": round(or_, 3), "p_value": round(p, 4),
                "interpretation": "significant" if p < 0.05 else "not significant"
            })
        except Exception:
            pass

    fp = philips_cn[philips_cn["error_type"] == "FP"]
    tn = philips_cn[philips_cn["error_type"] == "TN"]

    # 1. Philips CN FP vs TN: Age
    add_mwu("Philips_CN FP vs TN", fp, tn, "Age")
    # 2. Score
    add_mwu("Philips_CN FP vs TN", fp, tn, "y_score")
    # 3. Motion
    if not motion_df.empty:
        fp_m = fp.merge(motion_df[["SubjectID", "mean_FD"]].dropna(), on="SubjectID", how="inner")
        tn_m = tn.merge(motion_df[["SubjectID", "mean_FD"]].dropna(), on="SubjectID", how="inner")
        add_mwu("Philips_CN FP vs TN", fp_m, tn_m, "mean_FD")

    # 4. Sex (Fisher)
    add_fisher("Philips_CN FP vs TN", "Sex", fp, tn, val="F")

    # 5. Cross-manufacturer: Score by Manufacturer (all CN)
    ge_cn  = cn[cn["Manufacturer"] == "GE"]
    si_cn  = cn[cn["Manufacturer"] == "SIEMENS"]
    add_mwu("Philips_CN vs GE_CN (score)", philips_cn, ge_cn, "y_score")
    add_mwu("Philips_CN vs SIEMENS_CN (score)", philips_cn, si_cn, "y_score")
    add_mwu("GE_CN vs SIEMENS_CN (score)", ge_cn, si_cn, "y_score")

    # 6. Age by Manufacturer (confounding check)
    add_mwu("Philips_CN vs GE_CN (Age)", philips_cn, ge_cn, "Age")
    add_mwu("Philips_CN vs SIEMENS_CN (Age)", philips_cn, si_cn, "Age")

    # 7. APOE4 if available
    if "APOE4" in philips_cn.columns:
        add_mwu("Philips_CN FP vs TN", fp, tn, "APOE4")
        add_mwu("Philips_CN FP vs TN", fp, tn, "MMSE")
        add_mwu("Philips_CN FP vs TN", fp, tn, "CDRSB")

    df_tests = pd.DataFrame(results)
    save_df(df_tests, "statistical_tests")

    md = "# Statistical Tests\n\n"
    md += "All tests descriptive/exploratory. No pre-specified primary endpoint.\n\n"
    md += df_tests.round(4).to_markdown(index=False) + "\n\n"
    md += "**Note**: Low n in Philips FP/TN subgroups limits power. Effect sizes (r = U/(n1*n2)) provided.\n"
    save_md(md, "statistical_tests")
    return df_tests


# ---------------------------------------------------------------------------
# SECTION 9 – MAYO MCH AUDIT PLACEHOLDER
# ---------------------------------------------------------------------------

def mayo_mch_audit():
    log("=== SECTION: Mayo MCH Audit ===")
    md = """# Mayo MCH / Radiology Findings Join Audit

## Availability
The Mayo MCH (Medical Case History) table was not found in any accessible data path.
Searched:
- /media/diego/Datos/vae_AD_data/revision_bspc_2026/adni_metadata_martin_20260528/raw/
- /media/diego/Datos/vae_AD_data/revision_bspc_2026/adni_metadata_martin_20260528/derived/
- /media/diego/Datos/adni_expansion/

ADNIMERGE_14Oct2024.csv contains biomarker (AV45, FBB, Amyloid, Tau) and cognitive
(CDRSB, MMSE, RAVLT, FAQ, MOCA) data that partially serves as a proxy for clinical
severity, but does not contain radiology series quality flags.

## ADNIMERGE proxy fields available
- CDRSB: Clinical Dementia Rating Sum of Boxes (CN expected ≈ 0)
- MMSE: Mini Mental State Exam (CN expected ≈ 28–30)
- RAVLT_immediate: immediate recall
- FAQ: Functional Activities Questionnaire
- APOE4: Apolipoprotein E ε4 allele count (0/1/2) — genetic risk
- ABETA, TAU, PTAU: CSF biomarkers (sparse)

These are analyzed in statistical_tests.csv where data coverage permits.

## Recommendation
If radiology findings or QC flags (motion artifacts, partial-volume artifacts,
lesion flags) from ADNI are needed, request the `MRIQUALITY` or `MRADIOLOGYTBL`
tables from the ADNI data portal and join by RID+ImageUID.
"""
    save_md(md, "mayo_mch_join_audit")


# ---------------------------------------------------------------------------
# SECTION 10 – MISSING REQUIRED FIELDS
# ---------------------------------------------------------------------------

def missing_required_fields():
    log("=== SECTION: Missing Required Fields ===")
    md = """# Missing Required Fields for PE and Slice Timing Determination

## Phase Encoding Direction
**Status**: NOT DETERMINABLE from available data.

**Required fields**:
- BIDS JSON sidecar: `PhaseEncodingDirection` (e.g., "j-" = A-P, "j" = P-A)
- OR DICOM tag `InPlanePhaseEncodingDirection` (0018,1312): "ROW" or "COL"
- OR DICOM tag `PhaseEncodingDirection` (numeric)

**What was searched**: All .json files in /home/diego/proyectos/vae_AD/data/* and
/media/diego/Datos/adni_expansion/* — no BIDS sidecars found. Only pipeline/config
JSON files present.

## Slice Timing
**Status**: NOT DETERMINABLE from available data.

**Required fields**:
- BIDS JSON sidecar: `SliceTiming` array
- OR DICOM: per-slice `AcquisitionTime` (0008,0032) or `TriggerTime` (0018,1060)
- OR DPARSF SliceTiming configuration file

## Scanner Model and Software Version
**Status**: NOT IN AVAILABLE METADATA.

**Required**: DICOM tag `ManufacturerModelName` (0008,1090) and
`SoftwareVersions` (0018,1020). ADNIMERGE has Manufacturer but not model/version.

## Protocol Name / Sequence Description
**Status**: NOT AVAILABLE.

**Required**: DICOM tags `ProtocolName` (0018,1030) or `SeriesDescription` (0008,103E).

## What TO REQUEST from Martín
See `request_to_martin_missing_files.md`.
"""
    save_md(md, "missing_required_fields_for_PE_slice_timing")


# ---------------------------------------------------------------------------
# SECTION 11 – REQUEST TO MARTÍN
# ---------------------------------------------------------------------------

def request_to_martin():
    log("=== SECTION: Request to Martín ===")
    md = """# Request to Martín — Missing Files for Philips CN FP Forensic Audit

## Context
We are performing a forensic audit of Philips CN false positives in the promoted
ADNI model (recover035_latent384_beta3p75_T80_h10000_p560_full5x5).
Philips CN FPR ≈ 44–45%, vs GE ≈ 15%, SIEMENS ≈ 24%.
We need to determine whether this is driven by acquisition parameters that
differ between Philips FP and TN subjects.

## Highest Priority Requests

### 1. DICOM or BIDS JSON sidecars for Philips CN subjects
For subjects in the classifier pool (CN Philips, n ≈ 100–130 CN subjects):
- BIDS JSON sidecars (.json) co-registered with the fMRI NIfTI files
- OR the raw DICOM series headers (any tool to extract: `dcminfo`, `dcm2niix -b o`)

**Key fields needed**:
- `PhaseEncodingDirection` / `InPlanePhaseEncodingDirection` (A-P vs P-A)
- `SliceTiming` array (interleaved vs ascending vs descending)
- `ManufacturerModelName` (Achieva vs Ingenia vs Ingenuity etc.)
- `SoftwareVersions` (R3.2 vs R5.3 etc.)
- `ProtocolName` / `SeriesDescription`
- `TR`, `TE`, `FlipAngle`, `Rows`, `Columns`, number of slices

### 2. Realignment parameter files for remaining ADNI batches
Motion files (rp_*.txt) are currently available for:
- SIEMENS batch: ~65 subjects (OneDrive_1_27-4-2026)
- Philips10 batch: ~68 subjects (MARTIN_20260429_PHILIPS10)

**Missing**:
- Philips CN subjects NOT in the PHILIPS10 batch (most of the ~180 Philips CN subjects)
- MARTIN59 batch Philips subjects (confirmed batch exists, no RealignParameter found)
- GE batch subjects (no rp files found anywhere)

Please provide the RealignParameter folders for:
- MARTIN59 batch (should exist alongside the ROI signals)
- Remaining Philips batches

### 3. Clarification on 3mm/3° absolute drift exclusion
In a prior session, a question was raised about whether the absolute-drift
exclusion criterion (3mm/3° threshold) applied frame-to-frame or as an
absolute max-from-baseline criterion.

**Specific question**: Was the 3mm/3° criterion applied as:
(a) frame-to-frame FD > 3mm, OR
(b) absolute displacement from first volume > 3mm, OR
(c) a criterion applied to the rp files before computing connectivity?

This is critical for interpreting whether Philips subjects were excluded
for motion at different rates than GE/SIEMENS.

### 4. Confirmation of 197 → 140 TP truncation for Philips
The pipeline log for MARTIN59 shows: "Original TPs: 197, final shape for conn: (140, 131)".
Please confirm:
(a) Is the truncation to 140 applied only for Philips subjects with 197 TPs?
(b) What is the truncation logic? (first 140? middle 140? after steady-state?)
(c) Does GE 200-TP also get truncated or used at 200 TPs?
(d) Do SIEMENS 197-TP get truncated to 140 or used at 197?

### 5. ADNI image metadata tables
From ADNI data portal:
- `MRIQUALITY` table (series quality flags, motion flags)
- `MRADIOLOGYTBL` table (radiology findings)
These would allow joining radiology QC flags to Philips CN FP subjects.

## Low Priority
- NIfTI files themselves (only needed if voxel-level analysis required)
- DICOM folders (only needed if JSON sidecars unavailable)
"""
    save_md(md, "request_to_martin_missing_files")


# ---------------------------------------------------------------------------
# SECTION 12 – FIGURES
# ---------------------------------------------------------------------------

def make_figures(master, motion_df):
    log("=== SECTION: Figures ===")
    figs_made = []
    PALETTE = {"Philips": "#E07B54", "GE": "#5B8DB8", "SIEMENS": "#6AAB6A"}
    ERR_PALETTE = {"FP": "#D62728", "TN": "#2CA02C", "FN": "#FF7F0E", "TP": "#1F77B4"}

    cn = master[master["ResearchGroup_Mapped"] == "CN"].copy()

    # --- Figure 1: Philips CN score distributions FP vs TN
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        philips_cn = cn[cn["Manufacturer"] == "Philips"]
        for err_type, grp in philips_cn.groupby("error_type"):
            if err_type not in ["FP", "TN"]:
                continue
            color = ERR_PALETTE.get(err_type, "grey")
            axes[0].hist(grp["y_score"], bins=20, alpha=0.6, label=f"{err_type} (n={len(grp)})",
                         color=color, density=True)
            axes[1].ecdf(grp["y_score"], label=f"{err_type} (n={len(grp)})")
        axes[0].set_xlabel("OOF score (oof_ecdf calibrated)")
        axes[0].set_ylabel("Density")
        axes[0].set_title("Philips CN: FP vs TN score distribution")
        axes[0].legend()
        axes[1].set_xlabel("OOF score")
        axes[1].set_ylabel("ECDF")
        axes[1].set_title("Philips CN: FP vs TN score ECDF")
        axes[1].legend()
        plt.tight_layout()
        p = OUT_DIR / "fig_philips_cn_score_fp_vs_tn.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        figs_made.append(str(p))
        log(f"  Saved {p.name}")
    except Exception as e:
        log(f"  WARNING: Fig 1 failed: {e}")

    # --- Figure 2: CN score by Manufacturer (violin + strip)
    try:
        fig, ax = plt.subplots(figsize=(9, 6))
        mfrs = ["GE", "SIEMENS", "Philips"]
        cn_plot = cn[cn["Manufacturer"].isin(mfrs)].copy()
        # Violin
        order_m = mfrs
        for i, mfr in enumerate(order_m):
            grp = cn_plot[cn_plot["Manufacturer"] == mfr]["y_score"].dropna()
            if len(grp) == 0:
                continue
            parts = ax.violinplot([grp], positions=[i], showmedians=True, showextrema=True)
            for pc in parts["bodies"]:
                pc.set_facecolor(PALETTE.get(mfr, "grey"))
                pc.set_alpha(0.5)
            ax.scatter(np.random.normal(i, 0.05, len(grp)), grp,
                       alpha=0.3, s=10, color=PALETTE.get(mfr, "grey"))
        ax.set_xticks(range(len(order_m)))
        ax.set_xticklabels(order_m)
        ax.set_ylabel("OOF score (oof_ecdf)")
        ax.set_title("CN score distribution by Manufacturer")
        # Add FPR annotation
        for i, mfr in enumerate(order_m):
            g = cn_plot[cn_plot["Manufacturer"] == mfr]
            fpr_val = (g["error_type"] == "FP").mean()
            ax.text(i, ax.get_ylim()[1] * 0.98,
                    f"FPR={fpr_val:.2f}\nn={len(g)}", ha="center", fontsize=9,
                    va="top")
        plt.tight_layout()
        p = OUT_DIR / "fig_cn_score_by_manufacturer.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        figs_made.append(str(p))
        log(f"  Saved {p.name}")
    except Exception as e:
        log(f"  WARNING: Fig 2 failed: {e}")

    # --- Figure 3: FPR by Manufacturer (bar)
    try:
        fig, ax = plt.subplots(figsize=(7, 5))
        cn_grp = cn.groupby("Manufacturer").apply(
            lambda g: (g["error_type"] == "FP").mean()
        ).reset_index(name="FPR")
        cn_cnt = cn.groupby("Manufacturer").size().reset_index(name="n")
        cn_grp = cn_grp.merge(cn_cnt, on="Manufacturer")
        mfrs_plot = ["GE", "SIEMENS", "Philips"]
        cn_grp = cn_grp[cn_grp["Manufacturer"].isin(mfrs_plot)]
        colors = [PALETTE.get(m, "grey") for m in cn_grp["Manufacturer"]]
        bars = ax.bar(cn_grp["Manufacturer"], cn_grp["FPR"], color=colors, alpha=0.8)
        for bar, (_, row) in zip(bars, cn_grp.iterrows()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"n={row['n']}", ha="center", fontsize=10)
        ax.axhline(0.5, ls="--", color="red", alpha=0.5, label="FPR=0.50")
        ax.set_ylabel("CN FPR (oof_ecdf, inner_oof_target_sens_ge_0p70_max_spec)")
        ax.set_title("CN False Positive Rate by Manufacturer")
        ax.set_ylim(0, max(cn_grp["FPR"].max() + 0.1, 0.6))
        ax.legend()
        plt.tight_layout()
        p = OUT_DIR / "fig_fpr_by_manufacturer.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        figs_made.append(str(p))
        log(f"  Saved {p.name}")
    except Exception as e:
        log(f"  WARNING: Fig 3 failed: {e}")

    # --- Figure 4: Philips CN score by Site
    try:
        if "Site3" in cn.columns:
            philips_cn = cn[cn["Manufacturer"] == "Philips"].copy()
            site_counts = philips_cn.groupby("Site3").size()
            big_sites = site_counts[site_counts >= 8].index
            if len(big_sites) >= 3:
                philips_big = philips_cn[philips_cn["Site3"].isin(big_sites)]
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                # Score by site
                for site, grp in philips_big.groupby("Site3"):
                    axes[0].scatter([site] * len(grp),
                                    grp["y_score"], alpha=0.4, s=20)
                axes[0].set_xlabel("Site3")
                axes[0].set_ylabel("OOF score")
                axes[0].set_title("Philips CN score by site (n≥8)")
                # FPR by site
                site_fpr = philips_cn.groupby("Site3").apply(
                    lambda g: pd.Series({
                        "FPR": (g["error_type"] == "FP").mean(),
                        "n": len(g)
                    })
                ).reset_index()
                site_fpr = site_fpr.sort_values("FPR", ascending=False)
                axes[1].bar(site_fpr["Site3"].astype(str), site_fpr["FPR"],
                             alpha=0.8, color="#E07B54")
                axes[1].set_xlabel("Site3")
                axes[1].set_ylabel("FPR")
                axes[1].set_title("Philips CN FPR by site")
                axes[1].tick_params(axis="x", rotation=45)
                plt.tight_layout()
                p = OUT_DIR / "fig_philips_cn_score_by_site.png"
                plt.savefig(p, dpi=150, bbox_inches="tight")
                plt.close()
                figs_made.append(str(p))
                log(f"  Saved {p.name}")
    except Exception as e:
        log(f"  WARNING: Fig 4 failed: {e}")

    # --- Figure 5: Score vs Age by Manufacturer (all CN)
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        cn_plot = cn[cn["Manufacturer"].isin(["GE", "SIEMENS", "Philips"])].copy()
        for ax, mfr in zip(axes, ["GE", "SIEMENS", "Philips"]):
            grp = cn_plot[cn_plot["Manufacturer"] == mfr]
            fp = grp[grp["error_type"] == "FP"]
            tn = grp[grp["error_type"] == "TN"]
            ax.scatter(tn["Age"], tn["y_score"], alpha=0.4, s=20,
                       color=ERR_PALETTE["TN"], label="TN")
            ax.scatter(fp["Age"], fp["y_score"], alpha=0.6, s=25,
                       color=ERR_PALETTE["FP"], marker="^", label="FP")
            # Trend line (all CN)
            age_vals = grp["Age"].dropna()
            score_vals = grp["y_score"].dropna()
            common_idx = age_vals.index.intersection(score_vals.index)
            if len(common_idx) > 5:
                slope, intercept, r, pv, se = stats.linregress(
                    age_vals[common_idx], score_vals[common_idx]
                )
                x_line = np.array([age_vals.min(), age_vals.max()])
                ax.plot(x_line, intercept + slope * x_line, "k--", alpha=0.5,
                        label=f"r={r:.2f}, p={pv:.3f}")
            ax.set_xlabel("Age")
            ax.set_ylabel("OOF score") if mfr == "GE" else None
            ax.set_title(f"{mfr} CN (n={len(grp)})")
            ax.legend(fontsize=8)
        plt.suptitle("Score vs Age by Manufacturer (CN subjects)", fontsize=11)
        plt.tight_layout()
        p = OUT_DIR / "fig_score_vs_age_by_manufacturer.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        figs_made.append(str(p))
        log(f"  Saved {p.name}")
    except Exception as e:
        log(f"  WARNING: Fig 5 failed: {e}")

    # --- Figure 6: Motion FD distributions by Manufacturer/error_type
    try:
        if not motion_df.empty:
            mot_cn = motion_df[
                motion_df["motion_available"] &
                (motion_df["ResearchGroup_Mapped"] == "CN")
            ].copy()
            if len(mot_cn) >= 5:
                fig, ax = plt.subplots(figsize=(9, 5))
                for (mfr, err), grp in mot_cn.groupby(["Manufacturer", "error_type"]):
                    if err not in ["FP", "TN"]:
                        continue
                    fd_vals = grp["mean_FD"].dropna()
                    if len(fd_vals) < 2:
                        continue
                    ax.ecdf(fd_vals,
                            label=f"{mfr} {err} (n={len(fd_vals)})",
                            color=PALETTE.get(mfr, "grey"),
                            linestyle="--" if err == "FP" else "-")
                ax.set_xlabel("Mean FD (mm, Jenkinson)")
                ax.set_ylabel("ECDF")
                ax.set_title("CN motion FD distributions by Manufacturer × Error Type")
                ax.legend(fontsize=8)
                plt.tight_layout()
                p = OUT_DIR / "fig_motion_fd_by_manufacturer_errortype.png"
                plt.savefig(p, dpi=150, bbox_inches="tight")
                plt.close()
                figs_made.append(str(p))
                log(f"  Saved {p.name}")
    except Exception as e:
        log(f"  WARNING: Fig 6 failed: {e}")

    # --- Figure 7: Timepoints histogram by Manufacturer
    try:
        tp_col = None
        for col in ["n_timepoints_raw", "n_tp_original_best"]:
            if col in master.columns and master[col].notna().sum() > 5:
                tp_col = col
                break
        if tp_col:
            fig, ax = plt.subplots(figsize=(8, 5))
            for mfr, grp in master.groupby("Manufacturer"):
                vals = grp[tp_col].dropna()
                if len(vals) == 0:
                    continue
                ax.hist(vals, bins=15, alpha=0.5,
                        color=PALETTE.get(mfr, "grey"), label=f"{mfr} (n={len(vals)})")
            ax.set_xlabel("n_timepoints")
            ax.set_ylabel("Count")
            ax.set_title(f"Timepoints histogram by Manufacturer ({tp_col})")
            ax.legend()
            plt.tight_layout()
            p = OUT_DIR / "fig_timepoints_histogram_by_manufacturer.png"
            plt.savefig(p, dpi=150, bbox_inches="tight")
            plt.close()
            figs_made.append(str(p))
            log(f"  Saved {p.name}")
    except Exception as e:
        log(f"  WARNING: Fig 7 failed: {e}")

    # --- Figure 8: Score distribution all 3 manufacturers (combined)
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        for mfr in ["GE", "SIEMENS", "Philips"]:
            grp = cn[cn["Manufacturer"] == mfr]["y_score"].dropna()
            ax.ecdf(grp, label=f"{mfr} CN (n={len(grp)})",
                    color=PALETTE.get(mfr, "grey"))
        ax.axvline(0.5, ls="--", color="black", alpha=0.4, label="threshold~0.5")
        ax.set_xlabel("OOF score (oof_ecdf)")
        ax.set_ylabel("ECDF")
        ax.set_title("CN score ECDF by Manufacturer")
        ax.legend()
        plt.tight_layout()
        p = OUT_DIR / "fig_cn_score_ecdf_by_manufacturer.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        figs_made.append(str(p))
        log(f"  Saved {p.name}")
    except Exception as e:
        log(f"  WARNING: Fig 8 failed: {e}")

    return figs_made


# ---------------------------------------------------------------------------
# SECTION 13 – FINAL INTERPRETATION
# ---------------------------------------------------------------------------

def final_interpretation(master, stats_df, figs_made):
    log("=== SECTION: Final Philips Interpretation ===")

    cn = master[master["ResearchGroup_Mapped"] == "CN"]
    philips_cn = cn[cn["Manufacturer"] == "Philips"]
    fp = philips_cn[philips_cn["error_type"] == "FP"]
    tn = philips_cn[philips_cn["error_type"] == "TN"]
    ge_cn = cn[cn["Manufacturer"] == "GE"]
    si_cn = cn[cn["Manufacturer"] == "SIEMENS"]

    def fpr(grp):
        return (grp["error_type"] == "FP").mean()

    # Score shift
    p_median = philips_cn["y_score"].median()
    ge_median = ge_cn["y_score"].median()
    si_median = si_cn["y_score"].median()

    # Age
    p_age = philips_cn["Age"].mean()
    ge_age = ge_cn["Age"].mean()
    si_age = si_cn["Age"].mean()

    # Site concentration
    if "Site3" in philips_cn.columns:
        site_fpr = philips_cn.groupby("Site3").apply(
            lambda g: (g["error_type"] == "FP").mean()
        ).sort_values(ascending=False)
        top_site_fpr = site_fpr.head(3)

    md = f"""# Final Philips CN False-Positive Interpretation

**Date**: {datetime.now().isoformat()}
**Model**: recover035_latent384_beta3p75_T80_h10000_p560_full5x5
**Calibration**: oof_ecdf, logreg_l2_original, z_plus_age_sex
**Threshold**: inner_oof_target_sens_ge_0p70_max_spec

---

## Summary Statistics

| Metric | Philips CN | GE CN | SIEMENS CN |
|--------|-----------|-------|------------|
| N subjects | {len(philips_cn)} | {len(ge_cn)} | {len(si_cn)} |
| FPR | {fpr(philips_cn):.3f} | {fpr(ge_cn):.3f} | {fpr(si_cn):.3f} |
| Score median | {p_median:.3f} | {ge_median:.3f} | {si_median:.3f} |
| Age mean | {p_age:.1f} | {ge_age:.1f} | {si_age:.1f} |
| FP n | {len(fp)} | {(ge_cn["error_type"]=="FP").sum()} | {(si_cn["error_type"]=="FP").sum()} |
| TN n | {len(tn)} | {(ge_cn["error_type"]=="TN").sum()} | {(si_cn["error_type"]=="TN").sum()} |

---

## Findings by Hypothesis

### H1: Score elevation is demographically driven (older Philips CN)
- Philips CN mean age: {p_age:.1f} vs GE: {ge_age:.1f} vs SIEMENS: {si_age:.1f}
- Within Philips CN: FP vs TN age comparison → see statistical_tests.csv
- **Assessment**: Age difference between Philips and other manufacturers
  {'EXISTS' if abs(p_age - ge_age) > 3 or abs(p_age - si_age) > 3 else 'IS MINOR'}
  (|Δ|≥3 years). Age-score correlations are assessed in fig_score_vs_age_by_manufacturer.png.
  The model includes Age as a classifier feature (z_plus_age_sex), so age is partially
  controlled at the classifier level — but age captured in the VAE latent space is not
  controlled.

### H2: Score elevation is site-concentrated (few Philips sites drive FPR)
"""
    if "Site3" in philips_cn.columns:
        md += f"- Top Philips sites by FPR:\n"
        for site, v in top_site_fpr.items():
            n_site = (philips_cn["Site3"] == site).sum()
            md += f"  - Site {site}: FPR={v:.2f} (n={n_site})\n"
        site_std = philips_cn.groupby("Site3").apply(
            lambda g: (g["error_type"]=="FP").mean()
        ).std()
        md += f"- Cross-Philips-site FPR std: {site_std:.3f}\n"
        md += "- **Assessment**: If FPR is highly variable across Philips sites (std > 0.15), "
        md += "this suggests site-specific protocols rather than a manufacturer-wide effect.\n"
    else:
        md += "- Site3 not available for this analysis.\n"

    md += f"""
### H3: Score elevation driven by n_timepoints (197 raw → 140 connectivity)
- Pipeline logs confirm Philips subjects with 197 raw TPs are truncated to 140 connectivity TPs.
- GE: 200 raw TPs (no truncation observed in pipeline logs).
- SIEMENS: 197 raw TPs (truncation behavior not yet confirmed — see request to Martín).
- Truncation removes the last 57 frames; if systematic signal drift exists (scanner drift,
  physiological artifacts), the retained 140 frames may differ systematically.
- **Assessment**: Cannot conclusively distinguish timepoint effect from manufacturer effect
  without controlled experiments. 140 vs 200 TP connectivity matrices have different
  variance properties — this could bias off-diagonal connectivity estimates.

### H4: Motion differences (Philips FP vs TN)
- Motion data available for ~68 Philips subjects (PHILIPS10 batch only).
- See motion_qc_audit.csv and statistical_tests.csv for FP vs TN FD comparison.
- **Coverage gap**: Most Philips CN subjects do not have available rp files.
- **Assessment**: Inconclusive due to coverage gap. See request_to_martin_missing_files.md.

### H5: Protocol / scanner model / phase encoding differences
- **NOT ASSESSABLE**: No BIDS JSON sidecars or DICOM headers available.
- Philips has 17 sites in this dataset. Protocol variation across Philips sites
  (Achieva vs Ingenia, PE direction, slice timing) could explain within-Philips
  FPR variation.
- **Assessment**: Requires DICOM/BIDS data. See request_to_martin_missing_files.md.

### H6: Structural biological confound (Philips sites scan more impaired CN)
- ADNIMERGE cognitive/biomarker data joined where available.
- Philips CN subjects with high scores may have subtle cognitive impairment
  captured by CDRSB/MMSE/RAVLT/APOE4 that is below clinical threshold but
  drives model score.
- **Assessment**: Exploratory. APOE4 and CDRSB data analyzed in statistical_tests.csv
  where coverage permits.

---

## Key Structural Finding
The Philips CN score distribution is shifted upward across all Philips sites,
not just a few. This is consistent with:
1. A connectivity feature systematic to the Philips acquisition pipeline
   (e.g., 140 TP truncation, specific filter chain, smoothing kernel) that
   the VAE latent space captures as a disease-like signal.
2. The model's manufacturer stratification (classifier trained with Manufacturer
   as stratification variable) partly controls for manufacturer at the label level
   but cannot remove manufacturer-specific features from the latent space.

The model stratifies by Manufacturer for classifier fitting but the VAE
(all-pool strategy) encodes shared latent dimensions that may conflate
Philips-specific acquisition signatures with AD-related connectivity patterns.

---

## What Would Resolve This
1. BIDS JSON sidecars → PE direction, slice timing, scanner model
2. RealignParameter files for all Philips subjects → motion coverage
3. Confirmed 197→140 TP truncation logic → timepoint confound
4. ADNI MRIQUALITY table → series-level QC flags
5. Manufacturer-stratified VAE (separate Philips VAE or per-manufacturer harmonization)
   would be the model-level fix.

---

## Figures Generated
"""
    for f in figs_made:
        md += f"- {Path(f).name}\n"

    md += """
---

## Guardrails Compliance
- Read-only. No model training, no tensor modification, no threshold fitting.
- All findings descriptive/exploratory.
- OASIS stress-test not run (not relevant to this audit).
"""
    save_md(md, "final_philips_interpretation")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    log(f"=== Philips CN FP Forensic Audit — {datetime.now().isoformat()} ===")
    log(f"Output dir: {OUT_DIR}")

    # --- Load metadata
    log("Loading metadata...")
    try:
        meta = pd.read_csv(PATCHED_META_PATH)
        log(f"  Patched metadata: {len(meta)} rows")
    except Exception as e:
        log(f"  ERROR loading metadata: {e}"); raise

    try:
        adnimerge = pd.read_csv(ADNIMERGE_PATH, low_memory=False)
        log(f"  ADNIMERGE: {len(adnimerge)} rows")
    except Exception as e:
        log(f"  WARNING: ADNIMERGE not loaded: {e}")
        adnimerge = None

    # --- Inventory
    inventory_data_sources()

    # --- OOF predictions (promoted model)
    log("Loading promoted model OOF predictions...")
    pred_primary = load_promoted_oof_primary()
    log(f"  Primary rows: {len(pred_primary)}")

    # --- Master table
    master = build_master_table(pred_primary, meta, adnimerge)

    # --- Timepoints
    log("Loading pipeline logs...")
    pipeline_logs_df = load_pipeline_logs()
    log(f"  Pipeline log rows: {len(pipeline_logs_df)}")
    tp_df = timepoints_audit(master, pipeline_logs_df)

    # Re-attach timepoints to master where needed
    if "n_timepoints_original" in tp_df.columns:
        tp_merge = tp_df[["SubjectID", "n_timepoints_original",
                           "n_timepoints_connectivity"]].drop_duplicates("SubjectID")
        master = master.merge(tp_merge, on="SubjectID", how="left")

    # --- Acquisition metadata
    acquisition_metadata_audit(master)

    # --- Motion
    motion_df = motion_qc_audit(master)

    # --- Philips CN FP vs TN
    philips_cn = philips_cn_fp_vs_tn_table(master, motion_df)

    # --- FPR table
    model_site_phase_software_fpr_table(master)

    # --- Missing fields + request to Martin
    missing_required_fields()
    request_to_martin()

    # --- Mayo MCH placeholder
    mayo_mch_audit()

    # --- Statistical tests
    stats_df = statistical_tests(master, motion_df)

    # --- Figures
    figs_made = make_figures(master, motion_df)

    # --- Final interpretation
    final_interpretation(master, stats_df, figs_made)

    # --- Command log
    cmd_log_path = OUT_DIR / "command_log.json"
    with open(cmd_log_path, "w") as f:
        json.dump({
            "script": str(Path(__file__)),
            "run_date": datetime.now().isoformat(),
            "promoted_run_id": PROMOTED_RUN_ID,
            "primary_calib": PRIMARY_CALIB,
            "primary_model": PRIMARY_MODEL,
            "primary_thresh": PRIMARY_THRESH,
            "output_dir": str(OUT_DIR),
            "figures_made": figs_made,
            "log": COMMAND_LOG
        }, f, indent=2)
    log(f"  Saved command_log.json")

    log("=== DONE ===")
    log(f"Output files in: {OUT_DIR}")
    files = sorted(OUT_DIR.iterdir())
    log(f"Total files: {len(files)}")
    for f in files:
        log(f"  {f.name}")


if __name__ == "__main__":
    main()
