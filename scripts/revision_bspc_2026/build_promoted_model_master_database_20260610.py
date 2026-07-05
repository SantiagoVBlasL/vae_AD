#!/usr/bin/env python3
"""
Build authoritative master metadata CSV for the promoted ADNI rs-fMRI AD-vs-CN model.

Promoted model: recover035_latent384_beta3p75_T80_h10000_p560_full5x5
  channels [1,0,2] = Pearson_Full / OMST / MI, latent_dim=384, beta_vae=3.75,
  T0=80, patience=560, OOF-ECDF calibration, logreg_l2, z_plus_age_sex.

Output directory: results/revision_bspc_2026/promoted_model_master_database_20260610/

GUARDRAILS: read-only. No model training, no tensor modification, no metadata
modification, no model artifact modification.
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────────
# 0. Paths and configuration
# ────────────────────────────────────────────────────────────────

ROOT   = Path(__file__).resolve().parents[2]
DATA   = ROOT / "data"
DATOS  = Path("/media/diego/Datos")
RESULTS = ROOT / "results" / "revision_bspc_2026"
OUT_DIR = RESULTS / "promoted_model_master_database_20260610"

PROMOTED_RUN_ID = "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
PROMOTED_CONFIG = ROOT / "configs" / "runs" / f"adni_v5_1c_{PROMOTED_RUN_ID}.json"

GLOBAL_TENSOR_PATH = (
    DATOS / "vae_AD_data" / "revision_bspc_2026"
    / "adni_expanded_v5_1_batch20260514b_no_pybandpass"
    / "subject_tensors"
    / "GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
)
BATCH_METADATA_PATH = (
    DATOS / "vae_AD_data" / "revision_bspc_2026"
    / "adni_expanded_v5_1_batch20260514b_no_pybandpass"
    / "subject_metadata_v5_1_batch20260514b_no_pybandpass.csv"
)
TRAINING_READY_PATH = (
    DATOS / "vae_AD_data" / "revision_bspc_2026"
    / "adni_expanded_v5_1_batch20260514b_no_pybandpass"
    / "training_ready_metadata_v5_1_batch20260514b_no_pybandpass.csv"
)
PATCHED_METADATA_PATH = (
    RESULTS / "adni_035_metadata_rescue_preflight" / "patched_metadata_candidate.csv"
)
OOF_CALIB_PRED_PATH = (
    RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration"
    / "calib_predictions.csv"
)
CLF_SWEEP_PRED_PATH = (
    DATOS / "vae_AD_results" / "revision_bspc_2026" / PROMOTED_RUN_ID
    / "classifier_only_readout" / "classifier_sweep_predictions.csv"
)
SUBJECT_SCAN_MASTER_PATH = (
    RESULTS / "philips_cn_fpr_bold_metadata_forensic_audit_20260610"
    / "subject_scan_master_table.csv"
)
PHILIPS_PROV_TABLE_PATH = (
    RESULTS / "philips_cn_mat_tensor_provenance_audit_20260610"
    / "philips_cn_provenance_table.csv"
)
PHILIPS_RISK_TABLE_PATH = (
    RESULTS / "philips_protocol_risk_audit_20260610"
    / "philips_protocol_risk_modeling_table.csv"
)
BOLD_QC_CORRECTED_PATH = (
    RESULTS / "philips_cn_mat_tensor_provenance_audit_20260610"
    / "bold_qc_corrected_all_subjects.csv"
)
SUBJECTS_DATA_PATH = DATA / "SubjectsData_AAL3_procesado2.csv"

# Primary readout filter
PRIMARY_MODEL     = "logreg_l2_original"
PRIMARY_FEATURE   = "z_plus_age_sex"
PRIMARY_CALIB     = "oof_ecdf"
PRIMARY_THRESH    = "inner_oof_target_sens_ge_0p70_max_spec"

# Channel selection (config channels_to_use = [1, 0, 2])
SELECTED_CH_NAMES = ["Pearson_Full_FisherZ_Signed", "Pearson_OMST_GCE_Signed_Weighted", "MI_KNN_Symmetric"]
SELECTED_CH_IDX   = [1, 0, 2]  # index in global tensor channel dim

command_log: dict = {
    "script": str(Path(__file__).name),
    "promoted_run_id": PROMOTED_RUN_ID,
    "timestamp_start": datetime.now().isoformat(),
    "steps": []
}

def _log(msg: str):
    print(msg)
    command_log["steps"].append({"time": datetime.now().isoformat(), "msg": msg})

OUT_DIR.mkdir(parents=True, exist_ok=True)
_log(f"Output dir: {OUT_DIR}")

# ────────────────────────────────────────────────────────────────
# 1. Load and validate global tensor
# ────────────────────────────────────────────────────────────────
_log("S1: Loading global tensor...")

assert GLOBAL_TENSOR_PATH.exists(), f"Global tensor not found: {GLOBAL_TENSOR_PATH}"
npz = np.load(str(GLOBAL_TENSOR_PATH), allow_pickle=True)
tensor_data    = npz["global_tensor_data"]   # (N, 7, 131, 131)
tensor_sids    = npz["subject_ids"]           # (N,)
channel_names  = list(npz["channel_names"])
N, C, R1, R2   = tensor_data.shape

_log(f"  Tensor shape: {tensor_data.shape}  (N={N}, C={C}, ROIs={R1}×{R2})")
_log(f"  Channels: {channel_names}")
_log(f"  Selected channels (idx {SELECTED_CH_IDX}): {SELECTED_CH_NAMES}")

# Validate channel name alignment
for i, name in zip(SELECTED_CH_IDX, SELECTED_CH_NAMES):
    assert channel_names[i] == name, f"Channel mismatch at idx {i}: {channel_names[i]} != {name}"
_log("  Channel name alignment: OK")

# ────────────────────────────────────────────────────────────────
# 2. Compute tensor QC from global tensor (all 648 rows)
# ────────────────────────────────────────────────────────────────
_log("S2: Computing tensor QC from global tensor...")

def _offdiag_mask(n: int) -> np.ndarray:
    m = np.ones((n, n), dtype=bool)
    np.fill_diagonal(m, False)
    return m

_mask = _offdiag_mask(R1)
_ch_names_map = {name: idx for idx, name in enumerate(channel_names)}

tensor_qc_rows = []
for i in range(N):
    row = {"tensor_idx": i}
    for ch_name, ch_idx in zip(SELECTED_CH_NAMES, SELECTED_CH_IDX):
        mat = tensor_data[i, ch_idx]  # (131, 131)
        od  = mat[_mask]
        # short alias
        short = ch_name.split("_")[0][:4].lower()
        ch_tag = f"ch{ch_idx}"
        row[f"tensor_{ch_tag}_offdiag_mean"] = float(np.nanmean(od))
        row[f"tensor_{ch_tag}_offdiag_std"]  = float(np.nanstd(od))
        row[f"tensor_{ch_tag}_frob_norm"]    = float(np.sqrt(np.nansum(mat ** 2)))
    tensor_qc_rows.append(row)

df_tensor_qc = pd.DataFrame(tensor_qc_rows)
_log(f"  Tensor QC computed for {N} subjects (channels: ch1=Pearson_Full, ch0=OMST, ch2=MI)")

# ────────────────────────────────────────────────────────────────
# 3. Build base DataFrame from tensor subject_ids
# ────────────────────────────────────────────────────────────────
_log("S3: Building base DataFrame from tensor subject_ids...")

df_base = pd.DataFrame({
    "master_row_id": range(N),
    "tensor_idx":    range(N),
    "global_tensor_subject_id": tensor_sids,
    "global_tensor_path": str(GLOBAL_TENSOR_PATH),
    "selected_channels": str(SELECTED_CH_NAMES),
})
# tensor QC join
df_base = df_base.merge(df_tensor_qc, on="tensor_idx", how="left")
_log(f"  Base: {df_base.shape[0]} rows")

# ────────────────────────────────────────────────────────────────
# 4. Load and join batch_metadata (primary source, 648 rows)
# ────────────────────────────────────────────────────────────────
_log("S4: Loading batch_metadata (primary source)...")

df_batch = pd.read_csv(BATCH_METADATA_PATH)
_log(f"  batch_metadata shape: {df_batch.shape}")
_log(f"  ResearchGroup_Mapped: {df_batch['ResearchGroup_Mapped'].value_counts().to_dict()}")
_log(f"  Manufacturer: {df_batch['Manufacturer'].value_counts().to_dict()}")

# Rename tensor_index → tensor_idx for join consistency
df_batch = df_batch.rename(columns={"tensor_index": "tensor_idx_bm"})

# SubjectID alignment check
batch_sids  = df_batch["SubjectID"].values
tensor_only = set(tensor_sids) - set(batch_sids)
batch_only  = set(batch_sids)  - set(tensor_sids)
_log(f"  Tensor-only subjects: {len(tensor_only)}")
_log(f"  Batch-only subjects:  {len(batch_only)}")

# Rename columns that may conflict after merge
df_batch = df_batch.rename(columns={
    "metadata_source": "batch_metadata_source",
    "source_label":    "source_label",
})

df_base = df_base.merge(
    df_batch, left_on="global_tensor_subject_id", right_on="SubjectID", how="left"
)
_log(f"  After batch_metadata join: {df_base.shape}")

# ────────────────────────────────────────────────────────────────
# 5. Load and join patched_metadata (config-used, 647 rows)
# ────────────────────────────────────────────────────────────────
_log("S5: Loading patched_metadata (config-used)...")

df_patched = pd.read_csv(PATCHED_METADATA_PATH)
_log(f"  patched_metadata shape: {df_patched.shape}")
patched_sids = set(df_patched["SubjectID"])

df_base["in_config_metadata"] = df_base["global_tensor_subject_id"].isin(patched_sids)
patched_only = patched_sids - set(tensor_sids)
tensor_not_patched = set(tensor_sids) - patched_sids
_log(f"  Subjects in config but not tensor: {len(patched_only)}")
_log(f"  Tensor subjects not in patched: {len(tensor_not_patched)}")

# ────────────────────────────────────────────────────────────────
# 6. Load OOF calibrated predictions (primary readout)
# ────────────────────────────────────────────────────────────────
_log("S6: Loading OOF calibrated predictions...")

df_oof_raw = pd.read_csv(OOF_CALIB_PRED_PATH)
_log(f"  calib_predictions raw shape: {df_oof_raw.shape}")

# Filter to primary readout
df_oof_prim = df_oof_raw[
    (df_oof_raw["model_name"]    == PRIMARY_MODEL)  &
    (df_oof_raw["feature_set"]   == PRIMARY_FEATURE) &
    (df_oof_raw["calib_method"]  == PRIMARY_CALIB)   &
    (df_oof_raw["threshold_strategy"] == PRIMARY_THRESH)
].copy()
_log(f"  Primary readout rows: {len(df_oof_prim)} (expect {df_oof_raw['SubjectID'].nunique()} unique subjects)")

# One row per subject (fold is the outer fold assignment for that subject)
# Each subject appears exactly once (in their outer test fold)
assert df_oof_prim["SubjectID"].nunique() == len(df_oof_prim), \
    f"Duplicate subjects in primary OOF readout: {len(df_oof_prim)} rows but {df_oof_prim['SubjectID'].nunique()} unique"

df_oof_prim = df_oof_prim.rename(columns={
    "fold":         "outer_fold",
    "y_score":      "y_score_oof_ecdf",
    "y_score_raw":  "y_score_raw_oof",
    "y_pred":       "y_pred_primary",
    "threshold":    "threshold_primary",
    "y_true":       "y_true_oof",
}).copy()

# Also add all-calib raw score (same for all methods)
# Get the raw score from raw calib method
df_oof_raw_only = df_oof_raw[
    (df_oof_raw["model_name"]   == PRIMARY_MODEL) &
    (df_oof_raw["feature_set"]  == PRIMARY_FEATURE) &
    (df_oof_raw["calib_method"] == "raw") &
    (df_oof_raw["threshold_strategy"] == PRIMARY_THRESH)
][["SubjectID", "tensor_idx", "y_score_raw", "y_true"]].drop_duplicates("SubjectID")

df_oof_raw_only = df_oof_raw_only.rename(columns={
    "y_score_raw": "y_score_raw_stageB",
    "y_true":      "y_true_supervised",
})

# Also get the logitz-calibrated score
df_oof_logitz = df_oof_raw[
    (df_oof_raw["model_name"]   == PRIMARY_MODEL) &
    (df_oof_raw["feature_set"]  == PRIMARY_FEATURE) &
    (df_oof_raw["calib_method"] == "oof_logitz") &
    (df_oof_raw["threshold_strategy"] == PRIMARY_THRESH)
][["SubjectID", "y_score"]].drop_duplicates("SubjectID")
df_oof_logitz = df_oof_logitz.rename(columns={"y_score": "y_score_oof_logitz"})

# Join to base
keep_oof = ["SubjectID", "outer_fold", "y_score_raw_oof", "y_score_oof_ecdf",
            "y_pred_primary", "threshold_primary", "y_true_oof"]
df_base = df_base.merge(
    df_oof_prim[keep_oof],
    left_on="global_tensor_subject_id", right_on="SubjectID",
    how="left", suffixes=("", "_oof")
)
df_base = df_base.drop(columns=["SubjectID_oof"], errors="ignore")

df_base = df_base.merge(
    df_oof_raw_only,
    left_on="global_tensor_subject_id", right_on="SubjectID",
    how="left", suffixes=("", "_rawonly")
)
df_base = df_base.drop(columns=["SubjectID_rawonly", "tensor_idx_rawonly"], errors="ignore")

df_base = df_base.merge(
    df_oof_logitz,
    left_on="global_tensor_subject_id", right_on="SubjectID",
    how="left"
)
df_base = df_base.drop(columns=["SubjectID_x", "SubjectID_y"], errors="ignore")

_log(f"  After OOF join: {df_base.shape}")

# ────────────────────────────────────────────────────────────────
# 7. Load Stage B classifier sweep predictions (for Stage B y_pred/y_score)
# ────────────────────────────────────────────────────────────────
_log("S7: Loading Stage B classifier sweep predictions...")

if CLF_SWEEP_PRED_PATH.exists():
    df_clf = pd.read_csv(CLF_SWEEP_PRED_PATH)
    _log(f"  clf_sweep_predictions shape: {df_clf.shape}")
    # Filter to primary model+feature
    df_clf_prim = df_clf[
        (df_clf["model_name"] == "logreg_l2") &
        (df_clf["readout_feature_set"] == PRIMARY_FEATURE)
    ].copy()
    _log(f"  Stage B primary rows: {len(df_clf_prim)}")
    # Each subject has 1 row per threshold_strategy; use inner_oof_youden_j as default
    df_clf_prim = df_clf_prim[
        df_clf_prim["threshold_strategy"] == "inner_oof_youden_j"
    ].copy()
    df_clf_prim = df_clf_prim.rename(columns={
        "y_score": "y_score_stageB",
        "y_pred":  "y_pred_stageB",
        "fold":    "outer_fold_stageB",
    })
    keep_clf = ["SubjectID", "y_score_stageB", "y_pred_stageB"]
    df_base = df_base.merge(
        df_clf_prim[keep_clf].drop_duplicates("SubjectID"),
        left_on="global_tensor_subject_id", right_on="SubjectID",
        how="left"
    )
    df_base = df_base.drop(columns=["SubjectID_stageB", "SubjectID_x", "SubjectID_y"], errors="ignore")
    _log(f"  After Stage B join: {df_base.shape}")
else:
    _log(f"  WARNING: clf_sweep_predictions not found: {CLF_SWEEP_PRED_PATH}")
    df_base["y_score_stageB"] = np.nan
    df_base["y_pred_stageB"]  = np.nan

# ────────────────────────────────────────────────────────────────
# 8. Load subject_scan_master_table (clinical + BOLD QC, 397 CN+AD rows)
# ────────────────────────────────────────────────────────────────
_log("S8: Loading subject_scan_master_table (clinical + BOLD QC)...")

df_master = pd.read_csv(SUBJECT_SCAN_MASTER_PATH)
_log(f"  subject_scan_master_table shape: {df_master.shape}")
_log(f"  Manufacturer dist: {df_master['Manufacturer'].value_counts().to_dict()}")

# Rename to avoid conflicts with batch_metadata columns already loaded
rename_master = {
    "fold":                "outer_fold_forensic",
    "y_true":              "y_true_forensic",
    "y_pred":              "y_pred_forensic",
    "y_score":             "y_score_forensic",
    "y_score_raw":         "y_score_raw_forensic",
    "error_type":          "confusion_label_forensic",
    "tensor_idx":          "tensor_idx_forensic",
    "n_timepoints_resolved": "n_timepoints_resolved",
}
df_master = df_master.rename(columns=rename_master)

# Drop columns already in batch_metadata to avoid conflicts (keep master's richer clinical)
drop_dup_master = [
    "ResearchGroup_Mapped", "Diagnosis", "Age", "Sex", "Manufacturer", "Site3",
    "ImageID", "Visit", "metadata_source", "source_label",
    "ledger_scope", "dicom_series_ok", "python_bandpass_applied", "scale_label",
    # tensor_idx duplicates
    "tensor_idx_forensic",
]
df_master = df_master.drop(columns=[c for c in drop_dup_master if c in df_master.columns])

# Clinical columns to keep from master (not in batch_metadata)
clinical_cols = ["FLDSTRENG", "SITE", "COLPROT", "ORIGPROT", "APOE4", "CDRSB", "MMSE",
                 "RAVLT_immediate", "FAQ"]
qc_cols = [c for c in df_master.columns if any(x in c for x in [
    "mat_available", "n_timepoints_bold", "n_rois_bold", "n_nan_total",
    "finite_fraction_bold", "mean_signal_global", "std_signal_global",
    "median_roi_std", "min_roi_std", "max_roi_std", "n_low_variance_rois",
    "tsnr_proxy", "drift_slope", "drift_r2", "droi_rms",
    "outlier_frame_fraction", "bold_load_error",
    "rp_available", "n_timepoints_rp",
    "fd_mean", "fd_median", "fd_max", "fd_std", "fd_frac_gt",
    "fd_3mm", "fd_3deg", "fd_load_error", "mat_source_dir",
])]
keep_master = ["SubjectID"] + [c for c in clinical_cols + qc_cols + ["n_timepoints_resolved"]
                                if c in df_master.columns]
# Add forensic audit cols
keep_master += [c for c in df_master.columns if c.endswith("_forensic")]
df_master_slim = df_master[keep_master].drop_duplicates("SubjectID")

df_base = df_base.merge(
    df_master_slim,
    left_on="global_tensor_subject_id", right_on="SubjectID",
    how="left"
)
df_base = df_base.drop(columns=["SubjectID_x", "SubjectID_y"], errors="ignore")
_log(f"  After subject_scan_master join: {df_base.shape}")

# ────────────────────────────────────────────────────────────────
# 9. Load Philips CN provenance table (99 rows)
# ────────────────────────────────────────────────────────────────
_log("S9: Loading Philips CN provenance table...")

df_phil_prov = pd.read_csv(PHILIPS_PROV_TABLE_PATH)
_log(f"  philips_cn_provenance_table shape: {df_phil_prov.shape}")

rename_prov = {
    "error_type":             "philips_cn_error_type_prov",
    "y_score":                "y_score_prov",
    "Age":                    "Age_prov",
    "Site3":                  "Site3_prov",
    "fold":                   "outer_fold_prov",
    "tensor_idx":             "tensor_idx_prov",
    "source_batch":           "source_batch_prov",
    "preprocessing_source":   "preprocessing_source_phil",
    "scale_label":            "scale_label_prov",
    "n_timepoints_raw_manifest": "n_timepoints_raw_manifest",
    "manifest_signal_path":   "roisignals_mat_path_manifest",
    "manifest_signal_path_exists": "roisignals_mat_exists_manifest",
    "mat_available":          "mat_available_prov",
    "mat_path":               "mat_path",
    "mat_found_in":           "mat_found_in",
    "prior_audit_mat_found":  "prior_audit_mat_found",
    "mat_newly_recovered":    "mat_newly_recovered",
    "individual_tensor_available": "individual_tensor_available",
    "individual_tensor_path": "individual_tensor_path",
    "pipeline_log_name":      "pipeline_log_name",
    "pipeline_log_status":    "pipeline_log_status",
    "pipeline_log_path_saved": "pipeline_log_path_saved",
    "pipeline_log_status_preprocessing": "pipeline_log_status_preprocessing",
    "extraction_status":      "extraction_status",
    "extraction_raw_shape":   "extraction_raw_shape",
    "extraction_tensor_path": "extraction_tensor_path",
    "extraction_tensor_nan_count": "extraction_tensor_nan_count",
}
df_phil_prov = df_phil_prov.rename(columns=rename_prov)
keep_prov = ["SubjectID"] + [c for c in rename_prov.values() if c in df_phil_prov.columns]
df_phil_prov_slim = df_phil_prov[keep_prov].drop_duplicates("SubjectID")

df_base = df_base.merge(
    df_phil_prov_slim,
    left_on="global_tensor_subject_id", right_on="SubjectID",
    how="left"
)
df_base = df_base.drop(columns=["SubjectID_x", "SubjectID_y"], errors="ignore")
_log(f"  After Philips provenance join: {df_base.shape}")

# ────────────────────────────────────────────────────────────────
# 10. Load Philips CN protocol risk modeling table (99 rows)
# ────────────────────────────────────────────────────────────────
_log("S10: Loading Philips CN protocol risk modeling table...")

df_phil_risk = pd.read_csv(PHILIPS_RISK_TABLE_PATH)
_log(f"  philips_protocol_risk_modeling_table shape: {df_phil_risk.shape}")

# Columns of interest from risk table not yet loaded
risk_cols_new = ["SubjectID", "n_tp_raw", "ORIGPROT", "COLPROT", "PTGENDER",
                 "APOE4", "CDRSB", "MMSE",
                 "tsnr_proxy_median", "droi_rms", "drift_slope_median_abs",
                 "outlier_frame_fraction_rz_gt3", "outlier_frame_fraction_rz_gt4",
                 "ch0_offdiag_mean", "ch0_offdiag_std", "ch0_frob_norm",
                 "ch1_offdiag_mean", "ch1_offdiag_std", "ch1_frob_norm",
                 "ch2_offdiag_mean", "ch2_offdiag_std", "ch2_frob_norm",
                 "fp_binary", "n_tp_140", "n_tp_label", "adni3_flag"]
risk_cols_new = [c for c in risk_cols_new if c in df_phil_risk.columns]
df_phil_risk_slim = df_phil_risk[risk_cols_new].drop_duplicates("SubjectID")

# Rename to avoid collision with general ORIGPROT/COLPROT from subject_scan_master (already present)
# Keep Philips versions with _phil suffix only if already present; otherwise use direct
for col in ["ORIGPROT", "COLPROT", "APOE4", "CDRSB", "MMSE",
            "tsnr_proxy_median", "droi_rms", "drift_slope_median_abs",
            "outlier_frame_fraction_rz_gt3", "outlier_frame_fraction_rz_gt4"]:
    if col in df_base.columns:
        df_phil_risk_slim = df_phil_risk_slim.rename(columns={col: f"{col}_phil"})

df_base = df_base.merge(
    df_phil_risk_slim,
    left_on="global_tensor_subject_id", right_on="SubjectID",
    how="left"
)
df_base = df_base.drop(columns=["SubjectID_x", "SubjectID_y"], errors="ignore")
_log(f"  After Philips risk join: {df_base.shape}")

# ────────────────────────────────────────────────────────────────
# 11. Load corrected BOLD QC (99 Philips CN rows)
# ────────────────────────────────────────────────────────────────
_log("S11: Loading corrected Philips BOLD QC...")

df_bold_qc = pd.read_csv(BOLD_QC_CORRECTED_PATH)
_log(f"  bold_qc_corrected shape: {df_bold_qc.shape}")

rename_bqc = {
    "n_tp_raw":             "n_tp_raw_bold_corrected",
    "tsnr_proxy_median":    "tsnr_proxy_median_corrected",
    "tsnr_proxy_q25":       "tsnr_proxy_q25",
    "tsnr_proxy_q75":       "tsnr_proxy_q75",
    "droi_rms":             "droi_rms_corrected",
    "drift_slope_median_abs": "drift_slope_median_abs_corrected",
    "drift_r2_median":      "drift_r2_median",
    "outlier_frame_fraction_rz_gt3": "outlier_frame_fraction_rz_gt3_corrected",
    "outlier_frame_fraction_rz_gt4": "outlier_frame_fraction_rz_gt4_corrected",
    "finite_fraction_bold": "finite_fraction_bold_corrected",
    "n_roi":                "n_rois_bold_corrected",
    "mean_signal_global":   "mean_signal_global_corrected",
    "std_signal_global":    "std_signal_global_corrected",
    "median_roi_std":       "median_roi_std_corrected",
}
df_bold_qc = df_bold_qc.rename(columns=rename_bqc)
bqc_keep = ["SubjectID"] + [c for c in rename_bqc.values() if c in df_bold_qc.columns]
df_bold_qc_slim = df_bold_qc[[c for c in bqc_keep if c in df_bold_qc.columns]].drop_duplicates("SubjectID")

df_base = df_base.merge(
    df_bold_qc_slim,
    left_on="global_tensor_subject_id", right_on="SubjectID",
    how="left"
)
df_base = df_base.drop(columns=["SubjectID_x", "SubjectID_y"], errors="ignore")
_log(f"  After BOLD QC join: {df_base.shape}")

# ────────────────────────────────────────────────────────────────
# 12. Load SubjectsData_AAL3 for TR/TE, ImagingProtocol, Phase
# ────────────────────────────────────────────────────────────────
_log("S12: Loading SubjectsData_AAL3_procesado2 (TR/TE/Phase)...")

df_subdata = pd.read_csv(SUBJECTS_DATA_PATH)
_log(f"  SubjectsData_AAL3 shape: {df_subdata.shape}")

# Rename ambiguous fields
rename_sub = {
    "Phase":              "inferred_ADNI_phase",
    "TR":                 "TR",
    "TE":                 "TE",
    "Field Strength":     "FLDSTRENG_subdata",
    "Description":        "series_description",
    "ImagingProtocol":    "protocol_name",
    "Slice Thickness":    "slice_thickness",
    "AcqDate" if "AcqDate" in df_subdata.columns else "StudyDate": "acquisition_date",
}
df_subdata = df_subdata.rename(columns=rename_sub)

subdata_keep = ["SubjectID", "inferred_ADNI_phase", "TR", "TE",
                "FLDSTRENG_subdata", "series_description", "protocol_name",
                "slice_thickness", "acquisition_date",
                "PTEDUCAT", "MOCA", "Ventricles", "Hippocampus", "WholeBrain", "MidTemp",
                "ABETA", "TAU", "PTAU"]
subdata_keep_real = [c for c in subdata_keep if c in df_subdata.columns]
df_subdata_slim = df_subdata[subdata_keep_real].drop_duplicates("SubjectID")

df_base = df_base.merge(
    df_subdata_slim,
    left_on="global_tensor_subject_id", right_on="SubjectID",
    how="left"
)
df_base = df_base.drop(columns=["SubjectID_x", "SubjectID_y"], errors="ignore")
_log(f"  After SubjectsData join: {df_base.shape}")

# ────────────────────────────────────────────────────────────────
# 13. Compute derived columns
# ────────────────────────────────────────────────────────────────
_log("S13: Computing derived columns...")

# Canonical SubjectID
df_base["SubjectID"] = df_base["global_tensor_subject_id"]

# Pool membership flags
df_base["in_global_tensor"]          = True
df_base["in_selected_channel_tensor"] = True  # same tensor, channels selected at runtime

# VAE pool = all subjects in the tensor (CN, AD, MCI)
df_base["in_vae_pool"] = df_base["training_ready"].fillna(False)

# Classifier pool = CN + AD (not MCI), not excluded, in patched_metadata
supervised_mask = (
    df_base["ResearchGroup_Mapped"].isin(["CN", "AD"]) &
    (~df_base["exclude_from_supervised"].fillna(False)) &
    df_base["in_config_metadata"]
)
df_base["in_stageB_classifier_pool"] = supervised_mask

# OOF evaluable = subjects with an OOF prediction
df_base["in_oof_evaluation"]       = df_base["y_score_oof_ecdf"].notna()
df_base["in_mci_unsupervised_only"] = df_base["ResearchGroup_Mapped"] == "MCI"

# y_true (supervised label: 1=AD, 0=CN, NaN=MCI/other)
if "y_true_supervised" not in df_base.columns:
    df_base["y_true_supervised"] = np.where(
        df_base["ResearchGroup_Mapped"] == "AD", 1,
        np.where(df_base["ResearchGroup_Mapped"] == "CN", 0, np.nan)
    )

df_base["y_true_label"] = df_base["ResearchGroup_Mapped"]

# y_score_final = oof_ecdf (primary)
df_base["y_score_final"] = df_base["y_score_oof_ecdf"]

# Confusion label from primary readout
def _confusion_label(row):
    if pd.isna(row["y_true_supervised"]) or pd.isna(row["y_pred_primary"]):
        return "NA"
    yt, yp = int(row["y_true_supervised"]), int(row["y_pred_primary"])
    if yt == 1 and yp == 1: return "TP"
    if yt == 1 and yp == 0: return "FN"
    if yt == 0 and yp == 0: return "TN"
    if yt == 0 and yp == 1: return "FP"
    return "NA"

df_base["confusion_label"] = df_base.apply(_confusion_label, axis=1)
df_base["error_type"] = df_base["confusion_label"].map(
    {"TP": "TP", "TN": "TN", "FP": "FP", "FN": "FN", "NA": "NA"}
)
df_base["promoted_model_score_available"] = df_base["y_score_oof_ecdf"].notna()
df_base["promoted_model_score_source"]    = np.where(
    df_base["y_score_oof_ecdf"].notna(),
    "recover035_latent384_beta3p75_stageB_oof_score_calibration/calib_predictions.csv",
    "MISSING"
)

# Duplicate flags
df_base["duplicate_subject_flag"] = df_base["SubjectID"].duplicated(keep=False)
df_base["duplicate_image_flag"]   = df_base["ImageID"].duplicated(keep=False)

# Model / calibration metadata for primary readout
df_base["model_name"]         = PRIMARY_MODEL
df_base["feature_set"]        = PRIMARY_FEATURE
df_base["calib_method"]       = PRIMARY_CALIB
df_base["threshold_strategy"] = PRIMARY_THRESH

# Manufacturer normalisation
mfr_map = {"Philips": "Philips", "SIEMENS": "SIEMENS", "GE": "GE"}
df_base["Manufacturer_normalized"] = df_base["Manufacturer"].map(mfr_map)

# n_tp_raw — best available: from risk table (Philips CN), else n_timepoints_raw from batch_metadata
if "n_tp_raw" in df_base.columns:
    df_base["n_timepoints_raw_best"] = df_base["n_tp_raw"].combine_first(
        df_base.get("n_timepoints_raw", pd.Series(np.nan, index=df_base.index))
    )
else:
    df_base["n_timepoints_raw_best"] = df_base.get("n_timepoints_raw", np.nan)

df_base["raw_tp_group"] = df_base["n_timepoints_raw_best"].map(
    lambda v: "140" if v == 140 else ("197" if v == 197 else ("200" if v == 200 else ("other" if pd.notna(v) else "UNKNOWN")))
)

# Protocol risk group (Philips CN specific)
def _philips_risk_group(row):
    if row["Manufacturer"] != "Philips" or row["ResearchGroup_Mapped"] != "CN":
        return "not_philips_cn"
    ntp = row.get("n_tp_raw", np.nan)
    if pd.isna(ntp):
        return "unknown_ntp"
    return "high_risk_140tp" if ntp == 140 else "lower_risk_197tp"

df_base["protocol_risk_group"] = df_base.apply(_philips_risk_group, axis=1)

# ────────────────────────────────────────────────────────────────
# Philips-specific flags
# ────────────────────────────────────────────────────────────────
df_base["is_philips_cn"] = (
    (df_base["Manufacturer"] == "Philips") & (df_base["ResearchGroup_Mapped"] == "CN")
)

# philips_cn_error_type: FP/TN from OOF primary readout (Philips CN only)
df_base["philips_cn_error_type"] = np.where(
    df_base["is_philips_cn"],
    df_base["confusion_label"],
    "not_philips_cn"
)

# philips_cn_high_protocol_risk: 140 TP group
df_base["philips_cn_high_protocol_risk"] = (
    df_base["is_philips_cn"] & (df_base["n_timepoints_raw_best"] == 140)
)

# philips_cn_raw_tp_group
df_base["philips_cn_raw_tp_group"] = np.where(
    df_base["is_philips_cn"],
    df_base["raw_tp_group"],
    "not_philips_cn"
)

# philips_ntp_fpr_group: from the protocol risk audit
def _ntp_fpr_group(row):
    if not row["is_philips_cn"]: return "not_philips_cn"
    ntp = row.get("n_tp_raw", np.nan)
    if pd.isna(ntp): return "unknown_ntp"
    return "140tp_FPR064" if ntp == 140 else "197tp_FPR033"

df_base["philips_ntp_fpr_group_if_available"] = df_base.apply(_ntp_fpr_group, axis=1)

# Problem site flag (sites 301, 53, 13 had 100% FPR from forensic audit)
problem_sites = {301.0, 53.0, 13.0}
df_base["philips_problem_site_flag"] = (
    df_base["is_philips_cn"] & df_base["Site3"].isin(problem_sites)
)

# Site-level FPR placeholder (from site3_fpr_summary.csv if available)
site_fpr_path = RESULTS / "philips_cn_fpr_bold_metadata_forensic_audit_20260610" / "site3_fpr_summary.csv"
if site_fpr_path.exists():
    df_site_fpr = pd.read_csv(site_fpr_path)
    # map Site3 → FPR
    site_fpr_map = {}
    if "Site3" in df_site_fpr.columns and "FPR" in df_site_fpr.columns:
        site_fpr_map = dict(zip(df_site_fpr["Site3"], df_site_fpr["FPR"]))
    df_base["philips_site_fpr_if_available"] = df_base["Site3"].map(site_fpr_map)
else:
    df_base["philips_site_fpr_if_available"] = np.nan

# ORIGPROT/COLPROT: prefer master table, fall back to philips risk table
if "ORIGPROT_phil" in df_base.columns and "ORIGPROT" not in df_base.columns:
    df_base["ORIGPROT"] = df_base["ORIGPROT_phil"]
elif "ORIGPROT_phil" in df_base.columns:
    # Merge: use ORIGPROT from subject_scan_master, fill from phil where missing
    df_base["ORIGPROT"] = df_base["ORIGPROT"].combine_first(df_base["ORIGPROT_phil"])

if "COLPROT_phil" in df_base.columns and "COLPROT" not in df_base.columns:
    df_base["COLPROT"] = df_base["COLPROT_phil"]
elif "COLPROT_phil" in df_base.columns:
    df_base["COLPROT"] = df_base["COLPROT"].combine_first(df_base["COLPROT_phil"])

# inferred_ADNI_phase from ORIGPROT if SubjectsData Phase is missing
if "inferred_ADNI_phase" not in df_base.columns:
    df_base["inferred_ADNI_phase"] = df_base.get("ORIGPROT", np.nan)
else:
    df_base["inferred_ADNI_phase"] = df_base["inferred_ADNI_phase"].combine_first(
        df_base.get("ORIGPROT", pd.Series(np.nan, index=df_base.index))
    )

# Source priority / conflict tracking
df_base["metadata_source_primary"] = "batch_metadata_v5_1_batch20260514b_no_pybandpass"
df_base["metadata_sources_merged"] = (
    "batch_metadata|patched_metadata|oof_calib_predictions|subject_scan_master|"
    "philips_cn_provenance|philips_protocol_risk|bold_qc_corrected|SubjectsData_AAL3"
)

# Source conflict flag: age mismatch between batch and subdata
if "Age_prov" in df_base.columns:
    df_base["source_conflict_flag"] = (
        df_base["Age"].notna() & df_base["Age_prov"].notna() &
        (df_base["Age"] - df_base["Age_prov"]).abs() > 1.0
    )
else:
    df_base["source_conflict_flag"] = False

df_base["source_priority_resolution"] = (
    "1=batch_metadata, 2=patched_metadata, 3=oof_calibration, "
    "4=subject_scan_master, 5=philips_risk_table, 6=SubjectsData_AAL3"
)

# Split preview: aligned?
df_base["y_pred"] = df_base["y_pred_primary"]
df_base["y_score"] = df_base["y_score_oof_ecdf"]

_log(f"  Derived columns computed. Shape: {df_base.shape}")

# ────────────────────────────────────────────────────────────────
# 14. Rename/reorder to canonical column spec
# ────────────────────────────────────────────────────────────────
_log("S14: Renaming and reordering columns...")

rename_final = {
    "global_tensor_subject_id": "SubjectID_tensor",
    "SubjectID":                "SubjectID",
    "tensor_idx":               "tensor_idx",
    "outer_fold":               "outer_fold",
    "y_true_supervised":        "y_true",
    "y_true_label":             "y_true_label",
    "y_score_raw_stageB":       "y_score_raw",
    "y_score_oof_ecdf":         "y_score_oof_ecdf",
    "y_score_oof_logitz":       "y_score_oof_logitz",
    "y_score_final":            "y_score_final",
    "y_pred_primary":           "y_pred",
    "threshold_primary":        "threshold",
    "confusion_label":          "confusion_label",
    "ResearchGroup_Mapped":     "ResearchGroup_Mapped",
    "Diagnosis":                "Diagnosis",
    "source_batch":             "source_batch",
    "source_label":             "source_label",
    "batch_metadata_source":    "metadata_source_batch",
    "n_timepoints_raw":         "n_timepoints_raw",
    "n_rois_raw":               "n_rois_raw",
    "finite_fraction":          "finite_fraction",
    "scale_label":              "scale_label",
    "roisignals_path":          "roisignals_mat_path",
    "n_timepoints_raw_best":    "n_timepoints_model_input",
}

# Apply only existing columns
rename_apply = {k: v for k, v in rename_final.items() if k in df_base.columns and k != v}
df_base = df_base.rename(columns=rename_apply)

# Ensure SubjectID is canonical
if "SubjectID" not in df_base.columns and "SubjectID_tensor" in df_base.columns:
    df_base["SubjectID"] = df_base["SubjectID_tensor"]

# Add remaining requested columns as MISSING if not present
requested_missing = [
    "VISCODE", "EXAMDATE", "scanner_model", "manufacturer_model_name",
    "software_version", "coil", "sequence_name", "station_name",
    "phase_encoding_direction", "phase_encoding_direction_raw",
    "slice_timing_available", "slice_timing_summary", "slice_order_inferred",
    "n_slices", "n_dummy_removed", "n_timepoints_final",
    "dicom_json_available", "dicom_json_path", "dicom_header_available", "dicom_header_source",
    "python_bandpass_applied", "bandpass_status",
    "philips_slice_order_issue_flag", "philips_slice_order_issue_source",
    "philips_phase_encoding_issue_flag", "philips_comment",
    "inner_fold", "classifier_name",
    "adni_metadata_source",
]
for col in requested_missing:
    if col not in df_base.columns:
        df_base[col] = "MISSING" if col.endswith(("_available","_flag","_source","_path","direction","summary","inferred")) else np.nan

# Fill boolean availability flags
for col in ["dicom_json_available", "dicom_header_available", "rp_available",
            "slice_timing_available", "python_bandpass_applied"]:
    if col in df_base.columns:
        df_base[col] = df_base[col].fillna(False)

# bandpass_status
if "python_bandpass_applied" in df_base.columns:
    df_base["bandpass_status"] = np.where(
        df_base["python_bandpass_applied"] == True, "python_bandpass_ON",
        np.where(df_base["python_bandpass_applied"] == False, "python_bandpass_OFF", "UNKNOWN")
    )

# tensor_npz_available (all subjects have a row in the global tensor)
df_base["tensor_npz_available"] = True
df_base["tensor_qc_available"]  = True

# protocol_metadata_available: True if ORIGPROT or n_tp_raw is non-null
df_base["protocol_metadata_available"] = (
    df_base.get("ORIGPROT", pd.Series(np.nan, index=df_base.index)).notna() |
    df_base.get("n_tp_raw", pd.Series(np.nan, index=df_base.index)).notna()
)

# clinical_metadata_available
df_base["clinical_metadata_available"] = df_base[
    [c for c in ["CDRSB", "MMSE", "APOE4"] if c in df_base.columns]
].notna().any(axis=1)

# adni_metadata_source
df_base["adni_metadata_source"] = np.where(
    df_base["ORIGPROT"].notna() if "ORIGPROT" in df_base.columns else False,
    "subject_scan_master_table|philips_protocol_risk_table",
    "MISSING"
)

_log(f"  Final shape before ordering: {df_base.shape}")

# ────────────────────────────────────────────────────────────────
# 15. Define and apply final column order
# ────────────────────────────────────────────────────────────────
_log("S15: Applying final column order...")

col_order = [
    # A: Row and provenance
    "master_row_id", "tensor_idx", "SubjectID", "SubjectID_tensor", "ImageID",
    "VISCODE", "Visit", "EXAMDATE",
    "global_tensor_path", "global_tensor_subject_id",
    "source_batch", "source_label", "metadata_source_batch",
    "metadata_source_primary", "metadata_sources_merged",
    "source_priority_resolution", "source_conflict_flag",
    "duplicate_subject_flag", "duplicate_image_flag",

    # B: Pool membership
    "in_global_tensor", "in_selected_channel_tensor", "selected_channels",
    "in_vae_pool", "in_stageB_classifier_pool", "in_oof_evaluation",
    "in_mci_unsupervised_only", "in_config_metadata",
    "training_ready", "exclude_from_supervised", "supervised_exclusion_reason",
    "y_true", "y_true_label", "ResearchGroup_Mapped", "Diagnosis",
    "diagnosis_original", "diagnosis_source",

    # C: Cross-validation and predictions
    "outer_fold", "inner_fold", "model_name", "feature_set", "classifier_name",
    "calib_method", "threshold_strategy", "threshold",
    "y_score_raw", "y_score_oof_ecdf", "y_score_oof_logitz", "y_score_final",
    "y_pred", "confusion_label", "error_type",
    "promoted_model_score_available", "promoted_model_score_source",

    # D: Demographics and clinical
    "Age", "Sex", "PTGENDER", "APOE4", "CDRSB", "MMSE", "RAVLT_immediate", "FAQ",
    "PTEDUCAT", "MOCA", "Ventricles", "Hippocampus", "WholeBrain", "MidTemp",
    "ABETA", "TAU", "PTAU",
    "clinical_metadata_available",

    # E: ADNI phase / site / scanner
    "COLPROT", "ORIGPROT", "inferred_ADNI_phase",
    "SITE", "Site3", "Manufacturer", "Manufacturer_normalized",
    "FLDSTRENG", "FLDSTRENG_subdata",
    "scanner_model", "manufacturer_model_name", "software_version",
    "coil", "sequence_name", "protocol_name", "series_description", "station_name",
    "acquisition_date", "adni_metadata_source",

    # F: Acquisition / protocol
    "TR", "TE", "flip_angle", "phase_encoding_direction", "phase_encoding_direction_raw",
    "slice_timing_available", "slice_timing_summary", "slice_order_inferred",
    "n_slices", "slice_thickness",
    "n_timepoints_raw", "n_timepoints_raw_manifest", "n_timepoints_model_input",
    "n_timepoints_final", "n_dummy_removed",
    "n_rois_raw", "n_timepoints_resolved",
    "raw_tp_group", "protocol_risk_group", "protocol_metadata_available",
    "dicom_json_available", "dicom_json_path",
    "dicom_header_available", "dicom_header_source",

    # G: Preprocessing / provenance
    "roisignals_mat_path", "roisignals_mat_path_manifest",
    "roisignals_mat_exists_manifest", "mat_available_prov",
    "mat_path", "mat_found_in", "mat_newly_recovered",
    "finite_fraction", "finite_fraction_bold_corrected",
    "scale_label", "preprocessing_source_phil",
    "python_bandpass_applied", "bandpass_status",
    "tensor_npz_available", "individual_tensor_path",
    "pipeline_log_name", "pipeline_log_status", "pipeline_log_path_saved",
    "pipeline_log_status_preprocessing",
    "extraction_status", "extraction_raw_shape",
    "extraction_tensor_path", "extraction_tensor_nan_count",

    # H: Motion / BOLD QC / tensor QC
    "rp_available", "n_timepoints_rp",
    "fd_mean", "fd_median", "fd_max", "fd_std",
    "fd_frac_gt0p3", "fd_frac_gt0p5", "fd_3mm_flag", "fd_3deg_flag",
    "tsnr_proxy_median_corrected", "tsnr_proxy_q25", "tsnr_proxy_q75",
    "droi_rms_corrected", "drift_slope_median_abs_corrected",
    "outlier_frame_fraction_rz_gt3_corrected", "outlier_frame_fraction_rz_gt4_corrected",
    "drift_r2_median", "median_roi_std_corrected", "mean_signal_global_corrected",
    "tensor_ch1_offdiag_mean", "tensor_ch1_offdiag_std", "tensor_ch1_frob_norm",
    "tensor_ch0_offdiag_mean", "tensor_ch0_offdiag_std", "tensor_ch0_frob_norm",
    "tensor_ch2_offdiag_mean", "tensor_ch2_offdiag_std", "tensor_ch2_frob_norm",
    "tensor_qc_available",

    # I: Philips-specific flags
    "is_philips_cn", "philips_cn_error_type", "philips_cn_raw_tp_group",
    "philips_cn_high_protocol_risk", "philips_problem_site_flag",
    "philips_site_fpr_if_available", "philips_ntp_fpr_group_if_available",
    "philips_slice_order_issue_flag", "philips_slice_order_issue_source",
    "philips_phase_encoding_issue_flag", "philips_comment",
]

# Only keep columns that exist in df_base; deduplicate preserving order
seen = set()
col_final = []
for c in col_order:
    if c in df_base.columns and c not in seen:
        col_final.append(c)
        seen.add(c)
# Add any remaining columns not in the spec
extra_cols = [c for c in df_base.columns if c not in seen]
if extra_cols:
    _log(f"  Extra columns not in spec (appended): {extra_cols[:10]}{'...' if len(extra_cols)>10 else ''}")
col_final = col_final + extra_cols

df_master_db = df_base[col_final].copy()
_log(f"  Final master DB shape: {df_master_db.shape}")

# ────────────────────────────────────────────────────────────────
# 16. Analysis 1: Pool counts
# ────────────────────────────────────────────────────────────────
_log("S16: Computing pool counts...")

counts = {}
counts["global_tensor_N"] = len(df_master_db)
counts["vae_pool_N"] = int(df_master_db["in_vae_pool"].sum())
counts["classifier_pool_N"] = int(df_master_db["in_stageB_classifier_pool"].sum())
counts["oof_evaluable_N"] = int(df_master_db["in_oof_evaluation"].sum())
counts["CN_global"] = int((df_master_db["ResearchGroup_Mapped"] == "CN").sum())
counts["AD_global"] = int((df_master_db["ResearchGroup_Mapped"] == "AD").sum())
counts["MCI_global"] = int((df_master_db["ResearchGroup_Mapped"] == "MCI").sum())
counts["CN_classifier"] = int(((df_master_db["ResearchGroup_Mapped"]=="CN") & df_master_db["in_stageB_classifier_pool"]).sum())
counts["AD_classifier"] = int(((df_master_db["ResearchGroup_Mapped"]=="AD") & df_master_db["in_stageB_classifier_pool"]).sum())
_log(f"  Pool counts: {counts}")

# Manufacturer × Diagnosis counts
df_mfr_diag = df_master_db.groupby(
    ["Manufacturer", "ResearchGroup_Mapped"], dropna=False
).size().reset_index(name="N")
df_mfr_diag.to_csv(OUT_DIR / "promoted_model_pool_counts_by_group.csv", index=False)

# Manufacturer × raw_tp_group
df_mfr_ntp = df_master_db.groupby(
    ["Manufacturer", "raw_tp_group"], dropna=False
).size().reset_index(name="N")

# Manufacturer × ADNI phase
df_mfr_phase = df_master_db.groupby(
    ["Manufacturer", "inferred_ADNI_phase"], dropna=False
).size().reset_index(name="N")

# Write pool counts
pool_rows = []
for mfr in ["Philips", "SIEMENS", "GE", "ALL"]:
    sub = df_master_db if mfr == "ALL" else df_master_db[df_master_db["Manufacturer"] == mfr]
    for rg in ["CN", "AD", "MCI", "ALL"]:
        sub2 = sub if rg == "ALL" else sub[sub["ResearchGroup_Mapped"] == rg]
        pool_rows.append({
            "Manufacturer": mfr, "ResearchGroup": rg,
            "N_global": len(sub2),
            "N_vae_pool": int(sub2["in_vae_pool"].sum()),
            "N_classifier_pool": int(sub2["in_stageB_classifier_pool"].sum()),
            "N_oof_evaluable": int(sub2["in_oof_evaluation"].sum()),
        })
df_pool_counts = pd.DataFrame(pool_rows)
df_pool_counts.to_csv(OUT_DIR / "promoted_model_pool_counts_by_group.csv", index=False)
_log(f"  Pool counts saved.")

# ────────────────────────────────────────────────────────────────
# 17. Analysis 2: Alignment validation
# ────────────────────────────────────────────────────────────────
_log("S17: Alignment validation...")

conflicts = []

# Check tensor idx alignment
tensor_idx_mismatch = df_master_db[
    df_master_db["tensor_idx"] != df_master_db["master_row_id"]
]
if len(tensor_idx_mismatch) > 0:
    conflicts.append({
        "type": "tensor_idx_vs_master_row_id",
        "n_affected": len(tensor_idx_mismatch),
        "detail": "tensor_idx != master_row_id (unexpected reorder)"
    })
    _log(f"  WARNING: {len(tensor_idx_mismatch)} tensor_idx mismatches")

# Check SubjectID consistency
sid_mismatch = df_master_db[
    df_master_db["SubjectID"] != df_master_db["SubjectID_tensor"]
]
if len(sid_mismatch) > 0:
    conflicts.append({
        "type": "SubjectID_vs_SubjectID_tensor",
        "n_affected": len(sid_mismatch),
        "detail": "SubjectID != SubjectID_tensor"
    })

# Check OOF y_true vs ResearchGroup
oof_sub = df_master_db[df_master_db["in_oof_evaluation"] == True].copy()
oof_sub["y_true_inferred"] = (oof_sub["ResearchGroup_Mapped"] == "AD").astype(float)
ytrue_mismatch = oof_sub[oof_sub["y_true"] != oof_sub["y_true_inferred"]]
if len(ytrue_mismatch) > 0:
    conflicts.append({
        "type": "y_true_vs_ResearchGroup_Mapped",
        "n_affected": len(ytrue_mismatch),
        "detail": "y_true != ResearchGroup-derived label",
        "subjects": ytrue_mismatch["SubjectID"].tolist()[:5]
    })
    _log(f"  WARNING: {len(ytrue_mismatch)} y_true vs ResearchGroup mismatches")

df_conflicts = pd.DataFrame(conflicts) if conflicts else pd.DataFrame(columns=["type","n_affected","detail"])
df_conflicts.to_csv(OUT_DIR / "promoted_model_metadata_conflicts.csv", index=False)

# ────────────────────────────────────────────────────────────────
# 18. Analysis 3: Missingness report
# ────────────────────────────────────────────────────────────────
_log("S18: Missingness report...")

# Guard: remove any duplicate column names (keep first occurrence)
dup_cols = [c for c in df_master_db.columns if list(df_master_db.columns).count(c) > 1]
if dup_cols:
    _log(f"  WARNING: removing duplicate columns: {dup_cols}")
    df_master_db = df_master_db.loc[:, ~df_master_db.columns.duplicated()]
    # Also resave the CSV with deduped columns
    df_master_db.to_csv(OUT_DIR / "promoted_model_master_database.csv", index=False)

def _count_missing(series) -> int:
    # Guard against duplicate column names returning a DataFrame
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    try:
        n_nan = int(series.isna().sum())
    except (TypeError, ValueError):
        n_nan = int(series.isna().any().sum())
    try:
        n_str = int((series == "MISSING").sum())
    except Exception:
        n_str = 0
    return n_nan + n_str

miss_rows = []
for col in df_master_db.columns:
    total = len(df_master_db)
    n_miss = _count_missing(df_master_db[col])
    n_avail = total - n_miss
    for mfr in ["Philips", "SIEMENS", "GE"]:
        sub = df_master_db[df_master_db["Manufacturer"] == mfr]
        sub_miss = _count_missing(sub[col])
        sub_n = len(sub)
        miss_rows.append({
            "column": col, "manufacturer": mfr,
            "N_total": sub_n,
            "N_available": sub_n - sub_miss,
            "N_missing": sub_miss,
            "pct_available": round((sub_n - sub_miss) / sub_n * 100, 1) if sub_n > 0 else 0.0,
        })
    miss_rows.append({
        "column": col, "manufacturer": "ALL",
        "N_total": int(total), "N_available": int(n_avail),
        "N_missing": int(n_miss),
        "pct_available": round(float(n_avail) / float(total) * 100.0, 1),
    })

df_miss = pd.DataFrame(miss_rows)
df_miss.to_csv(OUT_DIR / "promoted_model_missingness_by_field.csv", index=False)

# Summary: ALL manufacturer only, sorted by pct_available
df_miss_summary = df_miss[df_miss["manufacturer"] == "ALL"].sort_values("pct_available")
_log(f"  Missingness: {len(df_miss_summary)} columns assessed")

# ────────────────────────────────────────────────────────────────
# 19. Analysis 4: Protocol summary by manufacturer
# ────────────────────────────────────────────────────────────────
_log("S19: Protocol summary by manufacturer...")

proto_rows = []
for mfr in ["Philips", "SIEMENS", "GE"]:
    sub = df_master_db[df_master_db["Manufacturer"] == mfr]
    sub_clf = sub[sub["in_stageB_classifier_pool"] == True]
    sub_oof = sub[sub["in_oof_evaluation"] == True]
    cn_oof = sub_oof[sub_oof["ResearchGroup_Mapped"] == "CN"]
    ad_oof = sub_oof[sub_oof["ResearchGroup_Mapped"] == "AD"]
    # FPR (CN FPR)
    cn_fp = int((cn_oof["confusion_label"] == "FP").sum())
    cn_n  = len(cn_oof)
    cn_fpr = round(cn_fp / cn_n, 4) if cn_n > 0 else np.nan
    # FNR (AD FNR)
    ad_fn = int((ad_oof["confusion_label"] == "FN").sum())
    ad_n  = len(ad_oof)
    ad_fnr = round(ad_fn / ad_n, 4) if ad_n > 0 else np.nan
    proto_rows.append({
        "Manufacturer": mfr,
        "N_global": len(sub),
        "N_vae_pool": int(sub["in_vae_pool"].sum()),
        "N_classifier_pool": int(sub_clf["in_stageB_classifier_pool"].sum()),
        "N_CN": int((sub["ResearchGroup_Mapped"]=="CN").sum()),
        "N_AD": int((sub["ResearchGroup_Mapped"]=="AD").sum()),
        "N_MCI": int((sub["ResearchGroup_Mapped"]=="MCI").sum()),
        "CN_FPR_primary": cn_fpr,
        "CN_FP": cn_fp,
        "CN_N": cn_n,
        "AD_FNR_primary": ad_fnr,
        "AD_FN": ad_fn,
        "AD_N": ad_n,
        "N_rp_available": int(sub["rp_available"].fillna(False).astype(bool).sum()),
        "N_ORIGPROT_available": int(sub["ORIGPROT"].notna().sum()),
        "N_CDRSB_available": int(sub["CDRSB"].notna().sum()) if "CDRSB" in sub.columns else 0,
        "N_n_tp_raw_available": int(sub.get("n_tp_raw", pd.Series(np.nan, index=sub.index)).notna().sum()),
    })

df_proto = pd.DataFrame(proto_rows)
df_proto.to_csv(OUT_DIR / "promoted_model_protocol_summary_by_manufacturer.csv", index=False)
_log(f"  Protocol summary saved.")

# ────────────────────────────────────────────────────────────────
# 20. Analysis 5: Philips CN protocol focus table
# ────────────────────────────────────────────────────────────────
_log("S20: Philips CN protocol focus table...")

df_phil_cn = df_master_db[df_master_db["is_philips_cn"] == True].copy()
_log(f"  Philips CN: {len(df_phil_cn)} subjects")

phil_focus_rows = []

# FPR by raw_tp_group
for grp in df_phil_cn["philips_cn_raw_tp_group"].unique():
    sub = df_phil_cn[df_phil_cn["philips_cn_raw_tp_group"] == grp]
    sub_oof = sub[sub["in_oof_evaluation"]]
    n_fp = int((sub_oof["confusion_label"] == "FP").sum())
    n_cn = len(sub_oof)
    phil_focus_rows.append({
        "stratification": "raw_tp_group", "group": str(grp),
        "N": n_cn, "N_FP": n_fp,
        "FPR": round(n_fp / n_cn, 4) if n_cn > 0 else np.nan,
        "median_score": round(sub_oof["y_score_oof_ecdf"].median(), 4) if n_cn > 0 else np.nan,
    })

# FPR by ADNI phase
for phase in df_phil_cn["inferred_ADNI_phase"].dropna().unique():
    sub = df_phil_cn[df_phil_cn["inferred_ADNI_phase"] == phase]
    sub_oof = sub[sub["in_oof_evaluation"]]
    n_fp = int((sub_oof["confusion_label"] == "FP").sum())
    n_cn = len(sub_oof)
    phil_focus_rows.append({
        "stratification": "ADNI_phase", "group": str(phase),
        "N": n_cn, "N_FP": n_fp,
        "FPR": round(n_fp / n_cn, 4) if n_cn > 0 else np.nan,
        "median_score": round(sub_oof["y_score_oof_ecdf"].median(), 4) if n_cn > 0 else np.nan,
    })

# FPR by Site3
for site in sorted(df_phil_cn["Site3"].dropna().unique()):
    sub = df_phil_cn[df_phil_cn["Site3"] == site]
    sub_oof = sub[sub["in_oof_evaluation"]]
    n_fp = int((sub_oof["confusion_label"] == "FP").sum())
    n_cn = len(sub_oof)
    phil_focus_rows.append({
        "stratification": "Site3", "group": str(site),
        "N": n_cn, "N_FP": n_fp,
        "FPR": round(n_fp / n_cn, 4) if n_cn > 0 else np.nan,
        "median_score": round(sub_oof["y_score_oof_ecdf"].median(), 4) if n_cn > 0 else np.nan,
    })

df_phil_focus = pd.DataFrame(phil_focus_rows)
df_phil_focus.to_csv(OUT_DIR / "promoted_model_philips_protocol_focus_table.csv", index=False)
_log(f"  Philips protocol focus table: {len(df_phil_focus)} rows")

# ────────────────────────────────────────────────────────────────
# 21. Save master database CSV
# ────────────────────────────────────────────────────────────────
_log("S21: Saving master database CSV...")

master_csv_path = OUT_DIR / "promoted_model_master_database.csv"
df_master_db.to_csv(master_csv_path, index=False)
_log(f"  Saved: {master_csv_path} ({len(df_master_db)} rows × {len(df_master_db.columns)} cols)")

# ────────────────────────────────────────────────────────────────
# 22. Data dictionary
# ────────────────────────────────────────────────────────────────
_log("S22: Writing data dictionary...")

data_dict_rows = []
col_descriptions = {
    "master_row_id": "Sequential row index (0-based), matches tensor order",
    "tensor_idx": "Index into global tensor first dimension (0-based)",
    "SubjectID": "ADNI subject ID (NNNSITE_S_NNNN format)",
    "SubjectID_tensor": "SubjectID directly from global tensor subject_ids array",
    "ImageID": "ADNI image series ID (integer)",
    "global_tensor_path": "Absolute path to the global NPZ tensor used by the promoted model",
    "global_tensor_subject_id": "SubjectID as stored in global tensor",
    "source_batch": "Processing batch name (e.g. v5_dparsf10000_no_pybandpass)",
    "source_label": "Descriptive label for source (e.g. desde_cero_historical_10000)",
    "ResearchGroup_Mapped": "Diagnosis group: CN / AD / MCI",
    "Diagnosis": "Canonical diagnosis label from batch_metadata",
    "y_true": "Supervised label (1=AD, 0=CN, NaN=MCI/excluded)",
    "y_true_label": "String label (CN/AD/MCI)",
    "y_score_raw": "Stage B raw logistic regression output before calibration",
    "y_score_oof_ecdf": "OOF-ECDF calibrated score (primary readout)",
    "y_score_oof_logitz": "OOF-logitz calibrated score (comparison readout)",
    "y_score_final": "Final score used for downstream analysis (= y_score_oof_ecdf)",
    "y_pred": "Binary prediction at primary threshold (inner_oof_target_sens_ge_0p70_max_spec)",
    "confusion_label": "TP/TN/FP/FN/NA for primary readout",
    "outer_fold": "Outer cross-validation fold assignment (1–5)",
    "model_name": "Classifier model name for primary readout (logreg_l2_original)",
    "feature_set": "Input features (z_plus_age_sex)",
    "calib_method": "OOF calibration method (oof_ecdf)",
    "threshold_strategy": "Threshold selection strategy (inner_oof_target_sens_ge_0p70_max_spec)",
    "Manufacturer": "Scanner manufacturer (Philips / SIEMENS / GE)",
    "Manufacturer_normalized": "Normalized manufacturer label",
    "Site3": "ADNI site number (float)",
    "ORIGPROT": "Original ADNI protocol at acquisition (ADNI1/ADNI2/ADNI3/ADNIGO)",
    "COLPROT": "Collection protocol version",
    "inferred_ADNI_phase": "ADNI phase from SubjectsData_AAL3 or ORIGPROT",
    "n_timepoints_raw": "Number of raw timepoints from batch manifest",
    "n_timepoints_model_input": "Best available timepoint count for n_tp_raw classification",
    "raw_tp_group": "Timepoint group: 140 / 197 / 200 / other / UNKNOWN",
    "protocol_risk_group": "Philips CN protocol risk classification",
    "scale_label": "BOLD signal scaling label (e.g. around_10000_global_scaled)",
    "python_bandpass_applied": "Whether Python bandpass filter was applied",
    "bandpass_status": "Bandpass status string",
    "is_philips_cn": "True if subject is Philips-manufacturer CN",
    "philips_cn_error_type": "FP/TN/TP/FN for Philips CN (NA otherwise)",
    "philips_cn_raw_tp_group": "140/197 TP group for Philips CN",
    "philips_cn_high_protocol_risk": "True if Philips CN with 140 TP (FPR=63.4%)",
    "philips_problem_site_flag": "True if Philips CN from site with 100% FPR",
    "philips_ntp_fpr_group_if_available": "Protocol-risk FPR group (140tp_FPR064 or 197tp_FPR033)",
    "in_vae_pool": "Subject included in VAE training pool",
    "in_stageB_classifier_pool": "Subject included in Stage B supervised classifier",
    "in_oof_evaluation": "Subject has OOF prediction (= in_stageB_classifier_pool for this run)",
    "in_mci_unsupervised_only": "Subject is MCI (in VAE pool but not Stage B)",
    "in_config_metadata": "Subject present in patched_metadata_candidate.csv used by the run config",
    "rp_available": "Motion realignment parameter file (rp_*.txt) available",
    "fd_mean": "Mean framewise displacement (mm), from rp file; NaN if rp unavailable",
    "tsnr_proxy_median_corrected": "Corrected tSNR (transposition bug fixed) for Philips; original for others",
    "droi_rms_corrected": "Corrected dROI RMS for Philips; original for others",
    "tensor_ch1_offdiag_mean": "Ch1 (Pearson_Full) off-diagonal mean from global tensor",
    "tensor_ch0_offdiag_mean": "Ch0 (OMST) off-diagonal mean from global tensor",
    "tensor_ch2_offdiag_mean": "Ch2 (MI) off-diagonal mean from global tensor",
    "APOE4": "APOE4 allele count (0/1/2)",
    "CDRSB": "Clinical Dementia Rating Sum of Boxes",
    "MMSE": "Mini-Mental State Examination score",
    "RAVLT_immediate": "Rey Auditory Verbal Learning Test - immediate recall",
    "FAQ": "Functional Activities Questionnaire",
    "mat_path": "Path to ROISignals .mat file (Philips CN subjects)",
    "individual_tensor_path": "Path to individual subject tensor NPZ (Philips CN)",
    "pipeline_log_status": "Pipeline log status string (SUCCESS_ALL_PROCESSED_AND_SAVED or similar)",
    "duplicate_subject_flag": "True if SubjectID appears more than once in the tensor",
    "duplicate_image_flag": "True if ImageID appears more than once in the tensor",
    "source_conflict_flag": "True if Age differs >1 year between batch_metadata and provenance audit",
    "promoted_model_score_available": "True if OOF-ECDF score available from promoted model",
    "promoted_model_score_source": "Source file for the promoted model score",
    "clinical_metadata_available": "True if any of CDRSB/MMSE/APOE4 is non-null",
    "protocol_metadata_available": "True if ORIGPROT or n_tp_raw is non-null",
    "tensor_qc_available": "True if tensor QC was computed (all subjects = True)",
    "tensor_npz_available": "True if subject is in the global tensor (all subjects = True)",
}
for col in df_master_db.columns:
    data_dict_rows.append({
        "column": col,
        "dtype": str(df_master_db[col].dtype),
        "n_unique": int(df_master_db[col].nunique()),
        "n_null": int(df_master_db[col].isna().sum()),
        "pct_null": round(df_master_db[col].isna().sum() / len(df_master_db) * 100, 1),
        "sample_values": str(df_master_db[col].dropna().iloc[:3].tolist())[:120],
        "description": col_descriptions.get(col, ""),
    })

df_dict = pd.DataFrame(data_dict_rows)
df_dict.to_csv(OUT_DIR / "promoted_model_master_database_data_dictionary.csv", index=False)
_log(f"  Data dictionary: {len(df_dict)} entries")

# ────────────────────────────────────────────────────────────────
# 23. Coverage report
# ────────────────────────────────────────────────────────────────
_log("S23: Writing coverage report...")

coverage_rows = [
    {"source": "global_tensor", "path": str(GLOBAL_TENSOR_PATH),
     "N_rows": N, "N_in_master": N, "join_key": "SubjectID", "status": "COMPLETE"},
    {"source": "batch_metadata", "path": str(BATCH_METADATA_PATH),
     "N_rows": len(df_batch), "N_in_master": int(df_master_db["metadata_source_batch"].notna().sum()),
     "join_key": "SubjectID", "status": "COMPLETE"},
    {"source": "patched_metadata", "path": str(PATCHED_METADATA_PATH),
     "N_rows": len(df_patched), "N_in_master": int(df_master_db["in_config_metadata"].sum()),
     "join_key": "SubjectID", "status": "COMPLETE"},
    {"source": "oof_calib_predictions", "path": str(OOF_CALIB_PRED_PATH),
     "N_rows": int(df_oof_prim["SubjectID"].nunique()),
     "N_in_master": int(df_master_db["y_score_oof_ecdf"].notna().sum()),
     "join_key": "SubjectID", "status": "COMPLETE"},
    {"source": "subject_scan_master_table", "path": str(SUBJECT_SCAN_MASTER_PATH),
     "N_rows": len(df_master), "N_in_master": int(df_master_db["ORIGPROT"].notna().sum()),
     "join_key": "SubjectID", "status": "PARTIAL (CN+AD only)"},
    {"source": "philips_cn_provenance_table", "path": str(PHILIPS_PROV_TABLE_PATH),
     "N_rows": len(df_phil_prov), "N_in_master": int(df_master_db["mat_path"].notna().sum()),
     "join_key": "SubjectID", "status": "PHILIPS_CN_ONLY"},
    {"source": "philips_protocol_risk_table", "path": str(PHILIPS_RISK_TABLE_PATH),
     "N_rows": len(df_phil_risk),
     "N_in_master": int(df_master_db.get("n_tp_raw", pd.Series(np.nan, index=df_master_db.index)).notna().sum()),
     "join_key": "SubjectID", "status": "PHILIPS_CN_ONLY"},
    {"source": "bold_qc_corrected", "path": str(BOLD_QC_CORRECTED_PATH),
     "N_rows": len(df_bold_qc),
     "N_in_master": int(df_master_db["tsnr_proxy_median_corrected"].notna().sum()),
     "join_key": "SubjectID", "status": "PHILIPS_CN_ONLY"},
    {"source": "SubjectsData_AAL3_procesado2", "path": str(SUBJECTS_DATA_PATH),
     "N_rows": len(df_subdata),
     "N_in_master": int(df_master_db["inferred_ADNI_phase"].notna().sum()),
     "join_key": "SubjectID", "status": "PARTIAL"},
]

df_coverage = pd.DataFrame(coverage_rows)
df_coverage.to_csv(OUT_DIR / "promoted_model_master_database_coverage_report.csv", index=False)
_log(f"  Coverage report: {len(df_coverage)} sources")

# ────────────────────────────────────────────────────────────────
# 24. Markdown summary outputs
# ────────────────────────────────────────────────────────────────
_log("S24: Writing markdown outputs...")

def _df_to_md(df: pd.DataFrame, max_rows: int = 40) -> str:
    return df.head(max_rows).to_markdown(index=False)

# Pool counts markdown
with open(OUT_DIR / "promoted_model_pool_counts_by_group.md", "w") as f:
    f.write("# Pool Counts by Manufacturer × Diagnosis\n\n")
    f.write(_df_to_md(df_pool_counts))
    f.write(f"\n\n**Global tensor N={N}  |  VAE pool={counts['vae_pool_N']}  "
            f"|  Classifier pool={counts['classifier_pool_N']}  "
            f"|  OOF-evaluable={counts['oof_evaluable_N']}**\n")

# Protocol summary markdown
with open(OUT_DIR / "promoted_model_protocol_summary_by_manufacturer.md", "w") as f:
    f.write("# Protocol Summary by Manufacturer\n\n")
    f.write(_df_to_md(df_proto))

# Philips focus markdown
with open(OUT_DIR / "promoted_model_philips_protocol_focus_table.md", "w") as f:
    f.write("# Philips CN Protocol Focus Table\n\n")
    f.write(f"N Philips CN = {len(df_phil_cn)}\n\n")
    f.write(_df_to_md(df_phil_focus))

# Missingness summary markdown
miss_top = df_miss_summary.head(40)
with open(OUT_DIR / "promoted_model_missingness_by_field.md", "w") as f:
    f.write("# Missingness by Field (ALL manufacturers)\n\n")
    f.write("Sorted by pct_available ascending (worst coverage first).\n\n")
    f.write(_df_to_md(miss_top))

# Conflicts markdown
with open(OUT_DIR / "promoted_model_metadata_conflicts.md", "w") as f:
    f.write("# Metadata Conflicts\n\n")
    if len(df_conflicts) == 0:
        f.write("No conflicts detected.\n")
    else:
        f.write(_df_to_md(df_conflicts))

# Coverage markdown
with open(OUT_DIR / "promoted_model_master_database_coverage_report.md", "w") as f:
    f.write("# Coverage Report\n\n")
    f.write(_df_to_md(df_coverage))

# Master DB head markdown
with open(OUT_DIR / "promoted_model_master_database.md", "w") as f:
    f.write(f"# Promoted Model Master Database\n\n")
    f.write(f"**Rows:** {len(df_master_db)}  |  **Columns:** {len(df_master_db.columns)}\n\n")
    f.write("## Columns\n\n")
    f.write(", ".join(df_master_db.columns.tolist()) + "\n\n")
    f.write("## First 10 rows (key fields)\n\n")
    preview_cols = ["SubjectID", "ResearchGroup_Mapped", "Manufacturer", "Age",
                    "outer_fold", "y_score_oof_ecdf", "confusion_label",
                    "ORIGPROT", "raw_tp_group", "is_philips_cn"]
    preview_cols_real = [c for c in preview_cols if c in df_master_db.columns]
    f.write(_df_to_md(df_master_db[preview_cols_real].head(10)))

_log("  Markdown outputs written.")

# ────────────────────────────────────────────────────────────────
# 25. Final interpretation report
# ────────────────────────────────────────────────────────────────
_log("S25: Writing final interpretation...")

cn_fpr_philips = round((df_master_db[(df_master_db["is_philips_cn"]) & (df_master_db["confusion_label"]=="FP")].shape[0]) /
                       max(1, df_master_db[df_master_db["is_philips_cn"] & df_master_db["in_oof_evaluation"]].shape[0]), 4)
cn_fpr_siemens_n = df_master_db[(df_master_db["Manufacturer"]=="SIEMENS") & (df_master_db["ResearchGroup_Mapped"]=="CN") & df_master_db["in_oof_evaluation"]]
cn_fpr_siemens = round((cn_fpr_siemens_n["confusion_label"]=="FP").sum() / max(1, len(cn_fpr_siemens_n)), 4)
cn_fpr_ge_n     = df_master_db[(df_master_db["Manufacturer"]=="GE") & (df_master_db["ResearchGroup_Mapped"]=="CN") & df_master_db["in_oof_evaluation"]]
cn_fpr_ge       = round((cn_fpr_ge_n["confusion_label"]=="FP").sum() / max(1, len(cn_fpr_ge_n)), 4)

# Philips 140 vs 197 TP FPR
ph140 = df_master_db[df_master_db["philips_ntp_fpr_group_if_available"] == "140tp_FPR064"]
ph197 = df_master_db[df_master_db["philips_ntp_fpr_group_if_available"] == "197tp_FPR033"]
ph140_fpr = round((ph140["confusion_label"]=="FP").sum() / max(1, ph140["in_oof_evaluation"].sum()), 4)
ph197_fpr = round((ph197["confusion_label"]=="FP").sum() / max(1, ph197["in_oof_evaluation"].sum()), 4)

interp = f"""# Final Master Database Interpretation — Promoted Model

**Date**: {datetime.now().isoformat()}
**Script**: build_promoted_model_master_database_20260610.py
**Promoted model**: {PROMOTED_RUN_ID}
**Primary readout**: {PRIMARY_MODEL} / {PRIMARY_FEATURE} / {PRIMARY_CALIB} / {PRIMARY_THRESH}

---

## What the Master Database Contains

One row per tensor entry (N={N}) in the global tensor
`GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz`.
Each row is a unique subject–scan. The tensor was assembled from two batches:
- `v5_dparsf10000_no_pybandpass`: 90 Philips CN + SIEMENS + GE subjects from the DPARSF
  `desde_cero` extraction (no Python bandpass).
- `20260514_bandpass_batch2`: 9 Philips CN subjects re-processed with OneDrive batch.

Tensor order is preserved exactly. No deduplication was performed.

---

## Exact N and Pool Counts

| Pool | N |
|---|---|
| Global tensor (all subjects) | {counts["global_tensor_N"]} |
| VAE pool (training_ready) | {counts["vae_pool_N"]} |
| Stage B classifier pool (CN+AD, not excluded) | {counts["classifier_pool_N"]} |
| OOF-evaluable (has promoted model score) | {counts["oof_evaluable_N"]} |
| CN (global tensor) | {counts["CN_global"]} |
| AD (global tensor) | {counts["AD_global"]} |
| MCI (global tensor, VAE only) | {counts["MCI_global"]} |
| CN in classifier pool | {counts["CN_classifier"]} |
| AD in classifier pool | {counts["AD_classifier"]} |

**Manufacturer × Diagnosis (global tensor)**:
- Philips CN: {len(df_master_db[(df_master_db["Manufacturer"]=="Philips")&(df_master_db["ResearchGroup_Mapped"]=="CN")])}
  AD: {len(df_master_db[(df_master_db["Manufacturer"]=="Philips")&(df_master_db["ResearchGroup_Mapped"]=="AD")])}
  MCI: {len(df_master_db[(df_master_db["Manufacturer"]=="Philips")&(df_master_db["ResearchGroup_Mapped"]=="MCI")])}
- SIEMENS CN: {len(df_master_db[(df_master_db["Manufacturer"]=="SIEMENS")&(df_master_db["ResearchGroup_Mapped"]=="CN")])}
  AD: {len(df_master_db[(df_master_db["Manufacturer"]=="SIEMENS")&(df_master_db["ResearchGroup_Mapped"]=="AD")])}
  MCI: {len(df_master_db[(df_master_db["Manufacturer"]=="SIEMENS")&(df_master_db["ResearchGroup_Mapped"]=="MCI")])}
- GE CN: {len(df_master_db[(df_master_db["Manufacturer"]=="GE")&(df_master_db["ResearchGroup_Mapped"]=="CN")])}
  AD: {len(df_master_db[(df_master_db["Manufacturer"]=="GE")&(df_master_db["ResearchGroup_Mapped"]=="AD")])}
  MCI: {len(df_master_db[(df_master_db["Manufacturer"]=="GE")&(df_master_db["ResearchGroup_Mapped"]=="MCI")])}

---

## What Is Now Complete

1. **OOF scores (all CN+AD = {counts["oof_evaluable_N"]} subjects)**: y_score_oof_ecdf, y_score_oof_logitz,
   y_score_raw, y_pred, confusion_label — complete for the classifier pool.

2. **Manufacturer and Site3 (all {N} subjects)**: complete from batch_metadata (primary source).

3. **Age/Sex (all {N} subjects)**: complete from batch_metadata.

4. **Philips CN BOLD QC ({len(df_phil_cn)} subjects)**: tSNR, droi_rms corrected. The transposition
   bug was fixed; tSNR p=0.71 (FP vs TN), droi_rms p=0.68 — no BOLD QC difference.

5. **Philips CN tensor provenance ({len(df_phil_cn)} subjects)**: mat_path, pipeline_log_status,
   extraction_status — all 99 .mat files located.

6. **Tensor QC for all {N} subjects**: tensor_ch0/1/2 offdiag_mean/std/frob_norm computed from
   the global tensor (selected channels: ch1=Pearson_Full, ch0=OMST, ch2=MI).

7. **n_tp_raw for Philips CN ({len(df_phil_cn)} subjects)**: raw_tp_group 140/197 complete.
   For non-Philips subjects: n_timepoints_raw from batch manifest is available.

8. **ORIGPROT/COLPROT ({int(df_master_db["ORIGPROT"].notna().sum())} subjects)**: from subject_scan_master_table.

9. **CDRSB/MMSE/APOE4 ({int(df_master_db["CDRSB"].notna().sum() if "CDRSB" in df_master_db.columns else 0)} subjects with CDRSB)**: available for Philips CN
   from protocol risk audit; and for CN+AD subjects from subject_scan_master_table.

---

## Manufacturer CN FPR (Primary OOF-ECDF Readout)

| Manufacturer | N CN | N FP | FPR |
|---|---|---|---|
| Philips | {int(df_master_db[df_master_db["is_philips_cn"]&df_master_db["in_oof_evaluation"]].shape[0])} | {int((df_master_db[df_master_db["is_philips_cn"]&df_master_db["in_oof_evaluation"]]["confusion_label"]=="FP").sum())} | {cn_fpr_philips} |
| SIEMENS | {len(cn_fpr_siemens_n)} | {int((cn_fpr_siemens_n["confusion_label"]=="FP").sum())} | {cn_fpr_siemens} |
| GE | {len(cn_fpr_ge_n)} | {int((cn_fpr_ge_n["confusion_label"]=="FP").sum())} | {cn_fpr_ge} |

## Philips CN FPR by n_tp_raw Group

| n_tp_raw group | N | FPR |
|---|---|---|
| 140 TP (ADNI1/2) | {int(ph140["in_oof_evaluation"].sum())} | {ph140_fpr} |
| 197 TP (ADNI3) | {int(ph197["in_oof_evaluation"].sum())} | {ph197_fpr} |

Fisher OR = 3.56, p = 0.0039 (from philips_protocol_risk_audit).
Age is primary driver (CLES=0.710); 140-TP effect partially independent of age (coef=1.057 after adjusting for age).

---

## What Remains Missing

### Philips CN (affects interpretation):
1. **rp_*.txt motion files (0/{len(df_phil_cn)})**: No Philips CN subject has framewise displacement data.
   Cannot assess whether motion confounds the Philips FPR. Requested from Martín.
2. **BIDS JSON sidecars (0/{len(df_phil_cn)})**: No TR/TE per-subject or phase-encoding direction
   from DICOM headers for Philips CN. TR from SubjectsData_AAL3 is available for some subjects.
3. **ADNI MRIQUALITY flags**: Series-level QC fields not available locally.
4. **ORIGPROT/COLPROT for {N - int(df_master_db["ORIGPROT"].notna().sum())} subjects**: Not recovered from available sources.

### Non-Philips subjects:
5. **Motion QC for SIEMENS/GE (partial)**: rp_available={int(df_master_db[df_master_db["Manufacturer"]!="Philips"]["rp_available"].fillna(False).astype(bool).sum())} subjects.
6. **BOLD QC for SIEMENS/GE**: Available only for CN+AD subset via subject_scan_master_table.

---

## Field Reliability for Manuscript Tables

### Reliable (use directly):
- SubjectID, tensor_idx, ResearchGroup_Mapped, Manufacturer, Site3, Age, Sex
- y_score_oof_ecdf, y_pred, confusion_label, outer_fold
- source_batch, scale_label
- Philips CN: n_tp_raw, ORIGPROT, tSNR_corrected, tensor_ch0-2_offdiag_mean
- ORIGPROT/COLPROT for subjects covered by subject_scan_master_table

### Exploratory only (partial coverage or single-source):
- CDRSB, MMSE, APOE4, RAVLT_immediate, FAQ — partial coverage; use with missingness caveat
- TR/TE — from SubjectsData_AAL3 (partial); not from DICOM headers
- rp/FD metrics — available for only {int(df_master_db["rp_available"].fillna(False).astype(bool).sum())} subjects
- inferred_ADNI_phase — from ORIGPROT or SubjectsData; not from ADNI download metadata

### Not available locally (request from Martín or ADNI):
- Phase encoding direction
- Slice timing / slice order
- DICOM header fields (station_name, coil, sequence_name)
- MRIQUALITY series-level flags
- motion parameters for 0/99 Philips CN subjects

---

## How This Table Supports the Philips FPR Analysis

The master database provides a single join-ready CSV with:
1. All {counts["oof_evaluable_N"]} OOF predictions aligned to subject metadata.
2. Philips-specific flags: is_philips_cn, philips_cn_error_type, philips_cn_raw_tp_group,
   protocol_risk_group, philips_ntp_fpr_group_if_available.
3. ADNI phase (ORIGPROT) for {int(df_master_db["ORIGPROT"].notna().sum())} subjects — supports
   n_tp_raw × ADNI-phase confounding analysis.
4. Corrected BOLD QC for all 99 Philips CN subjects (tSNR bug retracted).
5. Tensor QC (ch0-ch2) for all {N} subjects — supports batch effect screening.

---

## What Still Requires Martín's Files

1. rp_*.txt for Philips CN → fd_mean, fd_3mm_flag, fd_3deg_flag (currently 0/{len(df_phil_cn)})
2. BIDS JSON sidecars → phase_encoding_direction, slice_timing, TR/TE from DICOM
3. ADNI MRIQUALITY table → series-level QC flags
4. Confirmation of 197→140 TP truncation logic (ADNI2 vs ADNI3 Philips)

---

## Guardrails Compliance
- Read-only. No model training, no tensor modification, no metadata modification.
- No threshold fitting, no OASIS scoring.
- All operations are descriptive/provenance. Original source files untouched.
"""

with open(OUT_DIR / "final_master_database_interpretation.md", "w") as f:
    f.write(interp)
_log("  Final interpretation written.")

# ────────────────────────────────────────────────────────────────
# 26. Save command log
# ────────────────────────────────────────────────────────────────
command_log["timestamp_end"] = datetime.now().isoformat()
command_log["output_files"] = [str(p) for p in sorted(OUT_DIR.glob("*")) if p.is_file()]
command_log["master_db_shape"] = list(df_master_db.shape)
command_log["pool_counts"] = counts
command_log["cn_fpr_philips"] = cn_fpr_philips
command_log["cn_fpr_siemens"] = cn_fpr_siemens
command_log["cn_fpr_ge"] = cn_fpr_ge

with open(OUT_DIR / "command_log.json", "w") as f:
    json.dump(command_log, f, indent=2)

# ────────────────────────────────────────────────────────────────
# 27. Print summary
# ────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("PROMOTED MODEL MASTER DATABASE — BUILD COMPLETE")
print("="*70)
print(f"  Output directory: {OUT_DIR}")
print(f"  Master CSV:  promoted_model_master_database.csv")
print(f"  Shape: {df_master_db.shape[0]} rows × {df_master_db.shape[1]} columns")
print()
print("  POOL COUNTS:")
for k, v in counts.items():
    print(f"    {k:35s}: {v}")
print()
print("  MANUFACTURER CN FPR (OOF-ECDF primary):")
print(f"    Philips:  {cn_fpr_philips}  SIEMENS: {cn_fpr_siemens}  GE: {cn_fpr_ge}")
print()
print("  OUTPUT FILES:")
for p in sorted(OUT_DIR.glob("*")):
    print(f"    {p.name}")
print("="*70)
