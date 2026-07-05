#!/usr/bin/env python3
"""Export promoted-model full database and validity-mask preflight.

Read-only reproducibility package for Martin: one-row-per-subject/scan export
plus preflight masks for possible valid-only sensitivity retraining. The script
does not train models, modify tensors/metadata/predictions, refit thresholds,
or exclude subjects from any existing artifact.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results" / "revision_bspc_2026"
OUT = RESULTS / "full_database_for_martin_and_validity_preflight_20260612"
OUT.mkdir(parents=True, exist_ok=True)

MASTER_PATH = RESULTS / "promoted_model_master_database_20260610" / "promoted_model_master_database.csv"
FLAGS_PATH = RESULTS / "philips_foldwise_artifact_exposure_audit_20260612" / "subject_artifact_flags_master.csv"
MEMBERSHIP_PATH = RESULTS / "philips_foldwise_artifact_exposure_audit_20260612" / "fold_membership_reconstruction.csv"
SLICE_MERGED_PATH = RESULTS / "philips_fmri_slice_timing_audit_20260612" / "philips_cn_fmri_slice_timing_merged.csv"
RUN_DIR = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
OOF_DIR = RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration"


def write_md(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_df(stem: str, df: pd.DataFrame, md_rows: int | None = 200) -> None:
    df.to_csv(OUT / f"{stem}.csv", index=False)
    if df.empty:
        md = "_No rows._\n"
    else:
        show = df if md_rows is None or len(df) <= md_rows else df.head(md_rows)
        md = show.to_markdown(index=False) + "\n"
        if md_rows is not None and len(df) > md_rows:
            md += f"\n_Only first {md_rows} of {len(df)} rows shown. See CSV for full table._\n"
    (OUT / f"{stem}.md").write_text(md, encoding="utf-8")


def norm_missing(x: Any) -> str:
    if pd.isna(x):
        return "MISSING"
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return "MISSING"
    return s


def bool_series(s: pd.Series) -> pd.Series:
    return s.map(lambda x: bool(x) if isinstance(x, (bool, np.bool_)) else str(x).strip().lower() in {"true", "1", "yes", "y", "t"}).fillna(False)


def rid_from_subject(sid: Any) -> float:
    m = re.match(r"^\d+_S_(\d+)$", str(sid))
    return float(m.group(1)) if m else np.nan


def first_present(df: pd.DataFrame, cols: list[str], default: Any = np.nan) -> pd.Series:
    out = pd.Series(np.nan, index=df.index)
    for col in cols:
        if col in df.columns:
            out = out.where(out.notna(), df[col])
    return out.fillna(default)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for p in [MASTER_PATH, FLAGS_PATH, MEMBERSHIP_PATH, SLICE_MERGED_PATH, RUN_DIR, OOF_DIR]:
        if not Path(p).exists():
            raise FileNotFoundError(p)
    master = pd.read_csv(MASTER_PATH, low_memory=False)
    flags = pd.read_csv(FLAGS_PATH, low_memory=False)
    membership = pd.read_csv(MEMBERSHIP_PATH, low_memory=False)
    slice_merged = pd.read_csv(SLICE_MERGED_PATH, low_memory=False)
    return master, flags, membership, slice_merged


def enrich_master(master: pd.DataFrame, flags: pd.DataFrame, slice_merged: pd.DataFrame) -> pd.DataFrame:
    df = master.copy()
    df["SubjectID"] = df["SubjectID"].astype(str)
    df["RID"] = df["SubjectID"].map(rid_from_subject)
    df["diagnosis_group"] = first_present(df, ["ResearchGroup_Mapped", "Diagnosis"], "MISSING").map(norm_missing)
    df["Site3_final"] = first_present(df, ["Site3", "SITE"], np.nan)
    df["Site3_final"] = df["Site3_final"].map(lambda x: "MISSING" if pd.isna(x) else str(int(float(x))) if str(x).replace(".", "", 1).isdigit() else norm_missing(x))
    df["Manufacturer_final"] = first_present(df, ["Manufacturer_normalized", "Manufacturer"], "MISSING").map(norm_missing)
    df["raw_tp_group_final"] = first_present(df, ["raw_tp_group", "n_tp_label", "n_tp_raw"], "UNKNOWN").map(norm_missing)
    df["ORIGPROT_final"] = first_present(df, ["ORIGPROT", "ORIGPROT_phil"], "MISSING").map(norm_missing)
    df["COLPROT_final"] = first_present(df, ["COLPROT", "COLPROT_phil"], "MISSING").map(norm_missing)
    df["Sex_final"] = first_present(df, ["Sex", "PTGENDER"], "MISSING").map(norm_missing)
    df["Age_final"] = pd.to_numeric(first_present(df, ["Age", "Age_prov"], np.nan), errors="coerce")
    df["scanner_model_final"] = first_present(df, ["scanner_model", "scanner_model_flag_value"], "MISSING").map(norm_missing)
    df["manufacturer_model_name_final"] = first_present(df, ["manufacturer_model_name"], "MISSING").map(norm_missing)
    df["software_version_final"] = first_present(df, ["software_version"], "MISSING").map(norm_missing)
    df["phase_encoding_direction_final"] = first_present(df, ["phase_encoding_direction", "phase_encoding_direction_raw"], "MISSING").map(norm_missing)

    flag_cols = [
        "SubjectID",
        "high_confidence_slice_timing_match",
        "match_method",
        "match_confidence",
        "slice_order_class",
        "fmri_slice_order_len",
        "stc_mismatch_risk",
        "Site31_reverse_even_odd_48",
        "Site2_default_7of7_pattern",
        "problem_site_flag",
        "PHASEDIR_AP",
        "PHASEDIR_PA",
        "scanner_model_flag_value",
        "fmri_MEANTSNR",
        "fmri_MEDTSNR",
        "fmri_SDTSNR",
    ]
    flags_small = flags[[c for c in flag_cols if c in flags.columns]].drop_duplicates("SubjectID")
    df = df.merge(flags_small, on="SubjectID", how="left", suffixes=("", "_flag"))

    sm_cols = [
        "SubjectID",
        "fmri_PHASEDIR_norm",
        "fmri_SLICEORD",
        "fmri_SLICETIMING_NFQ",
        "fmri_STATUS",
        "fmri_MANUFACTURERSMODELNAME",
        "fmri_ScannerModel",
        "fmri_SoftwareVersion",
        "fmri_RepetitionTime",
        "fmri_EchoTime",
        "fmri_NumberVolumes",
        "fmri_SlicesPerVolume",
        "fmri_SliceThickness",
        "fmri_SERIES_QUALITY",
        "fmri_SliceTiming_MRINFQ",
    ]
    sm = slice_merged[[c for c in sm_cols if c in slice_merged.columns]].drop_duplicates("SubjectID")
    df = df.merge(sm, on="SubjectID", how="left", suffixes=("", "_slice"))

    df["PHASEDIR_final"] = first_present(df, ["fmri_PHASEDIR_norm", "phase_encoding_direction_final"], "MISSING").map(norm_missing)
    df["slice_order_class"] = df.get("slice_order_class", pd.Series("MISSING", index=df.index)).fillna("MISSING").map(norm_missing)
    df["match_method"] = df.get("match_method", pd.Series("none", index=df.index)).fillna("none")
    df["match_confidence"] = df.get("match_confidence", pd.Series("none", index=df.index)).fillna("none")
    df["high_confidence_slice_timing_match"] = bool_series(df.get("high_confidence_slice_timing_match", pd.Series(False, index=df.index)))
    df["matches_dparsf_default"] = df["slice_order_class"].eq("default_odd_even_48")
    for col in ["stc_mismatch_risk", "Site31_reverse_even_odd_48", "Site2_default_7of7_pattern", "problem_site_flag", "PHASEDIR_AP", "PHASEDIR_PA"]:
        df[col] = bool_series(df.get(col, pd.Series(False, index=df.index)))
    return df


def derive_validity_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    has_tensor = bool_series(out.get("in_selected_channel_tensor", pd.Series(False, index=out.index))) | bool_series(out.get("tensor_npz_available", pd.Series(False, index=out.index)))
    has_dx = out["diagnosis_group"].isin(["CN", "AD", "MCI"])
    has_age_sex = out["Age_final"].notna() & ~out["Sex_final"].eq("MISSING")
    has_mfr_site = ~out["Manufacturer_final"].eq("MISSING") & ~out["Site3_final"].eq("MISSING")
    tensor_stats = ["tensor_ch0_offdiag_mean", "tensor_ch1_offdiag_mean", "tensor_ch2_offdiag_mean"]
    has_tensor_stats = pd.Series(True, index=out.index)
    for col in tensor_stats:
        has_tensor_stats &= pd.to_numeric(out.get(col), errors="coerce").notna()
    not_supervised_excluded = ~bool_series(out.get("exclude_from_supervised", pd.Series(False, index=out.index)))

    out["valid_qc_minimal"] = has_tensor & has_dx & has_age_sex & has_mfr_site & has_tensor_stats & not_supervised_excluded
    out["valid_qc_minimal_fail_reasons"] = make_reasons(
        [
            ("missing_tensor", ~has_tensor),
            ("missing_diagnosis", ~has_dx),
            ("missing_age_or_sex", ~has_age_sex),
            ("missing_manufacturer_or_site", ~has_mfr_site),
            ("missing_selected_channel_tensor_stats", ~has_tensor_stats),
            ("original_supervised_exclusion", ~not_supervised_excluded),
        ],
        out.index,
    )

    raw_tp_known = ~out["raw_tp_group_final"].isin(["MISSING", "UNKNOWN", "nan"])
    no_known_stc_mismatch = ~out["stc_mismatch_risk"]
    no_confirmed_slice_order_mismatch = ~(out["high_confidence_slice_timing_match"] & ~out["matches_dparsf_default"])
    protocol_traceable = ~out["Manufacturer_final"].eq("MISSING") & ~out["Site3_final"].eq("MISSING") & raw_tp_known
    out["valid_qc_protocol"] = out["valid_qc_minimal"] & raw_tp_known & no_known_stc_mismatch & no_confirmed_slice_order_mismatch & protocol_traceable
    out["valid_qc_protocol_fail_reasons"] = make_reasons(
        [
            ("fails_minimal", ~out["valid_qc_minimal"]),
            ("raw_tp_group_unknown", ~raw_tp_known),
            ("known_high_confidence_stc_mismatch", ~no_known_stc_mismatch),
            ("confirmed_nondefault_slice_order", ~no_confirmed_slice_order_mismatch),
            ("protocol_not_traceable", ~protocol_traceable),
        ],
        out.index,
    )

    fd_available = pd.to_numeric(out.get("fd_mean"), errors="coerce").notna() | bool_series(out.get("rp_available", pd.Series(False, index=out.index)))
    fd_failure = bool_series(out.get("fd_3mm_flag", pd.Series(False, index=out.index))) | bool_series(out.get("fd_3deg_flag", pd.Series(False, index=out.index)))
    qc_failed = out.get("fmri_STATUS", pd.Series("", index=out.index)).astype(str).str.lower().str.contains("fail", na=False)
    qc_failed |= out.get("fmri_SERIES_QUALITY", pd.Series("", index=out.index)).astype(str).str.lower().str.contains("fail", na=False)
    scanner_annotated = ~out["scanner_model_final"].eq("MISSING") | ~out["manufacturer_model_name_final"].eq("MISSING") | ~out.get("scanner_model_flag_value", pd.Series("MISSING", index=out.index)).fillna("MISSING").eq("MISSING")
    phase_annotated = ~out["PHASEDIR_final"].eq("MISSING")
    is_philips = out["Manufacturer_final"].str.upper().eq("PHILIPS")
    philips_slice_annotated = ~is_philips | out["high_confidence_slice_timing_match"]

    out["valid_qc_strict_available_only"] = (
        out["valid_qc_protocol"]
        & fd_available
        & ~fd_failure
        & ~qc_failed
        & scanner_annotated
        & phase_annotated
        & philips_slice_annotated
    )
    out["valid_qc_strict_available_only_fail_reasons"] = make_reasons(
        [
            ("fails_protocol", ~out["valid_qc_protocol"]),
            ("motion_or_fd_unavailable", ~fd_available),
            ("fd_failure_flag", fd_failure),
            ("mayo_or_series_qc_failed", qc_failed),
            ("scanner_model_unannotated", ~scanner_annotated),
            ("phase_direction_unannotated", ~phase_annotated),
            ("philips_slice_timing_not_high_confidence", ~philips_slice_annotated),
        ],
        out.index,
    )

    fd_ok_if_available = ~fd_failure
    qc_ok_if_available = ~qc_failed
    scanner_ok_if_available = pd.Series(True, index=out.index)
    phase_ok_if_available = pd.Series(True, index=out.index)
    slice_ok_if_available = ~(out["high_confidence_slice_timing_match"] & ~out["matches_dparsf_default"])
    out["valid_qc_strict_missing_allowed"] = out["valid_qc_protocol"] & fd_ok_if_available & qc_ok_if_available & scanner_ok_if_available & phase_ok_if_available & slice_ok_if_available
    out["valid_qc_strict_missing_allowed_fail_reasons"] = make_reasons(
        [
            ("fails_protocol", ~out["valid_qc_protocol"]),
            ("fd_failure_flag", fd_failure),
            ("mayo_or_series_qc_failed", qc_failed),
            ("known_slice_order_mismatch", ~slice_ok_if_available),
        ],
        out.index,
    )
    return out


def make_reasons(reason_masks: list[tuple[str, pd.Series]], index: pd.Index) -> pd.Series:
    reasons: list[str] = []
    for i in index:
        row_reasons = [name for name, mask in reason_masks if bool(mask.loc[i])]
        reasons.append(";".join(row_reasons) if row_reasons else "PASS")
    return pd.Series(reasons, index=index)


EXPORT_COLUMNS = [
    "SubjectID", "ImageID", "RID", "diagnosis_group", "y_true", "y_pred", "y_score_final",
    "confusion_label", "outer_fold", "in_vae_pool", "in_stageB_classifier_pool",
    "in_oof_evaluation", "Manufacturer_final", "Site3_final", "Age_final", "Sex_final",
    "ORIGPROT_final", "COLPROT_final", "raw_tp_group_final", "n_timepoints_raw",
    "n_timepoints_model_input", "scanner_model_final", "manufacturer_model_name_final",
    "software_version_final", "coil", "TR", "TE", "phase_encoding_direction_final",
    "PHASEDIR_final", "slice_order_class", "fmri_slice_order_len", "matches_dparsf_default",
    "stc_mismatch_risk", "Site31_reverse_even_odd_48", "Site2_default_7of7_pattern",
    "problem_site_flag", "high_confidence_slice_timing_match", "match_method",
    "match_confidence", "rp_available", "fd_mean", "fd_median", "fd_max", "fd_frac_gt0p5",
    "fmri_MEANTSNR", "fmri_MEDTSNR", "fmri_SDTSNR", "tsnr_proxy_median_corrected",
    "tsnr_proxy_median", "droi_rms_corrected", "drift_slope_median_abs_corrected",
    "outlier_frame_fraction_rz_gt3_corrected", "outlier_frame_fraction_rz_gt4_corrected",
    "tensor_ch0_offdiag_mean", "tensor_ch1_offdiag_mean", "tensor_ch2_offdiag_mean",
    "mat_path", "roisignals_mat_path", "individual_tensor_path", "global_tensor_path",
    "source_batch", "source_label", "metadata_source_primary", "metadata_sources_merged",
    "valid_qc_minimal", "valid_qc_protocol", "valid_qc_strict_available_only",
    "valid_qc_strict_missing_allowed", "valid_qc_minimal_fail_reasons",
    "valid_qc_protocol_fail_reasons", "valid_qc_strict_available_only_fail_reasons",
    "valid_qc_strict_missing_allowed_fail_reasons",
]


def full_export(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in EXPORT_COLUMNS if c in df.columns]
    return df[cols].copy()


MASKS = [
    "valid_qc_minimal",
    "valid_qc_protocol",
    "valid_qc_strict_available_only",
    "valid_qc_strict_missing_allowed",
]


def counts_by_mask(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for mask in MASKS:
        for retained, label in [(True, "retained"), (False, "removed")]:
            sub = df[df[mask].eq(retained)]
            rows.append({"mask": mask, "status": label, "stratification": "total", "group": "ALL", "N": len(sub)})
            for col, strat in [
                ("diagnosis_group", "diagnosis"),
                ("Manufacturer_final", "Manufacturer"),
                ("Site3_final", "Site3"),
                ("raw_tp_group_final", "raw_tp_group"),
                ("outer_fold", "outer_fold"),
            ]:
                if col in sub:
                    for group, n in sub[col].fillna("MISSING").astype(str).value_counts(dropna=False).sort_index().items():
                        rows.append({"mask": mask, "status": label, "stratification": strat, "group": group, "N": int(n)})
    return pd.DataFrame(rows)


def counts_by_fold(membership: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    flag_cols = ["SubjectID"] + MASKS
    mem = membership.merge(flags[flag_cols], on="SubjectID", how="left")
    rows: list[dict[str, Any]] = []
    for mask in MASKS:
        for (fold, split), sub0 in mem.groupby(["outer_fold", "split"], dropna=False):
            sub = sub0[sub0[mask].fillna(False)]
            row = {"mask": mask, "outer_fold": int(fold), "split": split, "N_retained": len(sub)}
            for dx in ["CN", "AD", "MCI"]:
                row[f"N_{dx}"] = int(sub["diagnosis_group"].eq(dx).sum())
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["mask", "outer_fold", "split"])


def philips_retention(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    high_score = pd.to_numeric(df["y_score_final"], errors="coerce") > 0.75
    groups = {
        "Philips_CN_all": df["Manufacturer_final"].str.upper().eq("PHILIPS") & df["diagnosis_group"].eq("CN"),
        "Philips_CN_FP": df["Manufacturer_final"].str.upper().eq("PHILIPS") & df["diagnosis_group"].eq("CN") & df["confusion_label"].astype(str).eq("FP"),
        "Philips_CN_TN": df["Manufacturer_final"].str.upper().eq("PHILIPS") & df["diagnosis_group"].eq("CN") & df["confusion_label"].astype(str).eq("TN"),
        "Site31_reverse_even_odd_48": df["Site31_reverse_even_odd_48"],
        "Site2_default_7of7_pattern": df["Site2_default_7of7_pattern"],
        "High_score_Philips_FP_gt0p75": df["Manufacturer_final"].str.upper().eq("PHILIPS") & df["diagnosis_group"].eq("CN") & df["confusion_label"].astype(str).eq("FP") & high_score,
    }
    for mask in MASKS:
        for group, gmask in groups.items():
            total = int(gmask.sum())
            retained = int((gmask & df[mask]).sum())
            rows.append({
                "mask": mask,
                "group": group,
                "N_total": total,
                "N_retained": retained,
                "N_removed": total - retained,
                "retained_fraction": retained / total if total else np.nan,
            })
    return pd.DataFrame(rows)


def data_dictionary(export: pd.DataFrame) -> str:
    descriptions = {
        "SubjectID": "ADNI subject identifier.",
        "ImageID": "Promoted-model fMRI image identifier where available.",
        "RID": "RID parsed from SubjectID.",
        "diagnosis_group": "Mapped diagnosis group used by promoted model: CN, AD, MCI.",
        "y_true": "Binary AD-vs-CN label for supervised OOF rows where available.",
        "y_pred": "Promoted OOF prediction at the locked ADNI threshold where available.",
        "y_score_final": "Promoted score used in the master database, primarily OOF-ECDF score.",
        "confusion_label": "TP/TN/FP/FN for supervised OOF subjects.",
        "outer_fold": "Outer fold assignment for supervised OOF subjects.",
        "in_vae_pool": "Subject was eligible for diagnosis-agnostic VAE pool.",
        "in_stageB_classifier_pool": "Subject was eligible for CN/AD StageB classifier pool.",
        "in_oof_evaluation": "Subject was part of OOF evaluation.",
        "valid_qc_minimal": "Preflight minimal valid-only mask, not applied to existing model.",
        "valid_qc_protocol": "Preflight protocol-valid mask, excluding known high-confidence STC mismatch.",
        "valid_qc_strict_available_only": "Strict mask requiring available motion/scanner/phase/slice annotations.",
        "valid_qc_strict_missing_allowed": "Strict sensitivity mask that fails known QC issues but does not fail unknown/missing annotations.",
    }
    rows = []
    for col in export.columns:
        rows.append(f"- `{col}`: {descriptions.get(col, 'Trace/protocol/model/QC field exported from promoted master database or derived audit flags.')}")
    return "# Data Dictionary for Martín\n\n" + "\n".join(rows)


def validity_definitions() -> str:
    return """# Validity Mask Definitions

These masks are preflight-only. They do not modify the promoted model, its
predictions, thresholds, tensors, or metadata.

## valid_qc_minimal

Requires tensor availability, mapped diagnosis group, Age/Sex, Manufacturer and
Site3/SITE, finite selected-channel tensor summary fields, and no original
supervised exclusion flag.

## valid_qc_protocol

Requires `valid_qc_minimal`, known raw timepoint group, traceable
Manufacturer/Site/rawTP, no high-confidence STC mismatch risk, and no confirmed
non-default slice-order mismatch versus the DPARSF default.

## valid_qc_strict_available_only

Requires `valid_qc_protocol`, motion/FD availability, no FD failure flag, no
Mayo/series QC failure string where present, scanner-model annotation, phase
annotation, and high-confidence Philips slice-timing annotation. This is
intentionally stringent and has major coverage impact.

## valid_qc_strict_missing_allowed

Requires `valid_qc_protocol` and fails known FD/QC/slice-order problems, but
does not automatically invalidate subjects solely because optional strict
annotations are missing. This is included because motion and detailed scanner
metadata coverage is incomplete.
"""


def feasibility_text(df: pd.DataFrame, counts_fold: pd.DataFrame, philips: pd.DataFrame) -> str:
    lines = ["# Valid-Only Retraining Feasibility", ""]
    for mask in MASKS:
        sub = df[df[mask]]
        dx = sub["diagnosis_group"].value_counts().to_dict()
        mfr = sub["Manufacturer_final"].value_counts().to_dict()
        stage = counts_fold[(counts_fold["mask"].eq(mask)) & (counts_fold["split"].eq("stageB_test_oof"))]
        min_cn = int(stage["N_CN"].min()) if not stage.empty else 0
        min_ad = int(stage["N_AD"].min()) if not stage.empty else 0
        lines.extend([
            f"## {mask}",
            "",
            f"- Total retained: {len(sub)}",
            f"- Diagnosis counts: {dx}",
            f"- Manufacturer counts: {mfr}",
            f"- Minimum retained StageB OOF fold CN/AD: CN={min_cn}, AD={min_ad}",
            "",
        ])
    lines.extend([
        "## Recommendation",
        "",
        "A valid-only retraining should not be launched until Martín confirms which",
        "protocol/QC annotations constitute a true exclusion criterion rather than a",
        "post-hoc performance-correlated concern.",
        "",
        "Preflight recommendation: **A) no retraining yet, need more annotations**.",
        "`valid_qc_minimal` is the only currently feasible retraining mask because",
        "it preserves CN, AD, MCI, and all three manufacturers. It should still be",
        "treated as a sensitivity analysis, not a replacement for the locked primary",
        "model.",
        "",
        "`valid_qc_protocol` is not currently feasible for AD-vs-CN retraining:",
        "the available protocol/rawTP annotations retain CN only in this export.",
        "This is an annotation-coverage finding, not evidence that AD/MCI subjects",
        "are invalid. `valid_qc_strict_available_only` is also not feasible now",
        "because strict motion/scanner/slice-timing availability removes the entire",
        "cohort. A strict QC sensitivity should wait until Martín's annotation pass",
        "is complete.",
    ])
    return "\n".join(lines)


def final_recommendation(df: pd.DataFrame, philips: pd.DataFrame) -> str:
    retained = {m: int(df[m].sum()) for m in MASKS}
    philips_fp = philips[philips["group"].eq("Philips_CN_FP")][["mask", "N_retained", "N_removed"]]
    return f"""# Final Recommendation

The full promoted-model database export for Martín has been generated with one
row per promoted-pipeline subject/scan. The promoted model remains locked.

Retained totals by mask:

{pd.DataFrame([retained]).to_markdown(index=False)}

Philips CN FP retention by mask:

{philips_fp.to_markdown(index=False)}

Any valid-only retraining must be framed as a **pre-specified sensitivity
analysis**, not a replacement for the primary promoted model. If a mask removes
mostly Philips false positives, any lower Philips FPR could be partially
mechanical and must be reported transparently.

Recommended next action: **A) no retraining yet, need more annotations**. The
next useful step is Martín confirmation of which fields constitute confirmed
invalid acquisition/preprocessing versus descriptive protocol strata.

Current retraining feasibility: `valid_qc_minimal` is the only mask that
preserves a usable CN/AD/MCI cohort with GE/Philips/SIEMENS diversity. The
protocol and strict masks are not feasible for AD-vs-CN retraining at this
snapshot because protocol/rawTP and strict QC annotations are incomplete outside
the CN-focused Philips/source-audit strata.
"""


def main() -> None:
    master, flags, membership, slice_merged = load_inputs()
    enriched = enrich_master(master, flags, slice_merged)
    valid = derive_validity_flags(enriched)
    export = full_export(valid)
    export_path = OUT / "promoted_model_full_database_for_martin_20260612.csv"
    export.to_csv(export_path, index=False)
    write_md(
        OUT / "promoted_model_full_database_for_martin_20260612.md",
        f"# Promoted Model Full Database for Martín\n\nRows: `{len(export)}`\n\nColumns: `{len(export.columns)}`\n\n"
        + export.head(200).to_markdown(index=False)
        + ("\n\n_Only first 200 rows shown. See CSV for full database._" if len(export) > 200 else ""),
    )
    write_md(OUT / "data_dictionary_for_martin.md", data_dictionary(export))
    write_md(OUT / "validity_mask_definitions.md", validity_definitions())

    flag_cols = ["SubjectID", "tensor_idx", "diagnosis_group", "Manufacturer_final", "Site3_final", "raw_tp_group_final", "outer_fold"] + MASKS + [
        "valid_qc_minimal_fail_reasons",
        "valid_qc_protocol_fail_reasons",
        "valid_qc_strict_available_only_fail_reasons",
        "valid_qc_strict_missing_allowed_fail_reasons",
    ]
    valid[[c for c in flag_cols if c in valid.columns]].to_csv(OUT / "subject_validity_flags.csv", index=False)

    counts_mask = counts_by_mask(valid)
    write_df("validity_preflight_counts_by_mask", counts_mask, md_rows=500)
    counts_fold = counts_by_fold(membership, valid[["SubjectID"] + MASKS])
    write_df("validity_preflight_counts_by_fold", counts_fold, md_rows=None)
    philips = philips_retention(valid)
    write_df("philips_cn_retention_by_mask", philips, md_rows=None)
    write_md(OUT / "valid_only_retraining_feasibility.md", feasibility_text(valid, counts_fold, philips))
    write_md(OUT / "final_recommendation.md", final_recommendation(valid, philips))
    write_json(
        OUT / "command_log.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "argv": ["scripts/revision_bspc_2026/export_full_database_for_martin_validity_preflight_20260612.py"],
            "inputs": {
                "master": str(MASTER_PATH),
                "flags": str(FLAGS_PATH),
                "membership": str(MEMBERSHIP_PATH),
                "slice_merged": str(SLICE_MERGED_PATH),
                "run_dir": str(RUN_DIR),
                "oof_dir": str(OOF_DIR),
            },
            "outputs": str(OUT.relative_to(PROJECT_ROOT)),
            "guardrails": {
                "read_only": True,
                "model_training": False,
                "tensor_edits": False,
                "metadata_edits": False,
                "prediction_edits": False,
                "threshold_refitting": False,
                "subject_exclusion": False,
            },
            "row_counts": {
                "export_rows": len(export),
                "export_columns": len(export.columns),
                **{m: int(valid[m].sum()) for m in MASKS},
            },
        },
    )


if __name__ == "__main__":
    main()
