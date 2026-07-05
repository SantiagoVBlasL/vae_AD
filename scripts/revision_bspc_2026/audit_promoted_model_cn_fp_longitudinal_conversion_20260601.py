#!/usr/bin/env python3
"""Read-only longitudinal conversion audit for promoted-model CN false positives.

Promoted model:
  recover035_latent384_beta3p75_T80_h10000_p560_full5x5

Promoted readout:
  logreg_l2_original / z_plus_age_sex / oof_logitz /
  inner_oof_target_sens_ge_0p70_max_spec

This script is strictly post-hoc clinical interpretation. It does not train,
fit thresholds, modify tensors, modify metadata, or perform model selection.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.api as sm
except Exception:  # pragma: no cover - optional dependency
    sm = None


PROJECT_ROOT = Path("/home/diego/proyectos/vae_AD")
RESULTS_ROOT = PROJECT_ROOT / "results/revision_bspc_2026"
OUTPUT_DIR = RESULTS_ROOT / "promoted_model_cn_false_positive_conversion_audit_20260601"

STAGEB_DIR = RESULTS_ROOT / "recover035_latent384_beta3p75_stageB_oof_score_calibration"
PROMOTED_PREDICTIONS = STAGEB_DIR / "calib_predictions.csv"
METADATA_PATH = RESULTS_ROOT / "adni_035_metadata_rescue_preflight/patched_metadata_candidate.csv"
SUBJECTS_DATA_PATH = PROJECT_ROOT / "data/SubjectsData_AAL3_procesado2.csv"
SUBJECTS_TESTS_PATH = PROJECT_ROOT / "data/SubjctsDataAndTestsAAL3.csv"
AD_FMRI_EXTENDED_PATH = PROJECT_ROOT / "data/AD_fMRI_4_28_2026_extended.csv"
AD_FMRI_PATH = PROJECT_ROOT / "data/AD_fMRI_4_28_2026.csv"

MARTIN_RAW_DIR = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/adni_metadata_martin_20260528/raw"
)
ADNIMERGE_PATH = MARTIN_RAW_DIR / "ADNIMERGE_14Oct2024.csv"
DXSUM_PATH = MARTIN_RAW_DIR / "DXSUM_28May2026.csv"
DATADIC_PATH = MARTIN_RAW_DIR / "DATADIC_28May2026.csv"

PROMOTED_MODEL = "logreg_l2_original"
PROMOTED_FEATURE_SET = "z_plus_age_sex"
PROMOTED_CALIB = "oof_logitz"
PROMOTED_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"

DXSUM_CODE_MAP = {
    1: "CN",
    2: "MCI",
    3: "AD_or_Dementia",
}


def clean_subject_id(value: Any) -> str:
    return str(value).strip()


def md_table(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df.empty:
        return "_No rows._\n"
    view = df.head(max_rows).copy()
    try:
        return view.to_markdown(index=False) + "\n"
    except Exception:
        return view.to_string(index=False) + "\n"


def write_table(stem: str, df: pd.DataFrame, max_rows: int = 200) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / f"{stem}.csv", index=False)
    (OUTPUT_DIR / f"{stem}.md").write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


def write_md(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def months_between(later: pd.Timestamp | pd.NaT, earlier: pd.Timestamp | pd.NaT) -> float:
    if pd.isna(later) or pd.isna(earlier):
        return float("nan")
    return float((later - earlier).days / 30.4375)


def normalize_dx(value: Any) -> str:
    if pd.isna(value):
        return "Unknown"
    s = str(value).strip()
    if not s:
        return "Unknown"
    up = s.upper().replace("-", "_").replace(" ", "_")
    if up in {"CN", "NL", "NORMAL", "COGNITIVELY_NORMAL", "NORM"}:
        return "CN"
    if up in {"MCI", "EMCI", "LMCI", "SMC_TO_MCI", "MCI_TO_MCI"} or "MCI" in up:
        return "MCI"
    if up in {"AD", "DEMENTIA", "AD_DEMENTIA", "ALZHEIMER", "ALZHEIMERS"}:
        return "AD_or_Dementia"
    if "DEMENT" in up or up == "AD":
        return "AD_or_Dementia"
    return "Unknown"


def normalize_manufacturer(value: Any) -> str:
    if pd.isna(value):
        return "Unknown"
    s = str(value).strip()
    low = s.lower()
    if "philips" in low:
        return "Philips"
    if "siemens" in low:
        return "SIEMENS"
    if low in {"ge", "ge medical systems"} or "general electric" in low:
        return "GE"
    return s if s else "Unknown"


def normalize_site3(value: Any) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    if not s:
        return ""
    try:
        return f"{int(float(s)):03d}"
    except Exception:
        return s.zfill(3) if s.isdigit() else s


def fpath(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def source_inventory() -> pd.DataFrame:
    candidates = [
        (ADNIMERGE_PATH, "primary_longitudinal_dx", "ADNIMERGE visit-level DX text"),
        (DXSUM_PATH, "primary_longitudinal_dx", "DXSUM visit-level DIAGNOSIS codes"),
        (DATADIC_PATH, "code_dictionary", "ADNI data dictionary; used only if present"),
        (METADATA_PATH, "model_metadata_anchor", "Recovered all-eligible metadata candidate"),
        (SUBJECTS_DATA_PATH, "fmri_anchor_clinical", "fMRI StudyDate/Visit and clinical fields"),
        (SUBJECTS_TESTS_PATH, "fmri_anchor_clinical", "alternate fMRI clinical table"),
        (AD_FMRI_EXTENDED_PATH, "fmri_anchor_fallback", "AD fMRI download table with AcqDate"),
        (AD_FMRI_PATH, "fmri_anchor_fallback", "AD fMRI download table"),
        (PROMOTED_PREDICTIONS, "model_predictions", "Promoted OOF-logitz Stage B predictions"),
    ]
    rows: list[dict[str, Any]] = []
    for path, role, note in candidates:
        row = {
            "path": fpath(path),
            "exists": path.exists(),
            "role": role,
            "note": note,
            "n_rows": np.nan,
            "n_columns": np.nan,
            "candidate_subject_col": "",
            "candidate_date_cols": "",
            "candidate_dx_cols": "",
        }
        if path.exists():
            try:
                df = pd.read_csv(path, nrows=200, low_memory=False)
                row["n_columns"] = len(df.columns)
                # Full row count is useful for the small/medium CSVs.
                try:
                    row["n_rows"] = sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1
                except Exception:
                    row["n_rows"] = np.nan
                cols = [str(c) for c in df.columns]
                row["candidate_subject_col"] = ",".join(
                    [c for c in cols if c.upper() in {"PTID", "SUBJECTID", "SUBJECT", "RID"}]
                )
                row["candidate_date_cols"] = ",".join(
                    [c for c in cols if "DATE" in c.upper() or c.upper() in {"VISCODE", "VISCODE2"}]
                )
                row["candidate_dx_cols"] = ",".join(
                    [c for c in cols if c.upper() in {"DX", "DX_BL", "DIAGNOSIS", "GROUP", "RESEARCHGROUP_MAPPED"}]
                )
            except Exception as exc:
                row["note"] += f"; read_error={exc}"
        rows.append(row)
    return pd.DataFrame(rows)


def diagnosis_mapping_table() -> pd.DataFrame:
    rows = [
        {
            "source_file": "ADNIMERGE_14Oct2024.csv",
            "raw_field": "DX",
            "raw_code_or_value": "CN",
            "mapped_diagnosis": "CN",
            "mapping_note": "Text diagnosis; used as visit-level diagnosis when present.",
        },
        {
            "source_file": "ADNIMERGE_14Oct2024.csv",
            "raw_field": "DX",
            "raw_code_or_value": "MCI",
            "mapped_diagnosis": "MCI",
            "mapping_note": "Text diagnosis; used as visit-level diagnosis when present.",
        },
        {
            "source_file": "ADNIMERGE_14Oct2024.csv",
            "raw_field": "DX",
            "raw_code_or_value": "Dementia",
            "mapped_diagnosis": "AD_or_Dementia",
            "mapping_note": "Text diagnosis; mapped to AD/Dementia endpoint.",
        },
        {
            "source_file": "ADNIMERGE_14Oct2024.csv",
            "raw_field": "DX_bl",
            "raw_code_or_value": "AD",
            "mapped_diagnosis": "AD_or_Dementia",
            "mapping_note": "Used only as baseline fallback when DX is unavailable.",
        },
        {
            "source_file": "DXSUM_28May2026.csv",
            "raw_field": "DIAGNOSIS",
            "raw_code_or_value": "1",
            "mapped_diagnosis": "CN",
            "mapping_note": "ADNI DXSUM diagnostic-summary code.",
        },
        {
            "source_file": "DXSUM_28May2026.csv",
            "raw_field": "DIAGNOSIS",
            "raw_code_or_value": "2",
            "mapped_diagnosis": "MCI",
            "mapping_note": "ADNI DXSUM diagnostic-summary code.",
        },
        {
            "source_file": "DXSUM_28May2026.csv",
            "raw_field": "DIAGNOSIS",
            "raw_code_or_value": "3",
            "mapped_diagnosis": "AD_or_Dementia",
            "mapping_note": "ADNI DXSUM diagnostic-summary code.",
        },
    ]
    return pd.DataFrame(rows)


def load_promoted_predictions() -> pd.DataFrame:
    df = pd.read_csv(PROMOTED_PREDICTIONS)
    mask = (
        df["model_name"].astype(str).eq(PROMOTED_MODEL)
        & df["feature_set"].astype(str).eq(PROMOTED_FEATURE_SET)
        & df["calib_method"].astype(str).eq(PROMOTED_CALIB)
        & df["threshold_strategy"].astype(str).eq(PROMOTED_THRESHOLD)
    )
    out = df.loc[mask].copy()
    if out.empty:
        raise RuntimeError("No promoted prediction rows found in calib_predictions.csv")
    out["SubjectID"] = out["SubjectID"].apply(clean_subject_id)
    out["Manufacturer"] = out["Manufacturer"].map(normalize_manufacturer)
    out["model_error_type"] = np.select(
        [
            (out["y_true"].eq(0) & out["y_pred"].eq(0)),
            (out["y_true"].eq(0) & out["y_pred"].eq(1)),
            (out["y_true"].eq(1) & out["y_pred"].eq(0)),
            (out["y_true"].eq(1) & out["y_pred"].eq(1)),
        ],
        ["TN", "FP", "FN", "TP"],
        default="Unknown",
    )
    return out


def load_anchor_metadata() -> pd.DataFrame:
    meta_cols = [
        "SubjectID",
        "ResearchGroup_Mapped",
        "Age",
        "Sex",
        "Manufacturer",
        "Site3",
        "ImageID",
        "Visit",
        "source_batch",
        "metadata_source",
        "n_timepoints_raw",
    ]
    meta = pd.read_csv(METADATA_PATH, low_memory=False)
    meta["SubjectID"] = meta["SubjectID"].apply(clean_subject_id)
    meta = meta[[c for c in meta_cols if c in meta.columns]].drop_duplicates("SubjectID")

    frames = []
    if SUBJECTS_DATA_PATH.exists():
        sd = pd.read_csv(SUBJECTS_DATA_PATH, low_memory=False)
        sd["SubjectID"] = sd["SubjectID"].apply(clean_subject_id)
        keep = [
            "SubjectID",
            "Phase",
            "StudyDate",
            "ArchiveDate",
            "Visit",
            "ImageID",
            "CDRSB",
            "MMSE",
            "MOCA",
            "PTEDUCAT",
            "ABETA",
            "TAU",
            "PTAU",
            "Ventricles",
            "Hippocampus",
            "WholeBrain",
            "MidTemp",
            "TR",
        ]
        sd = sd[[c for c in keep if c in sd.columns]].copy()
        sd["anchor_source"] = "SubjectsData_AAL3_procesado2.csv"
        frames.append(sd)
    if SUBJECTS_TESTS_PATH.exists():
        st = pd.read_csv(SUBJECTS_TESTS_PATH, low_memory=False)
        st["SubjectID"] = st["SubjectID"].apply(clean_subject_id)
        keep = [
            "SubjectID",
            "Phase",
            "StudyDate",
            "ArchiveDate",
            "Visit",
            "ImageID",
            "CDRSB",
            "MMSE",
            "MOCA",
            "PTEDUCAT",
            "ABETA",
            "TAU",
            "PTAU",
            "Ventricles",
            "Hippocampus",
            "WholeBrain",
            "MidTemp",
        ]
        st = st[[c for c in keep if c in st.columns]].copy()
        st["anchor_source"] = "SubjctsDataAndTestsAAL3.csv"
        frames.append(st)

    if frames:
        anchor = pd.concat(frames, ignore_index=True, sort=False)
        anchor["StudyDate_parsed"] = parse_date(anchor.get("StudyDate", pd.Series(index=anchor.index)))
        anchor["ArchiveDate_parsed"] = parse_date(anchor.get("ArchiveDate", pd.Series(index=anchor.index)))
        anchor["nonmissing_clinical"] = anchor[
            [c for c in ["CDRSB", "MMSE", "MOCA", "ABETA", "TAU", "PTAU"] if c in anchor.columns]
        ].notna().sum(axis=1)
        anchor["source_priority"] = anchor["anchor_source"].map(
            {"SubjectsData_AAL3_procesado2.csv": 0, "SubjctsDataAndTestsAAL3.csv": 1}
        )
        anchor = anchor.sort_values(
            ["SubjectID", "StudyDate_parsed", "nonmissing_clinical", "source_priority"],
            ascending=[True, True, False, True],
        ).drop_duplicates("SubjectID", keep="first")
    else:
        anchor = pd.DataFrame({"SubjectID": meta["SubjectID"]})

    out = meta.merge(anchor, on="SubjectID", how="left", suffixes=("", "_anchor"))

    # Fallback acquisition date for AD rows from the downloaded AD table.
    ad_frames = []
    if AD_FMRI_EXTENDED_PATH.exists():
        ad = pd.read_csv(AD_FMRI_EXTENDED_PATH, low_memory=False)
        ad["SubjectID"] = ad["Subject"].apply(clean_subject_id)
        ad["AcqDate"] = ad["AcqDate"] if "AcqDate" in ad.columns else pd.NA
        ad["ImageID_ad_table"] = ad.get("ImageDataID", pd.NA).astype(str).str.replace("^I", "", regex=True)
        ad_frames.append(ad[["SubjectID", "AcqDate", "ImageID_ad_table"]].copy())
    if AD_FMRI_PATH.exists():
        ad = pd.read_csv(AD_FMRI_PATH, low_memory=False)
        ad["SubjectID"] = ad["Subject"].apply(clean_subject_id)
        ad["AcqDate"] = ad["Acq Date"] if "Acq Date" in ad.columns else pd.NA
        ad["ImageID_ad_table"] = ad.get("Image Data ID", pd.NA).astype(str).str.replace("^I", "", regex=True)
        ad_frames.append(ad[["SubjectID", "AcqDate", "ImageID_ad_table"]].copy())
    if ad_frames:
        ad_dates = pd.concat(ad_frames, ignore_index=True, sort=False)
        ad_dates["AcqDate_parsed"] = parse_date(ad_dates["AcqDate"])
        ad_dates = ad_dates.sort_values(["SubjectID", "AcqDate_parsed"]).drop_duplicates("SubjectID")
        out = out.merge(ad_dates, on="SubjectID", how="left")
    else:
        out["AcqDate"] = pd.NA
        out["AcqDate_parsed"] = pd.NaT
        out["ImageID_ad_table"] = pd.NA

    out["fmri_study_date"] = out.get("StudyDate_parsed", pd.Series(pd.NaT, index=out.index))
    out["fmri_study_date"] = out["fmri_study_date"].fillna(out.get("AcqDate_parsed", pd.Series(pd.NaT, index=out.index)))
    out["fmri_study_date"] = out["fmri_study_date"].fillna(out.get("ArchiveDate_parsed", pd.Series(pd.NaT, index=out.index)))
    out["Manufacturer"] = out["Manufacturer"].map(normalize_manufacturer)
    out["Site3"] = out["Site3"].map(normalize_site3)
    return out


def load_adnimerge_longitudinal(subjects: set[str]) -> pd.DataFrame:
    if not ADNIMERGE_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(ADNIMERGE_PATH, low_memory=False)
    df["SubjectID"] = df["PTID"].apply(clean_subject_id)
    df = df[df["SubjectID"].isin(subjects)].copy()
    df["exam_date"] = parse_date(df["EXAMDATE"])
    dx_raw = df["DX"] if "DX" in df.columns else pd.Series(pd.NA, index=df.index)
    # Baseline fallback when visit-level DX is missing.
    if "DX_bl" in df.columns:
        dx_raw = dx_raw.fillna(df["DX_bl"])
    out = pd.DataFrame(
        {
            "SubjectID": df["SubjectID"],
            "RID": df.get("RID", pd.NA),
            "VISCODE": df.get("VISCODE", pd.NA),
            "VISCODE2": df.get("VISCODE", pd.NA),
            "exam_date": df["exam_date"],
            "diagnosis_date": df["exam_date"],
            "source_file": "ADNIMERGE_14Oct2024.csv",
            "raw_diagnosis_code": dx_raw,
            "mapped_diagnosis": dx_raw.map(normalize_dx),
            "source_priority": 1,
        }
    )
    return out


def load_dxsum_longitudinal(subjects: set[str]) -> pd.DataFrame:
    if not DXSUM_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(DXSUM_PATH, low_memory=False)
    df["SubjectID"] = df["PTID"].apply(clean_subject_id)
    df = df[df["SubjectID"].isin(subjects)].copy()
    df["exam_date"] = parse_date(df["EXAMDATE"])
    raw = pd.to_numeric(df.get("DIAGNOSIS", pd.Series(pd.NA, index=df.index)), errors="coerce")
    out = pd.DataFrame(
        {
            "SubjectID": df["SubjectID"],
            "RID": df.get("RID", pd.NA),
            "VISCODE": df.get("VISCODE", pd.NA),
            "VISCODE2": df.get("VISCODE2", pd.NA),
            "exam_date": df["exam_date"],
            "diagnosis_date": df["exam_date"],
            "source_file": "DXSUM_28May2026.csv",
            "raw_diagnosis_code": raw,
            "mapped_diagnosis": raw.map(lambda x: DXSUM_CODE_MAP.get(int(x), "Unknown") if pd.notna(x) else "Unknown"),
            "source_priority": 0,
        }
    )
    return out


def build_longitudinal_table(subjects: set[str]) -> pd.DataFrame:
    frames = [load_dxsum_longitudinal(subjects), load_adnimerge_longitudinal(subjects)]
    long = pd.concat([f for f in frames if not f.empty], ignore_index=True, sort=False)
    if long.empty:
        return long
    long["diagnosis_date"] = pd.to_datetime(long["diagnosis_date"], errors="coerce")
    long["mapped_diagnosis"] = long["mapped_diagnosis"].fillna("Unknown")
    long = long.sort_values(["SubjectID", "diagnosis_date", "source_priority", "source_file"])
    return long.reset_index(drop=True)


def build_anchor_table(pred: pd.DataFrame, anchor_meta: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "SubjectID",
        "ResearchGroup_Mapped",
        "Age",
        "Sex",
        "Manufacturer",
        "Site3",
        "ImageID",
        "Visit",
        "Phase",
        "source_batch",
        "metadata_source",
        "n_timepoints_raw",
        "fmri_study_date",
        "StudyDate",
        "ArchiveDate",
        "AcqDate",
        "anchor_source",
        "CDRSB",
        "MMSE",
        "MOCA",
        "PTEDUCAT",
        "ABETA",
        "TAU",
        "PTAU",
        "Ventricles",
        "Hippocampus",
        "WholeBrain",
        "MidTemp",
    ]
    meta = anchor_meta[[c for c in keep if c in anchor_meta.columns]].copy()
    merged = pred.merge(meta, on="SubjectID", how="left", suffixes=("", "_meta"))
    for col in ["ResearchGroup_Mapped", "Age", "Sex", "Manufacturer"]:
        meta_col = f"{col}_meta"
        if meta_col in merged.columns:
            merged[col] = merged[col].where(merged[col].notna(), merged[meta_col])
            merged = merged.drop(columns=[meta_col])
    merged["Site3"] = merged.get("Site3", pd.Series(pd.NA, index=merged.index)).map(normalize_site3)
    merged["Phase"] = merged.get("Phase", pd.Series(pd.NA, index=merged.index))
    merged["fmri_study_date"] = pd.to_datetime(merged["fmri_study_date"], errors="coerce")
    return merged


def first_future_conversion(future: pd.DataFrame, dx_set: set[str]) -> tuple[Any, Any, float]:
    hit = future[future["mapped_diagnosis"].isin(dx_set)].copy()
    if hit.empty:
        return pd.NaT, "", float("nan")
    row = hit.sort_values(["diagnosis_date", "source_priority"]).iloc[0]
    return row["diagnosis_date"], row["mapped_diagnosis"], float(row["source_priority"])


def build_cn_conversion_table(anchor: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    cn = anchor[anchor["y_true"].eq(0)].copy()
    long_valid = long[long["mapped_diagnosis"].isin(["CN", "MCI", "AD_or_Dementia"])].copy()
    rows: list[dict[str, Any]] = []
    for _, row in cn.iterrows():
        sid = row["SubjectID"]
        anchor_date = row["fmri_study_date"]
        subj_long = long_valid[long_valid["SubjectID"].eq(sid)].copy()
        if pd.notna(anchor_date):
            future = subj_long[subj_long["diagnosis_date"].notna() & (subj_long["diagnosis_date"] > anchor_date)]
            all_after_or_equal = subj_long[subj_long["diagnosis_date"].notna() & (subj_long["diagnosis_date"] >= anchor_date)]
            prior_or_anchor = subj_long[subj_long["diagnosis_date"].notna() & (subj_long["diagnosis_date"] <= anchor_date)]
        else:
            future = subj_long.iloc[0:0]
            all_after_or_equal = subj_long.iloc[0:0]
            prior_or_anchor = subj_long.iloc[0:0]

        mci_date, mci_dx, _ = first_future_conversion(future, {"MCI"})
        ad_date, ad_dx, _ = first_future_conversion(future, {"AD_or_Dementia"})
        any_date, any_dx, _ = first_future_conversion(future, {"MCI", "AD_or_Dementia"})

        last_followup = future["diagnosis_date"].max() if not future.empty else pd.NaT
        if pd.isna(last_followup) and not all_after_or_equal.empty:
            last_followup = all_after_or_equal["diagnosis_date"].max()
        followup_months = months_between(last_followup, anchor_date)

        dx_sequence = " > ".join(
            f"{d.date()}:{dx}"
            for d, dx in subj_long[["diagnosis_date", "mapped_diagnosis"]].dropna().itertuples(index=False)
        )
        prior_dx_sequence = " > ".join(
            f"{d.date()}:{dx}"
            for d, dx in prior_or_anchor[["diagnosis_date", "mapped_diagnosis"]].dropna().itertuples(index=False)
        )
        future_dx_sequence = " > ".join(
            f"{d.date()}:{dx}"
            for d, dx in future[["diagnosis_date", "mapped_diagnosis"]].dropna().itertuples(index=False)
        )

        converted_mci = pd.notna(mci_date)
        converted_ad = pd.notna(ad_date)
        converted_any = pd.notna(any_date)
        has_followup = bool(len(future) > 0)
        prior_pathological = bool(prior_or_anchor["mapped_diagnosis"].isin(["MCI", "AD_or_Dementia"]).any())
        prior_ad = bool(prior_or_anchor["mapped_diagnosis"].eq("AD_or_Dementia").any())
        prior_mci = bool(prior_or_anchor["mapped_diagnosis"].eq("MCI").any())
        future_dx_values = future["mapped_diagnosis"].dropna().tolist()
        post_conversion_reversion_to_cn = False
        if converted_any and any_dx in future_dx_values:
            post_conversion_reversion_to_cn = "CN" in future_dx_values[future_dx_values.index(any_dx) + 1 :]

        if prior_pathological or post_conversion_reversion_to_cn:
            clinical_class = "ambiguous_diagnosis_trajectory"
        elif converted_any:
            clinical_class = "likely_prodromal_or_converter"
        elif has_followup and pd.notna(followup_months) and followup_months >= 36:
            clinical_class = "stable_CN_with_long_followup"
        elif has_followup:
            clinical_class = "insufficient_followup_no_conversion"
        else:
            clinical_class = "insufficient_followup_no_future_dx"

        rows.append(
            {
                "SubjectID": sid,
                "RID": first_nonmissing(subj_long.get("RID", pd.Series(dtype=object))),
                "ResearchGroup_Mapped": row.get("ResearchGroup_Mapped", ""),
                "Age": row.get("Age", np.nan),
                "Sex": row.get("Sex", ""),
                "Manufacturer": row.get("Manufacturer", ""),
                "Site3": row.get("Site3", ""),
                "Phase": row.get("Phase", ""),
                "fold": row.get("fold", np.nan),
                "y_true": row.get("y_true", np.nan),
                "y_score": row.get("y_score", np.nan),
                "threshold": row.get("threshold", np.nan),
                "y_pred": row.get("y_pred", np.nan),
                "model_error_type": row.get("model_error_type", ""),
                "fmri_study_date": anchor_date.date().isoformat() if pd.notna(anchor_date) else "",
                "Visit": row.get("Visit", ""),
                "ImageID": row.get("ImageID", ""),
                "n_longitudinal_dx_rows": int(len(subj_long)),
                "n_future_dx_rows_after_fmri": int(len(future)),
                "has_followup_after_fmri": has_followup,
                "converted_to_MCI_after_fMRI": converted_mci,
                "converted_to_AD_or_Dementia_after_fMRI": converted_ad,
                "converted_to_any_pathological_dx_after_fMRI": converted_any,
                "first_conversion_date": any_date.date().isoformat() if pd.notna(any_date) else "",
                "first_conversion_dx": any_dx,
                "months_to_conversion": months_between(any_date, anchor_date),
                "last_followup_date": last_followup.date().isoformat() if pd.notna(last_followup) else "",
                "followup_months": followup_months,
                "censored_no_conversion": bool(has_followup and not converted_any),
                "prior_or_anchor_pathological_dx_evidence": prior_pathological,
                "prior_or_anchor_MCI_evidence": prior_mci,
                "prior_or_anchor_AD_or_Dementia_evidence": prior_ad,
                "post_conversion_reversion_to_CN": post_conversion_reversion_to_cn,
                "dx_sequence_all_sources": dx_sequence,
                "prior_or_anchor_dx_sequence": prior_dx_sequence,
                "future_dx_sequence_after_fmri": future_dx_sequence,
                "clinical_interpretation_group": clinical_class,
                "CDRSB": row.get("CDRSB", np.nan),
                "MMSE": row.get("MMSE", np.nan),
                "MOCA": row.get("MOCA", np.nan),
                "ABETA": row.get("ABETA", np.nan),
                "TAU": row.get("TAU", np.nan),
                "PTAU": row.get("PTAU", np.nan),
                "n_timepoints_raw": row.get("n_timepoints_raw", np.nan),
            }
        )
    return pd.DataFrame(rows)


def first_nonmissing(series: pd.Series) -> Any:
    if series is None or len(series) == 0:
        return ""
    valid = series.dropna()
    if valid.empty:
        return ""
    return valid.iloc[0]


def summarize_groups(df: pd.DataFrame, group_col: str, groups: list[str], label: str) -> pd.DataFrame:
    rows = []
    for g in groups:
        sub = df[df[group_col].eq(g)].copy()
        with_follow = sub[sub["has_followup_after_fmri"].eq(True)]
        row = {
            "comparison": label,
            "group": g,
            "N_subjects": len(sub),
            "N_with_followup_after_fmri": len(with_follow),
            "median_followup_months": float(with_follow["followup_months"].median()) if len(with_follow) else np.nan,
            "mean_age": float(pd.to_numeric(sub["Age"], errors="coerce").mean()) if len(sub) else np.nan,
            "conversion_to_MCI_N": int(with_follow["converted_to_MCI_after_fMRI"].sum()) if len(with_follow) else 0,
            "conversion_to_MCI_rate": float(with_follow["converted_to_MCI_after_fMRI"].mean()) if len(with_follow) else np.nan,
            "conversion_to_AD_or_Dementia_N": int(with_follow["converted_to_AD_or_Dementia_after_fMRI"].sum()) if len(with_follow) else 0,
            "conversion_to_AD_or_Dementia_rate": float(with_follow["converted_to_AD_or_Dementia_after_fMRI"].mean()) if len(with_follow) else np.nan,
            "conversion_to_MCI_or_AD_N": int(with_follow["converted_to_any_pathological_dx_after_fMRI"].sum()) if len(with_follow) else 0,
            "conversion_to_MCI_or_AD_rate": float(with_follow["converted_to_any_pathological_dx_after_fMRI"].mean()) if len(with_follow) else np.nan,
            "N_stable_long_followup": int(sub["clinical_interpretation_group"].eq("stable_CN_with_long_followup").sum()),
            "N_insufficient_followup": int(sub["clinical_interpretation_group"].str.startswith("insufficient").sum()),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def add_comparison_labels(cn: pd.DataFrame) -> pd.DataFrame:
    cn = cn.copy()
    cn["philips_fp_tn_group"] = np.where(
        cn["Manufacturer"].eq("Philips") & cn["model_error_type"].eq("FP"),
        "Philips_CN_FP",
        np.where(
            cn["Manufacturer"].eq("Philips") & cn["model_error_type"].eq("TN"),
            "Philips_CN_TN",
            "Other_CN",
        ),
    )
    cn["all_cn_fp_tn_group"] = np.where(
        cn["model_error_type"].eq("FP"),
        "All_CN_FP",
        np.where(cn["model_error_type"].eq("TN"), "All_CN_TN", "Other_CN"),
    )
    cn["philips_vs_nonphilips_fp_group"] = np.where(
        cn["Manufacturer"].eq("Philips") & cn["model_error_type"].eq("FP"),
        "Philips_CN_FP",
        np.where(~cn["Manufacturer"].eq("Philips") & cn["model_error_type"].eq("FP"), "NonPhilips_CN_FP", "Other_CN"),
    )
    cn["score_quartile"] = pd.qcut(
        pd.to_numeric(cn["y_score"], errors="coerce"),
        q=4,
        labels=["Q1_lowest_score", "Q2", "Q3", "Q4_highest_score"],
        duplicates="drop",
    ).astype(str)
    return cn


def fisher_or_ci(a: int, b: int, c: int, d: int) -> tuple[float, float, float, float]:
    _, p = stats.fisher_exact([[a, b], [c, d]], alternative="two-sided")
    aa, bb, cc, dd = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    odds = (aa * dd) / (bb * cc)
    se = math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    lo = math.exp(math.log(odds) - 1.96 * se)
    hi = math.exp(math.log(odds) + 1.96 * se)
    return float(odds), float(p), float(lo), float(hi)


def fisher_tests(cn: pd.DataFrame) -> pd.DataFrame:
    tests = [
        ("Philips CN FP vs Philips CN TN", "philips_fp_tn_group", "Philips_CN_FP", "Philips_CN_TN"),
        ("All CN FP vs All CN TN", "all_cn_fp_tn_group", "All_CN_FP", "All_CN_TN"),
        ("Philips CN FP vs non-Philips CN FP", "philips_vs_nonphilips_fp_group", "Philips_CN_FP", "NonPhilips_CN_FP"),
        ("High-score Q4 CN vs low-score Q1 CN", "score_quartile", "Q4_highest_score", "Q1_lowest_score"),
    ]
    outcomes = [
        ("converted_to_MCI_after_fMRI", "MCI conversion"),
        ("converted_to_AD_or_Dementia_after_fMRI", "AD/Dementia conversion"),
        ("converted_to_any_pathological_dx_after_fMRI", "MCI_or_AD conversion"),
    ]
    rows = []
    for comp, group_col, exposed, ref in tests:
        for outcome_col, outcome_label in outcomes:
            sub = cn[cn[group_col].isin([exposed, ref]) & cn["has_followup_after_fmri"].eq(True)].copy()
            exp = sub[sub[group_col].eq(exposed)]
            ref_df = sub[sub[group_col].eq(ref)]
            a = int(exp[outcome_col].sum())
            b = int(len(exp) - a)
            c = int(ref_df[outcome_col].sum())
            d = int(len(ref_df) - c)
            if len(exp) == 0 or len(ref_df) == 0:
                odds = p = lo = hi = np.nan
                status = "not_feasible_missing_group"
            else:
                odds, p, lo, hi = fisher_or_ci(a, b, c, d)
                status = "computed"
            rows.append(
                {
                    "comparison": comp,
                    "outcome": outcome_label,
                    "analysis_subjects_with_followup": int(len(sub)),
                    "exposed_group": exposed,
                    "reference_group": ref,
                    "exposed_events": a,
                    "exposed_non_events": b,
                    "reference_events": c,
                    "reference_non_events": d,
                    "odds_ratio_haldane_anscombe": odds,
                    "odds_ratio_ci95_low": lo,
                    "odds_ratio_ci95_high": hi,
                    "p_fisher_exact": p,
                    "status": status,
                }
            )
    return pd.DataFrame(rows)


def age_adjusted_models(cn: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("Philips CN FP vs Philips CN TN", "philips_fp_tn_group", "Philips_CN_FP", "Philips_CN_TN"),
        ("All CN FP vs All CN TN", "all_cn_fp_tn_group", "All_CN_FP", "All_CN_TN"),
    ]
    rows = []
    for label, group_col, exposed, ref in specs:
        sub = cn[
            cn[group_col].isin([exposed, ref])
            & cn["has_followup_after_fmri"].eq(True)
        ].copy()
        sub["event"] = sub["converted_to_any_pathological_dx_after_fMRI"].astype(int)
        sub["exposed"] = sub[group_col].eq(exposed).astype(int)
        sub["Age_numeric"] = pd.to_numeric(sub["Age"], errors="coerce")
        sub = sub.dropna(subset=["Age_numeric", "event", "exposed"])
        status = "computed"
        if sm is None:
            status = "not_feasible_statsmodels_unavailable"
        elif len(sub) < 20:
            status = "not_feasible_n_lt_20"
        elif sub["event"].sum() < 5 or (len(sub) - sub["event"].sum()) < 5:
            status = "not_feasible_too_few_events_or_nonevents"
        elif sub["exposed"].nunique() < 2:
            status = "not_feasible_missing_comparison_group"

        if status != "computed":
            rows.append(
                {
                    "model": label,
                    "outcome": "MCI_or_AD conversion",
                    "n": int(len(sub)),
                    "events": int(sub["event"].sum()) if len(sub) else 0,
                    "coef_exposed": np.nan,
                    "or_exposed": np.nan,
                    "p_exposed": np.nan,
                    "coef_age": np.nan,
                    "or_age": np.nan,
                    "p_age": np.nan,
                    "status": status,
                }
            )
            continue
        try:
            x = sm.add_constant(sub[["exposed", "Age_numeric"]].astype(float))
            y = sub["event"].astype(float)
            fit = sm.Logit(y, x).fit(disp=False)
            rows.append(
                {
                    "model": label,
                    "outcome": "MCI_or_AD conversion",
                    "n": int(len(sub)),
                    "events": int(sub["event"].sum()),
                    "coef_exposed": float(fit.params.get("exposed", np.nan)),
                    "or_exposed": float(np.exp(fit.params.get("exposed", np.nan))),
                    "p_exposed": float(fit.pvalues.get("exposed", np.nan)),
                    "coef_age": float(fit.params.get("Age_numeric", np.nan)),
                    "or_age": float(np.exp(fit.params.get("Age_numeric", np.nan))),
                    "p_age": float(fit.pvalues.get("Age_numeric", np.nan)),
                    "status": "computed",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "model": label,
                    "outcome": "MCI_or_AD conversion",
                    "n": int(len(sub)),
                    "events": int(sub["event"].sum()),
                    "coef_exposed": np.nan,
                    "or_exposed": np.nan,
                    "p_exposed": np.nan,
                    "coef_age": np.nan,
                    "or_age": np.nan,
                    "p_age": np.nan,
                    "status": f"not_feasible_fit_error:{exc}",
                }
            )
    return pd.DataFrame(rows)


def high_score_gradient(cn: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for q, sub in cn.groupby("score_quartile", dropna=False):
        follow = sub[sub["has_followup_after_fmri"].eq(True)]
        rows.append(
            {
                "score_group": q,
                "N_subjects": len(sub),
                "N_with_followup": len(follow),
                "score_min": float(pd.to_numeric(sub["y_score"], errors="coerce").min()),
                "score_median": float(pd.to_numeric(sub["y_score"], errors="coerce").median()),
                "score_max": float(pd.to_numeric(sub["y_score"], errors="coerce").max()),
                "conversion_to_MCI_or_AD_N": int(follow["converted_to_any_pathological_dx_after_fMRI"].sum()),
                "conversion_to_MCI_or_AD_rate": float(follow["converted_to_any_pathological_dx_after_fMRI"].mean()) if len(follow) else np.nan,
                "conversion_to_AD_or_Dementia_N": int(follow["converted_to_AD_or_Dementia_after_fMRI"].sum()),
                "conversion_to_AD_or_Dementia_rate": float(follow["converted_to_AD_or_Dementia_after_fMRI"].mean()) if len(follow) else np.nan,
                "median_followup_months": float(follow["followup_months"].median()) if len(follow) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("score_group")


def secondary_strata(cn: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["Phase", "Site3", "Manufacturer"]:
        if col not in cn.columns:
            continue
        for val, sub in cn.groupby(col, dropna=False):
            follow = sub[sub["has_followup_after_fmri"].eq(True)]
            rows.append(
                {
                    "stratum_type": col,
                    "stratum_value": val,
                    "N_subjects": len(sub),
                    "N_FP": int(sub["model_error_type"].eq("FP").sum()),
                    "N_TN": int(sub["model_error_type"].eq("TN").sum()),
                    "N_with_followup": len(follow),
                    "conversion_to_MCI_or_AD_N": int(follow["converted_to_any_pathological_dx_after_fMRI"].sum()),
                    "conversion_to_MCI_or_AD_rate": float(follow["converted_to_any_pathological_dx_after_fMRI"].mean()) if len(follow) else np.nan,
                    "median_followup_months": float(follow["followup_months"].median()) if len(follow) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def km_curve(ax: plt.Axes, df: pd.DataFrame, label: str, color: str) -> None:
    sub = df.copy()
    sub["time"] = np.where(
        sub["converted_to_any_pathological_dx_after_fMRI"].eq(True),
        sub["months_to_conversion"],
        sub["followup_months"],
    )
    sub = sub[np.isfinite(sub["time"]) & (sub["time"] >= 0)]
    if sub.empty:
        return
    event_times = sorted(sub.loc[sub["converted_to_any_pathological_dx_after_fMRI"].eq(True), "time"].dropna().unique())
    times = [0.0]
    cuminc = [0.0]
    survival = 1.0
    for t in event_times:
        at_risk = int((sub["time"] >= t).sum())
        events = int(((sub["time"] == t) & sub["converted_to_any_pathological_dx_after_fMRI"].eq(True)).sum())
        if at_risk <= 0:
            continue
        survival *= (1 - events / at_risk)
        times.append(float(t))
        cuminc.append(float(1 - survival))
    max_time = float(sub["time"].max())
    if times[-1] < max_time:
        times.append(max_time)
        cuminc.append(cuminc[-1])
    ax.step(times, cuminc, where="post", label=f"{label} (N={len(sub)})", color=color)


def make_km_plot(cn: pd.DataFrame) -> bool:
    philips = cn[cn["philips_fp_tn_group"].isin(["Philips_CN_FP", "Philips_CN_TN"])].copy()
    if philips.empty:
        return False
    fig, ax = plt.subplots(figsize=(7, 5))
    km_curve(ax, philips[philips["philips_fp_tn_group"].eq("Philips_CN_FP")], "Philips CN FP", "#c0392b")
    km_curve(ax, philips[philips["philips_fp_tn_group"].eq("Philips_CN_TN")], "Philips CN TN", "#2980b9")
    ax.set_xlabel("Months after fMRI")
    ax.set_ylabel("Cumulative conversion to MCI/AD")
    ax.set_title("Descriptive Kaplan-Meier-style conversion curve")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "kaplan_meier_conversion_plot.png", dpi=150)
    plt.close(fig)
    return True


def make_final_interpretation(
    cn: pd.DataFrame,
    philips_summary: pd.DataFrame,
    all_summary: pd.DataFrame,
    fisher: pd.DataFrame,
    age_models: pd.DataFrame,
) -> str:
    philips_fp = cn[cn["philips_fp_tn_group"].eq("Philips_CN_FP")]
    philips_tn = cn[cn["philips_fp_tn_group"].eq("Philips_CN_TN")]
    all_fp = cn[cn["all_cn_fp_tn_group"].eq("All_CN_FP")]
    converter_fp = all_fp[all_fp["converted_to_any_pathological_dx_after_fMRI"].eq(True)]
    clean_likely_fp = all_fp[all_fp["clinical_interpretation_group"].eq("likely_prodromal_or_converter")]
    ambiguous_fp = all_fp[all_fp["clinical_interpretation_group"].eq("ambiguous_diagnosis_trajectory")]
    stable_fp = all_fp[all_fp["clinical_interpretation_group"].eq("stable_CN_with_long_followup")]

    key_fisher = fisher[
        fisher["comparison"].eq("Philips CN FP vs Philips CN TN")
        & fisher["outcome"].eq("MCI_or_AD conversion")
    ]
    if not key_fisher.empty:
        k = key_fisher.iloc[0]
        fisher_line = (
            f"Philips FP-vs-TN Fisher exact test for future MCI/AD conversion: "
            f"OR={k['odds_ratio_haldane_anscombe']:.3g}, "
            f"95% CI [{k['odds_ratio_ci95_low']:.3g}, {k['odds_ratio_ci95_high']:.3g}], "
            f"p={k['p_fisher_exact']:.3g}; analysis N={int(k['analysis_subjects_with_followup'])}."
        )
    else:
        fisher_line = "Philips FP-vs-TN Fisher exact test was not feasible."

    feasible_age = age_models[age_models["status"].eq("computed")]
    age_line = (
        "Age-adjusted logistic models were feasible and are reported in "
        "`conversion_age_adjusted_models.csv`."
        if not feasible_age.empty
        else "Age-adjusted logistic models were not feasible or underpowered; feasibility reasons are listed in `conversion_age_adjusted_models.csv`."
    )

    lines = [
        "# Final Interpretation",
        "",
        "This is a read-only, post-hoc clinical interpretation audit. Future longitudinal diagnoses were not used for model training, threshold selection, subject exclusion, or model selection.",
        "",
        "## Primary Philips CN FP Question",
        "",
        f"- Philips CN false positives in the promoted readout: `{len(philips_fp)}`.",
        f"- Philips CN true negatives in the promoted readout: `{len(philips_tn)}`.",
        f"- Philips CN FP with future diagnostic follow-up: `{int(philips_fp['has_followup_after_fmri'].sum())}`.",
        f"- Philips CN TN with future diagnostic follow-up: `{int(philips_tn['has_followup_after_fmri'].sum())}`.",
        f"- Philips CN FP future MCI/AD converters: `{int(philips_fp['converted_to_any_pathological_dx_after_fMRI'].sum())}`.",
        f"- Philips CN TN future MCI/AD converters: `{int(philips_tn['converted_to_any_pathological_dx_after_fMRI'].sum())}`.",
        f"- {fisher_line}",
        f"- {age_line}",
        "",
        "## Clinical Classification of CN False Positives",
        "",
        f"- CN false positives with any future MCI/AD diagnosis after fMRI: `{len(converter_fp)}`.",
        f"- Clean likely prodromal/converter CN false positives after excluding prior/reverting ambiguous trajectories: `{len(clean_likely_fp)}`.",
        f"- Ambiguous CN false-positive diagnosis trajectories: `{len(ambiguous_fp)}`.",
        f"- Stable CN false positives with at least 36 months of follow-up: `{len(stable_fp)}`.",
        f"- Remaining CN false positives had insufficient follow-up or ambiguous trajectories.",
        "",
        "## Guardrail Interpretation",
        "",
        "A future conversion signal, if present, supports clinical plausibility for a subset of false positives, but it does not justify relabeling, excluding subjects, or selecting a new model. Stable Philips CN false positives with long follow-up remain evidence for scanner/site/threshold limitations rather than occult prodromal AD.",
        "",
        "## Output Tables",
        "",
        "- `cn_subject_conversion_table.csv`: all CN subjects at fMRI with conversion status.",
        "- `philips_cn_fp_vs_tn_conversion_summary.csv`: primary Philips comparison.",
        "- `conversion_fisher_tests.csv`: exact tests for primary and secondary comparisons.",
        "- `likely_prodromal_false_positive_subjects.csv`: clean false-positive CN converters.",
        "- `ambiguous_diagnosis_trajectory_false_positive_subjects.csv`: false-positive CN subjects with prior or reverting pathological diagnosis evidence.",
        "- `stable_cn_false_positive_subjects.csv`: false-positive CN with long non-converting follow-up.",
    ]
    return "\n".join(lines) + "\n"


def make_martin_message(cn: pd.DataFrame, fisher: pd.DataFrame) -> str:
    philips_fp = cn[cn["philips_fp_tn_group"].eq("Philips_CN_FP")]
    philips_tn = cn[cn["philips_fp_tn_group"].eq("Philips_CN_TN")]
    conv_fp = int(philips_fp["converted_to_any_pathological_dx_after_fMRI"].sum())
    conv_tn = int(philips_tn["converted_to_any_pathological_dx_after_fMRI"].sum())
    clean_likely_fp = int(philips_fp["clinical_interpretation_group"].eq("likely_prodromal_or_converter").sum())
    ambiguous_fp = int(philips_fp["clinical_interpretation_group"].eq("ambiguous_diagnosis_trajectory").sum())
    follow_fp = int(philips_fp["has_followup_after_fmri"].sum())
    follow_tn = int(philips_tn["has_followup_after_fmri"].sum())
    key = fisher[
        fisher["comparison"].eq("Philips CN FP vs Philips CN TN")
        & fisher["outcome"].eq("MCI_or_AD conversion")
    ]
    if not key.empty and pd.notna(key.iloc[0]["p_fisher_exact"]):
        p_text = f"Fisher p={key.iloc[0]['p_fisher_exact']:.3g}"
    else:
        p_text = "Fisher test not feasible"
    return "\n".join(
        [
            "# Message for Martin",
            "",
            "I ran a read-only longitudinal ADNI diagnosis audit focused on the promoted model's Philips CN false positives.",
            "",
            f"- Philips CN FP: {len(philips_fp)} subjects; {follow_fp} had post-fMRI diagnostic follow-up.",
            f"- Philips CN TN: {len(philips_tn)} subjects; {follow_tn} had post-fMRI diagnostic follow-up.",
            f"- Future MCI/AD conversion: Philips FP {conv_fp}/{follow_fp if follow_fp else 0}, Philips TN {conv_tn}/{follow_tn if follow_tn else 0} ({p_text}).",
            f"- Among Philips CN FP, {clean_likely_fp} is a clean post-fMRI likely converter and {ambiguous_fp} have ambiguous prior/reverting diagnosis evidence.",
            "- This remains a post-hoc interpretation only. No labels were changed, no subjects were excluded, and the future diagnoses were not used for model selection.",
            "- Stable Philips CN false positives with long follow-up should be treated as evidence for residual scanner/site/threshold limits, while converters are clinically plausible false positives.",
        ]
    ) + "\n"


def make_readme(km_created: bool) -> str:
    return "\n".join(
        [
            "# Promoted Model CN False-Positive Longitudinal Conversion Audit",
            "",
            "This package audits whether promoted-model CN false positives later convert to MCI or AD/Dementia, with a primary focus on Philips CN false positives.",
            "",
            "## Promoted readout",
            "",
            "- Model: `recover035_latent384_beta3p75_T80_h10000_p560_full5x5`",
            "- Stage B: `logreg_l2_original`",
            "- Features: `z_plus_age_sex`",
            "- Score harmonization: `oof_logitz`",
            "- Threshold: `inner_oof_target_sens_ge_0p70_max_spec`",
            "",
            "## Longitudinal Sources",
            "",
            "- Primary longitudinal diagnosis source: `ADNIMERGE_14Oct2024.csv` (`DX`).",
            "- Primary diagnostic-summary source: `DXSUM_28May2026.csv` (`DIAGNOSIS`: 1=CN, 2=MCI, 3=Dementia).",
            "- fMRI anchor date source: local AAL3 metadata `StudyDate`, with AD fMRI download `AcqDate` as fallback.",
            "",
            "## Guardrails",
            "",
            "- No model training.",
            "- No threshold fitting.",
            "- No model selection.",
            "- No subject exclusion or relabeling.",
            "- No tensor or metadata modification.",
            "",
            f"Kaplan-Meier-style plot generated: `{km_created}`.",
        ]
    ) + "\n"


def main() -> None:
    t0 = datetime.now(timezone.utc)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sources = source_inventory()
    write_table("longitudinal_diagnosis_sources_found", sources)
    write_table("diagnosis_code_mapping_used", diagnosis_mapping_table())

    pred = load_promoted_predictions()
    anchor_meta = load_anchor_metadata()
    anchor = build_anchor_table(pred, anchor_meta)
    write_table("model_fmri_anchor_table", anchor, max_rows=500)

    subject_ids = set(anchor["SubjectID"].astype(str))
    long = build_longitudinal_table(subject_ids)
    write_table("subject_longitudinal_diagnosis_table", long, max_rows=1000)

    cn_conv = build_cn_conversion_table(anchor, long)
    cn_conv = add_comparison_labels(cn_conv)
    write_table("cn_subject_conversion_table", cn_conv, max_rows=500)

    philips_summary = summarize_groups(
        cn_conv[cn_conv["philips_fp_tn_group"].isin(["Philips_CN_FP", "Philips_CN_TN"])],
        "philips_fp_tn_group",
        ["Philips_CN_FP", "Philips_CN_TN"],
        "Philips CN FP vs Philips CN TN",
    )
    write_table("philips_cn_fp_vs_tn_conversion_summary", philips_summary)

    all_summary = summarize_groups(
        cn_conv[cn_conv["all_cn_fp_tn_group"].isin(["All_CN_FP", "All_CN_TN"])],
        "all_cn_fp_tn_group",
        ["All_CN_FP", "All_CN_TN"],
        "All CN FP vs All CN TN",
    )
    write_table("all_cn_fp_vs_tn_conversion_summary", all_summary)

    fisher = fisher_tests(cn_conv)
    write_table("conversion_fisher_tests", fisher)

    age_models = age_adjusted_models(cn_conv)
    write_table("conversion_age_adjusted_models", age_models)

    gradient = high_score_gradient(cn_conv)
    write_table("high_score_cn_conversion_gradient", gradient)

    strata = secondary_strata(cn_conv)
    write_table("secondary_strata_conversion_summary", strata)

    likely_prodromal = cn_conv[
        cn_conv["model_error_type"].eq("FP")
        & cn_conv["clinical_interpretation_group"].eq("likely_prodromal_or_converter")
    ].sort_values(["Manufacturer", "months_to_conversion", "y_score"])
    write_table("likely_prodromal_false_positive_subjects", likely_prodromal, max_rows=500)

    ambiguous_fp = cn_conv[
        cn_conv["model_error_type"].eq("FP")
        & cn_conv["clinical_interpretation_group"].eq("ambiguous_diagnosis_trajectory")
    ].sort_values(["Manufacturer", "months_to_conversion", "y_score"])
    write_table("ambiguous_diagnosis_trajectory_false_positive_subjects", ambiguous_fp, max_rows=500)

    stable_fp = cn_conv[
        cn_conv["model_error_type"].eq("FP")
        & cn_conv["clinical_interpretation_group"].eq("stable_CN_with_long_followup")
    ].sort_values(["Manufacturer", "followup_months", "y_score"], ascending=[True, False, False])
    write_table("stable_cn_false_positive_subjects", stable_fp, max_rows=500)

    km_created = make_km_plot(cn_conv)

    write_md(OUTPUT_DIR / "final_interpretation.md", make_final_interpretation(cn_conv, philips_summary, all_summary, fisher, age_models))
    write_md(OUTPUT_DIR / "martin_message.md", make_martin_message(cn_conv, fisher))
    write_md(OUTPUT_DIR / "README.md", make_readme(km_created))

    log = {
        "script": fpath(Path(__file__)),
        "created_utc": t0.isoformat(),
        "elapsed_seconds": (datetime.now(timezone.utc) - t0).total_seconds(),
        "output_dir": fpath(OUTPUT_DIR),
        "promoted_readout": {
            "model_name": PROMOTED_MODEL,
            "feature_set": PROMOTED_FEATURE_SET,
            "calib_method": PROMOTED_CALIB,
            "threshold_strategy": PROMOTED_THRESHOLD,
        },
        "n_promoted_prediction_subjects": int(pred["SubjectID"].nunique()),
        "n_cn_subjects": int(len(cn_conv)),
        "n_cn_fp": int(cn_conv["model_error_type"].eq("FP").sum()),
        "n_philips_cn_fp": int(cn_conv["philips_fp_tn_group"].eq("Philips_CN_FP").sum()),
        "guardrails": [
            "no_training",
            "no_threshold_fitting",
            "no_model_selection",
            "no_tensor_modification",
            "no_metadata_modification",
            "post_hoc_clinical_interpretation_only",
        ],
        "sources": sources.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "command_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    print(f"Audit complete: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
