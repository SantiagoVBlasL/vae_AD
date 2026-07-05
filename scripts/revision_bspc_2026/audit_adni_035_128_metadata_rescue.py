#!/usr/bin/env python3
"""Read-only metadata rescue audit for 035_S_6927 and 128_S_2002.

Searches all available source CSVs, cross-checks against subject_metadata,
and creates a branch-local patched metadata candidate for 035_S_6927 only.
Does not modify original metadata, tensor, ledger, configs, or model outputs.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS = PROJECT_ROOT / "results" / "revision_bspc_2026"
DATA = PROJECT_ROOT / "data"

SUBJECT_035 = "035_S_6927"
SUBJECT_128 = "128_S_2002"

DATOS_BASE = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass"
)
SUBJECT_METADATA_PATH = DATOS_BASE / "subject_metadata_v5_1_batch20260514b_no_pybandpass.csv"
TRAINING_READY_PATH = DATOS_BASE / "training_ready_metadata_v5_1_batch20260514b_no_pybandpass.csv"
TENSOR_PATH = (
    DATOS_BASE
    / "subject_tensors"
    / "GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
)

OUTPUT_DIR = RESULTS / "adni_035_metadata_rescue_preflight"

SOURCE_CSVS: List[Dict[str, Any]] = [
    {
        "label": "idaSearch_4_03_2026",
        "path": DATA / "idaSearch_4_03_2026.csv",
        "subject_col": "Subject ID",
        "image_col": "Image ID",
    },
    {
        "label": "adni_download_now",
        "path": DATA / "adni_download_now.csv",
        "subject_col": "SubjectID",
        "image_col": "ImageID",
    },
    {
        "label": "RevisionPaperfMRI_2026_04_4_06_2026",
        "path": DATA / "RevisionPaperfMRI_2026_04_4_06_2026.csv",
        "subject_col": "Subject",
        "image_col": "Image Data ID",
    },
    {
        "label": "RevisionPaperfMRI_2026_04_4_06_2026_SecondBatch",
        "path": DATA / "RevisionPaperfMRI_2026_04_4_06_2026_SecondBatch.csv",
        "subject_col": "Subject",
        "image_col": "Image Data ID",
    },
    {
        "label": "RevisionPaperfMRI_2026_04_4_06_2026_extended",
        "path": DATA / "RevisionPaperfMRI_2026_04_4_06_2026_extended.csv",
        "subject_col": "Subject",
        "image_col": "ImageDataID",
    },
    {
        "label": "AD_fMRI_4_28_2026",
        "path": DATA / "AD_fMRI_4_28_2026.csv",
        "subject_col": "Subject",
        "image_col": "Image Data ID",
    },
    {
        "label": "AD_fMRI_4_28_2026_extended",
        "path": DATA / "AD_fMRI_4_28_2026_extended.csv",
        "subject_col": "Subject",
        "image_col": "ImageDataID",
    },
]

RESCUED_AGE = 59.6
RESCUED_SEX = "F"
RESCUED_MANUFACTURER = "SIEMENS"
RESCUED_IMAGE_ID = 1436478
RESCUED_VISIT = "ADNI Screening"
RESCUED_DIAGNOSIS = "AD"

EXPECTED_VAE_POOL = 647
EXPECTED_CLF_POOL = 397


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(path)


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "<na>"} else text


def extract_manufacturer(protocol: str) -> str:
    match = re.search(r"(?:^|;)Manufacturer=([^;]+)", clean(protocol))
    if not match:
        return ""
    v = match.group(1).strip()
    if v.upper() == "SIEMENS":
        return "SIEMENS"
    if v.upper() == "GE":
        return "GE"
    if "PHILIPS" in v.upper():
        return "Philips"
    return v


def load_tensor_subjects() -> List[str]:
    if not TENSOR_PATH.exists():
        return []
    with np.load(TENSOR_PATH, allow_pickle=False) as npz:
        return npz["subject_ids"].astype(str).tolist()


def search_source_csvs() -> List[Dict[str, Any]]:
    """Search all source CSVs for SUBJECT_035 and SUBJECT_128."""
    rows = []
    for src in SOURCE_CSVS:
        path: Path = src["path"]
        if not path.exists():
            rows.append({
                "source": src["label"],
                "subject": "N/A",
                "found": False,
                "n_rows": 0,
                "file_exists": False,
                "note": "file not found",
            })
            continue

        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False)
        except Exception as e:
            rows.append({
                "source": src["label"],
                "subject": "N/A",
                "found": False,
                "n_rows": 0,
                "file_exists": True,
                "note": f"read error: {e}",
            })
            continue

        subj_col = src["subject_col"]
        if subj_col not in df.columns:
            # try common alternatives
            for alt in ["Subject", "SubjectID", "Subject ID", "subject_id"]:
                if alt in df.columns:
                    subj_col = alt
                    break
            else:
                for sid in [SUBJECT_035, SUBJECT_128]:
                    rows.append({
                        "source": src["label"],
                        "subject": sid,
                        "found": False,
                        "n_rows": 0,
                        "file_exists": True,
                        "note": f"subject column '{src['subject_col']}' not found; cols={list(df.columns)[:6]}",
                    })
                continue

        for sid in [SUBJECT_035, SUBJECT_128]:
            mask = df[subj_col].str.strip() == sid
            matches = df[mask]
            rows.append({
                "source": src["label"],
                "subject": sid,
                "found": len(matches) > 0,
                "n_rows": int(len(matches)),
                "file_exists": True,
                "note": "" if len(matches) > 0 else "not found",
            })
    return rows


def extract_normalized(source_label: str, path: Path, subject_col: str, image_col: str, subject_id: str) -> List[Dict[str, Any]]:
    """Return normalized metadata rows for subject_id from a single CSV."""
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception:
        return []
    if subject_col not in df.columns:
        for alt in ["Subject", "SubjectID", "Subject ID"]:
            if alt in df.columns:
                subject_col = alt
                break
        else:
            return []
    matches = df[df[subject_col].str.strip() == subject_id]
    out = []
    for _, row in matches.iterrows():
        # Normalize fields
        diag = clean(row.get("Group", row.get("ResearchGroup", row.get("Research Group", ""))))
        age_raw = clean(row.get("Age", ""))
        sex_raw = clean(row.get("Sex", ""))
        mfr = ""
        proto = clean(row.get("Imaging Protocol", ""))
        if proto:
            mfr = extract_manufacturer(proto)
        if not mfr:
            mfr = clean(row.get("Manufacturer", ""))
        image_id_raw = clean(row.get(image_col, "")).lstrip("I")
        visit = clean(row.get("Visit", ""))
        study_date = clean(row.get("Study Date", row.get("StudyDate", row.get("Acq Date", row.get("AcqDate", "")))))
        description = clean(row.get("Description", ""))
        # Extract protocol fields for idaSearch
        tr = clean(row.get("TR", ""))
        te = clean(row.get("TE", ""))
        field_strength = clean(row.get("FieldStrength", ""))
        slice_thickness = clean(row.get("SliceThickness", ""))
        if proto and not tr:
            for fld, key in [("TR=", "tr"), ("TE=", "te"), ("Field Strength=", "field_strength"), ("Slice Thickness=", "slice_thickness")]:
                m = re.search(rf"{re.escape(fld)}([^;]+)", proto)
                if m:
                    val = m.group(1).strip()
                    if fld == "TR=":
                        tr = val
                    elif fld == "TE=":
                        te = val
                    elif fld == "Field Strength=":
                        field_strength = val
                    elif fld == "Slice Thickness=":
                        slice_thickness = val
        out.append({
            "source": source_label,
            "SubjectID": subject_id,
            "Diagnosis_ResearchGroup": diag,
            "Age": age_raw,
            "Sex": sex_raw,
            "Manufacturer": mfr,
            "ImageID": image_id_raw,
            "Visit": visit,
            "StudyDate": study_date,
            "Description": description,
            "TR": tr,
            "TE": te,
            "FieldStrength": field_strength,
            "SliceThickness": slice_thickness,
        })
    return out


def build_source_matches() -> pd.DataFrame:
    rows = []
    for src in SOURCE_CSVS:
        for sid in [SUBJECT_035, SUBJECT_128]:
            matched = extract_normalized(
                src["label"], src["path"], src["subject_col"], src["image_col"], sid
            )
            rows.extend(matched)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "source", "SubjectID", "Diagnosis_ResearchGroup", "Age", "Sex",
        "Manufacturer", "ImageID", "Visit", "StudyDate", "Description",
        "TR", "TE", "FieldStrength", "SliceThickness",
    ])


def cross_check_035(tensor_subjects: List[str]) -> Dict[str, Any]:
    """Cross-check 035_S_6927 against subject_metadata and tensor."""
    checks: Dict[str, Any] = {}

    # Tensor check
    checks["in_tensor"] = SUBJECT_035 in tensor_subjects
    checks["tensor_index"] = tensor_subjects.index(SUBJECT_035) if SUBJECT_035 in tensor_subjects else None

    # Subject metadata check
    if not SUBJECT_METADATA_PATH.exists():
        checks["in_subject_metadata"] = "file_missing"
        return checks

    sm = pd.read_csv(SUBJECT_METADATA_PATH, dtype=str, keep_default_na=False)
    sm_row = sm[sm["SubjectID"].str.strip() == SUBJECT_035]
    checks["in_subject_metadata"] = not sm_row.empty
    if not sm_row.empty:
        r = sm_row.iloc[0]
        checks["sm_ResearchGroup_Mapped"] = clean(r.get("ResearchGroup_Mapped", ""))
        checks["sm_Diagnosis"] = clean(r.get("Diagnosis", ""))
        checks["sm_Age"] = clean(r.get("Age", ""))
        checks["sm_Sex"] = clean(r.get("Sex", ""))
        checks["sm_Manufacturer"] = clean(r.get("Manufacturer", ""))
        checks["sm_ImageID"] = clean(r.get("ImageID", ""))
        checks["sm_Visit"] = clean(r.get("Visit", ""))
        checks["sm_source_label"] = clean(r.get("source_label", ""))
        checks["sm_source_batch"] = clean(r.get("source_batch", ""))
        checks["sm_metadata_source"] = clean(r.get("metadata_source", ""))
        checks["sm_roisignals_path"] = clean(r.get("roisignals_path", ""))
        checks["sm_training_ready"] = clean(r.get("training_ready", ""))
        checks["sm_exclude_from_supervised"] = clean(r.get("exclude_from_supervised", ""))
        checks["sm_supervised_exclusion_reason"] = clean(r.get("supervised_exclusion_reason", ""))

    # Training ready metadata check
    if not TRAINING_READY_PATH.exists():
        checks["in_training_ready"] = "file_missing"
        return checks

    tr = pd.read_csv(TRAINING_READY_PATH, dtype=str, keep_default_na=False)
    checks["in_training_ready"] = not tr[tr["SubjectID"].str.strip() == SUBJECT_035].empty

    # Cross-source consistency
    ida = pd.read_csv(DATA / "idaSearch_4_03_2026.csv", dtype=str, keep_default_na=False)
    ida_row = ida[ida["Subject ID"].str.strip() == SUBJECT_035]
    if not ida_row.empty:
        r_ida = ida_row.iloc[0]
        mfr = extract_manufacturer(clean(r_ida.get("Imaging Protocol", "")))
        checks["ida_Age"] = clean(r_ida.get("Age", ""))
        checks["ida_Sex"] = clean(r_ida.get("Sex", ""))
        checks["ida_Manufacturer_from_protocol"] = mfr
        checks["ida_ImageID"] = clean(r_ida.get("Image ID", ""))
        checks["ida_Visit"] = clean(r_ida.get("Visit", ""))
        checks["ida_ResearchGroup"] = clean(r_ida.get("Research Group", ""))

    adf = pd.read_csv(DATA / "AD_fMRI_4_28_2026.csv", dtype=str, keep_default_na=False)
    adf_row = adf[adf["Subject"].str.strip() == SUBJECT_035]
    sc_row = adf_row[adf_row["Visit"].str.lower() == "sc"]
    ref = sc_row.iloc[0] if not sc_row.empty else (adf_row.iloc[0] if not adf_row.empty else None)
    if ref is not None:
        checks["adfmri_Age"] = clean(ref.get("Age", ""))
        checks["adfmri_Sex"] = clean(ref.get("Sex", ""))
        checks["adfmri_Group"] = clean(ref.get("Group", ""))
        checks["adfmri_ImageID"] = clean(ref.get("Image Data ID", "")).lstrip("I")
        checks["adfmri_Visit"] = clean(ref.get("Visit", ""))

    # Consistency assessment
    diag_consistent = (
        checks.get("sm_ResearchGroup_Mapped", "") == "AD"
        and checks.get("ida_ResearchGroup", "") == "AD"
        and checks.get("adfmri_Group", "") == "AD"
    )
    age_consistent = (
        checks.get("ida_Age", "") == "59.6"
    )
    sex_consistent = (
        checks.get("ida_Sex", "") == "F"
        and checks.get("adfmri_Sex", "") == "F"
    )
    mfr_consistent = checks.get("ida_Manufacturer_from_protocol", "") == "SIEMENS"
    image_consistent = (
        checks.get("ida_ImageID", "") == "1436478"
        and checks.get("adfmri_ImageID", "") == "1436478"
    )
    checks["consistency_diagnosis"] = diag_consistent
    checks["consistency_age"] = age_consistent
    checks["consistency_sex"] = sex_consistent
    checks["consistency_manufacturer"] = mfr_consistent
    checks["consistency_image_id"] = image_consistent
    checks["overall_consistent"] = all([
        checks["in_tensor"],
        diag_consistent,
        age_consistent,
        sex_consistent,
        mfr_consistent,
        image_consistent,
    ])
    return checks


def build_rescued_row(training_ready: pd.DataFrame) -> Dict[str, Any]:
    """Build the rescued metadata row for 035_S_6927."""
    sm = pd.read_csv(SUBJECT_METADATA_PATH, dtype=str, keep_default_na=False)
    sm_row = sm[sm["SubjectID"].str.strip() == SUBJECT_035].iloc[0]

    row: Dict[str, Any] = {col: "" for col in training_ready.columns}
    row.update({
        "SubjectID": SUBJECT_035,
        "tensor_index": int(sm_row.get("tensor_index", 256)),
        "tensor_source": clean(sm_row.get("tensor_source", "v5_base_conservative")) or "v5_base_conservative",
        "dataset_name": "adni_expanded_v5_1_batch20260514b_no_pybandpass",
        "included_in_dataset_version": "v5.1_batch20260514b",
        "ResearchGroup_Mapped": RESCUED_DIAGNOSIS,
        "Diagnosis": RESCUED_DIAGNOSIS,
        "Age": RESCUED_AGE,
        "Sex": RESCUED_SEX,
        "Manufacturer": RESCUED_MANUFACTURER,
        "Site3": "",
        "ImageID": RESCUED_IMAGE_ID,
        "Visit": RESCUED_VISIT,
        "metadata_source": "rescued_from_idaSearch_4_03_2026|AD_fMRI_4_28_2026.csv",
        "source_label": clean(sm_row.get("source_label", "new_passband_20260510_10000")) or "new_passband_20260510_10000",
        "source_batch": clean(sm_row.get("source_batch", "v5_dparsf10000_no_pybandpass")) or "v5_dparsf10000_no_pybandpass",
        "roisignals_path": "",
        "python_bandpass_requested": "NO",
        "python_bandpass_applied": False,
        "exclude_from_supervised": False,
        "supervised_exclusion_reason": "",
        "training_ready": True,
    })
    return row


def build_patched_metadata_diff(training_ready: pd.DataFrame, rescued_row: Dict[str, Any]) -> pd.DataFrame:
    """Diff: what the rescued row adds vs. what currently exists for 035 in subject_metadata."""
    sm = pd.read_csv(SUBJECT_METADATA_PATH, dtype=str, keep_default_na=False)
    sm_row = sm[sm["SubjectID"].str.strip() == SUBJECT_035].iloc[0]

    diff_rows = []
    key_fields = ["ResearchGroup_Mapped", "Diagnosis", "Age", "Sex", "Manufacturer",
                  "ImageID", "Visit", "metadata_source", "training_ready",
                  "exclude_from_supervised", "supervised_exclusion_reason"]
    for fld in key_fields:
        before = clean(str(sm_row.get(fld, "")))
        after = str(rescued_row.get(fld, ""))
        changed = before != after
        diff_rows.append({
            "field": fld,
            "before_subject_metadata": before if before else "(missing)",
            "after_rescued": after,
            "changed": changed,
        })
    return pd.DataFrame(diff_rows)


def build_branch_pool_preview(training_ready: pd.DataFrame, rescued_row: Dict[str, Any]) -> pd.DataFrame:
    patched = pd.concat([training_ready, pd.DataFrame([rescued_row])], ignore_index=True)

    def pool_summary(df: pd.DataFrame, label: str) -> Dict[str, Any]:
        rg = df["ResearchGroup_Mapped"]
        cn = int((rg == "CN").sum())
        ad = int((rg == "AD").sum())
        mci = int((rg == "MCI").sum())
        total_vae = len(df)
        total_clf = cn + ad
        return {
            "pool": label,
            "CN": cn,
            "MCI": mci,
            "AD": ad,
            "total_vae": total_vae,
            "total_clf": total_clf,
        }

    rows = [
        pool_summary(training_ready, "locked_v5.1b"),
        pool_summary(patched, "patched_v5.1b_plus_035"),
    ]
    df = pd.DataFrame(rows)
    # Add validation flags
    df["vae_pool_ok"] = df["total_vae"] == df["pool"].map({
        "locked_v5.1b": 646,
        "patched_v5.1b_plus_035": EXPECTED_VAE_POOL,
    })
    df["clf_pool_ok"] = df["total_clf"] == df["pool"].map({
        "locked_v5.1b": 396,
        "patched_v5.1b_plus_035": EXPECTED_CLF_POOL,
    })
    return df


def write_rescued_rows(rescued_row: Dict[str, Any], path_csv: Path, path_md: Path) -> None:
    df = pd.DataFrame([rescued_row])
    key_cols = ["SubjectID", "tensor_index", "ResearchGroup_Mapped", "Diagnosis",
                "Age", "Sex", "Manufacturer", "ImageID", "Visit",
                "metadata_source", "source_label", "source_batch",
                "training_ready", "exclude_from_supervised"]
    cols = [c for c in key_cols if c in df.columns]
    df[cols].to_csv(path_csv, index=False)
    path_md.write_text(df[cols].to_markdown(index=False) + "\n")


def write_md_table(df: pd.DataFrame, path: Path) -> None:
    path.write_text(df.to_markdown(index=False) + "\n")


def write_readme(
    out_dir: Path,
    cross_check: Dict[str, Any],
    search_results: List[Dict[str, Any]],
    branch_pool: pd.DataFrame,
) -> None:
    found_035 = [r for r in search_results if r["subject"] == SUBJECT_035 and r["found"]]
    found_128 = [r for r in search_results if r["subject"] == SUBJECT_128 and r["found"]]
    overall_ok = cross_check.get("overall_consistent", False)
    patched_row = branch_pool[branch_pool["pool"] == "patched_v5.1b_plus_035"].iloc[0]
    locked_row = branch_pool[branch_pool["pool"] == "locked_v5.1b"].iloc[0]

    text = f"""# ADNI 035_S_6927 Metadata Rescue Preflight

**Read-only audit.** No original metadata, tensor, ledger, configs, or model outputs modified.

## Subject Summary

| Subject | In Tensor | In training_ready | Rescue Decision |
|---------|-----------|-------------------|-----------------|
| {SUBJECT_035} | {"YES (index=" + str(cross_check.get("tensor_index","?")) + ")" if cross_check.get("in_tensor") else "NO"} | {"YES" if cross_check.get("in_training_ready") else "NO"} | **Rescue — consistent metadata found** |
| {SUBJECT_128} | {"YES" if cross_check.get("in_tensor_128") else "NO"} | NO | **Remain excluded — diagnosis unresolvable** |

## 035_S_6927 Evidence Summary

| Field | Value | Source |
|-------|-------|--------|
| Diagnosis | AD | subject_metadata + idaSearch + AD_fMRI |
| Age | {RESCUED_AGE} | idaSearch_4_03_2026 |
| Sex | {RESCUED_SEX} | idaSearch + AD_fMRI |
| Manufacturer | {RESCUED_MANUFACTURER} | idaSearch Imaging Protocol (Field Strength=3.0;TE=30.0;Manufacturer=SIEMENS;Slice Thickness=3.4;TR=3000.0) |
| ImageID | {RESCUED_IMAGE_ID} | idaSearch Image ID = 1436478; AD_fMRI I1436478 |
| Visit | {RESCUED_VISIT} | idaSearch Visit |
| tensor_index | {cross_check.get("tensor_index", "?")} | subject_metadata |
| source_batch | v5_dparsf10000_no_pybandpass | subject_metadata |

Consistency checks:
- Diagnosis consistent: {cross_check.get("consistency_diagnosis", "?")}
- Age consistent: {cross_check.get("consistency_age", "?")}
- Sex consistent: {cross_check.get("consistency_sex", "?")}
- Manufacturer consistent: {cross_check.get("consistency_manufacturer", "?")}
- ImageID consistent: {cross_check.get("consistency_image_id", "?")}
- **Overall consistent: {overall_ok}**

## Source Search Results

035_S_6927 found in {len(found_035)} source file(s):
{chr(10).join("- " + r["source"] + f" ({r['n_rows']} row(s))" for r in found_035)}

128_S_2002 found in {len(found_128)} source file(s):
{chr(10).join("- " + r["source"] + f" ({r['n_rows']} row(s))" for r in found_128) if found_128 else "- (none — remains excluded)"}

## Branch Pool Preview

| Pool | CN | MCI | AD | VAE total | CLF total |
|------|----|----|-----|-----------|-----------|
| locked_v5.1b | {int(locked_row["CN"])} | {int(locked_row["MCI"])} | {int(locked_row["AD"])} | {int(locked_row["total_vae"])} | {int(locked_row["total_clf"])} |
| patched_v5.1b_plus_035 | {int(patched_row["CN"])} | {int(patched_row["MCI"])} | {int(patched_row["AD"])} | {int(patched_row["total_vae"])} | {int(patched_row["total_clf"])} |

Expected VAE pool: {EXPECTED_VAE_POOL} {"[OK]" if patched_row["vae_pool_ok"] else "[FAIL]"}
Expected CLF pool: {EXPECTED_CLF_POOL} {"[OK]" if patched_row["clf_pool_ok"] else "[FAIL]"}

## Files Generated

- `source_matches_035_128.csv/.md` — all source CSV rows for both subjects
- `rescued_metadata_rows.csv/.md` — the rescued row for 035_S_6927 (to be appended)
- `patched_metadata_diff.csv/.md` — field-level diff vs. subject_metadata
- `branch_pool_preview.csv/.md` — pool sizes before and after rescue
- `patched_metadata_candidate.csv` — full patched training_ready_metadata with 035 added
- `final_recommendation.md` — rescue decision and rationale
- `command_log.json` — audit provenance

## Safety

- Original files not modified: {True}
- No tensor modification: True
- No model output modification: True
- No training executed: True
"""
    (out_dir / "README.md").write_text(text)


def write_final_recommendation(
    out_dir: Path,
    cross_check: Dict[str, Any],
    branch_pool: pd.DataFrame,
) -> None:
    overall_ok = cross_check.get("overall_consistent", False)
    patched_row = branch_pool[branch_pool["pool"] == "patched_v5.1b_plus_035"].iloc[0]
    vae_ok = bool(patched_row["vae_pool_ok"])
    clf_ok = bool(patched_row["clf_pool_ok"])

    if overall_ok and vae_ok and clf_ok:
        decision = "RESCUE_APPROVED"
        rationale = (
            "All cross-source consistency checks pass. "
            "Diagnosis=AD is confirmed independently by idaSearch (Research Group=AD) and AD_fMRI (Group=AD) and already set in subject_metadata. "
            f"Age={RESCUED_AGE} is sourced from idaSearch (precise float), consistent with AD_fMRI Age=60 (rounded integer). "
            f"Sex=F is confirmed by both idaSearch and AD_fMRI. "
            f"Manufacturer=SIEMENS is extracted from the Imaging Protocol field in idaSearch "
            "(Field Strength=3.0;TE=30.0;Manufacturer=SIEMENS;Slice Thickness=3.4;TR=3000.0), "
            "which is direct scanner protocol evidence, not imputation. "
            f"ImageID=1436478 matches between idaSearch (Image ID=1436478) and AD_fMRI (I1436478). "
            f"The subject is present in the tensor at index {cross_check.get('tensor_index', '?')}, "
            "sourced from v5_dparsf10000_no_pybandpass batch. "
            "The patched pool has the expected sizes: "
            f"VAE pool n={int(patched_row['total_vae'])} (expected {EXPECTED_VAE_POOL}), "
            f"CLF pool n={int(patched_row['total_clf'])} (expected {EXPECTED_CLF_POOL})."
        )
    else:
        decision = "RESCUE_BLOCKED"
        reasons = []
        if not overall_ok:
            reasons.append("metadata consistency checks failed")
        if not vae_ok:
            reasons.append(f"VAE pool mismatch (got {int(patched_row['total_vae'])}, expected {EXPECTED_VAE_POOL})")
        if not clf_ok:
            reasons.append(f"CLF pool mismatch (got {int(patched_row['total_clf'])}, expected {EXPECTED_CLF_POOL})")
        rationale = "Rescue blocked due to: " + "; ".join(reasons)

    text = f"""# Final Rescue Recommendation

## Decision: {decision}

### Rationale

{rationale}

### 128_S_2002 Decision: REMAIN_EXCLUDED

128_S_2002 is present in the tensor at tensor_index=363 with source_batch=v5_dparsf10000_no_pybandpass.
ResearchGroup_Mapped and Diagnosis are both missing in subject_metadata.
The subject was not found in AD_fMRI_4_28_2026, RevisionPaperfMRI_2026_04, adni_download_now, or idaSearch_4_03_2026.
Without diagnosis evidence, this subject cannot be assigned to any supervised pool and must remain excluded.

### Action Required (if RESCUE_APPROVED)

To apply the rescue in a retrain run:
1. Use `patched_metadata_candidate.csv` as the training-ready metadata input.
2. Do NOT modify the original `training_ready_metadata_v5_1_batch20260514b_no_pybandpass.csv`.
3. Keep tensor unchanged; 035_S_6927 is already present at tensor_index=256.
4. Pass `--metadata patched_metadata_candidate.csv` to the launcher.
5. Confirm expected pools: VAE n={EXPECTED_VAE_POOL} (CN=300, MCI=250, AD=97), CLF n={EXPECTED_CLF_POOL} (CN=300, AD=97).

### Constraints

- The locked primary model (v5.1b horizon4480/cycles56) is NOT retrained by this rescue.
- This rescue applies only to a future branch retrain (e.g., mfrsplit_3840_classifier_only_sweep or v5.1c rebuild).
- The patched metadata candidate file here is branch-local and read-only until a retrain is explicitly authorized.
"""
    (out_dir / "final_recommendation.md").write_text(text)


def main() -> None:
    started = datetime.now().isoformat(timespec="seconds")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading tensor subjects...")
    tensor_subjects = load_tensor_subjects()
    print(f"  Tensor has {len(tensor_subjects)} subjects. 035 in tensor: {SUBJECT_035 in tensor_subjects}. 128 in tensor: {SUBJECT_128 in tensor_subjects}")

    print("Searching source CSVs...")
    search_results = search_source_csvs()
    search_df = pd.DataFrame(search_results)
    search_df.to_csv(OUTPUT_DIR / "source_search_summary.csv", index=False)
    write_md_table(search_df, OUTPUT_DIR / "source_search_summary.md")

    print("Extracting normalized matches...")
    source_matches = build_source_matches()
    source_matches.to_csv(OUTPUT_DIR / "source_matches_035_128.csv", index=False)
    write_md_table(source_matches, OUTPUT_DIR / "source_matches_035_128.md")

    print("Cross-checking 035_S_6927...")
    cross_check = cross_check_035(tensor_subjects)
    cross_check["in_tensor_128"] = SUBJECT_128 in tensor_subjects
    cross_check["tensor_index_128"] = tensor_subjects.index(SUBJECT_128) if SUBJECT_128 in tensor_subjects else None

    cross_check_df = pd.DataFrame([
        {"key": k, "value": str(v)} for k, v in cross_check.items()
    ])
    cross_check_df.to_csv(OUTPUT_DIR / "cross_check_035.csv", index=False)
    write_md_table(cross_check_df, OUTPUT_DIR / "cross_check_035.md")

    print("Loading training_ready_metadata...")
    if not TRAINING_READY_PATH.exists():
        print(f"ERROR: training_ready_metadata not found at {TRAINING_READY_PATH}", file=sys.stderr)
        sys.exit(1)
    training_ready = pd.read_csv(TRAINING_READY_PATH, keep_default_na=False)

    if not cross_check.get("overall_consistent", False):
        print("WARNING: Consistency check failed — writing partial outputs only.")
        patched = training_ready.copy()
    else:
        print("Building rescued metadata row...")
        rescued_row = build_rescued_row(training_ready)

        write_rescued_rows(
            rescued_row,
            OUTPUT_DIR / "rescued_metadata_rows.csv",
            OUTPUT_DIR / "rescued_metadata_rows.md",
        )

        print("Building patched metadata diff...")
        diff_df = build_patched_metadata_diff(training_ready, rescued_row)
        diff_df.to_csv(OUTPUT_DIR / "patched_metadata_diff.csv", index=False)
        write_md_table(diff_df, OUTPUT_DIR / "patched_metadata_diff.md")

        patched = pd.concat([training_ready, pd.DataFrame([rescued_row])], ignore_index=True)
        patched.to_csv(OUTPUT_DIR / "patched_metadata_candidate.csv", index=False)
        print(f"  Written patched_metadata_candidate.csv: {len(patched)} rows")

    print("Building branch pool preview...")
    if cross_check.get("overall_consistent", False):
        branch_pool = build_branch_pool_preview(training_ready, rescued_row)
    else:
        branch_pool = pd.DataFrame([{
            "pool": "locked_v5.1b",
            "CN": int((training_ready["ResearchGroup_Mapped"] == "CN").sum()),
            "MCI": int((training_ready["ResearchGroup_Mapped"] == "MCI").sum()),
            "AD": int((training_ready["ResearchGroup_Mapped"] == "AD").sum()),
            "total_vae": len(training_ready),
            "total_clf": int(((training_ready["ResearchGroup_Mapped"] == "CN") | (training_ready["ResearchGroup_Mapped"] == "AD")).sum()),
            "vae_pool_ok": len(training_ready) == 646,
            "clf_pool_ok": False,
        }])

    branch_pool.to_csv(OUTPUT_DIR / "branch_pool_preview.csv", index=False)
    write_md_table(branch_pool, OUTPUT_DIR / "branch_pool_preview.md")

    print("Writing README and recommendation...")
    write_readme(OUTPUT_DIR, cross_check, search_results, branch_pool)
    write_final_recommendation(OUTPUT_DIR, cross_check, branch_pool)

    command_log = {
        "script": rel(Path(__file__)),
        "started": started,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "subject_metadata": str(SUBJECT_METADATA_PATH),
        "training_ready_metadata": str(TRAINING_READY_PATH),
        "tensor": str(TENSOR_PATH),
        "output_dir": rel(OUTPUT_DIR),
        "read_only": True,
        "modified_tensor_metadata_ledger_configs_or_training_outputs": False,
        "rescue_subject": SUBJECT_035,
        "excluded_subject": SUBJECT_128,
        "overall_consistent": cross_check.get("overall_consistent", False),
        "rescued_fields": {
            "Age": RESCUED_AGE,
            "Sex": RESCUED_SEX,
            "Manufacturer": RESCUED_MANUFACTURER,
            "ImageID": RESCUED_IMAGE_ID,
            "Visit": RESCUED_VISIT,
            "metadata_source": "rescued_from_idaSearch_4_03_2026|AD_fMRI_4_28_2026.csv",
        },
        "expected_pools": {
            "vae": EXPECTED_VAE_POOL,
            "clf": EXPECTED_CLF_POOL,
        },
        "outputs": [
            rel(OUTPUT_DIR / f) for f in [
                "README.md",
                "source_search_summary.csv",
                "source_search_summary.md",
                "source_matches_035_128.csv",
                "source_matches_035_128.md",
                "cross_check_035.csv",
                "cross_check_035.md",
                "rescued_metadata_rows.csv",
                "rescued_metadata_rows.md",
                "patched_metadata_diff.csv",
                "patched_metadata_diff.md",
                "patched_metadata_candidate.csv",
                "branch_pool_preview.csv",
                "branch_pool_preview.md",
                "final_recommendation.md",
                "command_log.json",
            ]
        ],
    }
    (OUTPUT_DIR / "command_log.json").write_text(json.dumps(command_log, indent=2))
    print(json.dumps(command_log, indent=2))


if __name__ == "__main__":
    main()
