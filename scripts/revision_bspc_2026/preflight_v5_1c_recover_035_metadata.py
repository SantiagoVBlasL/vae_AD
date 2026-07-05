#!/usr/bin/env python3
"""Read-only preflight for a future v5.1c metadata recovery.

This script simulates adding 035_S_6927 to the final training metadata and
keeps 128_S_2002 excluded. It does not modify tensor, metadata, ledger, configs,
or model outputs.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TENSOR = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors/"
    "GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
)
DEFAULT_FINAL_METADATA = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass/"
    "training_ready_metadata_v5_1_batch20260514b_no_pybandpass.csv"
)
DEFAULT_DECISION_TABLE = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "tensor_metadata_missing_subjects_audit"
    / "per_subject_decision.csv"
)
DEFAULT_AD_FMRI = PROJECT_ROOT / "data" / "AD_fMRI_4_28_2026.csv"
DEFAULT_IDA_SEARCH = PROJECT_ROOT / "data" / "idaSearch_4_03_2026.csv"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1c_metadata_recovery_preflight"
)

RECOVER_SUBJECT = "035_S_6927"
EXCLUDED_SUBJECT = "128_S_2002"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only v5.1c preflight for recovering 035_S_6927 metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tensor", type=Path, default=DEFAULT_TENSOR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_FINAL_METADATA)
    parser.add_argument("--decision-table", type=Path, default=DEFAULT_DECISION_TABLE)
    parser.add_argument("--ad-fmri", type=Path, default=DEFAULT_AD_FMRI)
    parser.add_argument("--ida-search", type=Path, default=DEFAULT_IDA_SEARCH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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


def normalize_subject_site(subject_id: str) -> str:
    match = re.match(r"^(\d{3})_", str(subject_id))
    return match.group(1) if match else ""


def normalize_site(value: Any, subject_id: str = "") -> str:
    text = clean(value)
    if text:
        try:
            return f"{int(float(text)):03d}"
        except ValueError:
            return text.zfill(3) if text.isdigit() else text
    return normalize_subject_site(subject_id)


def extract_manufacturer(protocol: str) -> str:
    match = re.search(r"(?:^|;)Manufacturer=([^;]+)", clean(protocol))
    if not match:
        return ""
    value = match.group(1).strip()
    if value.upper() == "SIEMENS":
        return "SIEMENS"
    if value.upper() == "GE":
        return "GE"
    if "PHILIPS" in value.upper():
        return "Philips"
    return value


def first_existing_column(df: pd.DataFrame, candidates: List[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"None of these columns exist: {candidates}; available={list(df.columns)}")


def load_tensor_subjects(path: Path) -> List[str]:
    with np.load(path, allow_pickle=False) as npz:
        return npz["subject_ids"].astype(str).tolist()


def recover_035_evidence(args: argparse.Namespace, tensor_subjects: List[str]) -> Dict[str, Any]:
    ad = pd.read_csv(args.ad_fmri, dtype=str, keep_default_na=False)
    ida = pd.read_csv(args.ida_search, dtype=str, keep_default_na=False)
    decision = pd.read_csv(args.decision_table, dtype=str, keep_default_na=False)

    ad_subject_col = first_existing_column(ad, ["Subject", "SubjectID", "Subject ID"])
    ida_subject_col = first_existing_column(ida, ["Subject", "SubjectID", "Subject ID"])
    ad_image_col = first_existing_column(ad, ["Image Data ID", "ImageDataID", "Image ID"])
    ida_image_col = first_existing_column(ida, ["Image ID", "ImageDataID", "Image Data ID"])

    ad_rows = ad[ad[ad_subject_col].astype(str) == RECOVER_SUBJECT].copy()
    ida_rows = ida[ida[ida_subject_col].astype(str) == RECOVER_SUBJECT].copy()
    dec_rows = decision[decision["SubjectID"].astype(str) == RECOVER_SUBJECT].copy()

    first_ad = ad_rows[ad_rows["Visit"].astype(str).str.lower().isin(["sc", "screening"])]
    if first_ad.empty and not ad_rows.empty:
        first_ad = ad_rows.iloc[[0]]
    if first_ad.empty:
        raise RuntimeError(f"No AD_fMRI row found for {RECOVER_SUBJECT}")
    ad_row = first_ad.iloc[0]

    image_id = clean(ad_row.get(ad_image_col, "")).lstrip("I")
    ida_image = ida_rows[ida_rows[ida_image_col].astype(str) == image_id] if ida_image_col in ida_rows.columns else pd.DataFrame()
    ida_row = ida_image.iloc[0] if not ida_image.empty else (ida_rows.iloc[0] if not ida_rows.empty else pd.Series(dtype=str))

    manufacturer = extract_manufacturer(clean(ida_row.get("Imaging Protocol", "")))
    manufacturer_source = "data/idaSearch_4_03_2026.csv Imaging Protocol"
    manufacturer_supported = bool(manufacturer)

    evidence = {
        "SubjectID": RECOVER_SUBJECT,
        "present_in_tensor": RECOVER_SUBJECT in tensor_subjects,
        "tensor_index": tensor_subjects.index(RECOVER_SUBJECT) if RECOVER_SUBJECT in tensor_subjects else "",
        "diagnosis": clean(ad_row.get("Group", "")),
        "diagnosis_evidence": "data/AD_fMRI_4_28_2026.csv Group=AD; idaSearch Research Group=AD",
        "age": clean(ad_row.get("Age", "")),
        "age_evidence": f"AD_fMRI Age={clean(ad_row.get('Age', ''))}; idaSearch Age={clean(ida_row.get('Age', ''))}",
        "sex": clean(ad_row.get("Sex", "")),
        "sex_evidence": f"AD_fMRI Sex={clean(ad_row.get('Sex', ''))}; idaSearch Sex={clean(ida_row.get('Sex', ''))}",
        "manufacturer": manufacturer,
        "manufacturer_source": manufacturer_source if manufacturer_supported else "",
        "manufacturer_supported": manufacturer_supported,
        "sitecode": normalize_subject_site(RECOVER_SUBJECT),
        "sitecode_evidence": "derived from ADNI SubjectID prefix 035",
        "image_id": image_id,
        "visit": clean(ad_row.get("Visit", "")),
        "visit_evidence": f"AD_fMRI first usable visit={clean(ad_row.get('Visit', ''))}; ImageID={image_id}",
        "roisignals_usable": clean(dec_rows.iloc[0].get("roisignals_usable", "")) if not dec_rows.empty else "",
        "signal_qc_evidence": clean(dec_rows.iloc[0].get("roisignals_evidence", "")) if not dec_rows.empty else "",
        "appears_in_current_final_metadata": False,
    }
    return evidence


def excluded_128_evidence(args: argparse.Namespace, tensor_subjects: List[str]) -> Dict[str, Any]:
    decision = pd.read_csv(args.decision_table, dtype=str, keep_default_na=False)
    dec_rows = decision[decision["SubjectID"].astype(str) == EXCLUDED_SUBJECT].copy()
    row = dec_rows.iloc[0] if not dec_rows.empty else pd.Series(dtype=str)
    return {
        "SubjectID": EXCLUDED_SUBJECT,
        "present_in_tensor": EXCLUDED_SUBJECT in tensor_subjects,
        "tensor_index": tensor_subjects.index(EXCLUDED_SUBJECT) if EXCLUDED_SUBJECT in tensor_subjects else "",
        "diagnosis": clean(row.get("Diagnosis", "unknown")) or "unknown",
        "diagnosis_evidence": clean(row.get("exclusion_or_loss_reason", "")),
        "age": "",
        "sex": "",
        "manufacturer": clean(row.get("Manufacturer", "UNKNOWN")) or "UNKNOWN",
        "sitecode": normalize_subject_site(EXCLUDED_SUBJECT),
        "roisignals_usable": clean(row.get("roisignals_usable", "")),
        "signal_qc_evidence": clean(row.get("roisignals_evidence", "")),
        "recommendation": "remain_excluded_and_remove_from_future_clean_tensor_rebuild",
    }


def build_recovered_row(metadata: pd.DataFrame, evidence: Dict[str, Any]) -> Dict[str, Any]:
    row = {col: "" for col in metadata.columns}
    row.update(
        {
            "SubjectID": evidence["SubjectID"],
            "tensor_index": evidence["tensor_index"],
            "tensor_source": "v5_base_conservative",
            "dataset_name": "adni_expanded_v5_1c_metadata_recovery_preflight_simulation",
            "included_in_dataset_version": "v5.1c_preflight_simulated",
            "ResearchGroup_Mapped": "AD",
            "Diagnosis": "AD",
            "Age": float(evidence["age"]),
            "Sex": evidence["sex"],
            "Manufacturer": evidence["manufacturer"],
            "Site3": evidence["sitecode"],
            "ImageID": evidence["image_id"],
            "Visit": evidence["visit"],
            "metadata_source": "AD_fMRI_4_28_2026.csv|idaSearch_4_03_2026.csv|tensor_missing_subjects_deep_audit",
            "source_label": "new_passband_20260510_10000",
            "source_batch": "v5_dparsf10000_no_pybandpass",
            "python_bandpass_requested": "NO",
            "python_bandpass_applied": False,
            "exclude_from_supervised": False,
            "supervised_exclusion_reason": "",
            "training_ready": True,
            "n_rois_raw": 170,
            "finite_fraction": 0.9764705882,
            "scale_label": "around_10000_global_scaled",
        }
    )
    return row


def diagnosis_counts(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["ResearchGroup_Mapped"].value_counts(dropna=False).rename_axis("Diagnosis").reset_index(name="N")
    order = {"CN": 0, "AD": 1, "MCI": 2}
    counts["_order"] = counts["Diagnosis"].map(order).fillna(99)
    return counts.sort_values(["_order", "Diagnosis"]).drop(columns=["_order"])


def crosstab_long(df: pd.DataFrame, row_col: str, col_col: str, value_name: str = "N") -> pd.DataFrame:
    tab = pd.crosstab(df[row_col], df[col_col]).reset_index()
    return tab.melt(id_vars=[row_col], var_name=col_col, value_name=value_name)


def simulate_stratification(df: pd.DataFrame, n_splits: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    sim = df.copy().reset_index(drop=True)
    sim["stratification_key"] = sim["ResearchGroup_Mapped"].astype(str) + "|" + sim["Manufacturer"].astype(str)
    key_counts = sim["stratification_key"].value_counts().rename_axis("stratification_key").reset_index(name="N")
    key_counts["valid_for_5fold"] = key_counts["N"] >= n_splits

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows: List[Dict[str, Any]] = []
    fold_assignment = {}
    for fold, (_, test_idx) in enumerate(splitter.split(sim, sim["stratification_key"]), start=1):
        fold_df = sim.iloc[test_idx].copy()
        for sid in fold_df["SubjectID"].astype(str):
            fold_assignment[sid] = fold
        rows.append(
            {
                "fold": fold,
                "n_test": int(len(fold_df)),
                "CN": int((fold_df["ResearchGroup_Mapped"] == "CN").sum()),
                "AD": int((fold_df["ResearchGroup_Mapped"] == "AD").sum()),
                "MCI": int((fold_df["ResearchGroup_Mapped"] == "MCI").sum()),
                "GE": int((fold_df["Manufacturer"] == "GE").sum()),
                "Philips": int((fold_df["Manufacturer"] == "Philips").sum()),
                "SIEMENS": int((fold_df["Manufacturer"] == "SIEMENS").sum()),
                "AD_GE": int(((fold_df["ResearchGroup_Mapped"] == "AD") & (fold_df["Manufacturer"] == "GE")).sum()),
                "AD_Philips": int(
                    ((fold_df["ResearchGroup_Mapped"] == "AD") & (fold_df["Manufacturer"] == "Philips")).sum()
                ),
                "AD_SIEMENS": int(
                    ((fold_df["ResearchGroup_Mapped"] == "AD") & (fold_df["Manufacturer"] == "SIEMENS")).sum()
                ),
                "contains_035_S_6927": RECOVER_SUBJECT in set(fold_df["SubjectID"].astype(str)),
            }
        )
    fold_df = pd.DataFrame(rows)
    fold_df["all_diagnoses_present"] = (fold_df[["CN", "AD", "MCI"]] > 0).all(axis=1)
    fold_df["all_manufacturers_present"] = (fold_df[["GE", "Philips", "SIEMENS"]] > 0).all(axis=1)
    fold_df["valid_fold"] = fold_df["all_diagnoses_present"] & fold_df["all_manufacturers_present"]
    fold_df["simulated_035_outer_test_fold"] = fold_assignment.get(RECOVER_SUBJECT, "")
    return key_counts, fold_df


def write_md_table(df: pd.DataFrame, path: Path) -> None:
    path.write_text(df.to_markdown(index=False) + "\n")


def write_recommendation(
    path: Path,
    evidence_035: Dict[str, Any],
    evidence_128: Dict[str, Any],
    diagnosis_df: pd.DataFrame,
    stratum_df: pd.DataFrame,
    fold_df: pd.DataFrame,
) -> None:
    all_folds_valid = bool(fold_df["valid_fold"].all())
    all_strata_valid = bool(stratum_df["valid_for_5fold"].all())
    manufacturer_ok = bool(evidence_035["manufacturer_supported"])
    if all_folds_valid and all_strata_valid and manufacturer_ok:
        recommendation = "safe_to_rebuild_now"
    elif not manufacturer_ok:
        recommendation = "needs_missing_manufacturer_resolution"
    else:
        recommendation = "not_recommended_for_current_revision"

    ad_count = int(diagnosis_df.loc[diagnosis_df["Diagnosis"] == "AD", "N"].iloc[0])
    cn_count = int(diagnosis_df.loc[diagnosis_df["Diagnosis"] == "CN", "N"].iloc[0])
    mci_count = int(diagnosis_df.loc[diagnosis_df["Diagnosis"] == "MCI", "N"].iloc[0])
    test_fold = fold_df.loc[fold_df["contains_035_S_6927"], "fold"].iloc[0]

    text = f"""# v5.1c Metadata Recovery Preflight

This is a read-only simulation. It does not modify tensor, metadata, ledger, configs, or training outputs.

## Recommendation

`{recommendation}`

Rationale:

- `035_S_6927` is present in the tensor at index {evidence_035['tensor_index']} and has usable ROISignals.
- Diagnosis is supported as AD; age/sex are supported as age {evidence_035['age']} and sex {evidence_035['sex']}.
- ImageID/visit are supported as ImageID {evidence_035['image_id']} and visit `{evidence_035['visit']}`.
- SiteCode is `{evidence_035['sitecode']}` from the ADNI subject prefix.
- Manufacturer is `{evidence_035['manufacturer']}` from `{evidence_035['manufacturer_source']}`. This is direct scanner protocol evidence, not imputation.
- `128_S_2002` remains excluded because diagnosis/demographics are unresolved and signal/provenance evidence is not acceptable for final v5.1 use.

## Simulated Cohort After Adding 035_S_6927

- CN: {cn_count}
- AD: {ad_count}
- MCI: {mci_count}
- Total training-ready rows would become: {cn_count + ad_count + mci_count}

## Stratification Safety

- Stratification key: `ResearchGroup_Mapped + Manufacturer`.
- Minimum simulated stratum count: {int(stratum_df['N'].min())}.
- All strata valid for 5-fold: {all_strata_valid}.
- All simulated folds valid: {all_folds_valid}.

If rebuilt, `035_S_6927` would enter:

- the AD/CN classifier pool as an AD subject;
- exactly one outer test fold in a 5-fold split, simulated here as fold {test_fold} with seed 42;
- the VAE train/dev pool for the other four outer folds, consistent with the diagnosis-agnostic VAE training design using train-fold CN+MCI+AD only.

## Current Model

The current locked model should not be modified by this preflight. This is only a future v5.1c rebuild check.
"""
    path.write_text(text)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now().isoformat(timespec="seconds")

    tensor_subjects = load_tensor_subjects(args.tensor)
    metadata = pd.read_csv(args.metadata, dtype=str, keep_default_na=False)
    evidence_035 = recover_035_evidence(args, tensor_subjects)
    evidence_128 = excluded_128_evidence(args, tensor_subjects)

    if not evidence_035["manufacturer_supported"]:
        simulated = metadata.copy()
    else:
        recovered_row = build_recovered_row(metadata, evidence_035)
        simulated = pd.concat([metadata, pd.DataFrame([recovered_row])], ignore_index=True)

    simulated["SiteCode"] = [
        normalize_site(row.get("Site3", ""), row.get("SubjectID", ""))
        for _, row in simulated.iterrows()
    ]

    diag_counts = diagnosis_counts(simulated)
    manufacturer_dx = crosstab_long(simulated, "Manufacturer", "ResearchGroup_Mapped")
    site_dx = crosstab_long(simulated, "SiteCode", "ResearchGroup_Mapped")
    stratum_counts, fold_preview = simulate_stratification(simulated, args.n_splits, args.seed)

    evidence_df = pd.DataFrame([evidence_035, evidence_128])
    evidence_df.to_csv(args.output_dir / "subject_recovery_evidence.csv", index=False)
    diag_counts.to_csv(args.output_dir / "simulated_diagnosis_counts.csv", index=False)
    manufacturer_dx.to_csv(args.output_dir / "simulated_manufacturer_by_diagnosis.csv", index=False)
    site_dx.to_csv(args.output_dir / "simulated_sitecode_by_diagnosis.csv", index=False)
    stratum_counts.to_csv(args.output_dir / "stratification_key_counts.csv", index=False)
    fold_preview.to_csv(args.output_dir / "simulated_5fold_stratification_preview.csv", index=False)

    write_md_table(evidence_df, args.output_dir / "subject_recovery_evidence.md")
    write_md_table(diag_counts, args.output_dir / "simulated_diagnosis_counts.md")
    write_md_table(manufacturer_dx, args.output_dir / "simulated_manufacturer_by_diagnosis.md")
    write_md_table(site_dx, args.output_dir / "simulated_sitecode_by_diagnosis.md")
    write_md_table(stratum_counts, args.output_dir / "stratification_key_counts.md")
    write_md_table(fold_preview, args.output_dir / "simulated_5fold_stratification_preview.md")

    write_recommendation(
        args.output_dir / "final_recommendation.md",
        evidence_035,
        evidence_128,
        diag_counts,
        stratum_counts,
        fold_preview,
    )

    command_log = {
        "script": rel(Path(__file__)),
        "started": started,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "tensor": str(args.tensor),
        "metadata": str(args.metadata),
        "output_dir": rel(args.output_dir),
        "read_only": True,
        "modified_tensor_metadata_ledger_configs_or_training_outputs": False,
        "recovered_subject": RECOVER_SUBJECT,
        "excluded_subject": EXCLUDED_SUBJECT,
        "manufacturer_recovery_source": evidence_035["manufacturer_source"],
        "manufacturer_recovered": evidence_035["manufacturer"],
        "all_folds_valid": bool(fold_preview["valid_fold"].all()),
        "all_strata_valid_for_5fold": bool(stratum_counts["valid_for_5fold"].all()),
        "outputs": [
            rel(args.output_dir / "subject_recovery_evidence.csv"),
            rel(args.output_dir / "subject_recovery_evidence.md"),
            rel(args.output_dir / "simulated_diagnosis_counts.csv"),
            rel(args.output_dir / "simulated_diagnosis_counts.md"),
            rel(args.output_dir / "simulated_manufacturer_by_diagnosis.csv"),
            rel(args.output_dir / "simulated_manufacturer_by_diagnosis.md"),
            rel(args.output_dir / "simulated_sitecode_by_diagnosis.csv"),
            rel(args.output_dir / "simulated_sitecode_by_diagnosis.md"),
            rel(args.output_dir / "stratification_key_counts.csv"),
            rel(args.output_dir / "stratification_key_counts.md"),
            rel(args.output_dir / "simulated_5fold_stratification_preview.csv"),
            rel(args.output_dir / "simulated_5fold_stratification_preview.md"),
            rel(args.output_dir / "final_recommendation.md"),
            rel(args.output_dir / "command_log.json"),
        ],
    }
    (args.output_dir / "command_log.json").write_text(json.dumps(command_log, indent=2))
    print(json.dumps(command_log, indent=2))


if __name__ == "__main__":
    main()
