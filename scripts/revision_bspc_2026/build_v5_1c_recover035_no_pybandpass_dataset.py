#!/usr/bin/env python3
"""Build ADNI v5.1c recover035 dataset branch.

This creates a new dataset branch without modifying v5.1b:
- recover 035_S_6927 into metadata using local evidence;
- exclude 128_S_2002 from the clean tensor;
- align tensor rows and metadata rows 1:1 by SubjectID.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_NAME = "adni_expanded_v5_1c_recover035_no_pybandpass"
FINAL_TENSOR_NAME = "GLOBAL_TENSOR_ADNI_expanded_v5_1c_recover035_no_pybandpass.npz"
RECOVER_SUBJECT = "035_S_6927"
EXCLUDE_SUBJECT = "128_S_2002"

DEFAULT_SOURCE_TENSOR = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors/"
    "GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
)
DEFAULT_SOURCE_METADATA = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass/"
    "training_ready_metadata_v5_1_batch20260514b_no_pybandpass.csv"
)
DEFAULT_SOURCE_SUBJECT_METADATA = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass/"
    "subject_metadata_v5_1_batch20260514b_no_pybandpass.csv"
)
DEFAULT_DECISION_TABLE = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "tensor_metadata_missing_subjects_audit"
    / "per_subject_decision.csv"
)
DEFAULT_PRECHECK_EVIDENCE = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1c_metadata_recovery_preflight"
    / "subject_recovery_evidence.csv"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/adni_expanded_v5_1c_recover035_no_pybandpass"
)
DEFAULT_LOCAL_SYMLINK = PROJECT_ROOT / "data/revision_bspc_2026/adni_expanded_v5_1c_recover035_no_pybandpass"
DEFAULT_QC_DIR = PROJECT_ROOT / "results/revision_bspc_2026/adni_v5_1c_recover035_dataset_qc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source-tensor", type=Path, default=DEFAULT_SOURCE_TENSOR)
    parser.add_argument("--source-metadata", type=Path, default=DEFAULT_SOURCE_METADATA)
    parser.add_argument("--source-subject-metadata", type=Path, default=DEFAULT_SOURCE_SUBJECT_METADATA)
    parser.add_argument("--decision-table", type=Path, default=DEFAULT_DECISION_TABLE)
    parser.add_argument("--precheck-evidence", type=Path, default=DEFAULT_PRECHECK_EVIDENCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--local-symlink", type=Path, default=DEFAULT_LOCAL_SYMLINK)
    parser.add_argument("--qc-dir", type=Path, default=DEFAULT_QC_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-symlink", action="store_true")
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


def sitecode(subject_id: str) -> str:
    match = re.match(r"^(\d{3})_", subject_id)
    return match.group(1) if match else ""


def normalize_site(value: Any, subject_id: str) -> str:
    text = clean(value)
    if text:
        try:
            return f"{int(float(text)):03d}"
        except ValueError:
            return text.zfill(3) if text.isdigit() else text
    return sitecode(subject_id)


def ensure_empty_or_overwrite(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise RuntimeError(f"Output exists and is not empty; pass --overwrite: {path}")
    path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def evidence_for_035(precheck: pd.DataFrame, decision: pd.DataFrame) -> Dict[str, Any]:
    row = precheck[precheck["SubjectID"].astype(str).eq(RECOVER_SUBJECT)]
    if row.empty:
        raise RuntimeError(f"{RECOVER_SUBJECT} missing from {DEFAULT_PRECHECK_EVIDENCE}")
    ev = row.iloc[0].to_dict()
    dec = decision[decision["SubjectID"].astype(str).eq(RECOVER_SUBJECT)]
    dec_row = dec.iloc[0].to_dict() if not dec.empty else {}
    if clean(ev.get("manufacturer")) != "SIEMENS" or clean(ev.get("manufacturer_supported")) not in {"True", "true", "1"}:
        raise RuntimeError("035 manufacturer must be directly supported as SIEMENS before v5.1c build.")
    if clean(ev.get("diagnosis")) != "AD":
        raise RuntimeError("035 diagnosis must be AD before v5.1c build.")
    if clean(dec_row.get("roisignals_usable")) != "yes":
        raise RuntimeError("035 must have roisignals_usable=yes before v5.1c build.")
    return {**dec_row, **ev}


def build_035_metadata_row(columns: Sequence[str], evidence: Dict[str, Any]) -> Dict[str, Any]:
    row = {col: "" for col in columns}
    row.update(
        {
            "SubjectID": RECOVER_SUBJECT,
            "tensor_source": "v5_base_conservative",
            "dataset_name": DATASET_NAME,
            "included_in_dataset_version": "v5.1c_recover035",
            "ResearchGroup_Mapped": "AD",
            "Diagnosis": "AD",
            "Age": "60",
            "Sex": "F",
            "Manufacturer": "SIEMENS",
            "Site3": "035",
            "ImageID": "1436478",
            "Visit": "sc",
            "metadata_source": "metadata_recovery_v5_1c|AD_fMRI_4_28_2026.csv|idaSearch_4_03_2026.csv",
            "source_label": "new_passband_20260510_10000",
            "source_batch": "v5_dparsf10000_no_pybandpass",
            "roisignals_path": "",
            "stage_guess": "ARWSDCFN",
            "spectral_class": "bandpassed_like",
            "priority_batch": "",
            "ledger_scope": "metadata_recovery_v5_1c",
            "dicom_series_ok": "",
            "python_bandpass_requested": "NO",
            "python_bandpass_applied": "False",
            "exclude_from_supervised": "False",
            "supervised_exclusion_reason": "",
            "training_ready": "True",
            "n_timepoints_raw": "197",
            "n_rois_raw": "170",
            "finite_fraction": "0.9764705882",
            "scale_label": "around_10000_global_scaled",
        }
    )
    return row


def align_tensor_and_metadata(
    source_tensor: Path,
    source_metadata: pd.DataFrame,
    recovered_row: Dict[str, Any],
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    with np.load(source_tensor, allow_pickle=False) as npz:
        arrays = {key: npz[key] for key in npz.files}

    subject_ids = arrays["subject_ids"].astype(str)
    tensor_data = arrays["global_tensor_data"]
    subject_to_old_idx = {sid: idx for idx, sid in enumerate(subject_ids)}
    if RECOVER_SUBJECT not in subject_to_old_idx:
        raise RuntimeError(f"{RECOVER_SUBJECT} missing from source tensor")
    if EXCLUDE_SUBJECT not in subject_to_old_idx:
        raise RuntimeError(f"{EXCLUDE_SUBJECT} missing from source tensor; expected explicit removal")

    meta = source_metadata.copy()
    if RECOVER_SUBJECT in set(meta["SubjectID"].astype(str)):
        raise RuntimeError(f"{RECOVER_SUBJECT} unexpectedly already present in source training metadata")
    if EXCLUDE_SUBJECT in set(meta["SubjectID"].astype(str)):
        raise RuntimeError(f"{EXCLUDE_SUBJECT} unexpectedly present in source training metadata")
    meta = pd.concat([meta, pd.DataFrame([recovered_row])], ignore_index=True)

    meta_subjects = set(meta["SubjectID"].astype(str))
    keep_subjects = [sid for sid in subject_ids if sid != EXCLUDE_SUBJECT and sid in meta_subjects]
    tensor_only = [sid for sid in subject_ids if sid != EXCLUDE_SUBJECT and sid not in meta_subjects]
    metadata_only = sorted(meta_subjects - set(keep_subjects))
    if tensor_only or metadata_only:
        raise RuntimeError(f"Alignment failed: tensor_only={tensor_only}, metadata_only={metadata_only}")

    keep_indices = [subject_to_old_idx[sid] for sid in keep_subjects]
    new_tensor = tensor_data[keep_indices].astype(np.float32, copy=False)
    arrays["global_tensor_data"] = new_tensor
    arrays["subject_ids"] = np.asarray(keep_subjects, dtype=subject_ids.dtype)

    aligned = meta.set_index("SubjectID").loc[keep_subjects].reset_index()
    aligned["dataset_name"] = DATASET_NAME
    aligned["tensor_index"] = np.arange(len(aligned), dtype=int)
    aligned["training_ready"] = True
    aligned["exclude_from_supervised"] = False
    aligned["python_bandpass_applied"] = False
    aligned["python_bandpass_requested"] = "NO"

    alignment = pd.DataFrame(
        {
            "new_tensor_index": np.arange(len(keep_subjects), dtype=int),
            "old_tensor_index": keep_indices,
            "SubjectID": keep_subjects,
            "ResearchGroup_Mapped": aligned["ResearchGroup_Mapped"].tolist(),
            "Manufacturer": aligned["Manufacturer"].tolist(),
            "SiteCode": [normalize_site(v, sid) for v, sid in zip(aligned["Site3"], keep_subjects)],
            "recovered_035": [sid == RECOVER_SUBJECT for sid in keep_subjects],
            "removed_128": False,
        }
    )
    return arrays, aligned, alignment


def diagnosis_counts(df: pd.DataFrame) -> pd.DataFrame:
    out = df["ResearchGroup_Mapped"].value_counts(dropna=False).rename_axis("Diagnosis").reset_index(name="N")
    order = {"CN": 0, "AD": 1, "MCI": 2}
    out["_order"] = out["Diagnosis"].map(order).fillna(99)
    return out.sort_values(["_order", "Diagnosis"]).drop(columns=["_order"])


def crosstab_long(df: pd.DataFrame, row: str, col: str) -> pd.DataFrame:
    tab = pd.crosstab(df[row], df[col]).reset_index()
    return tab.melt(id_vars=[row], var_name=col, value_name="N")


def stratification_feasibility(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work["SiteCode"] = [normalize_site(v, sid) for v, sid in zip(work["Site3"], work["SubjectID"])]
    work["stratification_key"] = work["ResearchGroup_Mapped"].astype(str) + "|" + work["Manufacturer"].astype(str)
    key_counts = work["stratification_key"].value_counts().rename_axis("stratification_key").reset_index(name="N")
    key_counts["valid_for_5fold"] = key_counts["N"] >= 5
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows: List[Dict[str, Any]] = []
    for fold, (_, test_idx) in enumerate(splitter.split(work, work["stratification_key"]), start=1):
        sub = work.iloc[test_idx]
        rows.append(
            {
                "fold": fold,
                "N": int(len(sub)),
                "AD": int((sub["ResearchGroup_Mapped"] == "AD").sum()),
                "CN": int((sub["ResearchGroup_Mapped"] == "CN").sum()),
                "MCI": int((sub["ResearchGroup_Mapped"] == "MCI").sum()),
                "GE": int((sub["Manufacturer"] == "GE").sum()),
                "Philips": int((sub["Manufacturer"] == "Philips").sum()),
                "SIEMENS": int((sub["Manufacturer"] == "SIEMENS").sum()),
                "AD_GE": int(((sub["ResearchGroup_Mapped"] == "AD") & (sub["Manufacturer"] == "GE")).sum()),
                "AD_Philips": int(((sub["ResearchGroup_Mapped"] == "AD") & (sub["Manufacturer"] == "Philips")).sum()),
                "AD_SIEMENS": int(((sub["ResearchGroup_Mapped"] == "AD") & (sub["Manufacturer"] == "SIEMENS")).sum()),
                "contains_035_S_6927": RECOVER_SUBJECT in set(sub["SubjectID"]),
            }
        )
    preview = pd.DataFrame(rows)
    preview["valid_fold"] = (preview[["AD", "CN", "MCI"]] > 0).all(axis=1) & (
        preview[["GE", "Philips", "SIEMENS"]] > 0
    ).all(axis=1)
    return key_counts, preview


def write_md(df: pd.DataFrame, path: Path) -> None:
    path.write_text(df.to_markdown(index=False) + "\n")


def create_symlink(target: Path, link: Path) -> None:
    if link.exists() or link.is_symlink():
        if link.is_symlink() and link.resolve() == target.resolve():
            return
        raise RuntimeError(f"Local symlink path already exists and points elsewhere: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)


def main() -> None:
    args = parse_args()
    started = datetime.now().isoformat(timespec="seconds")
    ensure_empty_or_overwrite(args.output_root, args.overwrite)
    args.qc_dir.mkdir(parents=True, exist_ok=True)

    source_meta = read_csv(args.source_metadata)
    decision = read_csv(args.decision_table)
    precheck = read_csv(args.precheck_evidence)
    evidence = evidence_for_035(precheck, decision)
    recovered = build_035_metadata_row(source_meta.columns, evidence)
    arrays, aligned_meta, alignment = align_tensor_and_metadata(args.source_tensor, source_meta, recovered)

    if aligned_meta["SubjectID"].duplicated().any():
        raise RuntimeError("Duplicate SubjectID in v5.1c metadata")
    if list(aligned_meta["SubjectID"].astype(str)) != list(arrays["subject_ids"].astype(str)):
        raise RuntimeError("SubjectID order mismatch between tensor and metadata")
    if EXCLUDE_SUBJECT in set(aligned_meta["SubjectID"].astype(str)):
        raise RuntimeError(f"{EXCLUDE_SUBJECT} still present in aligned metadata")
    if EXCLUDE_SUBJECT in set(arrays["subject_ids"].astype(str)):
        raise RuntimeError(f"{EXCLUDE_SUBJECT} still present in aligned tensor")

    tensor_dir = args.output_root / "subject_tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = tensor_dir / FINAL_TENSOR_NAME
    np.savez_compressed(tensor_path, **arrays)

    subject_metadata_path = args.output_root / "subject_metadata_v5_1c_recover035_no_pybandpass.csv"
    training_metadata_path = args.output_root / "training_ready_metadata_v5_1c_recover035_no_pybandpass.csv"
    aligned_meta.to_csv(subject_metadata_path, index=False)
    aligned_meta.to_csv(training_metadata_path, index=False)

    if not args.no_symlink:
        create_symlink(args.output_root, args.local_symlink)

    work = aligned_meta.copy()
    work["SiteCode"] = [normalize_site(v, sid) for v, sid in zip(work["Site3"], work["SubjectID"])]
    diagnosis = diagnosis_counts(work)
    manufacturer_dx = crosstab_long(work, "Manufacturer", "ResearchGroup_Mapped")
    site_dx = crosstab_long(work, "SiteCode", "ResearchGroup_Mapped")
    site035 = work[work["SiteCode"] == "035"].copy()
    strat_counts, split_preview = stratification_feasibility(work)
    duplicate_subjects = work[work["SubjectID"].duplicated(keep=False)].copy()
    duplicate_visits = work[work.duplicated(["SubjectID", "Visit"], keep=False)].copy()

    qc_outputs = {
        "subject_alignment": alignment,
        "diagnosis_counts": diagnosis,
        "manufacturer_x_diagnosis": manufacturer_dx,
        "sitecode_x_diagnosis": site_dx,
        "sitecode035_subjects": site035,
        "stratification_key_counts": strat_counts,
        "split_feasibility_5fold": split_preview,
        "duplicate_subjects": duplicate_subjects,
        "duplicate_subject_visit_rows": duplicate_visits,
    }
    for stem, df in qc_outputs.items():
        csv_path = args.qc_dir / f"{stem}.csv"
        df.to_csv(csv_path, index=False)
        write_md(df, args.qc_dir / f"{stem}.md")

    readme = f"""# ADNI v5.1c Recover035 Dataset QC

Created: {started}

This build created a new dataset branch and did not modify v5.1b files.

## Actions

- Recovered `035_S_6927` into metadata as AD, F, age 60, ImageID 1436478, Visit sc, Site3 035, Manufacturer SIEMENS.
- Removed `128_S_2002` from the clean tensor branch.
- Recomputed `tensor_index` so tensor rows and metadata rows align 1:1 by SubjectID.

## Counts

- Tensor rows: {arrays['global_tensor_data'].shape[0]}
- Metadata rows: {len(aligned_meta)}
- AD/CN/MCI: {diagnosis.set_index('Diagnosis')['N'].to_dict()}
- Duplicate SubjectID rows: {len(duplicate_subjects)}
- Duplicate SubjectID/Visit rows: {len(duplicate_visits)}
- All `ResearchGroup_Mapped + Manufacturer` strata valid for 5-fold: {bool(strat_counts['valid_for_5fold'].all())}
- All simulated folds valid: {bool(split_preview['valid_fold'].all())}

## Python Bandpass

`python_bandpass_applied=False` is preserved from the source tensor.
"""
    (args.qc_dir / "README.md").write_text(readme)

    command_log = {
        "script": rel(Path(__file__)),
        "started": started,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "source_tensor": str(args.source_tensor),
        "source_metadata": str(args.source_metadata),
        "output_root": str(args.output_root),
        "tensor_path": str(tensor_path),
        "subject_metadata_path": str(subject_metadata_path),
        "training_metadata_path": str(training_metadata_path),
        "qc_dir": rel(args.qc_dir),
        "recovered_subject": RECOVER_SUBJECT,
        "excluded_subject": EXCLUDE_SUBJECT,
        "n_tensor": int(arrays["global_tensor_data"].shape[0]),
        "n_metadata": int(len(aligned_meta)),
        "python_bandpass_applied": bool(arrays["python_bandpass_applied"].item())
        if np.asarray(arrays["python_bandpass_applied"]).shape == ()
        else str(arrays["python_bandpass_applied"]),
        "modified_v5_1b_files": False,
        "outputs": [
            str(tensor_path),
            str(subject_metadata_path),
            str(training_metadata_path),
            rel(args.qc_dir / "README.md"),
            rel(args.qc_dir / "subject_alignment.csv"),
        ],
    }
    (args.qc_dir / "command_log.json").write_text(json.dumps(command_log, indent=2))
    print(json.dumps(command_log, indent=2))


if __name__ == "__main__":
    main()
