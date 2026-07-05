#!/usr/bin/env python3
"""
Build ADNI_expanded_v4_all_available by appending Martin Philips7 to v3.

Inputs (read-only):
  data/revision_bspc_2026/adni_expanded_v3_all_available/
  /media/diego/Datos/adni_expansion/MARTIN_20260429_PHILIPS10/
    AAL3_v6_5_17_MARTIN_PHILIPS7_ARWSDCF/
    metadata_philips7/

Outputs:
  data/revision_bspc_2026/adni_expanded_v4_all_available/
    GLOBAL_TENSOR_ADNI_expanded_v4_all_available.npz
    subject_metadata_adni_expanded_v4_all_available.csv
    build_report.json
    tensor_qc_adni_expanded_v4_all_available.csv
    cohort_balance_tables.csv
    README.md
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

V3_DIR = PROJECT_ROOT / "data" / "revision_bspc_2026" / "adni_expanded_v3_all_available"
V3_TENSOR = V3_DIR / "GLOBAL_TENSOR_ADNI_expanded_v3_all_available.npz"
V3_METADATA = V3_DIR / "subject_metadata_adni_expanded_v3_all_available.csv"

PHILIPS7_TENSOR = Path(
    "/media/diego/Datos/adni_expansion"
    "/MARTIN_20260429_PHILIPS10"
    "/AAL3_v6_5_17_MARTIN_PHILIPS7_ARWSDCF"
    "/GLOBAL_TENSOR_from_AAL3_v6_5_17_MARTIN_PHILIPS7_ARWSDCF.npz"
)
PHILIPS7_METADATA = Path(
    "/media/diego/Datos/adni_expansion"
    "/MARTIN_20260429_PHILIPS10"
    "/metadata_philips7"
    "/subject_metadata_philips7.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "revision_bspc_2026" / "adni_expanded_v4_all_available"
NEW_SOURCE_COHORT = "martin_philips7_20260429"
NEW_INPUT_COHORT = "martin_philips7"

EXPECTED_SUBJECT_SHAPE = (7, 131, 131)
SUBJECT_ID_RE = re.compile(r"(\d{3}_S_\d{4})")
METADATA_COLUMNS = [
    "SubjectID",
    "ResearchGroup_Mapped",
    "Age",
    "Sex",
    "Manufacturer",
    "Site3",
    "SourceCohort",
    "InputCohort",
    "metadata_source",
]
VALIDATE_NPZ_KEYS = ["channel_names", "roi_names_in_order", "network_labels_in_order"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Append Martin Philips7 to v3 tensor and write v4 dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def normalize_subject_id(v: object) -> str:
    return str(v).strip()


def normalize_diagnosis(v: object) -> str:
    text = str(v).strip().upper()
    if text in {"CN", "NL", "NORMAL"}:
        return "CN"
    if text in {"AD", "DEMENTIA"}:
        return "AD"
    if text in {"MCI", "EMCI", "LMCI"}:
        return "MCI"
    return text


def normalize_manufacturer(v: object) -> str:
    upper = str(v).strip().upper()
    if "SIEMENS" in upper:
        return "SIEMENS"
    if "GE" in upper:
        return "GE MEDICAL SYSTEMS"
    if "PHILIPS" in upper:
        return "Philips"
    return str(v).strip()


def normalize_site3(subject_id: str) -> str:
    m = SUBJECT_ID_RE.match(subject_id.strip())
    if m:
        prefix = m.group(1).split("_")[0]
        return str(int(prefix)) if prefix.isdigit() else prefix
    return subject_id.split("_")[0]


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as f:
        return {k: np.asarray(f[k]) for k in f.files}


def validate_subject_tensor(tensor: np.ndarray, label: str) -> None:
    if tuple(tensor.shape) != EXPECTED_SUBJECT_SHAPE:
        raise RuntimeError(f"{label}: shape {tensor.shape} != {EXPECTED_SUBJECT_SHAPE}")
    if int(np.isnan(tensor).sum()) != 0:
        raise RuntimeError(f"{label}: contains NaN")
    if int(np.isinf(tensor).sum()) != 0:
        raise RuntimeError(f"{label}: contains Inf")


def validate_compatibility(v3_payload: Dict[str, np.ndarray], p7_payload: Dict[str, np.ndarray]) -> None:
    for key in VALIDATE_NPZ_KEYS:
        if key in v3_payload and key in p7_payload:
            v3_vals = np.asarray(v3_payload[key]).astype(str)
            p7_vals = np.asarray(p7_payload[key]).astype(str)
            if not np.array_equal(v3_vals, p7_vals):
                raise RuntimeError(
                    f"NPZ key '{key}' differs between v3 and Philips7:\n"
                    f"  v3: {list(v3_vals)}\n  p7: {list(p7_vals)}"
                )


def build_philips7_metadata(
    subject_ids: Sequence[str],
    raw_meta: pd.DataFrame,
) -> pd.DataFrame:
    meta_by_id = raw_meta.copy()
    meta_by_id["SubjectID"] = meta_by_id["SubjectID"].map(normalize_subject_id)
    meta_by_id = meta_by_id.set_index("SubjectID")

    rows = []
    missing_meta = []
    for sid in subject_ids:
        if sid in meta_by_id.index:
            row_src = meta_by_id.loc[sid]
            rg = normalize_diagnosis(row_src.get("ResearchGroup_Mapped", row_src.get("ResearchGroup", "")))
            age = pd.to_numeric(row_src.get("Age", np.nan), errors="coerce")
            sex = str(row_src.get("Sex", "")).strip()
            mfr = normalize_manufacturer(row_src.get("Manufacturer", "Philips"))
        else:
            missing_meta.append(sid)
            rg, age, sex, mfr = "", np.nan, "", "Philips"

    if missing_meta:
        raise RuntimeError(f"Philips7 subjects missing from metadata CSV: {missing_meta}")

    for sid in subject_ids:
        row_src = meta_by_id.loc[sid]
        rg = normalize_diagnosis(row_src.get("ResearchGroup_Mapped", row_src.get("ResearchGroup", "")))
        age = pd.to_numeric(row_src.get("Age", np.nan), errors="coerce")
        sex = str(row_src.get("Sex", "")).strip()
        mfr = normalize_manufacturer(row_src.get("Manufacturer", "Philips"))
        rows.append({
            "SubjectID": sid,
            "ResearchGroup_Mapped": rg,
            "Age": age,
            "Sex": sex,
            "Manufacturer": mfr,
            "Site3": normalize_site3(sid),
            "SourceCohort": NEW_SOURCE_COHORT,
            "InputCohort": NEW_INPUT_COHORT,
            "metadata_source": "cohort:martin_philips7",
        })

    df = pd.DataFrame(rows, columns=METADATA_COLUMNS)

    missing_required = []
    for _, row in df.iterrows():
        if not row["ResearchGroup_Mapped"] or not row["Manufacturer"] or not row["Sex"] or pd.isna(row["Age"]):
            missing_required.append(row["SubjectID"])
    if missing_required:
        raise RuntimeError(f"Missing required metadata fields for Philips7 subjects: {missing_required}")

    return df


def tensor_qc(tensor: np.ndarray, metadata: pd.DataFrame) -> pd.DataFrame:
    flat = tensor.reshape(tensor.shape[0], -1)
    return pd.DataFrame({
        "SubjectID": metadata["SubjectID"].to_numpy(str),
        "SourceCohort": metadata["SourceCohort"].to_numpy(str),
        "InputCohort": metadata["InputCohort"].to_numpy(str),
        "tensor_idx": np.arange(tensor.shape[0], dtype=int),
        "shape": [str(tuple(tensor[i].shape)) for i in range(tensor.shape[0])],
        "dtype": str(tensor.dtype),
        "nan_count": np.isnan(flat).sum(axis=1).astype(int),
        "inf_count": np.isinf(flat).sum(axis=1).astype(int),
        "abs_max": np.nanmax(np.abs(flat), axis=1),
    })


def count_table(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    return (
        df.groupby(list(columns), dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )


def balance_tables(metadata: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {
        "ResearchGroup_Mapped": count_table(metadata, ["ResearchGroup_Mapped"]),
        "Manufacturer": count_table(metadata, ["Manufacturer"]),
        "SourceCohort": count_table(metadata, ["SourceCohort"]),
        "Manufacturer_x_ResearchGroup": count_table(metadata, ["Manufacturer", "ResearchGroup_Mapped"]),
        "SourceCohort_x_ResearchGroup": count_table(metadata, ["SourceCohort", "ResearchGroup_Mapped"]),
        "Site3_x_ResearchGroup": count_table(metadata, ["Site3", "ResearchGroup_Mapped"]),
    }


def combined_balance_csv(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for name, table in tables.items():
        t = table.copy()
        t.insert(0, "table_name", name)
        frames.append(t)
    return pd.concat(frames, ignore_index=True, sort=False)


def readme_text(n_v3: int, n_added: int, n_dup: int, n_total: int, tables: Dict[str, pd.DataFrame]) -> str:
    diag = tables["ResearchGroup_Mapped"].to_string(index=False)
    mfr = tables["Manufacturer_x_ResearchGroup"].to_string(index=False)
    src = tables["SourceCohort_x_ResearchGroup"].to_string(index=False)
    return "\n".join([
        "# ADNI Expanded v4 All Available",
        "",
        "v3 tensor + Martin Philips7 (20260429) appended.",
        "No retraining performed in this step.",
        "",
        f"- v3 subjects: {n_v3}",
        f"- Philips7 new subjects added: {n_added}",
        f"- Philips7 duplicates excluded (already in v3): {n_dup}",
        f"- Total v4 subjects: {n_total}",
        "",
        "Diagnosis counts:",
        diag,
        "",
        "Manufacturer × ResearchGroup_Mapped:",
        mfr,
        "",
        "SourceCohort × ResearchGroup_Mapped:",
        src,
        "",
        "Primary outputs:",
        "- GLOBAL_TENSOR_ADNI_expanded_v4_all_available.npz",
        "- subject_metadata_adni_expanded_v4_all_available.csv",
        "- tensor_qc_adni_expanded_v4_all_available.csv",
        "- cohort_balance_tables.csv",
        "- build_report.json",
        "",
    ])


def build(args: argparse.Namespace) -> int:
    print("Loading v3 tensor …")
    v3_payload = load_npz(V3_TENSOR)
    v3_tensor = v3_payload["global_tensor_data"].astype(np.float32)
    v3_ids = list(v3_payload["subject_ids"].astype(str))
    v3_id_set = set(v3_ids)
    print(f"  v3 subjects: {len(v3_ids)}, shape: {v3_tensor.shape}")

    print("Loading v3 metadata …")
    v3_meta = pd.read_csv(V3_METADATA)
    assert len(v3_meta) == len(v3_ids), (
        f"v3 metadata rows ({len(v3_meta)}) != tensor subjects ({len(v3_ids)})"
    )

    print("Loading Philips7 tensor …")
    p7_payload = load_npz(PHILIPS7_TENSOR)
    p7_tensor = p7_payload["global_tensor_data"].astype(np.float32)
    p7_ids_all = list(p7_payload["subject_ids"].astype(str))
    print(f"  Philips7 subjects: {len(p7_ids_all)}, shape: {p7_tensor.shape}")

    print("Loading Philips7 metadata …")
    p7_raw_meta = pd.read_csv(PHILIPS7_METADATA)

    print("Validating tensor compatibility (channel_names, roi_names_in_order) …")
    validate_compatibility(v3_payload, p7_payload)

    # Filter duplicates: keep only Philips7 subjects not already in v3
    new_indices: List[int] = []
    dup_ids: List[str] = []
    for i, sid in enumerate(p7_ids_all):
        if sid in v3_id_set:
            dup_ids.append(sid)
        else:
            new_indices.append(i)

    new_ids = [p7_ids_all[i] for i in new_indices]
    print(f"  Philips7 new (not in v3): {len(new_ids)}, duplicates excluded: {len(dup_ids)}")
    if dup_ids:
        print(f"  Excluded IDs: {dup_ids}")

    # Validate shapes and NaN/Inf for new subjects
    print("Validating Philips7 subject tensors …")
    for local_idx, global_idx in enumerate(new_indices):
        sid = new_ids[local_idx]
        validate_subject_tensor(p7_tensor[global_idx], f"Philips7:{sid}")

    # Build Philips7 metadata for new subjects
    p7_meta_new = build_philips7_metadata(new_ids, p7_raw_meta)

    # Concatenate
    if new_indices:
        p7_tensor_new = p7_tensor[new_indices]
        combined_tensor = np.concatenate([v3_tensor, p7_tensor_new], axis=0)
    else:
        combined_tensor = v3_tensor.copy()

    combined_meta = pd.concat([v3_meta, p7_meta_new], ignore_index=True)
    assert len(combined_meta) == combined_tensor.shape[0]

    # Final NaN/Inf check
    flat = combined_tensor.reshape(combined_tensor.shape[0], -1)
    nan_total = int(np.isnan(flat).sum())
    inf_total = int(np.isinf(flat).sum())
    if nan_total or inf_total:
        raise RuntimeError(f"Combined tensor has NaN={nan_total}, Inf={inf_total}")

    qc = tensor_qc(combined_tensor, combined_meta)
    tables = balance_tables(combined_meta)

    n_v3 = len(v3_ids)
    n_added = len(new_ids)
    n_dup = len(dup_ids)
    n_total = len(combined_meta)

    # Print summary
    print(f"\nv4 summary:")
    print(f"  Total subjects: {n_total}")
    print(f"  v3 base: {n_v3}, Philips7 added: {n_added}, duplicates skipped: {n_dup}")
    print("\nDiagnosis counts:")
    print(tables["ResearchGroup_Mapped"].to_string(index=False))
    print("\nManufacturer × ResearchGroup_Mapped:")
    print(tables["Manufacturer_x_ResearchGroup"].to_string(index=False))
    print("\nSourceCohort × ResearchGroup_Mapped:")
    print(tables["SourceCohort_x_ResearchGroup"].to_string(index=False))

    if args.dry_run:
        print("\nDry-run complete. No files written.")
        return 0

    # Write outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)

    out_payload = dict(v3_payload)
    out_payload["global_tensor_data"] = combined_tensor
    out_payload["subject_ids"] = combined_meta["SubjectID"].to_numpy(dtype=str)
    out_payload["source_cohort"] = combined_meta["SourceCohort"].to_numpy(dtype=str)
    out_payload["input_cohort"] = combined_meta["InputCohort"].to_numpy(dtype=str)
    tensor_path = args.output_dir / "GLOBAL_TENSOR_ADNI_expanded_v4_all_available.npz"
    np.savez_compressed(tensor_path, **out_payload)
    print(f"Tensor written: {tensor_path}")

    meta_path = args.output_dir / "subject_metadata_adni_expanded_v4_all_available.csv"
    combined_meta.to_csv(meta_path, index=False)
    print(f"Metadata written: {meta_path}")

    qc_path = args.output_dir / "tensor_qc_adni_expanded_v4_all_available.csv"
    qc.to_csv(qc_path, index=False)
    print(f"QC written: {qc_path}")

    balance_path = args.output_dir / "cohort_balance_tables.csv"
    combined_balance_csv(tables).to_csv(balance_path, index=False)
    print(f"Balance tables written: {balance_path}")

    excluded_df = pd.DataFrame({"SubjectID": dup_ids, "reason": "already_in_v3"})
    excluded_path = args.output_dir / "excluded_duplicates_adni_expanded_v4_all_available.csv"
    excluded_df.to_csv(excluded_path, index=False)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": False,
        "v3_source": str(V3_TENSOR),
        "philips7_source": str(PHILIPS7_TENSOR),
        "philips7_metadata_source": str(PHILIPS7_METADATA),
        "new_source_cohort": NEW_SOURCE_COHORT,
        "counts": {
            "n_v3_base": n_v3,
            "n_philips7_total": len(p7_ids_all),
            "n_philips7_new": n_added,
            "n_philips7_dup_excluded": n_dup,
            "n_total_v4": n_total,
        },
        "counts_by_diagnosis": tables["ResearchGroup_Mapped"].to_dict("records"),
        "counts_by_manufacturer": tables["Manufacturer"].to_dict("records"),
        "counts_by_source_cohort": tables["SourceCohort"].to_dict("records"),
        "excluded_duplicates": dup_ids,
        "paths": {
            "tensor": str(tensor_path),
            "metadata": str(meta_path),
            "qc": str(qc_path),
            "balance": str(balance_path),
        },
    }
    report_path = args.output_dir / "build_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Build report: {report_path}")

    readme_path = args.output_dir / "README.md"
    readme_path.write_text(readme_text(n_v3, n_added, n_dup, n_total, tables), encoding="utf-8")
    print(f"README: {readme_path}")

    print(f"\nAll files written to: {args.output_dir}")
    return 0


def main() -> int:
    args = parse_args()
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
