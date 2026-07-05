#!/usr/bin/env python3
"""
Build ADNI_expanded_v3_all_available for the BSPC 2026 revision.

Priority order for duplicate SubjectID handling:
historical_adni > martin59 > santiago_siemens > santiago_ge > santiago_philips.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADNI_EXPANSION_ROOT = Path("/media/diego/Datos/adni_expansion")
PILOT_ROOT = PROJECT_ROOT / "data" / "RevisionPaperfMRI_2026_04_pilot"

HISTORICAL_TENSOR_DIR = (
    PROJECT_ROOT
    / "data"
    / "AAL3_dynamicROIs_fmri_tensor_NeuroEnhanced_v6.5.17_"
    "AAL3_131ROIs_OMST_GCE_Signed_GrangerLag1_ChNorm_ROIreorderedYeo17_"
    "ParallelTuned"
)
HISTORICAL_METADATA = PROJECT_ROOT / "data" / "SubjectsData_AAL3_procesado2.csv"
MARTIN59_DIR = ADNI_EXPANSION_ROOT / "MARTIN59" / "AAL3_v6_5_17_MARTIN59_ARWSDCF"
MARTIN59_METADATA = (
    PROJECT_ROOT
    / "data"
    / "OneDrive_1_27-4-2026"
    / "metadata_martin59"
    / "subject_metadata_martin59.csv"
)
SIEMENS_DIR = ADNI_EXPANSION_ROOT / "SIEMENS_available" / "AAL3_v6_5_17_SIEMENS_available"
GE3_DIR = ADNI_EXPANSION_ROOT / "GE_smoketest3" / "AAL3_v6_5_17_GE_smoketest3"
GE7_DIR = ADNI_EXPANSION_ROOT / "GE_batch7" / "AAL3_v6_5_17_GE_batch7"
PHILIPS1_DIR = (
    ADNI_EXPANSION_ROOT / "PHILIPS_CN_STRESS" / "AAL3_v6_5_17_PhilipsCN_first1"
)
PHILIPS2_DIR = (
    ADNI_EXPANSION_ROOT / "PHILIPS_CN_STRESS" / "AAL3_v6_5_17_PhilipsCN_plus2"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "revision_bspc_2026" / "adni_expanded_v3_all_available"
OUTPUT_TENSOR = OUTPUT_DIR / "GLOBAL_TENSOR_ADNI_expanded_v3_all_available.npz"
OUTPUT_METADATA = OUTPUT_DIR / "subject_metadata_adni_expanded_v3_all_available.csv"
OUTPUT_QC = OUTPUT_DIR / "tensor_qc_adni_expanded_v3_all_available.csv"
OUTPUT_BALANCE = OUTPUT_DIR / "cohort_balance_tables.csv"
OUTPUT_README = OUTPUT_DIR / "README.md"

EXPECTED_SHAPE = (7, 131, 131)
SUBJECT_ID_RE = re.compile(r"(\d{3}_S_\d{4})")
MINIMAL_COLUMNS = [
    "SubjectID",
    "ResearchGroup_Mapped",
    "Age",
    "Sex",
    "Manufacturer",
    "Site3",
    "SourceCohort",
]


@dataclass(frozen=True)
class CohortSpec:
    input_name: str
    source_cohort: str
    tensor_dir: Path
    metadata_path: Optional[Path]
    fallback_manufacturer: Optional[str]
    fallback_diagnosis: Optional[str] = None


@dataclass
class LoadedCohort:
    spec: CohortSpec
    tensor: np.ndarray
    subject_ids: np.ndarray
    payload: Dict[str, np.ndarray]
    tensor_source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ADNI_expanded_v3_all_available from all currently processed ADNI tensors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def normalize_subject_id(value: object) -> str:
    return str(value).strip()


def normalize_site3(value: object, subject_id: Optional[str] = None) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if not text and subject_id:
        text = normalize_subject_id(subject_id).split("_", 1)[0]
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if text.isdigit():
        return str(int(text))
    return text


def normalize_diagnosis(value: object) -> str:
    text = str(value).strip().upper()
    if text in {"CN", "NL", "NORMAL"}:
        return "CN"
    if text in {"AD", "DEMENTIA"}:
        return "AD"
    if text in {"MCI", "EMCI", "LMCI"}:
        return "MCI"
    return text


def normalize_manufacturer(value: object) -> str:
    text = str(value).strip()
    upper = text.upper()
    if "SIEMENS" in upper:
        return "SIEMENS"
    if "GE" in upper:
        return "GE MEDICAL SYSTEMS"
    if "PHILIPS" in upper:
        return "Philips"
    return text


def subject_id_from_tensor_filename(path: Path) -> Optional[str]:
    match = SUBJECT_ID_RE.search(path.name)
    return match.group(1) if match else None


def find_global_tensor(tensor_dir: Path) -> Optional[Path]:
    if tensor_dir.is_file() and tensor_dir.suffix == ".npz":
        return tensor_dir
    candidates = sorted(tensor_dir.glob("GLOBAL_TENSOR*.npz")) if tensor_dir.exists() else []
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple GLOBAL_TENSOR*.npz files in {tensor_dir}: {candidates}")
    return candidates[0] if candidates else None


def load_npz_payload(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as npz:
        return {key: np.asarray(npz[key]) for key in npz.files}


def validate_tensor(tensor: np.ndarray, label: str) -> None:
    if tensor.shape != EXPECTED_SHAPE:
        raise RuntimeError(f"{label} shape {tensor.shape} != {EXPECTED_SHAPE}")
    if int(np.isnan(tensor).sum()) != 0:
        raise RuntimeError(f"{label} contains NaN values")
    if int(np.isinf(tensor).sum()) != 0:
        raise RuntimeError(f"{label} contains Inf values")


def load_global_cohort(spec: CohortSpec, path: Path) -> LoadedCohort:
    payload = load_npz_payload(path)
    if "global_tensor_data" not in payload:
        raise RuntimeError(f"{path} missing global_tensor_data")
    if "subject_ids" not in payload:
        raise RuntimeError(f"{path} missing subject_ids")
    tensor = np.asarray(payload["global_tensor_data"], dtype=np.float32)
    ids = np.asarray(payload["subject_ids"]).reshape(-1).astype(str)
    if tensor.ndim != 4 or tuple(tensor.shape[1:]) != EXPECTED_SHAPE:
        raise RuntimeError(f"{path} shape {tensor.shape} != (N, 7, 131, 131)")
    if len(ids) != tensor.shape[0]:
        raise RuntimeError(f"{path}: subject_ids length {len(ids)} != tensor N {tensor.shape[0]}")
    for idx, sid in enumerate(ids):
        validate_tensor(tensor[idx], f"{spec.input_name}:{sid}")
    return LoadedCohort(spec, tensor, ids, payload, str(path))


def load_individual_cohort(spec: CohortSpec) -> LoadedCohort:
    ind_dir = spec.tensor_dir / "individual_subject_tensors"
    files = sorted(ind_dir.glob("tensor_7ch_131rois_*.npz")) if ind_dir.exists() else []
    if not files:
        raise FileNotFoundError(f"No global tensor or individual tensors found for {spec.input_name}: {spec.tensor_dir}")
    tensors: List[np.ndarray] = []
    ids: List[str] = []
    first_payload: Dict[str, np.ndarray] = {}
    for path in files:
        payload = load_npz_payload(path)
        if not first_payload:
            first_payload = payload
        if "tensor_data" in payload:
            tensor = np.asarray(payload["tensor_data"], dtype=np.float32)
        elif "global_tensor_data" in payload and np.asarray(payload["global_tensor_data"]).ndim == 3:
            tensor = np.asarray(payload["global_tensor_data"], dtype=np.float32)
        else:
            raise RuntimeError(f"Cannot identify tensor key in {path}")
        sid = str(payload["subject_id"]) if "subject_id" in payload else subject_id_from_tensor_filename(path)
        if not sid:
            raise RuntimeError(f"Cannot infer SubjectID from {path}")
        validate_tensor(tensor, f"{spec.input_name}:{sid}")
        tensors.append(tensor)
        ids.append(sid)
    payload = dict(first_payload)
    payload["global_tensor_data"] = np.stack(tensors, axis=0)
    payload["subject_ids"] = np.asarray(ids, dtype=str)
    return LoadedCohort(spec, payload["global_tensor_data"], payload["subject_ids"], payload, str(ind_dir))


def load_cohort(spec: CohortSpec) -> LoadedCohort:
    global_path = find_global_tensor(spec.tensor_dir)
    if global_path is not None:
        return load_global_cohort(spec, global_path)
    return load_individual_cohort(spec)


def standardize_metadata(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    out = df.copy()
    rename = {
        "Subject": "SubjectID",
        "Subject ID": "SubjectID",
        "Group": "ResearchGroup_Mapped",
        "ResearchGroup": "ResearchGroup_Mapped",
        "Age_master": "Age",
        "Age_meta": "Age",
        "Sex_master": "Sex",
        "Sex_meta": "Sex",
    }
    for old, new in rename.items():
        if old in out.columns and new not in out.columns:
            out[new] = out[old]
    if "SubjectID" not in out.columns:
        return pd.DataFrame(columns=["SubjectID"])
    if "Site3" not in out.columns:
        out["Site3"] = out["SubjectID"].map(lambda sid: normalize_site3(None, sid))
    if "Manufacturer" not in out.columns:
        out["Manufacturer"] = ""
    if "ResearchGroup_Mapped" not in out.columns:
        out["ResearchGroup_Mapped"] = ""
    if "Age" not in out.columns:
        out["Age"] = np.nan
    if "Sex" not in out.columns:
        out["Sex"] = ""
    keep = ["SubjectID", "ResearchGroup_Mapped", "Age", "Sex", "Manufacturer", "Site3"]
    out = out[keep].copy()
    out["SubjectID"] = out["SubjectID"].map(normalize_subject_id)
    out["ResearchGroup_Mapped"] = out["ResearchGroup_Mapped"].map(normalize_diagnosis)
    out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
    out["Sex"] = out["Sex"].astype(str).str.strip()
    out["Manufacturer"] = out["Manufacturer"].map(normalize_manufacturer)
    out["Site3"] = [normalize_site3(site, sid) for site, sid in zip(out["Site3"], out["SubjectID"])]
    out["metadata_source"] = source_label
    return out.drop_duplicates("SubjectID", keep="first")


def load_metadata_sources(specs: Sequence[CohortSpec]) -> pd.DataFrame:
    paths: List[Tuple[str, Path]] = []
    for spec in specs:
        if spec.metadata_path is not None:
            paths.append((f"cohort:{spec.input_name}", spec.metadata_path))
    paths.extend(
        [
            ("expanded_v2_lookup", PROJECT_ROOT / "data" / "revision_bspc_2026" / "adni_expanded_v2" / "subject_metadata_adni_expanded_v2.csv"),
            ("adni_download_now", PROJECT_ROOT / "data" / "adni_download_now.csv"),
            ("candidate_batch1", PROJECT_ROOT / "data" / "RevisionPaperfMRI_2026_04_4_06_2026.csv"),
            ("candidate_batch2", PROJECT_ROOT / "data" / "RevisionPaperfMRI_2026_04_4_06_2026_SecondBatch.csv"),
            ("assignment_report", PILOT_ROOT / "batches" / "adni_expansion_assignment_report.csv"),
        ]
    )
    frames = []
    seen_paths = set()
    for label, path in paths:
        if path in seen_paths or not path.exists():
            continue
        seen_paths.add(path)
        frames.append(standardize_metadata(pd.read_csv(path), label))
    if not frames:
        return pd.DataFrame(columns=["SubjectID", "ResearchGroup_Mapped", "Age", "Sex", "Manufacturer", "Site3", "metadata_source"])
    return pd.concat(frames, ignore_index=True).drop_duplicates("SubjectID", keep="first")


def build_metadata_for_subjects(
    subject_ids: Sequence[str],
    source_cohorts: Sequence[str],
    input_names: Sequence[str],
    specs_by_input: Dict[str, CohortSpec],
    lookup: pd.DataFrame,
) -> pd.DataFrame:
    lookup_map = lookup.set_index("SubjectID").to_dict("index") if not lookup.empty else {}
    rows = []
    missing = []
    for sid, source, input_name in zip(subject_ids, source_cohorts, input_names):
        spec = specs_by_input[input_name]
        base = lookup_map.get(sid, {})
        row = {
            "SubjectID": sid,
            "ResearchGroup_Mapped": base.get("ResearchGroup_Mapped", "") or spec.fallback_diagnosis or "",
            "Age": base.get("Age", np.nan),
            "Sex": base.get("Sex", ""),
            "Manufacturer": base.get("Manufacturer", "") or spec.fallback_manufacturer or "",
            "Site3": base.get("Site3", "") or normalize_site3(None, sid),
            "SourceCohort": source,
            "InputCohort": input_name,
            "metadata_source": base.get("metadata_source", "fallback_from_source"),
        }
        row["ResearchGroup_Mapped"] = normalize_diagnosis(row["ResearchGroup_Mapped"])
        row["Manufacturer"] = normalize_manufacturer(row["Manufacturer"])
        row["Site3"] = normalize_site3(row["Site3"], sid)
        rows.append(row)
        if not row["ResearchGroup_Mapped"] or not row["Manufacturer"] or not row["Site3"] or not row["Sex"] or pd.isna(row["Age"]):
            missing.append(sid)
    if missing:
        raise RuntimeError(f"Missing required metadata for included subjects: {missing[:20]}")
    return pd.DataFrame(rows)


def build_specs() -> List[CohortSpec]:
    return [
        CohortSpec("historical_adni", "historical_adni", HISTORICAL_TENSOR_DIR, HISTORICAL_METADATA, None),
        CohortSpec("martin59", "martin59", MARTIN59_DIR, MARTIN59_METADATA, None),
        CohortSpec(
            "santiago_siemens5",
            "santiago_siemens",
            SIEMENS_DIR,
            PILOT_ROOT / "metadata_SIEMENS_available" / "subject_metadata_SIEMENS_available.csv",
            "SIEMENS",
            "CN",
        ),
        CohortSpec(
            "santiago_ge_smoketest3",
            "santiago_ge",
            GE3_DIR,
            PILOT_ROOT / "metadata_GE_smoketest3" / "subject_metadata_GE_smoketest3.csv",
            "GE MEDICAL SYSTEMS",
            "CN",
        ),
        CohortSpec(
            "santiago_ge_batch7",
            "santiago_ge",
            GE7_DIR,
            PILOT_ROOT / "metadata_GE_batch7" / "subject_metadata_GE_batch7.csv",
            "GE MEDICAL SYSTEMS",
            "CN",
        ),
        CohortSpec("santiago_philips_first1", "santiago_philips", PHILIPS1_DIR, None, "Philips", "CN"),
        CohortSpec("santiago_philips_plus2", "santiago_philips", PHILIPS2_DIR, None, "Philips", "CN"),
    ]


def validate_reference_metadata(reference: Dict[str, np.ndarray], loaded: LoadedCohort) -> None:
    for key in ["channel_names", "roi_names_in_order", "network_labels_in_order"]:
        if key in reference and key in loaded.payload:
            if not np.array_equal(np.asarray(reference[key]).astype(str), np.asarray(loaded.payload[key]).astype(str)):
                raise RuntimeError(f"{loaded.spec.input_name}: tensor metadata key differs from reference: {key}")


def tensor_qc(tensor: np.ndarray, metadata: pd.DataFrame) -> pd.DataFrame:
    flat = tensor.reshape(tensor.shape[0], -1)
    return pd.DataFrame(
        {
            "SubjectID": metadata["SubjectID"].to_numpy(str),
            "SourceCohort": metadata["SourceCohort"].to_numpy(str),
            "InputCohort": metadata["InputCohort"].to_numpy(str),
            "tensor_idx": np.arange(tensor.shape[0], dtype=int),
            "shape": [str(tuple(tensor[i].shape)) for i in range(tensor.shape[0])],
            "dtype": str(tensor.dtype),
            "nan_count": np.isnan(flat).sum(axis=1).astype(int),
            "inf_count": np.isinf(flat).sum(axis=1).astype(int),
            "abs_max": np.nanmax(np.abs(flat), axis=1),
        }
    )


def count_table(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    return df.groupby(list(columns), dropna=False).size().reset_index(name="n").sort_values("n", ascending=False)


def balance_tables(metadata: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    tables = {
        "ResearchGroup_Mapped": count_table(metadata, ["ResearchGroup_Mapped"]),
        "Manufacturer": count_table(metadata, ["Manufacturer"]),
        "SourceCohort": count_table(metadata, ["SourceCohort"]),
        "Manufacturer_x_ResearchGroup": count_table(metadata, ["Manufacturer", "ResearchGroup_Mapped"]),
        "SourceCohort_x_ResearchGroup": count_table(metadata, ["SourceCohort", "ResearchGroup_Mapped"]),
        "Site3_x_ResearchGroup": count_table(metadata, ["Site3", "ResearchGroup_Mapped"]),
        "Manufacturer_x_Site3_x_ResearchGroup": count_table(metadata, ["Manufacturer", "Site3", "ResearchGroup_Mapped"]),
    }
    return tables


def combined_balance_csv(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for name, table in tables.items():
        t = table.copy()
        t.insert(0, "table_name", name)
        frames.append(t)
    return pd.concat(frames, ignore_index=True, sort=False)


def write_outputs(
    args: argparse.Namespace,
    reference_payload: Dict[str, np.ndarray],
    tensor: np.ndarray,
    metadata: pd.DataFrame,
    qc: pd.DataFrame,
    excluded: pd.DataFrame,
    tables: Dict[str, pd.DataFrame],
    summary: Dict[str, object],
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(reference_payload)
    payload["global_tensor_data"] = tensor
    payload["subject_ids"] = metadata["SubjectID"].to_numpy(dtype=str)
    payload["source_cohort"] = metadata["SourceCohort"].to_numpy(dtype=str)
    payload["input_cohort"] = metadata["InputCohort"].to_numpy(dtype=str)
    np.savez_compressed(args.output_dir / OUTPUT_TENSOR.name, **payload)
    metadata.to_csv(args.output_dir / OUTPUT_METADATA.name, index=False)
    qc.to_csv(args.output_dir / OUTPUT_QC.name, index=False)
    combined_balance_csv(tables).to_csv(args.output_dir / OUTPUT_BALANCE.name, index=False)
    excluded.to_csv(args.output_dir / "excluded_duplicates_adni_expanded_v3_all_available.csv", index=False)
    with (args.output_dir / "build_report.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    (args.output_dir / OUTPUT_README.name).write_text(readme_text(summary), encoding="utf-8")


def readme_text(summary: Dict[str, object]) -> str:
    counts = summary["counts"]
    return "\n".join(
        [
            "# ADNI Expanded v3 All Available",
            "",
            "Composition: historical ADNI + Martin59 + Santiago Siemens/GE/Philips processed tensors.",
            "",
            f"- Total subjects: {counts['n_total']}",
            f"- Duplicates excluded: {counts['duplicates_excluded']}",
            "",
            "Primary outputs:",
            "- GLOBAL_TENSOR_ADNI_expanded_v3_all_available.npz",
            "- subject_metadata_adni_expanded_v3_all_available.csv",
            "- tensor_qc_adni_expanded_v3_all_available.csv",
            "- cohort_balance_tables.csv",
            "",
        ]
    )


def build_dataset(args: argparse.Namespace) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, object]]:
    specs = build_specs()
    loaded = [load_cohort(spec) for spec in specs]
    reference_payload = loaded[0].payload
    for cohort in loaded[1:]:
        validate_reference_metadata(reference_payload, cohort)

    tensors = []
    subject_ids = []
    source_cohorts = []
    input_names = []
    excluded_rows = []
    seen: Dict[str, str] = {}
    for cohort in loaded:
        for idx, sid in enumerate(cohort.subject_ids.astype(str)):
            if sid in seen:
                excluded_rows.append(
                    {
                        "SubjectID": sid,
                        "excluded_input_cohort": cohort.spec.input_name,
                        "excluded_source_cohort": cohort.spec.source_cohort,
                        "kept_input_cohort": seen[sid],
                        "reason": "duplicate_subject_lower_priority",
                    }
                )
                continue
            seen[sid] = cohort.spec.input_name
            tensors.append(cohort.tensor[idx])
            subject_ids.append(sid)
            source_cohorts.append(cohort.spec.source_cohort)
            input_names.append(cohort.spec.input_name)

    tensor = np.stack(tensors, axis=0).astype(np.float32)
    lookup = load_metadata_sources(specs)
    specs_by_input = {spec.input_name: spec for spec in specs}
    metadata = build_metadata_for_subjects(subject_ids, source_cohorts, input_names, specs_by_input, lookup)
    qc = tensor_qc(tensor, metadata)
    if int(qc["nan_count"].sum()) or int(qc["inf_count"].sum()):
        raise RuntimeError("Unexpected NaN/Inf in final tensor")
    excluded = pd.DataFrame(excluded_rows)
    tables = balance_tables(metadata)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(args.dry_run),
        "paths": {
            "output_dir": str(args.output_dir),
            "output_tensor": str(args.output_dir / OUTPUT_TENSOR.name),
            "output_metadata": str(args.output_dir / OUTPUT_METADATA.name),
        },
        "loaded_cohorts": [
            {
                "input_name": c.spec.input_name,
                "source_cohort": c.spec.source_cohort,
                "tensor_source": c.tensor_source,
                "n_subjects": int(c.tensor.shape[0]),
            }
            for c in loaded
        ],
        "counts": {
            "n_total": int(tensor.shape[0]),
            "duplicates_excluded": int(len(excluded)),
        },
    }
    summary["counts_by_diagnosis"] = tables["ResearchGroup_Mapped"].to_dict("records")
    summary["counts_by_manufacturer"] = tables["Manufacturer"].to_dict("records")
    summary["counts_by_source_cohort"] = tables["SourceCohort"].to_dict("records")
    return reference_payload, tensor, metadata, qc, excluded, tables, summary


def print_summary(metadata: pd.DataFrame, excluded: pd.DataFrame, tables: Dict[str, pd.DataFrame]) -> None:
    print("\nADNI_expanded_v3_all_available summary")
    print(f"N total: {len(metadata)}")
    print(f"Duplicates excluded: {len(excluded)}")
    print("\nN CN/AD/MCI:")
    print(tables["ResearchGroup_Mapped"].to_string(index=False))
    print("\nN by Manufacturer:")
    print(tables["Manufacturer"].to_string(index=False))
    print("\nN by SourceCohort:")
    print(tables["SourceCohort"].to_string(index=False))
    man_diag = tables["Manufacturer_x_ResearchGroup"]
    print("\nManufacturer x ResearchGroup_Mapped:")
    print(man_diag.to_string(index=False))


def main() -> int:
    args = parse_args()
    reference_payload, tensor, metadata, qc, excluded, tables, summary = build_dataset(args)
    print_summary(metadata, excluded, tables)
    if args.dry_run:
        print("\nDry-run complete. No output files written.")
    else:
        write_outputs(args, reference_payload, tensor, metadata, qc, excluded, tables, summary)
        print(f"\nFiles written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
