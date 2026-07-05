#!/usr/bin/env python3
"""
Build ADNI_expanded_v1 for the BSPC 2026 revision.

Compatible with scripts/run_vae_clf_ad_inference.py:
  - NPZ key is global_tensor_data (what betavae_xai.data.preprocessing.load_data() requires).
  - No tensor_data alias needed or written.
  - metadata CSV omits tensor_idx; load_data() creates it from subject_ids.

Output files:
  expanded_subject_manifest.csv   — 490 rows, one per subject, with provenance
  expanded_dataset_summary.csv    — file-level summary (role/path/sha256)
  subject_metadata_adni_expanded_v1.csv
  GLOBAL_TENSOR_ADNI_expanded_v1.npz
  qc_adni_expanded_v1.csv
  duplicate_subject_report.csv
  build_report.json
  README.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

HISTORICAL_TENSOR_DIR = (
    PROJECT_ROOT
    / "data"
    / "AAL3_dynamicROIs_fmri_tensor_NeuroEnhanced_v6.5.17_"
    "AAL3_131ROIs_OMST_GCE_Signed_GrangerLag1_ChNorm_ROIreorderedYeo17_"
    "ParallelTuned"
)
HISTORICAL_METADATA = PROJECT_ROOT / "data" / "SubjectsData_AAL3_procesado2.csv"

MARTIN59_TENSOR_DIR = Path(
    "/media/diego/Datos/adni_expansion/MARTIN59/"
    "AAL3_v6_5_17_MARTIN59_ARWSDCF"
)
MARTIN59_TENSOR = MARTIN59_TENSOR_DIR / "GLOBAL_TENSOR_from_AAL3_v6_5_17_MARTIN59_ARWSDCF.npz"
MARTIN59_METADATA = (
    PROJECT_ROOT
    / "data"
    / "OneDrive_1_27-4-2026"
    / "metadata_martin59"
    / "subject_metadata_martin59.csv"
)
MARTIN59_QC = MARTIN59_TENSOR_DIR / "tensor_qc_MARTIN59_ARWSDCF.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "revision_bspc_2026" / "adni_expanded_v1"
OUTPUT_TENSOR = OUTPUT_DIR / "GLOBAL_TENSOR_ADNI_expanded_v1.npz"
OUTPUT_METADATA = OUTPUT_DIR / "subject_metadata_adni_expanded_v1.csv"
OUTPUT_QC = OUTPUT_DIR / "qc_adni_expanded_v1.csv"
OUTPUT_SUBJECT_MANIFEST = OUTPUT_DIR / "expanded_subject_manifest.csv"  # 490-row per-subject
OUTPUT_SUMMARY = OUTPUT_DIR / "expanded_dataset_summary.csv"            # file-level summary
OUTPUT_DUPLICATES = OUTPUT_DIR / "duplicate_subject_report.csv"
OUTPUT_REPORT = OUTPUT_DIR / "build_report.json"
OUTPUT_README = OUTPUT_DIR / "README.md"

EXPECTED_TENSOR_SHAPE_TAIL = (7, 131, 131)
REQUIRED_METADATA_COLUMNS = {
    "SubjectID",
    "ResearchGroup_Mapped",
    "Manufacturer",
    "Site3",
    "Age",
    "Sex",
}

SUBJECT_ID_RE = re.compile(r"(\d{3}_S_\d{4})")


@dataclass
class CohortData:
    name: str
    source_cohort: str
    tensor_path: Path
    metadata_path: Path
    tensor: np.ndarray
    subject_ids: np.ndarray
    metadata: pd.DataFrame
    npz_payload: Dict[str, np.ndarray]
    id_source: str


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ADNI_expanded_v1 = historical ADNI + non-duplicate Martin59 CN subjects.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--historical-tensor-dir", type=Path, default=HISTORICAL_TENSOR_DIR)
    parser.add_argument("--historical-tensor-path", type=Path, default=None)
    parser.add_argument("--historical-metadata-path", type=Path, default=HISTORICAL_METADATA)
    parser.add_argument("--martin59-tensor-path", type=Path, default=MARTIN59_TENSOR)
    parser.add_argument("--martin59-metadata-path", type=Path, default=MARTIN59_METADATA)
    parser.add_argument("--martin59-qc-path", type=Path, default=MARTIN59_QC)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and summarize without writing any outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files if present.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the already-materialized dataset without rebuilding it.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def resolve_historical_tensor(args: argparse.Namespace) -> Path:
    if args.historical_tensor_path is not None:
        path = args.historical_tensor_path
        if not path.exists():
            raise FileNotFoundError(f"Historical tensor not found: {path}")
        return path

    tensor_dir = args.historical_tensor_dir
    candidates = sorted(tensor_dir.glob("GLOBAL_TENSOR*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No GLOBAL_TENSOR*.npz found in {tensor_dir}")
    if len(candidates) > 1:
        rendered = "\n".join(f"  - {p}" for p in candidates)
        raise RuntimeError(
            "Multiple historical global tensors found; pass --historical-tensor-path.\n"
            f"{rendered}"
        )
    return candidates[0]


def normalize_subject_ids(values: Iterable[object]) -> np.ndarray:
    return np.asarray([str(v).strip() for v in values], dtype=str)


def ensure_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_git_hash() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def tensor_digest(x: np.ndarray) -> str:
    arr = np.ascontiguousarray(x)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def subject_id_from_path(path: Path) -> Optional[str]:
    match = SUBJECT_ID_RE.search(path.name)
    return match.group(1) if match else None


def find_individual_tensor_dir(global_tensor_path: Path) -> Optional[Path]:
    candidates = [
        global_tensor_path.parent / "individual_subject_tensors",
        global_tensor_path.parent / "individual_tensors",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def load_individual_tensor(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as npz:
        if "tensor" in npz:
            arr = np.asarray(npz["tensor"])
        elif "connectivity_tensor" in npz:
            arr = np.asarray(npz["connectivity_tensor"])
        elif "data" in npz:
            arr = np.asarray(npz["data"])
        else:
            array_keys = [k for k in npz.files if isinstance(npz[k], np.ndarray) and npz[k].ndim >= 3]
            if len(array_keys) != 1:
                raise RuntimeError(f"Cannot identify tensor key in individual tensor: {path}")
            arr = np.asarray(npz[array_keys[0]])
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.shape != EXPECTED_TENSOR_SHAPE_TAIL:
        raise RuntimeError(f"Individual tensor has unexpected shape {arr.shape}: {path}")
    return arr


def reconstruct_subject_ids_from_individual_tensors(
    tensor: np.ndarray,
    global_tensor_path: Path,
) -> Tuple[np.ndarray, str]:
    individual_dir = find_individual_tensor_dir(global_tensor_path)
    if individual_dir is None:
        raise RuntimeError(
            f"{global_tensor_path} does not include subject_ids and no individual tensor dir exists."
        )

    files = sorted(individual_dir.glob("*.npz"))
    if len(files) != tensor.shape[0]:
        raise RuntimeError(
            "Cannot guarantee subject order: individual tensor count "
            f"({len(files)}) != global tensor N ({tensor.shape[0]})."
        )

    digest_to_subject: Dict[str, str] = {}
    duplicate_digests = set()
    for path in files:
        sid = subject_id_from_path(path)
        if sid is None:
            raise RuntimeError(f"Cannot parse SubjectID from individual tensor filename: {path}")
        digest = tensor_digest(load_individual_tensor(path))
        if digest in digest_to_subject:
            duplicate_digests.add(digest)
        digest_to_subject[digest] = sid

    if duplicate_digests:
        raise RuntimeError(
            "Cannot guarantee subject order: duplicate tensor content hashes in individual tensors."
        )

    subject_ids: List[str] = []
    for idx in range(tensor.shape[0]):
        digest = tensor_digest(tensor[idx])
        sid = digest_to_subject.get(digest)
        if sid is None:
            raise RuntimeError(
                f"Cannot match global tensor row {idx} to any individual tensor by content hash."
            )
        subject_ids.append(sid)

    if len(set(subject_ids)) != len(subject_ids):
        raise RuntimeError("Reconstructed subject_ids contain duplicates; aborting.")

    return normalize_subject_ids(subject_ids), "reconstructed_from_individual_tensor_hashes"


def load_npz_payload(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as npz:
        return {key: np.asarray(npz[key]) for key in npz.files}


def pick_subject_ids(
    payload: Dict[str, np.ndarray], tensor: np.ndarray, tensor_path: Path
) -> Tuple[np.ndarray, str]:
    candidate_keys = ["subject_ids", "subjects", "SubjectID", "subject_id", "participant_ids", "ids"]
    for key in candidate_keys:
        if key in payload:
            ids = normalize_subject_ids(np.asarray(payload[key]).reshape(-1))
            if len(ids) != tensor.shape[0]:
                raise RuntimeError(
                    f"{tensor_path} key '{key}' has {len(ids)} IDs, tensor has {tensor.shape[0]} rows."
                )
            return ids, f"npz:{key}"
    return reconstruct_subject_ids_from_individual_tensors(tensor, tensor_path)


def validate_tensor_shape(tensor: np.ndarray, path: Path) -> None:
    if tensor.ndim != 4:
        raise RuntimeError(f"Tensor must be 4D [N,7,131,131], got shape {tensor.shape}: {path}")
    if tuple(tensor.shape[1:]) != EXPECTED_TENSOR_SHAPE_TAIL:
        raise RuntimeError(
            f"Tensor shape mismatch for {path}: got {tensor.shape}, expected (N, 7, 131, 131)."
        )


# ---------------------------------------------------------------------------
# Cohort loading and alignment
# ---------------------------------------------------------------------------

def load_cohort(
    name: str, source_cohort: str, tensor_path: Path, metadata_path: Path
) -> CohortData:
    ensure_file(tensor_path, f"{name} tensor")
    ensure_file(metadata_path, f"{name} metadata")

    payload = load_npz_payload(tensor_path)
    if "global_tensor_data" not in payload:
        raise RuntimeError(f"{tensor_path} does not contain 'global_tensor_data'.")

    tensor = np.asarray(payload["global_tensor_data"])
    validate_tensor_shape(tensor, tensor_path)

    subject_ids, id_source = pick_subject_ids(payload, tensor, tensor_path)
    if len(set(subject_ids.tolist())) != len(subject_ids):
        raise RuntimeError(f"{name} tensor has duplicate subject_ids.")

    metadata = pd.read_csv(metadata_path)
    if "SubjectID" not in metadata.columns:
        raise RuntimeError(f"{metadata_path} does not contain SubjectID.")
    metadata["SubjectID"] = metadata["SubjectID"].astype(str).str.strip()

    missing_cols = sorted(REQUIRED_METADATA_COLUMNS - set(metadata.columns))
    if missing_cols:
        raise RuntimeError(f"{metadata_path} is missing required columns: {missing_cols}")

    return CohortData(
        name=name,
        source_cohort=source_cohort,
        tensor_path=tensor_path,
        metadata_path=metadata_path,
        tensor=tensor,
        subject_ids=subject_ids,
        metadata=metadata,
        npz_payload=payload,
        id_source=id_source,
    )


def align_metadata_to_tensor(cohort: CohortData) -> Tuple[pd.DataFrame, pd.DataFrame]:
    duplicate_rows: List[Dict[str, object]] = []
    dup_mask = cohort.metadata["SubjectID"].duplicated(keep=False)
    if dup_mask.any():
        for sid in sorted(cohort.metadata.loc[dup_mask, "SubjectID"].unique()):
            duplicate_rows.append(
                {
                    "duplicate_type": "metadata_within_source",
                    "SubjectID": sid,
                    "sources": cohort.source_cohort,
                    "action": "kept_first_metadata_row",
                }
            )
        metadata_unique = cohort.metadata.drop_duplicates("SubjectID", keep="first")
    else:
        metadata_unique = cohort.metadata

    tensor_df = pd.DataFrame({"SubjectID": cohort.subject_ids})
    aligned = tensor_df.merge(metadata_unique, on="SubjectID", how="left", validate="one_to_one")

    metadata_cols = [c for c in aligned.columns if c != "SubjectID"]
    missing = aligned[metadata_cols].isna().all(axis=1)
    if missing.any():
        missing_ids = aligned.loc[missing, "SubjectID"].tolist()
        raise RuntimeError(
            f"{cohort.name}: {len(missing_ids)} tensor SubjectIDs are missing metadata. "
            f"First missing IDs: {missing_ids[:10]}"
        )

    extras = sorted(set(metadata_unique["SubjectID"]) - set(cohort.subject_ids))
    for sid in extras:
        duplicate_rows.append(
            {
                "duplicate_type": "metadata_not_in_tensor",
                "SubjectID": sid,
                "sources": cohort.source_cohort,
                "action": "excluded_from_expanded_dataset",
            }
        )

    aligned["SourceCohort"] = cohort.source_cohort
    aligned["IsExpansionSubject"] = cohort.source_cohort == "martin59_arwsdcf"
    aligned["IsStressCandidate"] = (
        aligned["IsExpansionSubject"].astype(bool)
        & aligned["ResearchGroup_Mapped"].astype(str).eq("CN")
    )

    if "tensor_idx" in aligned.columns:
        aligned = aligned.drop(columns=["tensor_idx"])

    return aligned, pd.DataFrame(duplicate_rows)


def validate_common_tensor_metadata(historical: CohortData, martin59: CohortData) -> List[str]:
    warnings: List[str] = []
    required_equal = ["channel_names", "roi_names_in_order", "network_labels_in_order", "rois_count"]
    for key in required_equal:
        if key not in historical.npz_payload or key not in martin59.npz_payload:
            raise RuntimeError(f"Required tensor metadata key missing from one cohort: {key}")
        left = historical.npz_payload[key].astype(str)
        right = martin59.npz_payload[key].astype(str)
        if not np.array_equal(left, right):
            raise RuntimeError(f"Tensor metadata key differs between cohorts: {key}")

    optional_compare = [
        "target_len_ts",
        "tr_seconds",
        "filter_low_hz",
        "filter_high_hz",
        "hrf_deconvolution_applied",
        "hrf_model",
        "channel_normalization_method_subject",
        "roi_order_name",
    ]
    for key in optional_compare:
        if key in historical.npz_payload and key in martin59.npz_payload:
            if not np.array_equal(historical.npz_payload[key], martin59.npz_payload[key]):
                warnings.append(f"Optional tensor metadata differs: {key}")
    return warnings


def build_duplicate_report(
    historical: CohortData,
    martin59: CohortData,
    historical_meta_dups: pd.DataFrame,
    martin_meta_dups: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    rows.extend(historical_meta_dups.to_dict("records"))
    rows.extend(martin_meta_dups.to_dict("records"))

    overlap = sorted(set(historical.subject_ids.tolist()) & set(martin59.subject_ids.tolist()))
    for sid in overlap:
        rows.append(
            {
                "duplicate_type": "historical_vs_martin59_tensor",
                "SubjectID": sid,
                "sources": "historical_adni;martin59_arwsdcf",
                "action": "kept_historical_excluded_martin59",
            }
        )

    columns = ["duplicate_type", "SubjectID", "sources", "action"]
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# QC and summaries
# ---------------------------------------------------------------------------

def tensor_qc(tensor: np.ndarray, metadata: pd.DataFrame) -> pd.DataFrame:
    flat = tensor.reshape(tensor.shape[0], -1)
    return pd.DataFrame(
        {
            "SubjectID": metadata["SubjectID"].to_numpy(str),
            "SourceCohort": metadata["SourceCohort"].to_numpy(str),
            "IsExpansionSubject": metadata["IsExpansionSubject"].to_numpy(bool),
            "IsStressCandidate": metadata["IsStressCandidate"].to_numpy(bool),
            "tensor_idx_expanded": np.arange(tensor.shape[0], dtype=int),
            "shape": [str(tuple(tensor[i].shape)) for i in range(tensor.shape[0])],
            "dtype": str(tensor.dtype),
            "nan_count": np.isnan(flat).sum(axis=1).astype(int),
            "inf_count": np.isinf(flat).sum(axis=1).astype(int),
            "abs_max": np.nanmax(np.abs(flat), axis=1),
            "ResearchGroup_Mapped": metadata["ResearchGroup_Mapped"].to_numpy(str),
            "Manufacturer": metadata["Manufacturer"].astype(str).to_numpy(),
            "Site3": metadata["Site3"].astype(str).to_numpy(),
        }
    )


def count_table(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    available = [c for c in columns if c in df.columns]
    if not available:
        return pd.DataFrame()
    return (
        df.groupby(available, dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
        .reset_index(drop=True)
    )


def print_count_summary(title: str, df: pd.DataFrame, columns: Sequence[str]) -> None:
    table = count_table(df, columns)
    print(f"\n## {title}")
    if table.empty:
        print("  (no data)")
    else:
        print(table.to_string(index=False))


def build_subject_manifest(
    expanded_metadata: pd.DataFrame,
    historical: CohortData,
    martin59: CohortData,
    keep_martin_mask: np.ndarray,
) -> pd.DataFrame:
    """490-row per-subject manifest with full provenance."""
    n_hist = historical.tensor.shape[0]
    n_martin_retained = int(keep_martin_mask.sum())

    original_indices = np.concatenate([
        np.arange(n_hist, dtype=int),
        np.where(keep_martin_mask)[0],
    ])
    tensor_source_paths = (
        [str(historical.tensor_path)] * n_hist
        + [str(martin59.tensor_path)] * n_martin_retained
    )
    metadata_source_paths = (
        [str(historical.metadata_path)] * n_hist
        + [str(martin59.metadata_path)] * n_martin_retained
    )

    cols_to_include = [
        "SubjectID",
        "SourceCohort",
        "IsExpansionSubject",
        "IsStressCandidate",
        "ResearchGroup_Mapped",
        "Manufacturer",
        "Site3",
        "Age",
        "Sex",
    ]
    available = [c for c in cols_to_include if c in expanded_metadata.columns]
    manifest = expanded_metadata[available].copy().reset_index(drop=True)
    manifest.insert(0, "RowIndex", np.arange(len(manifest), dtype=int))
    manifest["TensorSourcePath"] = tensor_source_paths
    manifest["MetadataSourcePath"] = metadata_source_paths
    manifest["OriginalIndexWithinSource"] = original_indices
    manifest["Included"] = True
    manifest["ExclusionReason"] = ""
    return manifest


def dataset_summary_rows(paths: Dict[str, Path], dry_run: bool) -> pd.DataFrame:
    """File-level summary table (role / path / exists / size / sha256)."""
    rows: List[Dict[str, object]] = []
    for role, path in paths.items():
        exists = path.exists()
        rows.append(
            {
                "role": role,
                "path": str(path),
                "exists": bool(exists),
                "size_bytes": int(path.stat().st_size) if exists else None,
                "sha256": None if (dry_run or not exists) else sha256_file(path),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Internal and post-build validation
# ---------------------------------------------------------------------------

def final_internal_validate(
    expanded_tensor: np.ndarray,
    expanded_metadata: pd.DataFrame,
    subject_manifest: pd.DataFrame,
    npz_subject_ids: np.ndarray,
) -> None:
    errors: List[str] = []

    n_t = expanded_tensor.shape[0]
    n_m = len(expanded_metadata)
    n_s = len(subject_manifest)

    if n_t != n_m:
        errors.append(f"tensor N={n_t} != metadata N={n_m}")
    if n_t != n_s:
        errors.append(f"tensor N={n_t} != subject_manifest N={n_s}")
    if tuple(expanded_tensor.shape[1:]) != EXPECTED_TENSOR_SHAPE_TAIL:
        errors.append(
            f"tensor shape tail {expanded_tensor.shape[1:]} != expected {EXPECTED_TENSOR_SHAPE_TAIL}"
        )

    flat = expanded_tensor.reshape(n_t, -1)
    n_nan = int(np.isnan(flat).sum())
    n_inf = int(np.isinf(flat).sum())
    if n_nan:
        errors.append(f"tensor has {n_nan} NaN values")
    if n_inf:
        errors.append(f"tensor has {n_inf} Inf values")

    meta_ids = expanded_metadata["SubjectID"].astype(str).tolist()
    npz_ids = npz_subject_ids.astype(str).tolist()
    if meta_ids != npz_ids:
        n_diff = sum(a != b for a, b in zip(meta_ids, npz_ids))
        errors.append(f"SubjectID mismatch: metadata vs NPZ subject_ids ({n_diff} positions differ)")

    if "SubjectID" in subject_manifest.columns:
        mfst_ids = subject_manifest["SubjectID"].astype(str).tolist()
        if meta_ids != mfst_ids:
            errors.append("SubjectID order differs between metadata and subject_manifest")

    for col in ["SourceCohort", "IsExpansionSubject", "IsStressCandidate"]:
        if col not in expanded_metadata.columns:
            errors.append(f"metadata missing required column: {col}")
        if col not in subject_manifest.columns:
            errors.append(f"subject_manifest missing required column: {col}")

    if errors:
        raise RuntimeError(
            "Internal validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )


def validate_existing_dataset(output_dir: Path) -> int:
    """Validate already-materialized dataset without rebuilding."""
    tensor_path = output_dir / OUTPUT_TENSOR.name
    metadata_path = output_dir / OUTPUT_METADATA.name
    subject_manifest_path = output_dir / OUTPUT_SUBJECT_MANIFEST.name

    errors: List[str] = []

    for path, label in [(tensor_path, "tensor"), (metadata_path, "metadata")]:
        if not path.exists():
            errors.append(f"{label} not found: {path}")

    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return 1

    print(f"Loading tensor from {tensor_path} ...")
    with np.load(tensor_path, allow_pickle=True) as npz:
        if "global_tensor_data" not in npz.files:
            print("ERROR: NPZ missing key: global_tensor_data")
            return 1
        tensor = np.asarray(npz["global_tensor_data"])
        npz_keys = list(npz.files)
        npz_sids = np.asarray(npz["subject_ids"]).astype(str) if "subject_ids" in npz.files else None

    metadata = pd.read_csv(metadata_path)
    N_t = tensor.shape[0]
    N_m = len(metadata)

    if N_t != N_m:
        errors.append(f"N mismatch: tensor={N_t}, metadata={N_m}")
    if tuple(tensor.shape[1:]) != EXPECTED_TENSOR_SHAPE_TAIL:
        errors.append(f"tensor tail shape: got {tensor.shape[1:]}, expected {EXPECTED_TENSOR_SHAPE_TAIL}")

    flat = tensor.reshape(N_t, -1)
    n_nan = int(np.isnan(flat).sum())
    n_inf = int(np.isinf(flat).sum())
    if n_nan:
        errors.append(f"tensor has {n_nan} NaN values")
    if n_inf:
        errors.append(f"tensor has {n_inf} Inf values")

    for col in [
        "SubjectID", "SourceCohort", "IsExpansionSubject", "IsStressCandidate",
        "ResearchGroup_Mapped", "Manufacturer", "Site3", "Age", "Sex",
    ]:
        if col not in metadata.columns:
            errors.append(f"metadata missing column: {col}")

    if "SubjectID" in metadata.columns and npz_sids is not None:
        meta_ids = metadata["SubjectID"].astype(str).tolist()
        npz_ids = npz_sids.tolist()
        if meta_ids != npz_ids:
            n_diff = sum(a != b for a, b in zip(meta_ids, npz_ids))
            errors.append(f"SubjectID mismatch: metadata vs NPZ ({n_diff} positions differ)")

    if subject_manifest_path.exists():
        sm = pd.read_csv(subject_manifest_path)
        if len(sm) != N_t:
            errors.append(f"subject_manifest has {len(sm)} rows, tensor has {N_t}")
        for col in ["SubjectID", "SourceCohort", "IsExpansionSubject", "IsStressCandidate"]:
            if col not in sm.columns:
                errors.append(f"subject_manifest missing column: {col}")
    else:
        print(f"WARNING: subject_manifest not found: {subject_manifest_path}")
        print("         Re-run the builder (without --validate-only) to generate it.")

    print(f"\nValidation of: {output_dir}")
    print(f"  Tensor shape:  {tensor.shape}")
    print(f"  Metadata rows: {N_m}")
    print(f"  NPZ keys:      {npz_keys}")
    if "SourceCohort" in metadata.columns:
        print_count_summary("SourceCohort", metadata, ["SourceCohort"])
    if "ResearchGroup_Mapped" in metadata.columns:
        print_count_summary("ResearchGroup_Mapped", metadata, ["ResearchGroup_Mapped"])
    if "Manufacturer" in metadata.columns:
        print_count_summary("Manufacturer", metadata, ["Manufacturer"])
    if "Site3" in metadata.columns:
        print_count_summary("Site3", metadata, ["Site3"])

    if errors:
        print("\nVALIDATION FAILED:")
        for e in errors:
            print(f"  ERROR: {e}")
        return 1

    print("\nVALIDATION PASSED: all checks OK.")
    return 0


# ---------------------------------------------------------------------------
# README and report
# ---------------------------------------------------------------------------

def build_readme(report: Dict[str, object], expanded_metadata: pd.DataFrame) -> str:
    counts = report["counts"]

    def vcounts(col: str) -> Dict[str, int]:
        if col in expanded_metadata.columns:
            return expanded_metadata[col].value_counts().to_dict()
        return {}

    rg = vcounts("ResearchGroup_Mapped")
    mfr = vcounts("Manufacturer")
    cohort = vcounts("SourceCohort")
    stress_n = int(expanded_metadata["IsStressCandidate"].sum()) if "IsStressCandidate" in expanded_metadata.columns else "?"

    def fmt_dict(d: Dict) -> List[str]:
        return [f"  - {k}: {v}" for k, v in sorted(d.items())]

    lines = [
        "# ADNI Expanded v1",
        "",
        "Dataset built for the BSPC 2026 revision.",
        "",
        "## Composition",
        "",
        "ADNI_expanded_v1 = historical_adni + martin59_arwsdcf",
        f"N = {counts['expanded_tensor_subjects']}",
        f"Tensor shape: ({counts['expanded_tensor_subjects']}, 7, 131, 131)",
        "",
        "### By SourceCohort",
        *fmt_dict(cohort),
        "",
        "### By ResearchGroup_Mapped (diagnostic label)",
        *fmt_dict(rg),
        "",
        "### By Manufacturer",
        *fmt_dict(mfr),
        "",
        f"Stress candidates (IsStressCandidate=True, Martin59 CN): {stress_n}",
        "",
        "## Sources",
        "",
        "- **historical_adni**: the canonical ADNI fMRI cohort used in prior BSPC analyses.",
        f"  N={counts['historical_tensor_subjects']}.",
        "- **martin59_arwsdcf**: CN subjects preprocessed by Martín Alcaraz.",
        "  Pipeline: ROISignals_AAL3_FunImgARWSDCF → feature_extraction → global tensor.",
        f"  Non-duplicates retained: {counts['martin59_retained_subjects']}.",
        f"  Duplicates excluded: {counts['martin59_duplicates_excluded']}.",
        "",
        "## Purpose",
        "",
        "1. Expanded retraining: train VAE+classifier on the full 490-subject cohort.",
        "2. Stress testing: retained Martin59 CN subjects (IsStressCandidate=True) serve as",
        "   a held-out FPR stress test for both the original and expanded models.",
        "3. Comparison: original_vs_expanded tables in results/revision_bspc_2026/.",
        "",
        "## Key fields",
        "",
        "- **SourceCohort**: 'historical_adni' or 'martin59_arwsdcf'.",
        "- **IsExpansionSubject**: True for retained Martin59 rows.",
        "- **IsStressCandidate**: True for retained Martin59 CN rows (used for FPR stress test).",
        "",
        "## NPZ key compatibility",
        "",
        "Key used: global_tensor_data (required by betavae_xai.data.preprocessing.load_data()).",
        "No tensor_data alias is stored to avoid doubling file size (~300 MB).",
        "The expanded NPZ is a drop-in replacement for scripts/run_vae_clf_ad_inference.py",
        "--global_tensor_path.",
        "",
        "## Files",
        "",
        "- GLOBAL_TENSOR_ADNI_expanded_v1.npz         — merged tensor",
        "- subject_metadata_adni_expanded_v1.csv       — 490-row per-subject metadata",
        "- expanded_subject_manifest.csv               — 490-row per-subject provenance",
        "- expanded_dataset_summary.csv                — file-level summary (role/path/sha256)",
        "- qc_adni_expanded_v1.csv                    — per-subject QC metrics",
        "- duplicate_subject_report.csv                — subjects excluded due to duplicates",
        "- build_report.json                           — full build audit log",
        "",
    ]
    return "\n".join(lines)


def json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict("records")
    raise TypeError(f"Object is not JSON serializable: {type(value)}")


# ---------------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------------

def write_outputs(
    output_dir: Path,
    historical: CohortData,
    expanded_tensor: np.ndarray,
    expanded_metadata: pd.DataFrame,
    qc: pd.DataFrame,
    subject_manifest: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    duplicates: pd.DataFrame,
    report: Dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = dict(historical.npz_payload)
    payload["global_tensor_data"] = expanded_tensor
    payload["subject_ids"] = expanded_metadata["SubjectID"].to_numpy(dtype=str)
    payload["source_cohort"] = expanded_metadata["SourceCohort"].to_numpy(dtype=str)
    payload["is_expansion_subject"] = expanded_metadata["IsExpansionSubject"].to_numpy(dtype=bool)
    payload["is_stress_candidate"] = expanded_metadata["IsStressCandidate"].to_numpy(dtype=bool)

    np.savez_compressed(output_dir / OUTPUT_TENSOR.name, **payload)
    expanded_metadata.to_csv(output_dir / OUTPUT_METADATA.name, index=False)
    qc.to_csv(output_dir / OUTPUT_QC.name, index=False)
    subject_manifest.to_csv(output_dir / OUTPUT_SUBJECT_MANIFEST.name, index=False)
    dataset_summary.to_csv(output_dir / OUTPUT_SUMMARY.name, index=False)
    duplicates.to_csv(output_dir / OUTPUT_DUPLICATES.name, index=False)
    with (output_dir / OUTPUT_REPORT.name).open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=json_safe)
        f.write("\n")
    (output_dir / OUTPUT_README.name).write_text(
        build_readme(report, expanded_metadata), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    if args.validate_only:
        return validate_existing_dataset(args.output_dir)

    historical_tensor_path = resolve_historical_tensor(args)

    output_tensor = args.output_dir / OUTPUT_TENSOR.name
    if output_tensor.exists() and not args.overwrite and not args.dry_run:
        raise RuntimeError(
            f"Output tensor already exists: {output_tensor}\n"
            "Use --overwrite to rebuild, --dry-run to simulate, "
            "or --validate-only to validate without rebuilding."
        )

    historical = load_cohort(
        "historical", "historical_adni", historical_tensor_path, args.historical_metadata_path
    )
    martin59 = load_cohort(
        "martin59", "martin59_arwsdcf", args.martin59_tensor_path, args.martin59_metadata_path
    )

    warnings = validate_common_tensor_metadata(historical, martin59)

    hist_meta, hist_meta_dups = align_metadata_to_tensor(historical)
    martin_meta, martin_meta_dups = align_metadata_to_tensor(martin59)
    duplicate_report = build_duplicate_report(historical, martin59, hist_meta_dups, martin_meta_dups)

    historical_ids = set(historical.subject_ids.tolist())
    keep_martin_mask = np.asarray(
        [sid not in historical_ids for sid in martin59.subject_ids], dtype=bool
    )

    retained_martin_tensor = martin59.tensor[keep_martin_mask]
    retained_martin_meta = martin_meta.loc[keep_martin_mask].reset_index(drop=True)

    expanded_tensor = np.concatenate([historical.tensor, retained_martin_tensor], axis=0)
    expanded_metadata = pd.concat([hist_meta, retained_martin_meta], ignore_index=True)

    if "tensor_idx" in expanded_metadata.columns:
        raise RuntimeError("Expanded metadata unexpectedly contains tensor_idx; this would break load_data().")

    expanded_ids = expanded_metadata["SubjectID"].astype(str).tolist()
    if len(set(expanded_ids)) != len(expanded_ids):
        raise RuntimeError("Expanded dataset still contains duplicate SubjectID values.")

    subject_manifest = build_subject_manifest(
        expanded_metadata, historical, martin59, keep_martin_mask
    )
    npz_subject_ids = expanded_metadata["SubjectID"].to_numpy(str)

    final_internal_validate(expanded_tensor, expanded_metadata, subject_manifest, npz_subject_ids)

    qc = tensor_qc(expanded_tensor, expanded_metadata)

    git_hash = get_git_hash()

    npz_keys_out = sorted(
        set(historical.npz_payload.keys())
        | {"global_tensor_data", "subject_ids", "source_cohort", "is_expansion_subject", "is_stress_candidate"}
    )

    output_paths: Dict[str, Path] = {
        "historical_tensor": historical.tensor_path,
        "historical_metadata": historical.metadata_path,
        "martin59_tensor": martin59.tensor_path,
        "martin59_metadata": martin59.metadata_path,
        "martin59_tensor_qc": args.martin59_qc_path,
        "expanded_tensor": args.output_dir / OUTPUT_TENSOR.name,
        "expanded_metadata": args.output_dir / OUTPUT_METADATA.name,
        "expanded_qc": args.output_dir / OUTPUT_QC.name,
        "expanded_subject_manifest": args.output_dir / OUTPUT_SUBJECT_MANIFEST.name,
        "expanded_dataset_summary": args.output_dir / OUTPUT_SUMMARY.name,
        "duplicate_subject_report": args.output_dir / OUTPUT_DUPLICATES.name,
        "build_report": args.output_dir / OUTPUT_REPORT.name,
        "readme": args.output_dir / OUTPUT_README.name,
    }
    summary_df = dataset_summary_rows(output_paths, dry_run=args.dry_run)

    report: Dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(args.dry_run),
        "git_hash": git_hash,
        "dataset_name": "ADNI_expanded_v1",
        "paths": {k: str(v) for k, v in output_paths.items()},
        "id_sources": {
            "historical": historical.id_source,
            "martin59": martin59.id_source,
        },
        "tensor_shapes": {
            "historical": list(historical.tensor.shape),
            "martin59": list(martin59.tensor.shape),
            "expanded": list(expanded_tensor.shape),
        },
        "npz_keys": npz_keys_out,
        "counts": {
            "historical_tensor_subjects": int(historical.tensor.shape[0]),
            "martin59_tensor_subjects": int(martin59.tensor.shape[0]),
            "martin59_retained_subjects": int(retained_martin_tensor.shape[0]),
            "martin59_duplicates_excluded": int((~keep_martin_mask).sum()),
            "expanded_tensor_subjects": int(expanded_tensor.shape[0]),
            "stress_candidates": int(expanded_metadata["IsStressCandidate"].sum()),
        },
        "warnings": warnings,
        "counts_by_source_cohort": count_table(expanded_metadata, ["SourceCohort"]),
        "counts_by_research_group": count_table(expanded_metadata, ["ResearchGroup_Mapped"]),
        "counts_by_manufacturer": count_table(expanded_metadata, ["Manufacturer"]),
        "counts_by_site3": count_table(expanded_metadata, ["Site3"]),
    }

    print("\nADNI_expanded_v1 build summary")
    print(f"Historical tensor: {historical.tensor_path}")
    print(f"Martin59 tensor:   {martin59.tensor_path}")
    print(f"Output dir:        {args.output_dir}")
    print(f"Dry run:           {args.dry_run}")
    print(f"Git hash:          {git_hash}")
    print(f"Historical shape:  {historical.tensor.shape}")
    print(f"Martin59 shape:    {martin59.tensor.shape}")
    print(f"Expanded shape:    {expanded_tensor.shape}")
    print(f"Historical IDs:    {historical.id_source}")
    print(f"Martin59 IDs:      {martin59.id_source}")
    print(f"Martin59 duplicates excluded: {int((~keep_martin_mask).sum())}")
    print(f"Stress candidates: {int(expanded_metadata['IsStressCandidate'].sum())}")
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")

    print_count_summary("SourceCohort", expanded_metadata, ["SourceCohort"])
    print_count_summary("ResearchGroup_Mapped", expanded_metadata, ["ResearchGroup_Mapped"])
    print_count_summary("Manufacturer", expanded_metadata, ["Manufacturer"])
    print_count_summary("Site3", expanded_metadata, ["Site3"])

    if args.dry_run:
        print("\nDry-run complete. No files written.")
    else:
        write_outputs(
            args.output_dir,
            historical,
            expanded_tensor,
            expanded_metadata,
            qc,
            subject_manifest,
            summary_df,
            duplicate_report,
            report,
        )
        print("\nFiles written:")
        for role, path in output_paths.items():
            if role.startswith("expanded") or role in {
                "duplicate_subject_report", "build_report", "readme",
            }:
                print(f"  {role}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
