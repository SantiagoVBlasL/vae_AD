#!/usr/bin/env python3
"""
Build ADNI_expanded_v2 = historical_adni + martin59_arwsdcf + santi_processed_qc_pass.

santi_processed cohorts (processed by Santiago Blas Laguzza):
  - siemens_available   (5 CN subjects, Site3=941)
  - ge_batch7           (7 CN subjects, Site3=135)
  - ge_smoketest3       (3 CN subjects, Site3=135)

NPZ includes both global_tensor_data (primary) and tensor_data (alias) keys.
Deduplication order (highest priority first): historical > martin59 > siemens > ge_batch7 >
ge_smoketest3 > CLI extras > auto-discovered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADNI_EXPANSION_ROOT = Path("/media/diego/Datos/adni_expansion")
PILOT_METADATA_ROOT = PROJECT_ROOT / "data" / "RevisionPaperfMRI_2026_04_pilot"

# ── historical ───────────────────────────────────────────────────────────────
HISTORICAL_TENSOR_DIR = (
    PROJECT_ROOT / "data"
    / "AAL3_dynamicROIs_fmri_tensor_NeuroEnhanced_v6.5.17_"
    "AAL3_131ROIs_OMST_GCE_Signed_GrangerLag1_ChNorm_ROIreorderedYeo17_ParallelTuned"
)
HISTORICAL_METADATA = PROJECT_ROOT / "data" / "SubjectsData_AAL3_procesado2.csv"

# ── martin59 ─────────────────────────────────────────────────────────────────
MARTIN59_TENSOR = (
    ADNI_EXPANSION_ROOT / "MARTIN59" / "AAL3_v6_5_17_MARTIN59_ARWSDCF"
    / "GLOBAL_TENSOR_from_AAL3_v6_5_17_MARTIN59_ARWSDCF.npz"
)
MARTIN59_METADATA = (
    PROJECT_ROOT / "data" / "OneDrive_1_27-4-2026"
    / "metadata_martin59" / "subject_metadata_martin59.csv"
)

# ── santi_processed ───────────────────────────────────────────────────────────
SIEMENS_TENSOR = (
    ADNI_EXPANSION_ROOT / "SIEMENS_available" / "AAL3_v6_5_17_SIEMENS_available"
    / "GLOBAL_TENSOR_from_AAL3_v6_5_17_SIEMENS_available.npz"
)
SIEMENS_METADATA = (
    PILOT_METADATA_ROOT / "metadata_SIEMENS_available" / "subject_metadata_SIEMENS_available.csv"
)
GE7_TENSOR = (
    ADNI_EXPANSION_ROOT / "GE_batch7" / "AAL3_v6_5_17_GE_batch7"
    / "GLOBAL_TENSOR_from_AAL3_v6_5_17_GE_batch7.npz"
)
GE7_METADATA = PILOT_METADATA_ROOT / "metadata_GE_batch7" / "subject_metadata_GE_batch7.csv"
GE3_TENSOR = (
    ADNI_EXPANSION_ROOT / "GE_smoketest3" / "AAL3_v6_5_17_GE_smoketest3"
    / "GLOBAL_TENSOR_from_AAL3_v6_5_17_GE_smoketest3.npz"
)
GE3_METADATA = PILOT_METADATA_ROOT / "metadata_GE_smoketest3" / "subject_metadata_GE_smoketest3.csv"

# ── output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = PROJECT_ROOT / "data" / "revision_bspc_2026" / "adni_expanded_v2"
OUTPUT_TENSOR = OUTPUT_DIR / "GLOBAL_TENSOR_ADNI_expanded_v2.npz"
OUTPUT_METADATA = OUTPUT_DIR / "subject_metadata_adni_expanded_v2.csv"
OUTPUT_QC = OUTPUT_DIR / "qc_adni_expanded_v2.csv"
OUTPUT_SUBJECT_MANIFEST = OUTPUT_DIR / "expanded_subject_manifest.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "expanded_dataset_summary.csv"
OUTPUT_EXCLUDED = OUTPUT_DIR / "excluded_subjects_adni_expanded_v2.csv"
OUTPUT_REPORT = OUTPUT_DIR / "build_report.json"
OUTPUT_README = OUTPUT_DIR / "README.md"

EXPECTED_TENSOR_SHAPE_TAIL = (7, 131, 131)
EXPECTED_CHANNEL_NAMES = [
    "Pearson_OMST_GCE_Signed_Weighted",
    "Pearson_Full_FisherZ_Signed",
    "MI_KNN_Symmetric",
    "dFC_AbsDiffMean",
    "dFC_StdDev",
    "DistanceCorr",
    "Granger_F_lag1",
]
REQUIRED_METADATA_COLUMNS = {
    "SubjectID", "ResearchGroup_Mapped", "Manufacturer", "Site3", "Age", "Sex",
}
SUBJECT_ID_RE = re.compile(r"(\d{3}_S_\d{4})")

# Known santi_processed cohort names (lowercased) for auto-discovery de-duplication.
_KNOWN_SANTI_NAMES = {"siemens_available", "ge_batch7", "ge_smoketest3"}


@dataclass
class SantiCohortSpec:
    name: str           # e.g. "siemens_available"
    source_cohort: str  # stored in metadata/NPZ
    tensor_path: Path
    metadata_path: Path


@dataclass
class LoadedCohort:
    spec: SantiCohortSpec
    tensor: np.ndarray
    subject_ids: np.ndarray
    metadata: pd.DataFrame  # aligned to tensor order (no tensor_idx)
    npz_payload: Dict[str, np.ndarray]
    id_source: str


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build ADNI_expanded_v2 = historical + martin59 + santi_processed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── base cohort paths ────────────────────────────────────────────────────
    p.add_argument("--historical-tensor-dir", type=Path, default=HISTORICAL_TENSOR_DIR)
    p.add_argument("--historical-tensor-path", type=Path, default=None)
    p.add_argument("--historical-metadata-path", type=Path, default=HISTORICAL_METADATA)
    p.add_argument("--martin59-tensor-path", type=Path, default=MARTIN59_TENSOR)
    p.add_argument("--martin59-metadata-path", type=Path, default=MARTIN59_METADATA)
    # ── known santi cohort overrides ─────────────────────────────────────────
    p.add_argument("--siemens-tensor-path", type=Path, default=SIEMENS_TENSOR)
    p.add_argument("--siemens-metadata-path", type=Path, default=SIEMENS_METADATA)
    p.add_argument("--ge-batch7-tensor-path", type=Path, default=GE7_TENSOR)
    p.add_argument("--ge-batch7-metadata-path", type=Path, default=GE7_METADATA)
    p.add_argument("--ge-smoketest3-tensor-path", type=Path, default=GE3_TENSOR)
    p.add_argument("--ge-smoketest3-metadata-path", type=Path, default=GE3_METADATA)
    # ── skip known santi cohorts ─────────────────────────────────────────────
    p.add_argument("--skip-siemens", action="store_true")
    p.add_argument("--skip-ge-batch7", action="store_true")
    p.add_argument("--skip-ge-smoketest3", action="store_true")
    # ── extra cohorts (repeatable) ───────────────────────────────────────────
    p.add_argument(
        "--extra-cohort-name", action="append", dest="extra_names", default=[],
        metavar="NAME",
        help="Name for an extra cohort (repeatable, must match --extra-global-tensor count).",
    )
    p.add_argument(
        "--extra-global-tensor", action="append", dest="extra_tensors", default=[],
        metavar="PATH",
        help="Global tensor NPZ for an extra cohort (repeatable).",
    )
    p.add_argument(
        "--extra-metadata", action="append", dest="extra_metadatas", default=[],
        metavar="PATH",
        help="Metadata CSV for an extra cohort (repeatable).",
    )
    # ── auto-discovery ────────────────────────────────────────────────────────
    p.add_argument(
        "--no-auto-discover", action="store_true",
        help="Disable auto-discovery of additional cohorts under ADNI_EXPANSION_ROOT.",
    )
    # ── output ────────────────────────────────────────────────────────────────
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def get_git_hash() -> Optional[str]:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=str(PROJECT_ROOT), timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while blk := f.read(chunk):
            h.update(blk)
    return h.hexdigest()


def normalize_subject_ids(values: Iterable[object]) -> np.ndarray:
    return np.asarray([str(v).strip() for v in values], dtype=str)


def ensure_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def validate_tensor_shape(tensor: np.ndarray, path: Path) -> None:
    if tensor.ndim != 4:
        raise RuntimeError(f"Tensor must be 4D, got {tensor.shape}: {path}")
    if tuple(tensor.shape[1:]) != EXPECTED_TENSOR_SHAPE_TAIL:
        raise RuntimeError(
            f"Tensor shape tail {tensor.shape[1:]} != expected {EXPECTED_TENSOR_SHAPE_TAIL}: {path}"
        )


def load_npz_payload(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as npz:
        return {k: np.asarray(npz[k]) for k in npz.files}


def pick_subject_ids(
    payload: Dict[str, np.ndarray], tensor: np.ndarray, tensor_path: Path,
) -> Tuple[np.ndarray, str]:
    for key in ["subject_ids", "subjects", "SubjectID", "subject_id", "participant_ids"]:
        if key in payload:
            ids = normalize_subject_ids(payload[key].reshape(-1))
            if len(ids) != tensor.shape[0]:
                raise RuntimeError(
                    f"Key '{key}' has {len(ids)} IDs, tensor has {tensor.shape[0]} rows: {tensor_path}"
                )
            return ids, f"npz:{key}"
    raise RuntimeError(
        f"No subject_ids key found in {tensor_path}. Keys: {list(payload)}"
    )


def resolve_historical_tensor(args: argparse.Namespace) -> Path:
    if args.historical_tensor_path is not None:
        ensure_file(args.historical_tensor_path, "historical tensor")
        return args.historical_tensor_path
    candidates = sorted(args.historical_tensor_dir.glob("GLOBAL_TENSOR*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No GLOBAL_TENSOR*.npz in {args.historical_tensor_dir}")
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple global tensors in {args.historical_tensor_dir}; pass --historical-tensor-path."
        )
    return candidates[0]


def count_table(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    avail = [c for c in cols if c in df.columns]
    if not avail:
        return pd.DataFrame()
    return (
        df.groupby(avail, dropna=False).size().reset_index(name="n")
        .sort_values("n", ascending=False).reset_index(drop=True)
    )


def print_count_summary(title: str, df: pd.DataFrame, cols: Sequence[str]) -> None:
    t = count_table(df, cols)
    print(f"\n## {title}")
    print(t.to_string(index=False) if not t.empty else "  (no data)")


def json_safe(v: object) -> object:
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, pd.DataFrame):
        return v.to_dict("records")
    raise TypeError(f"Not JSON serializable: {type(v)}")


def dataset_summary_rows(paths: Dict[str, Path], dry_run: bool) -> pd.DataFrame:
    rows = []
    for role, path in paths.items():
        ex = path.exists()
        rows.append({
            "role": role, "path": str(path), "exists": bool(ex),
            "size_bytes": int(path.stat().st_size) if ex else None,
            "sha256": None if (dry_run or not ex) else sha256_file(path),
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Cohort loading (base: historical & martin59)
# ──────────────────────────────────────────────────────────────────────────────

def load_base_cohort(
    name: str, source_cohort: str, tensor_path: Path, metadata_path: Path,
) -> LoadedCohort:
    """Load historical or martin59 cohort (same logic as v1 builder)."""
    ensure_file(tensor_path, f"{name} tensor")
    ensure_file(metadata_path, f"{name} metadata")

    payload = load_npz_payload(tensor_path)
    if "global_tensor_data" not in payload:
        raise RuntimeError(f"'global_tensor_data' missing from {tensor_path}")
    tensor = np.asarray(payload["global_tensor_data"])
    validate_tensor_shape(tensor, tensor_path)

    ids, id_source = pick_subject_ids(payload, tensor, tensor_path)
    if len(set(ids.tolist())) != len(ids):
        raise RuntimeError(f"{name}: duplicate subject_ids in tensor")

    meta = pd.read_csv(metadata_path)
    meta["SubjectID"] = meta["SubjectID"].astype(str).str.strip()
    missing = sorted(REQUIRED_METADATA_COLUMNS - set(meta.columns))
    if missing:
        raise RuntimeError(f"{name} metadata missing columns: {missing}")

    spec = SantiCohortSpec(name=name, source_cohort=source_cohort,
                           tensor_path=tensor_path, metadata_path=metadata_path)
    # align metadata to tensor order
    tensor_df = pd.DataFrame({"SubjectID": ids})
    meta_u = meta.drop_duplicates("SubjectID", keep="first")
    aligned = tensor_df.merge(meta_u, on="SubjectID", how="left", validate="one_to_one")
    all_nan = aligned[[c for c in aligned.columns if c != "SubjectID"]].isna().all(axis=1)
    if all_nan.any():
        bad = aligned.loc[all_nan, "SubjectID"].tolist()
        raise RuntimeError(f"{name}: {len(bad)} tensor IDs missing metadata: {bad[:10]}")
    if "tensor_idx" in aligned.columns:
        aligned = aligned.drop(columns=["tensor_idx"])

    return LoadedCohort(spec=spec, tensor=tensor, subject_ids=ids,
                        metadata=aligned, npz_payload=payload, id_source=id_source)


def validate_channel_compatibility(
    reference: LoadedCohort, candidate: LoadedCohort,
) -> List[str]:
    """Return list of warning strings; raise on critical mismatches."""
    warnings: List[str] = []
    required = ["channel_names", "roi_names_in_order", "network_labels_in_order", "rois_count"]
    for key in required:
        if key not in reference.npz_payload or key not in candidate.npz_payload:
            raise RuntimeError(
                f"Key '{key}' missing from one of the cohorts "
                f"(reference={reference.spec.name}, candidate={candidate.spec.name})"
            )
        if not np.array_equal(
            reference.npz_payload[key].astype(str),
            candidate.npz_payload[key].astype(str),
        ):
            raise RuntimeError(
                f"Critical tensor metadata key '{key}' differs between "
                f"{reference.spec.name} and {candidate.spec.name}"
            )
    for key in ["target_len_ts", "tr_seconds", "filter_low_hz", "filter_high_hz",
                "hrf_model", "channel_normalization_method_subject"]:
        if key in reference.npz_payload and key in candidate.npz_payload:
            if not np.array_equal(reference.npz_payload[key], candidate.npz_payload[key]):
                warnings.append(
                    f"Optional metadata '{key}' differs: "
                    f"{reference.spec.name} vs {candidate.spec.name}"
                )
    return warnings


# ──────────────────────────────────────────────────────────────────────────────
# Santi cohort loading with QC
# ──────────────────────────────────────────────────────────────────────────────

def load_santi_cohort(spec: SantiCohortSpec, reference: LoadedCohort) -> LoadedCohort:
    """Load a santi_processed cohort and validate compatibility with reference (historical)."""
    ensure_file(spec.tensor_path, f"{spec.name} tensor")
    ensure_file(spec.metadata_path, f"{spec.name} metadata")

    payload = load_npz_payload(spec.tensor_path)
    if "global_tensor_data" not in payload:
        raise RuntimeError(f"'global_tensor_data' missing from {spec.tensor_path}")
    tensor = np.asarray(payload["global_tensor_data"])
    validate_tensor_shape(tensor, spec.tensor_path)

    # dtype check — convert if needed
    if tensor.dtype not in (np.float32, np.float64):
        if not np.issubdtype(tensor.dtype, np.floating):
            raise RuntimeError(
                f"{spec.name}: tensor dtype {tensor.dtype} is not float32/64 "
                "and cannot be safely cast"
            )
    tensor = tensor.astype(np.float32, copy=False)

    ids, id_source = pick_subject_ids(payload, tensor, spec.tensor_path)
    if len(set(ids.tolist())) != len(ids):
        raise RuntimeError(f"{spec.name}: duplicate subject_ids in tensor")

    meta = pd.read_csv(spec.metadata_path)
    meta["SubjectID"] = meta["SubjectID"].astype(str).str.strip()
    missing_cols = sorted(REQUIRED_METADATA_COLUMNS - set(meta.columns))
    if missing_cols:
        raise RuntimeError(f"{spec.name} metadata missing required columns: {missing_cols}")

    meta_u = meta.drop_duplicates("SubjectID", keep="first")
    tensor_df = pd.DataFrame({"SubjectID": ids})
    aligned = tensor_df.merge(meta_u, on="SubjectID", how="left", validate="one_to_one")
    all_nan = aligned[[c for c in aligned.columns if c != "SubjectID"]].isna().all(axis=1)
    if all_nan.any():
        bad = aligned.loc[all_nan, "SubjectID"].tolist()
        raise RuntimeError(
            f"{spec.name}: {len(bad)} tensor IDs have no metadata row: {bad[:10]}"
        )
    if "tensor_idx" in aligned.columns:
        aligned = aligned.drop(columns=["tensor_idx"])

    cohort = LoadedCohort(spec=spec, tensor=tensor, subject_ids=ids,
                          metadata=aligned, npz_payload=payload, id_source=id_source)
    # validate channel/ROI compatibility with historical
    validate_channel_compatibility(reference, cohort)
    return cohort


def process_santi_cohort(
    cohort: LoadedCohort,
    seen_ids: set,
    excluded: List[Dict],
) -> Tuple[np.ndarray, pd.DataFrame, List[int]]:
    """
    Apply deduplication + per-subject QC to a santi cohort.

    Updates seen_ids in-place with retained IDs.
    Appends excluded rows to excluded list.
    Returns (retained_tensor, retained_metadata, retained_global_indices).
    """
    retain: List[int] = []
    retained_global: List[int] = []  # index within this cohort's tensor

    for i, sid in enumerate(cohort.subject_ids.tolist()):
        if sid in seen_ids:
            excluded.append({
                "SubjectID": sid,
                "SourceCohort": cohort.spec.source_cohort,
                "Reason": "duplicate_prior_cohort",
                "TensorPath": str(cohort.spec.tensor_path),
                "MetadataPath": str(cohort.spec.metadata_path),
            })
            continue

        row_flat = cohort.tensor[i].reshape(-1)
        n_nan = int(np.isnan(row_flat).sum())
        n_inf = int(np.isinf(row_flat).sum())
        if n_nan:
            excluded.append({
                "SubjectID": sid,
                "SourceCohort": cohort.spec.source_cohort,
                "Reason": f"qc_nan:{n_nan}",
                "TensorPath": str(cohort.spec.tensor_path),
                "MetadataPath": str(cohort.spec.metadata_path),
            })
            continue
        if n_inf:
            excluded.append({
                "SubjectID": sid,
                "SourceCohort": cohort.spec.source_cohort,
                "Reason": f"qc_inf:{n_inf}",
                "TensorPath": str(cohort.spec.tensor_path),
                "MetadataPath": str(cohort.spec.metadata_path),
            })
            continue

        retain.append(i)
        retained_global.append(i)
        seen_ids.add(sid)

    if not retain:
        empty_t = np.empty((0, *EXPECTED_TENSOR_SHAPE_TAIL), dtype=cohort.tensor.dtype)
        empty_m = cohort.metadata.iloc[[]]
        return empty_t, empty_m.reset_index(drop=True), []

    idx = np.array(retain, dtype=int)
    ret_tensor = cohort.tensor[idx]
    ret_meta = cohort.metadata.iloc[idx].reset_index(drop=True)
    return ret_tensor, ret_meta, retain


# ──────────────────────────────────────────────────────────────────────────────
# Auto-discovery
# ──────────────────────────────────────────────────────────────────────────────

def discover_santi_cohorts(args: argparse.Namespace) -> List[SantiCohortSpec]:
    """
    Scan ADNI_EXPANSION_ROOT for AAL3_v6_5_17_* dirs not already in known cohorts.
    Match metadata in PILOT_METADATA_ROOT/metadata_COHORT_NAME/.
    """
    discovered: List[SantiCohortSpec] = []
    if not ADNI_EXPANSION_ROOT.exists():
        return discovered

    pattern = "*/AAL3_v6_5_17_*/GLOBAL_TENSOR_from_*.npz"
    for npz_path in sorted(ADNI_EXPANSION_ROOT.glob(pattern)):
        tensor_dir = npz_path.parent
        # Extract cohort name from dir name (strip AAL3_v6_5_17_ prefix)
        dir_name = tensor_dir.name
        prefix = "AAL3_v6_5_17_"
        if not dir_name.startswith(prefix):
            continue
        cohort_name = dir_name[len(prefix):].lower()

        if cohort_name in _KNOWN_SANTI_NAMES:
            continue  # already in known list

        # Look for metadata
        meta_path = PILOT_METADATA_ROOT / f"metadata_{dir_name[len(prefix):]}" / f"subject_metadata_{dir_name[len(prefix):]}.csv"
        if not meta_path.exists():
            print(
                f"AUTO-DISCOVER: found tensor {npz_path} but no metadata at {meta_path}. "
                "Skipping (no complete metadata)."
            )
            continue

        print(f"AUTO-DISCOVER: including {cohort_name} from {npz_path}")
        discovered.append(SantiCohortSpec(
            name=cohort_name,
            source_cohort=cohort_name,
            tensor_path=npz_path,
            metadata_path=meta_path,
        ))

    return discovered


# ──────────────────────────────────────────────────────────────────────────────
# QC table and manifests
# ──────────────────────────────────────────────────────────────────────────────

def tensor_qc_v2(tensor: np.ndarray, metadata: pd.DataFrame) -> pd.DataFrame:
    flat = tensor.reshape(tensor.shape[0], -1)
    row = {
        "SubjectID": metadata["SubjectID"].to_numpy(str),
        "SourceCohort": metadata["SourceCohort"].to_numpy(str),
        "IsExpansionSubject": metadata["IsExpansionSubject"].to_numpy(bool),
        "IsMartin59": metadata["IsMartin59"].to_numpy(bool),
        "IsSantiProcessed": metadata["IsSantiProcessed"].to_numpy(bool),
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
    return pd.DataFrame(row)


def build_subject_manifest_v2(
    expanded_metadata: pd.DataFrame,
    hist: LoadedCohort,
    m59_original: LoadedCohort,
    keep_m59: np.ndarray,
    santi_cohorts_and_retain: List[Tuple[LoadedCohort, List[int]]],
) -> pd.DataFrame:
    """Build per-subject manifest (490+ rows) with full provenance."""
    n_hist = hist.tensor.shape[0]
    n_m59_retained = int(keep_m59.sum())

    orig_within_source = list(range(n_hist))
    tensor_paths = [str(hist.spec.tensor_path)] * n_hist
    meta_paths = [str(hist.spec.metadata_path)] * n_hist

    m59_orig_idx = np.where(keep_m59)[0].tolist()
    orig_within_source += m59_orig_idx
    tensor_paths += [str(m59_original.spec.tensor_path)] * n_m59_retained
    meta_paths += [str(m59_original.spec.metadata_path)] * n_m59_retained

    for cohort, retain_idx in santi_cohorts_and_retain:
        orig_within_source += retain_idx
        tensor_paths += [str(cohort.spec.tensor_path)] * len(retain_idx)
        meta_paths += [str(cohort.spec.metadata_path)] * len(retain_idx)

    cols = [
        "SubjectID", "SourceCohort", "IsExpansionSubject", "IsMartin59",
        "IsSantiProcessed", "IsStressCandidate", "ResearchGroup_Mapped",
        "Manufacturer", "Site3", "Age", "Sex",
    ]
    avail = [c for c in cols if c in expanded_metadata.columns]
    mfst = expanded_metadata[avail].copy().reset_index(drop=True)
    mfst.insert(0, "RowIndex", np.arange(len(mfst), dtype=int))
    mfst["TensorSourcePath"] = tensor_paths
    mfst["MetadataSourcePath"] = meta_paths
    mfst["OriginalIndexWithinSource"] = orig_within_source
    mfst["Included"] = True
    mfst["ExclusionReason"] = ""
    return mfst


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

def final_internal_validate(
    tensor: np.ndarray,
    metadata: pd.DataFrame,
    subject_manifest: pd.DataFrame,
    npz_subject_ids: np.ndarray,
) -> None:
    errors: List[str] = []
    n_t, n_m, n_s = tensor.shape[0], len(metadata), len(subject_manifest)
    if n_t != n_m:
        errors.append(f"tensor N={n_t} != metadata N={n_m}")
    if n_t != n_s:
        errors.append(f"tensor N={n_t} != manifest N={n_s}")
    if tuple(tensor.shape[1:]) != EXPECTED_TENSOR_SHAPE_TAIL:
        errors.append(f"tensor tail {tensor.shape[1:]} != expected {EXPECTED_TENSOR_SHAPE_TAIL}")
    flat = tensor.reshape(n_t, -1)
    if np.isnan(flat).any():
        errors.append(f"tensor has {int(np.isnan(flat).sum())} NaNs")
    if np.isinf(flat).any():
        errors.append(f"tensor has {int(np.isinf(flat).sum())} Infs")
    meta_ids = metadata["SubjectID"].astype(str).tolist()
    if meta_ids != npz_subject_ids.astype(str).tolist():
        errors.append("SubjectID mismatch: metadata vs NPZ subject_ids")
    if "SubjectID" in subject_manifest.columns:
        if meta_ids != subject_manifest["SubjectID"].astype(str).tolist():
            errors.append("SubjectID mismatch: metadata vs subject_manifest")
    for col in ["SourceCohort", "IsExpansionSubject", "IsMartin59", "IsSantiProcessed", "IsStressCandidate"]:
        for src, label in [(metadata, "metadata"), (subject_manifest, "manifest")]:
            if col not in src.columns:
                errors.append(f"{label} missing column: {col}")
    if errors:
        raise RuntimeError("Internal validation failed:\n" + "\n".join(f"  - {e}" for e in errors))


def validate_existing_dataset(output_dir: Path) -> int:
    """Validate already-materialized v2 dataset without rebuilding."""
    tensor_path = output_dir / OUTPUT_TENSOR.name
    metadata_path = output_dir / OUTPUT_METADATA.name
    manifest_path = output_dir / OUTPUT_SUBJECT_MANIFEST.name

    for p, label in [(tensor_path, "tensor"), (metadata_path, "metadata")]:
        if not p.exists():
            print(f"ERROR: {label} not found: {p}")
            return 1

    print(f"Loading tensor from {tensor_path} ...")
    with np.load(tensor_path, allow_pickle=True) as npz:
        if "global_tensor_data" not in npz.files:
            print("ERROR: NPZ missing key: global_tensor_data")
            return 1
        tensor = np.asarray(npz["global_tensor_data"])
        npz_keys = list(npz.files)
        npz_sids = np.asarray(npz["subject_ids"]).astype(str) if "subject_ids" in npz.files else None

    meta = pd.read_csv(metadata_path)
    errors: List[str] = []

    if tensor.shape[0] != len(meta):
        errors.append(f"N mismatch: tensor={tensor.shape[0]}, metadata={len(meta)}")
    if tuple(tensor.shape[1:]) != EXPECTED_TENSOR_SHAPE_TAIL:
        errors.append(f"tensor tail {tensor.shape[1:]} != {EXPECTED_TENSOR_SHAPE_TAIL}")
    flat = tensor.reshape(tensor.shape[0], -1)
    if np.isnan(flat).any():
        errors.append(f"tensor has {int(np.isnan(flat).sum())} NaNs")
    if np.isinf(flat).any():
        errors.append(f"tensor has {int(np.isinf(flat).sum())} Infs")
    for col in ["SubjectID", "SourceCohort", "IsExpansionSubject", "IsMartin59",
                "IsSantiProcessed", "IsStressCandidate", "ResearchGroup_Mapped",
                "Manufacturer", "Site3", "Age", "Sex"]:
        if col not in meta.columns:
            errors.append(f"metadata missing column: {col}")
    if "SubjectID" in meta.columns and npz_sids is not None:
        if meta["SubjectID"].astype(str).tolist() != npz_sids.tolist():
            errors.append("SubjectID mismatch: metadata vs NPZ")

    if manifest_path.exists():
        sm = pd.read_csv(manifest_path)
        if len(sm) != tensor.shape[0]:
            errors.append(f"manifest rows={len(sm)}, tensor N={tensor.shape[0]}")
        for col in ["SubjectID", "SourceCohort", "IsMartin59", "IsSantiProcessed"]:
            if col not in sm.columns:
                errors.append(f"manifest missing column: {col}")
    else:
        print(f"WARNING: manifest not found: {manifest_path}")

    print(f"\nValidation: {output_dir}")
    print(f"  Tensor shape:  {tensor.shape}")
    print(f"  Metadata rows: {len(meta)}")
    print(f"  NPZ keys:      {npz_keys}")
    for col in ["SourceCohort", "ResearchGroup_Mapped", "Manufacturer"]:
        if col in meta.columns:
            print_count_summary(col, meta, [col])

    if errors:
        print("\nVALIDATION FAILED:")
        for e in errors:
            print(f"  ERROR: {e}")
        return 1
    print("\nVALIDATION PASSED: all checks OK.")
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# README and report
# ──────────────────────────────────────────────────────────────────────────────

def build_readme_v2(report: Dict, meta: pd.DataFrame) -> str:
    c = report["counts"]

    def vc(col: str) -> str:
        if col not in meta.columns:
            return "  (column not found)"
        rows = meta[col].value_counts().sort_values(ascending=False)
        return "\n".join(f"  - {k}: {v}" for k, v in rows.items())

    lines = [
        "# ADNI Expanded v2",
        "",
        "Dataset built for the BSPC 2026 revision.",
        "",
        "## Composition",
        "",
        "ADNI_expanded_v2 = historical_adni + martin59_arwsdcf + santi_processed_qc_pass",
        f"N = {c['expanded_tensor_subjects']}",
        f"Tensor shape: ({c['expanded_tensor_subjects']}, 7, 131, 131)",
        "",
        "### By SourceCohort",
        vc("SourceCohort"),
        "",
        "### By ResearchGroup_Mapped",
        vc("ResearchGroup_Mapped"),
        "",
        "### By Manufacturer",
        vc("Manufacturer"),
        "",
        f"Stress candidates (IsStressCandidate=True): {c['stress_candidates']}",
        f"  IsMartin59 retained: {c['martin59_retained']}",
        f"  IsSantiProcessed retained: {c['santi_retained_total']}",
        "",
        "## Sources",
        "",
        "- **historical_adni**: canonical ADNI fMRI cohort, N=" + str(c["historical_n"]) + ".",
        "- **martin59_arwsdcf**: 59 CN subjects from Martín Alcaraz, pipeline ARWSDCF.",
        f"  Retained: {c['martin59_retained']}, duplicates excluded: {c['martin59_dup_excluded']}.",
        "- **santi_processed**: subjects processed by Santiago Blas Laguzza.",
        "  Cohorts: siemens_available (Site3=941), ge_batch7 (Site3=135), ge_smoketest3 (Site3=135).",
        f"  Retained: {c['santi_retained_total']}, excluded: {c['santi_excluded_total']}.",
        "",
        "## Purpose",
        "",
        "1. Expanded retraining on full 505+ subject cohort.",
        "2. Stress testing: all IsExpansionSubject CN subjects as held-out FPR test.",
        "3. Comparison with v1 results.",
        "",
        "## Key flags",
        "",
        "- SourceCohort: 'historical_adni' | 'martin59_arwsdcf' | 'siemens_available' |",
        "  'ge_batch7' | 'ge_smoketest3'",
        "- IsExpansionSubject: True if not historical_adni",
        "- IsMartin59: True for martin59_arwsdcf subjects",
        "- IsSantiProcessed: True for siemens/GE subjects",
        "- IsStressCandidate: IsExpansionSubject AND ResearchGroup_Mapped == 'CN'",
        "",
        "## NPZ keys",
        "",
        "- global_tensor_data  (primary, used by betavae_xai.data.preprocessing.load_data())",
        "- tensor_data         (alias; stored as separate compressed block, ~doubles tensor storage)",
        "- subject_ids, channel_names, roi_names_in_order, network_labels_in_order",
        "- source_cohort, is_expansion_subject, is_martin59, is_santi_processed, is_stress_candidate",
        "",
        "## Files",
        "",
        "- GLOBAL_TENSOR_ADNI_expanded_v2.npz",
        "- subject_metadata_adni_expanded_v2.csv",
        "- expanded_subject_manifest.csv",
        "- expanded_dataset_summary.csv",
        "- qc_adni_expanded_v2.csv",
        "- excluded_subjects_adni_expanded_v2.csv",
        "- build_report.json",
        "",
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Write outputs
# ──────────────────────────────────────────────────────────────────────────────

def write_outputs_v2(
    output_dir: Path,
    hist: LoadedCohort,
    expanded_tensor: np.ndarray,
    expanded_metadata: pd.DataFrame,
    qc: pd.DataFrame,
    subject_manifest: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    excluded_df: pd.DataFrame,
    report: Dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = dict(hist.npz_payload)
    payload["global_tensor_data"] = expanded_tensor
    payload["tensor_data"] = expanded_tensor          # alias key as required
    payload["subject_ids"] = expanded_metadata["SubjectID"].to_numpy(dtype=str)
    payload["source_cohort"] = expanded_metadata["SourceCohort"].to_numpy(dtype=str)
    payload["is_expansion_subject"] = expanded_metadata["IsExpansionSubject"].to_numpy(dtype=bool)
    payload["is_martin59"] = expanded_metadata["IsMartin59"].to_numpy(dtype=bool)
    payload["is_santi_processed"] = expanded_metadata["IsSantiProcessed"].to_numpy(dtype=bool)
    payload["is_stress_candidate"] = expanded_metadata["IsStressCandidate"].to_numpy(dtype=bool)

    np.savez_compressed(output_dir / OUTPUT_TENSOR.name, **payload)
    expanded_metadata.to_csv(output_dir / OUTPUT_METADATA.name, index=False)
    qc.to_csv(output_dir / OUTPUT_QC.name, index=False)
    subject_manifest.to_csv(output_dir / OUTPUT_SUBJECT_MANIFEST.name, index=False)
    dataset_summary.to_csv(output_dir / OUTPUT_SUMMARY.name, index=False)
    excluded_df.to_csv(output_dir / OUTPUT_EXCLUDED.name, index=False)
    with (output_dir / OUTPUT_REPORT.name).open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=json_safe)
        f.write("\n")
    (output_dir / OUTPUT_README.name).write_text(
        build_readme_v2(report, expanded_metadata), encoding="utf-8"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    if args.validate_only:
        return validate_existing_dataset(args.output_dir)

    # Overwrite guard
    out_tensor = args.output_dir / OUTPUT_TENSOR.name
    if out_tensor.exists() and not args.overwrite and not args.dry_run:
        raise RuntimeError(
            f"Output tensor already exists: {out_tensor}\n"
            "Use --overwrite, --dry-run, or --validate-only."
        )

    # ── 1. Load base cohorts ──────────────────────────────────────────────────
    hist_tensor_path = resolve_historical_tensor(args)
    hist = load_base_cohort("historical", "historical_adni", hist_tensor_path,
                            args.historical_metadata_path)
    m59 = load_base_cohort("martin59", "martin59_arwsdcf",
                           args.martin59_tensor_path, args.martin59_metadata_path)
    validate_channel_compatibility(hist, m59)

    # ── 2. Deduplicate martin59 against historical ────────────────────────────
    hist_ids = set(hist.subject_ids.tolist())
    seen_ids: set = set(hist_ids)

    excluded: List[Dict] = []
    m59_dup = np.array([sid in hist_ids for sid in m59.subject_ids], dtype=bool)
    for sid in m59.subject_ids[m59_dup].tolist():
        excluded.append({
            "SubjectID": sid, "SourceCohort": "martin59_arwsdcf",
            "Reason": "duplicate_historical_adni",
            "TensorPath": str(m59.spec.tensor_path),
            "MetadataPath": str(m59.spec.metadata_path),
        })
    keep_m59 = ~m59_dup
    seen_ids |= set(m59.subject_ids[keep_m59].tolist())

    # ── 3. Build santi cohort specs list ─────────────────────────────────────
    santi_specs: List[SantiCohortSpec] = []
    if not args.skip_siemens:
        santi_specs.append(SantiCohortSpec(
            "siemens_available", "siemens_available",
            args.siemens_tensor_path, args.siemens_metadata_path,
        ))
    if not args.skip_ge_batch7:
        santi_specs.append(SantiCohortSpec(
            "ge_batch7", "ge_batch7",
            args.ge_batch7_tensor_path, args.ge_batch7_metadata_path,
        ))
    if not args.skip_ge_smoketest3:
        santi_specs.append(SantiCohortSpec(
            "ge_smoketest3", "ge_smoketest3",
            args.ge_smoketest3_tensor_path, args.ge_smoketest3_metadata_path,
        ))

    # CLI extras
    if len(args.extra_names) != len(args.extra_tensors) or len(args.extra_names) != len(args.extra_metadatas):
        raise RuntimeError(
            f"--extra-cohort-name ({len(args.extra_names)}), "
            f"--extra-global-tensor ({len(args.extra_tensors)}), "
            f"--extra-metadata ({len(args.extra_metadatas)}) must all have the same count."
        )
    for name, tp, mp in zip(args.extra_names, args.extra_tensors, args.extra_metadatas):
        santi_specs.append(SantiCohortSpec(name, name, Path(tp), Path(mp)))

    # Auto-discovery
    if not args.no_auto_discover:
        discovered = discover_santi_cohorts(args)
        known_names = {s.name for s in santi_specs}
        for spec in discovered:
            if spec.name not in known_names:
                santi_specs.append(spec)

    # ── 4. Load and process each santi cohort ────────────────────────────────
    santi_tensors: List[np.ndarray] = []
    santi_metas: List[pd.DataFrame] = []
    santi_cohorts_and_retain: List[Tuple[LoadedCohort, List[int]]] = []
    all_warnings: List[str] = []
    santi_counts: Dict[str, Dict] = {}

    for spec in santi_specs:
        if not spec.tensor_path.exists():
            print(f"WARNING: {spec.name} tensor not found: {spec.tensor_path}. Skipping.")
            continue
        if not spec.metadata_path.exists():
            print(f"WARNING: {spec.name} metadata not found: {spec.metadata_path}. Skipping.")
            continue

        cohort = load_santi_cohort(spec, hist)
        ret_tensor, ret_meta, retain_idx = process_santi_cohort(cohort, seen_ids, excluded)

        ret_meta = ret_meta.copy()
        ret_meta["SourceCohort"] = spec.source_cohort
        ret_meta["IsExpansionSubject"] = True
        ret_meta["IsMartin59"] = False
        ret_meta["IsSantiProcessed"] = True
        ret_meta["IsStressCandidate"] = ret_meta["ResearchGroup_Mapped"].astype(str).eq("CN")
        if "tensor_idx" in ret_meta.columns:
            ret_meta = ret_meta.drop(columns=["tensor_idx"])

        santi_tensors.append(ret_tensor)
        santi_metas.append(ret_meta)
        santi_cohorts_and_retain.append((cohort, retain_idx))
        santi_counts[spec.name] = {
            "total": cohort.tensor.shape[0],
            "retained": len(retain_idx),
            "excluded": cohort.tensor.shape[0] - len(retain_idx),
        }

    # ── 5. Align and flag base cohort metadata ────────────────────────────────
    hist_meta = hist.metadata.copy()
    hist_meta["SourceCohort"] = "historical_adni"
    hist_meta["IsExpansionSubject"] = False
    hist_meta["IsMartin59"] = False
    hist_meta["IsSantiProcessed"] = False
    hist_meta["IsStressCandidate"] = False

    m59_meta = m59.metadata.loc[keep_m59].reset_index(drop=True).copy()
    m59_meta["SourceCohort"] = "martin59_arwsdcf"
    m59_meta["IsExpansionSubject"] = True
    m59_meta["IsMartin59"] = True
    m59_meta["IsSantiProcessed"] = False
    m59_meta["IsStressCandidate"] = m59_meta["ResearchGroup_Mapped"].astype(str).eq("CN")
    if "tensor_idx" in m59_meta.columns:
        m59_meta = m59_meta.drop(columns=["tensor_idx"])

    # ── 6. Concatenate ────────────────────────────────────────────────────────
    m59_tensor_retained = m59.tensor[keep_m59]
    all_tensors = [hist.tensor, m59_tensor_retained] + santi_tensors
    all_metas = [hist_meta, m59_meta] + santi_metas
    expanded_tensor = np.concatenate(all_tensors, axis=0)
    expanded_metadata = pd.concat(all_metas, ignore_index=True)

    if "tensor_idx" in expanded_metadata.columns:
        raise RuntimeError("Expanded metadata unexpectedly contains tensor_idx.")

    exp_ids = expanded_metadata["SubjectID"].astype(str).tolist()
    if len(set(exp_ids)) != len(exp_ids):
        dupes = [s for s in set(exp_ids) if exp_ids.count(s) > 1]
        raise RuntimeError(f"Expanded dataset has duplicate SubjectIDs: {dupes[:10]}")

    # ── 7. Build manifest and run final validation ────────────────────────────
    subject_manifest = build_subject_manifest_v2(
        expanded_metadata, hist, m59, keep_m59, santi_cohorts_and_retain
    )
    npz_subject_ids = expanded_metadata["SubjectID"].to_numpy(str)
    final_internal_validate(expanded_tensor, expanded_metadata, subject_manifest, npz_subject_ids)

    qc = tensor_qc_v2(expanded_tensor, expanded_metadata)
    excluded_df = pd.DataFrame(excluded, columns=[
        "SubjectID", "SourceCohort", "Reason", "TensorPath", "MetadataPath"
    ]) if excluded else pd.DataFrame(
        columns=["SubjectID", "SourceCohort", "Reason", "TensorPath", "MetadataPath"]
    )

    # ── 8. Report ─────────────────────────────────────────────────────────────
    git_hash = get_git_hash()
    santi_retained_total = sum(v["retained"] for v in santi_counts.values())
    santi_excluded_total = sum(v["excluded"] for v in santi_counts.values())

    output_paths: Dict[str, Path] = {
        "historical_tensor": hist.spec.tensor_path,
        "historical_metadata": hist.spec.metadata_path,
        "martin59_tensor": m59.spec.tensor_path,
        "martin59_metadata": m59.spec.metadata_path,
        "expanded_tensor": args.output_dir / OUTPUT_TENSOR.name,
        "expanded_metadata": args.output_dir / OUTPUT_METADATA.name,
        "expanded_qc": args.output_dir / OUTPUT_QC.name,
        "expanded_subject_manifest": args.output_dir / OUTPUT_SUBJECT_MANIFEST.name,
        "expanded_dataset_summary": args.output_dir / OUTPUT_SUMMARY.name,
        "excluded_subjects": args.output_dir / OUTPUT_EXCLUDED.name,
        "build_report": args.output_dir / OUTPUT_REPORT.name,
        "readme": args.output_dir / OUTPUT_README.name,
    }
    for spec in santi_specs:
        output_paths[f"santi_{spec.name}_tensor"] = spec.tensor_path
        output_paths[f"santi_{spec.name}_metadata"] = spec.metadata_path

    summary_df = dataset_summary_rows(output_paths, dry_run=args.dry_run)

    npz_keys_out = sorted(
        set(hist.npz_payload.keys())
        | {"global_tensor_data", "tensor_data", "subject_ids", "source_cohort",
           "is_expansion_subject", "is_martin59", "is_santi_processed", "is_stress_candidate"}
    )

    report: Dict = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(args.dry_run),
        "git_hash": git_hash,
        "dataset_name": "ADNI_expanded_v2",
        "paths": {k: str(v) for k, v in output_paths.items()},
        "id_sources": {"historical": hist.id_source, "martin59": m59.id_source},
        "tensor_shapes": {
            "historical": list(hist.tensor.shape),
            "martin59": list(m59.tensor.shape),
            "expanded": list(expanded_tensor.shape),
        },
        "npz_keys": npz_keys_out,
        "counts": {
            "historical_n": int(hist.tensor.shape[0]),
            "martin59_total": int(m59.tensor.shape[0]),
            "martin59_retained": int(keep_m59.sum()),
            "martin59_dup_excluded": int((~keep_m59).sum()),
            "santi_cohorts": santi_counts,
            "santi_retained_total": santi_retained_total,
            "santi_excluded_total": santi_excluded_total,
            "expanded_tensor_subjects": int(expanded_tensor.shape[0]),
            "stress_candidates": int(expanded_metadata["IsStressCandidate"].sum()),
            "excluded_total": len(excluded_df),
        },
        "warnings": all_warnings,
        "counts_by_source_cohort": count_table(expanded_metadata, ["SourceCohort"]),
        "counts_by_research_group": count_table(expanded_metadata, ["ResearchGroup_Mapped"]),
        "counts_by_manufacturer": count_table(expanded_metadata, ["Manufacturer"]),
        "counts_by_site3": count_table(expanded_metadata, ["Site3"]),
    }

    # ── 9. Print summary ──────────────────────────────────────────────────────
    print("\nADNI_expanded_v2 build summary")
    print(f"Historical:        {hist.tensor.shape}")
    print(f"Martin59 total:    {m59.tensor.shape[0]}, retained: {int(keep_m59.sum())}")
    for spec in santi_specs:
        sc = santi_counts.get(spec.name, {})
        print(f"{spec.name}: total={sc.get('total','?')}, retained={sc.get('retained','?')}, excluded={sc.get('excluded','?')}")
    print(f"Expanded shape:    {expanded_tensor.shape}")
    print(f"Stress candidates: {int(expanded_metadata['IsStressCandidate'].sum())}")
    print(f"Excluded total:    {len(excluded_df)}")
    print(f"Git hash:          {git_hash}")

    print_count_summary("SourceCohort", expanded_metadata, ["SourceCohort"])
    print_count_summary("ResearchGroup_Mapped", expanded_metadata, ["ResearchGroup_Mapped"])
    print_count_summary("Manufacturer", expanded_metadata, ["Manufacturer"])

    if args.dry_run:
        print("\nDry-run complete. No files written.")
        return 0

    write_outputs_v2(
        args.output_dir, hist, expanded_tensor, expanded_metadata,
        qc, subject_manifest, summary_df, excluded_df, report,
    )
    print("\nFiles written:")
    for role, path in output_paths.items():
        if "expanded" in role or role in {"excluded_subjects", "build_report", "readme"}:
            print(f"  {role}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
