#!/usr/bin/env python3
"""Assemble ADNI v5.1 batch-20260514b final candidate dataset.

This is an assembly-only step:
- reuse v5.1_batch20260514;
- append only QC-pass, non-duplicate subjects from Martin bandpass batch 20260514b;
- do not use the quarantined v5.1_gecn9 branch;
- do not apply Python bandpass;
- do not train.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_NAME = "adni_expanded_v5_1_batch20260514b_no_pybandpass"
INCLUDED_DATASET_VERSION = "v5.1_batch20260514b"
SOURCE_BATCH = "20260514b_bandpass_batch3"
EXPECTED_CHANNELS = 7
EXPECTED_ROIS = 131

PAPER_ORIGINAL = {
    "version": "paper_original",
    "N_tensor": 431,
    "N_training_ready": 431,
    "AD": 95,
    "CN": 89,
    "MCI": 247,
    "Unknown": 0,
    "duplicate_subjects": 0,
}

DEFAULT_BASE_ROOT = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514_no_pybandpass"
)
DEFAULT_BASE_TENSOR = (
    DEFAULT_BASE_ROOT
    / "subject_tensors"
    / "GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514_no_pybandpass.npz"
)
DEFAULT_BASE_METADATA = DEFAULT_BASE_ROOT / "subject_metadata_v5_1_batch20260514_no_pybandpass.csv"
DEFAULT_BASE_TRAINING_METADATA = DEFAULT_BASE_ROOT / "training_ready_metadata_v5_1_batch20260514_no_pybandpass.csv"

DEFAULT_V5_BASE_ROOT = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_dparsf10000_no_pybandpass"
)
DEFAULT_V5_BASE_TENSOR = (
    DEFAULT_V5_BASE_ROOT
    / "subject_tensors"
    / "GLOBAL_TENSOR_ADNI_expanded_v5_dparsf10000_no_pybandpass.npz"
)
DEFAULT_V5_BASE_METADATA = DEFAULT_V5_BASE_ROOT / "subject_metadata_v5_dparsf10000_no_pybandpass.csv"
DEFAULT_V5_BASE_TRAINING_METADATA = DEFAULT_V5_BASE_ROOT / "training_ready_metadata_v5_dparsf10000_no_pybandpass.csv"

DEFAULT_INCREMENTAL_TENSOR = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_v5_1_batch20260514b_incremental_no_pybandpass/"
    "INCREMENTAL_TENSOR_MARTIN_BANDPASS_BATCH20260514B.npz"
)
DEFAULT_INCREMENTAL_METADATA = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_v5_1_batch20260514b_incremental_no_pybandpass/"
    "incremental_subject_metadata.csv"
)
DEFAULT_INCREMENTAL_QC = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_v5_1_batch20260514b_incremental_no_pybandpass/"
    "incremental_qc_summary.csv"
)
DEFAULT_BATCH_DECISION = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "martin_bandpass_batch_20260514b"
    / "batch_import_decision.csv"
)
DEFAULT_LEDGER = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "martin_bandpass_batch_20260514b"
    / "ADNI_v5_1_DATA_LEDGER_20260514b_candidate.csv"
)
DEFAULT_LEDGER_IMPORTED = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "martin_bandpass_batch_20260514b"
    / "ADNI_v5_1_DATA_LEDGER_20260514b_imported_candidate.csv"
)
DEFAULT_MASTER_MANIFEST = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_master_manifest"
    / "adni_v5_1_master_subject_manifest_first_visit.csv"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass"
)
DEFAULT_LOCAL_SYMLINK = (
    PROJECT_ROOT
    / "data"
    / "revision_bspc_2026"
    / "adni_expanded_v5_1_batch20260514b_no_pybandpass"
)
DEFAULT_QC_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_batch20260514b_full_build_qc"
)

FINAL_TENSOR_NAME = "GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ADNI v5.1 batch20260514b final candidate tensor and metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-tensor", type=Path, default=DEFAULT_BASE_TENSOR)
    parser.add_argument("--base-metadata", type=Path, default=DEFAULT_BASE_METADATA)
    parser.add_argument("--base-training-metadata", type=Path, default=DEFAULT_BASE_TRAINING_METADATA)
    parser.add_argument("--v5-base-tensor", type=Path, default=DEFAULT_V5_BASE_TENSOR)
    parser.add_argument("--v5-base-metadata", type=Path, default=DEFAULT_V5_BASE_METADATA)
    parser.add_argument("--v5-base-training-metadata", type=Path, default=DEFAULT_V5_BASE_TRAINING_METADATA)
    parser.add_argument("--incremental-tensor", type=Path, default=DEFAULT_INCREMENTAL_TENSOR)
    parser.add_argument("--incremental-metadata", type=Path, default=DEFAULT_INCREMENTAL_METADATA)
    parser.add_argument("--incremental-qc", type=Path, default=DEFAULT_INCREMENTAL_QC)
    parser.add_argument("--batch-decision", type=Path, default=DEFAULT_BATCH_DECISION)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--ledger-imported", type=Path, default=DEFAULT_LEDGER_IMPORTED)
    parser.add_argument("--master-manifest", type=Path, default=DEFAULT_MASTER_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--local-symlink", type=Path, default=DEFAULT_LOCAL_SYMLINK)
    parser.add_argument("--qc-dir", type=Path, default=DEFAULT_QC_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-symlink", action="store_true")
    parser.add_argument("--no-symlink", action="store_true")
    return parser.parse_args()


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def append_reason(current: str, reason: str) -> str:
    parts = [part for part in clean(current).split("|") if part]
    if reason and reason not in parts:
        parts.append(reason)
    return "|".join(parts)


def normalize_diagnosis(value: Any) -> str:
    text = clean(value).upper()
    if text in {"AD", "CN", "MCI"}:
        return text
    if "DEMENT" in text or "ALZ" in text:
        return "AD"
    if "CONTROL" in text or text == "NORMAL":
        return "CN"
    if "MCI" in text:
        return "MCI"
    return ""


def normalize_manufacturer(value: Any) -> str:
    text = clean(value).upper()
    if not text:
        return ""
    if "GE" in text:
        return "GE"
    if "PHILIPS" in text:
        return "Philips"
    if "SIEMENS" in text:
        return "SIEMENS"
    return clean(value)


def is_true(value: Any) -> bool:
    return clean(value).lower() in {"true", "1", "yes", "y"}


def ready_subset(metadata: pd.DataFrame) -> pd.DataFrame:
    if "training_ready" not in metadata.columns:
        return metadata.iloc[0:0].copy()
    return metadata[metadata["training_ready"].map(is_true)].copy()


def read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def prepare_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(f"Output directory exists and is not empty; pass --overwrite: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ensure_output_file(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise RuntimeError(f"Output file exists; pass --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def scalar_str(zf: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in zf.files:
        return default
    arr = zf[key]
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0])
    return "|".join(str(x) for x in arr.astype(str).reshape(-1).tolist())


def scalar_bool(zf: np.lib.npyio.NpzFile, key: str, default: bool = False) -> bool:
    if key not in zf.files:
        return default
    arr = zf[key]
    if arr.shape == ():
        return bool(arr.item())
    if arr.size == 1:
        return bool(arr.reshape(-1)[0])
    raise RuntimeError(f"Expected scalar bool for {key}")


def load_tensor_npz(path: Path, label: str) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as zf:
        required = ["global_tensor_data", "subject_ids", "channel_names", "python_bandpass_applied"]
        missing = [key for key in required if key not in zf.files]
        if missing:
            raise RuntimeError(f"{label} tensor missing required keys {missing}: {path}")
        tensor = zf["global_tensor_data"].astype(np.float32, copy=False)
        return {
            "label": label,
            "path": path,
            "tensor": tensor,
            "subject_ids": [str(x) for x in zf["subject_ids"].astype(str).tolist()],
            "channel_names": [str(x) for x in zf["channel_names"].astype(str).tolist()],
            "python_bandpass_applied": scalar_bool(zf, "python_bandpass_applied"),
            "rois_count": int(np.asarray(zf["rois_count"]).reshape(-1)[0]) if "rois_count" in zf.files else tensor.shape[-1],
            "target_len_ts": int(np.asarray(zf["target_len_ts"]).reshape(-1)[0]) if "target_len_ts" in zf.files else 140,
            "tr_seconds": float(np.asarray(zf["tr_seconds"]).reshape(-1)[0]) if "tr_seconds" in zf.files else 3.0,
            "preprocessing_source": scalar_str(zf, "preprocessing_source", "DPARSF_ROISignals_AAL3_10000"),
            "roi_order_name": scalar_str(zf, "roi_order_name", ""),
            "roi_names_in_order": zf["roi_names_in_order"].astype(str) if "roi_names_in_order" in zf.files else np.asarray([], dtype="U1"),
            "network_labels_in_order": zf["network_labels_in_order"].astype(str)
            if "network_labels_in_order" in zf.files
            else np.asarray([], dtype="U1"),
        }


def load_subject_ids(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as zf:
        if "subject_ids" not in zf.files:
            raise RuntimeError(f"Tensor missing subject_ids: {path}")
        return [str(x) for x in zf["subject_ids"].astype(str).tolist()]


def validate_single_tensor(payload: Dict[str, Any]) -> None:
    tensor = payload["tensor"]
    ids = payload["subject_ids"]
    label = payload["label"]
    if tensor.ndim != 4:
        raise RuntimeError(f"{label} tensor must be 4D, got {tensor.shape}")
    if tensor.shape[0] != len(ids):
        raise RuntimeError(f"{label} tensor first dimension does not match subject_ids")
    if tensor.shape[1:] != (EXPECTED_CHANNELS, EXPECTED_ROIS, EXPECTED_ROIS):
        raise RuntimeError(f"{label} tensor shape must be (*, 7, 131, 131), got {tensor.shape}")
    if tensor.dtype != np.float32:
        raise RuntimeError(f"{label} tensor dtype must be float32, got {tensor.dtype}")
    if np.isnan(tensor).any():
        raise RuntimeError(f"{label} tensor contains NaNs")
    if payload["python_bandpass_applied"]:
        raise RuntimeError(f"{label} has python_bandpass_applied=True")
    if len(ids) != len(set(ids)):
        duplicated = sorted({sid for sid in ids if ids.count(sid) > 1})
        raise RuntimeError(f"{label} has duplicate SubjectID values: {duplicated}")


def validate_tensor_compatibility(base: Dict[str, Any], incremental: Dict[str, Any]) -> None:
    validate_single_tensor(base)
    validate_single_tensor(incremental)
    if base["channel_names"] != incremental["channel_names"]:
        raise RuntimeError(f"channel_names mismatch: {base['channel_names']} vs {incremental['channel_names']}")
    if base["rois_count"] != EXPECTED_ROIS or incremental["rois_count"] != EXPECTED_ROIS:
        raise RuntimeError(f"rois_count mismatch: base={base['rois_count']} incremental={incremental['rois_count']}")
    if base["target_len_ts"] != incremental["target_len_ts"]:
        raise RuntimeError(f"target_len mismatch: base={base['target_len_ts']} incremental={incremental['target_len_ts']}")
    if abs(base["tr_seconds"] - incremental["tr_seconds"]) > 1e-6:
        raise RuntimeError(f"TR mismatch: base={base['tr_seconds']} incremental={incremental['tr_seconds']}")


def index_by_subject(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    if df.empty or "SubjectID" not in df.columns:
        return {}
    return df.drop_duplicates("SubjectID", keep="first").set_index("SubjectID", drop=False).to_dict(orient="index")


def first_value(sid: str, field: str, sources: Sequence[Tuple[str, Dict[str, Dict[str, Any]]]]) -> Tuple[str, str]:
    for label, indexed in sources:
        if sid not in indexed:
            continue
        value = clean(indexed[sid].get(field, ""))
        if value:
            return value, label
    return "", ""


def status_ok(value: Any) -> bool:
    return clean(value).lower() in {"ok", "yes", "true", "1", "qc_pass"}


def build_incremental_include_table(
    incremental_ids: Sequence[str],
    base_ids: Sequence[str],
    incremental_metadata: pd.DataFrame,
    batch_decision: pd.DataFrame,
    incremental_qc: pd.DataFrame,
) -> pd.DataFrame:
    meta = index_by_subject(incremental_metadata)
    decisions = index_by_subject(batch_decision)
    qc = index_by_subject(incremental_qc)
    base_id_set = set(base_ids)
    rows: List[Dict[str, Any]] = []
    for idx, sid in enumerate(incremental_ids):
        decision = decisions.get(sid, {})
        qc_row = qc.get(sid, {})
        import_ready = clean(decision.get("import_ready_new", "")).lower() == "yes"
        qc_status = clean(qc_row.get("status", "ok" if incremental_qc.empty else ""))
        qc_pass = status_ok(qc_status) if qc_status else False
        duplicate = sid in base_id_set
        reason = ""
        if sid not in meta:
            reason = append_reason(reason, "missing_incremental_metadata")
        if sid not in decisions:
            reason = append_reason(reason, "missing_batch_decision")
        if not import_ready:
            reason = append_reason(reason, f"import_ready_new_not_yes:{clean(decision.get('import_ready_new', 'missing'))}")
        if not incremental_qc.empty and not qc_pass:
            reason = append_reason(reason, f"incremental_qc_not_ok:{qc_status or 'missing'}")
        if duplicate:
            reason = append_reason(reason, "duplicate_with_v5_1_batch20260514_excluded")
        include = sid in meta and sid in decisions and import_ready and (qc_pass or incremental_qc.empty) and not duplicate
        rows.append(
            {
                "SubjectID": sid,
                "incremental_tensor_index": idx,
                "in_base_v5_1_batch20260514": duplicate,
                "in_incremental_metadata": sid in meta,
                "in_batch_decision": sid in decisions,
                "import_ready_new": clean(decision.get("import_ready_new", "")),
                "incremental_qc_status": qc_status,
                "incremental_qc_pass": bool(qc_pass or incremental_qc.empty),
                "include_in_final": bool(include),
                "exclusion_reason": reason,
                "selected_path": clean(decision.get("selected_path", meta.get(sid, {}).get("selected_path", ""))),
            }
        )
    return pd.DataFrame(rows)


def build_duplicate_report(include_table: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "SubjectID",
        "incremental_tensor_index",
        "in_base_v5_1_batch20260514",
        "in_incremental_metadata",
        "in_batch_decision",
        "import_ready_new",
        "incremental_qc_status",
        "include_in_final",
        "action",
        "reason",
        "selected_path",
    ]
    duplicates = include_table[include_table["in_base_v5_1_batch20260514"]].copy()
    if duplicates.empty:
        return pd.DataFrame(columns=columns)
    duplicates["action"] = "excluded_incremental_duplicate"
    duplicates["reason"] = duplicates["exclusion_reason"]
    return duplicates[columns]


def build_incremental_metadata_rows(
    included_ids: Sequence[str],
    start_index: int,
    incremental_metadata: pd.DataFrame,
    batch_decision: pd.DataFrame,
    ledger: pd.DataFrame,
    master_manifest: pd.DataFrame,
) -> pd.DataFrame:
    inc_meta = index_by_subject(incremental_metadata)
    decisions = index_by_subject(batch_decision)
    ledger_map = index_by_subject(ledger)
    master = index_by_subject(master_manifest)
    sources = [
        ("master_manifest", master),
        ("ledger", ledger_map),
        ("incremental_metadata", inc_meta),
        ("batch_decision", decisions),
    ]
    rows: List[Dict[str, Any]] = []
    for offset, sid in enumerate(included_ids):
        dx_raw, dx_source = first_value(sid, "ResearchGroup_Mapped", sources)
        if not dx_raw:
            dx_raw, dx_source = first_value(sid, "Diagnosis", sources)
        if not dx_raw:
            dx_raw, dx_source = first_value(sid, "master_diagnosis", sources)
        dx = normalize_diagnosis(dx_raw)
        age, age_source = first_value(sid, "Age", sources)
        sex, sex_source = first_value(sid, "Sex", sources)
        manufacturer_raw, man_source = first_value(sid, "Manufacturer", sources)
        if not manufacturer_raw:
            manufacturer_raw, man_source = first_value(sid, "master_manufacturer", sources)
        manufacturer = normalize_manufacturer(manufacturer_raw)
        image_id, image_source = first_value(sid, "ImageID", sources)
        if not image_id:
            image_id, image_source = first_value(sid, "master_image_id", sources)
        visit, visit_source = first_value(sid, "Visit", sources)
        if not visit:
            visit, visit_source = first_value(sid, "master_visit", sources)
        site3, site_source = first_value(sid, "Site3", sources)
        decision = decisions.get(sid, {})
        inc = inc_meta.get(sid, {})
        led = ledger_map.get(sid, {})
        exclusion_reason = ""
        training_ready = True
        if not dx:
            training_ready = False
            exclusion_reason = append_reason(exclusion_reason, "missing_diagnosis")
        if dx in {"AD", "CN"} and (not age or not sex):
            training_ready = False
            exclusion_reason = append_reason(exclusion_reason, "missing_age_or_sex_for_ad_cn")
        if clean(led.get("dicom_series_ok", "")).upper() == "NO":
            training_ready = False
            exclusion_reason = append_reason(exclusion_reason, "dicom_series_not_ok")
        selected_path = clean(decision.get("selected_path", inc.get("selected_path", "")))
        metadata_sources = sorted(
            {
                source
                for source in [dx_source, age_source, sex_source, man_source, image_source, visit_source, site_source]
                if source
            }
        )
        rows.append(
            {
                "SubjectID": sid,
                "tensor_index": start_index + offset,
                "tensor_source": "martin_bandpass_batch20260514b_incremental",
                "dataset_name": DATASET_NAME,
                "included_in_dataset_version": INCLUDED_DATASET_VERSION,
                "ResearchGroup_Mapped": dx,
                "Diagnosis": dx,
                "Age": age,
                "Sex": sex,
                "Manufacturer": manufacturer,
                "Site3": site3,
                "ImageID": image_id,
                "Visit": visit,
                "metadata_source": "|".join(metadata_sources),
                "source_label": "martin_bandpass_batch20260514b",
                "source_batch": SOURCE_BATCH,
                "roisignals_path": selected_path,
                "stage_guess": clean(decision.get("stage_guess", inc.get("stage_guess", ""))),
                "spectral_class": clean(decision.get("spectral_class", inc.get("spectral_class", ""))),
                "priority_batch": clean(led.get("priority_batch", inc.get("priority_batch", decision.get("priority_batch", "")))),
                "ledger_scope": clean(led.get("ledger_scope", inc.get("ledger_scope", decision.get("ledger_scope", "")))),
                "dicom_series_ok": clean(led.get("dicom_series_ok", inc.get("dicom_series_ok", decision.get("dicom_series_ok", "")))),
                "python_bandpass_requested": "NO",
                "python_bandpass_applied": False,
                "exclude_from_supervised": not training_ready,
                "supervised_exclusion_reason": exclusion_reason,
                "training_ready": bool(training_ready),
                "n_timepoints_raw": clean(decision.get("n_timepoints", inc.get("n_timepoints_raw", ""))),
                "n_rois_raw": clean(decision.get("n_rois", inc.get("n_rois_raw", ""))),
                "finite_fraction": clean(decision.get("finite_fraction", inc.get("finite_fraction", ""))),
                "scale_label": clean(decision.get("scale_label", inc.get("scale_label", ""))),
            }
        )
    return pd.DataFrame(rows)


def write_tensor_npz(
    output_root: Path,
    base: Dict[str, Any],
    incremental: Dict[str, Any],
    final_tensor: np.ndarray,
    final_subject_ids: Sequence[str],
) -> Path:
    tensor_dir = output_root / "subject_tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = tensor_dir / FINAL_TENSOR_NAME
    np.savez_compressed(
        tensor_path,
        global_tensor_data=final_tensor.astype(np.float32, copy=False),
        subject_ids=np.asarray(list(final_subject_ids), dtype="U32"),
        channel_names=np.asarray(base["channel_names"], dtype="U64"),
        rois_count=np.asarray(EXPECTED_ROIS, dtype=np.int32),
        target_len_ts=np.asarray(base["target_len_ts"], dtype=np.int32),
        tr_seconds=np.asarray(base["tr_seconds"], dtype=np.float32),
        python_bandpass_applied=np.asarray(False, dtype=np.bool_),
        preprocessing_source=np.asarray(base["preprocessing_source"], dtype="U96"),
        dataset_name=np.asarray(DATASET_NAME, dtype="U96"),
        source_global_tensors=np.asarray([str(base["path"]), str(incremental["path"])], dtype="U256"),
        uses_v5_1_gecn9=np.asarray(False, dtype=np.bool_),
        assembled_with_v5_1_batch20260514=np.asarray(True, dtype=np.bool_),
        source_batch=np.asarray(SOURCE_BATCH, dtype="U32"),
        roi_order_name=np.asarray(base["roi_order_name"], dtype="U96"),
        roi_names_in_order=np.asarray(base["roi_names_in_order"], dtype="U128"),
        network_labels_in_order=np.asarray(base["network_labels_in_order"], dtype="U64"),
    )
    return tensor_path


def create_symlink(link: Path, target: Path, overwrite: bool) -> str:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        current = link.resolve()
        if current == target.resolve():
            return "already_correct"
        if not overwrite:
            return f"existing_symlink_points_elsewhere:{current}"
        link.unlink()
    elif link.exists():
        return "existing_non_symlink_not_modified"
    link.symlink_to(target, target_is_directory=True)
    return "created"


def make_subject_alignment(
    final_subject_ids: Sequence[str],
    base_ids: Sequence[str],
    incremental_ids: Sequence[str],
    include_table: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    meta = metadata.set_index("SubjectID", drop=False).to_dict(orient="index")
    base_pos = {sid: idx for idx, sid in enumerate(base_ids)}
    inc_pos = {sid: idx for idx, sid in enumerate(incremental_ids)}
    include_map = include_table.set_index("SubjectID", drop=False).to_dict(orient="index")
    rows: List[Dict[str, Any]] = []
    final_set = set(final_subject_ids)
    for final_idx, sid in enumerate(final_subject_ids):
        m = meta.get(sid, {})
        rows.append(
            {
                "SubjectID": sid,
                "final_tensor_index": final_idx,
                "base_v5_1_batch20260514_tensor_index": base_pos.get(sid, ""),
                "incremental_batch20260514b_tensor_index": inc_pos.get(sid, ""),
                "tensor_source": m.get("tensor_source", ""),
                "included_in_final": True,
                "exclusion_reason": "",
                "ResearchGroup_Mapped": m.get("ResearchGroup_Mapped", ""),
                "Age": m.get("Age", ""),
                "Sex": m.get("Sex", ""),
                "Manufacturer": m.get("Manufacturer", ""),
                "ImageID": m.get("ImageID", ""),
                "Visit": m.get("Visit", ""),
                "training_ready": m.get("training_ready", ""),
            }
        )
    for sid, row in include_map.items():
        if sid in final_set:
            continue
        rows.append(
            {
                "SubjectID": sid,
                "final_tensor_index": "",
                "base_v5_1_batch20260514_tensor_index": base_pos.get(sid, ""),
                "incremental_batch20260514b_tensor_index": row.get("incremental_tensor_index", ""),
                "tensor_source": "incremental_excluded",
                "included_in_final": False,
                "exclusion_reason": row.get("exclusion_reason", ""),
                "ResearchGroup_Mapped": "",
                "Age": "",
                "Sex": "",
                "Manufacturer": "",
                "ImageID": "",
                "Visit": "",
                "training_ready": False,
            }
        )
    return pd.DataFrame(rows)


def diagnosis_counts(metadata: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for scope, sub in [("all_tensor_subjects", metadata), ("training_ready", ready_subset(metadata))]:
        counts = sub["ResearchGroup_Mapped"].replace("", "UNKNOWN").value_counts(dropna=False).to_dict()
        for diagnosis in ["AD", "CN", "MCI", "UNKNOWN"]:
            rows.append({"scope": scope, "Diagnosis": diagnosis, "n": int(counts.get(diagnosis, 0))})
        rows.append({"scope": scope, "Diagnosis": "TOTAL", "n": int(len(sub))})
    return pd.DataFrame(rows)


def diagnosis_x_manufacturer(metadata: pd.DataFrame) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for scope, sub in [("all_tensor_subjects", metadata), ("training_ready", ready_subset(metadata))]:
        table = (
            sub.assign(
                ResearchGroup_Mapped=sub["ResearchGroup_Mapped"].replace("", "UNKNOWN"),
                Manufacturer=sub["Manufacturer"].replace("", "UNKNOWN"),
            )
            .groupby(["ResearchGroup_Mapped", "Manufacturer"], dropna=False)
            .size()
            .reset_index(name="n")
            .sort_values(["ResearchGroup_Mapped", "Manufacturer"])
        )
        table.insert(0, "scope", scope)
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["scope", "ResearchGroup_Mapped", "Manufacturer", "n"])


EVOLUTION_COLUMNS = [
    "version",
    "N_tensor",
    "N_training_ready",
    "AD",
    "CN",
    "MCI",
    "Unknown",
    "CN_GE",
    "CN_Siemens",
    "CN_Philips",
    "AD_GE",
    "AD_Siemens",
    "AD_Philips",
    "MCI_GE",
    "MCI_Siemens",
    "MCI_Philips",
    "AgeSex_missing_CNAD",
    "duplicate_subjects",
]


def evolution_row(version: str, metadata: pd.DataFrame, training: pd.DataFrame) -> Dict[str, Any]:
    train = training.copy()
    train["ResearchGroup_Mapped"] = train["ResearchGroup_Mapped"].map(normalize_diagnosis)
    train["Manufacturer"] = train["Manufacturer"].map(normalize_manufacturer)
    row: Dict[str, Any] = {
        "version": version,
        "N_tensor": len(metadata),
        "N_training_ready": len(train),
        "AD": int(train["ResearchGroup_Mapped"].eq("AD").sum()),
        "CN": int(train["ResearchGroup_Mapped"].eq("CN").sum()),
        "MCI": int(train["ResearchGroup_Mapped"].eq("MCI").sum()),
        "Unknown": int(train["ResearchGroup_Mapped"].eq("").sum()),
        "AgeSex_missing_CNAD": int(
            (
                train["ResearchGroup_Mapped"].isin(["AD", "CN"])
                & (train["Age"].map(clean).eq("") | train["Sex"].map(clean).eq(""))
            ).sum()
        ),
        "duplicate_subjects": int(metadata["SubjectID"].duplicated().sum()) if "SubjectID" in metadata.columns else 0,
    }
    for dx in ["CN", "AD", "MCI"]:
        for manufacturer in ["GE", "SIEMENS", "Philips"]:
            col = f"{dx}_{'Siemens' if manufacturer == 'SIEMENS' else manufacturer}"
            row[col] = int((train["ResearchGroup_Mapped"].eq(dx) & train["Manufacturer"].eq(manufacturer)).sum())
    return row


def paper_original_row() -> Dict[str, Any]:
    row = {col: "" for col in EVOLUTION_COLUMNS}
    row.update(PAPER_ORIGINAL)
    return row


def dataset_evolution_summary(
    v5_meta: pd.DataFrame,
    v5_train: pd.DataFrame,
    base_meta: pd.DataFrame,
    base_train: pd.DataFrame,
    final_meta: pd.DataFrame,
    final_train: pd.DataFrame,
) -> pd.DataFrame:
    batch20260513_meta = base_meta[
        ~base_meta["tensor_source"].eq("martin_bandpass_batch20260514_incremental")
    ].copy()
    batch20260513_train = ready_subset(batch20260513_meta)
    rows = [
        paper_original_row(),
        evolution_row("v5_base", v5_meta, v5_train),
        evolution_row("v5_1_batch20260513", batch20260513_meta, batch20260513_train),
        evolution_row("v5_1_batch20260514", base_meta, base_train),
        evolution_row("v5_1_batch20260514b", final_meta, final_train),
    ]
    return pd.DataFrame(rows)[EVOLUTION_COLUMNS]


ADDED_COLUMNS = [
    "step",
    "n_added",
    "AD_added",
    "CN_added",
    "MCI_added",
    "GE_added",
    "Siemens_added",
    "Philips_added",
    "CN_GE_added",
    "CN_Siemens_added",
    "CN_Philips_added",
]


def added_step_row(step: str, added: pd.DataFrame) -> Dict[str, Any]:
    df = added.copy()
    df["ResearchGroup_Mapped"] = df["ResearchGroup_Mapped"].map(normalize_diagnosis)
    df["Manufacturer"] = df["Manufacturer"].map(normalize_manufacturer)
    return {
        "step": step,
        "n_added": len(df),
        "AD_added": int(df["ResearchGroup_Mapped"].eq("AD").sum()),
        "CN_added": int(df["ResearchGroup_Mapped"].eq("CN").sum()),
        "MCI_added": int(df["ResearchGroup_Mapped"].eq("MCI").sum()),
        "GE_added": int(df["Manufacturer"].eq("GE").sum()),
        "Siemens_added": int(df["Manufacturer"].eq("SIEMENS").sum()),
        "Philips_added": int(df["Manufacturer"].eq("Philips").sum()),
        "CN_GE_added": int((df["ResearchGroup_Mapped"].eq("CN") & df["Manufacturer"].eq("GE")).sum()),
        "CN_Siemens_added": int((df["ResearchGroup_Mapped"].eq("CN") & df["Manufacturer"].eq("SIEMENS")).sum()),
        "CN_Philips_added": int((df["ResearchGroup_Mapped"].eq("CN") & df["Manufacturer"].eq("Philips")).sum()),
    }


def added_subjects_by_step_summary(base_meta: pd.DataFrame, added_20260514b: pd.DataFrame) -> pd.DataFrame:
    added_20260513 = base_meta[base_meta["tensor_source"].eq("martin_bandpass_batch20260513_incremental")].copy()
    added_20260514 = base_meta[base_meta["tensor_source"].eq("martin_bandpass_batch20260514_incremental")].copy()
    rows = [
        added_step_row("batch20260513 added", added_20260513),
        added_step_row("batch20260514 added", added_20260514),
        added_step_row("batch20260514b added", added_20260514b),
    ]
    return pd.DataFrame(rows)[ADDED_COLUMNS]


def filter_metadata_to_subject_ids(metadata: pd.DataFrame, subject_ids: Sequence[str], label: str) -> pd.DataFrame:
    indexed = index_by_subject(metadata)
    missing = [sid for sid in subject_ids if sid not in indexed]
    if missing:
        raise RuntimeError(f"{label} metadata missing tensor subjects: {missing[:20]}")
    return pd.DataFrame([indexed[sid] for sid in subject_ids]).reset_index(drop=True)


def update_ledger_candidate(
    ledger: pd.DataFrame,
    included_incremental_metadata: pd.DataFrame,
    batch_decision: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    updated = ledger.copy()
    decisions = index_by_subject(batch_decision)
    included_ids = included_incremental_metadata["SubjectID"].astype(str).tolist()
    for col in [
        "roisignals_status",
        "processing_status",
        "uploaded_batch",
        "uploaded_path",
        "included_in_dataset_version",
        "python_bandpass_requested",
    ]:
        if col not in updated.columns:
            updated[col] = ""
    missing_in_ledger = []
    for sid in included_ids:
        if not updated["SubjectID"].eq(sid).any():
            missing_in_ledger.append(sid)
            continue
        selected_path = clean(decisions.get(sid, {}).get("selected_path", ""))
        row_idx = updated.index[updated["SubjectID"].eq(sid)]
        updated.loc[row_idx, "roisignals_status"] = "qc_pass"
        updated.loc[row_idx, "processing_status"] = "imported"
        updated.loc[row_idx, "uploaded_batch"] = SOURCE_BATCH
        updated.loc[row_idx, "uploaded_path"] = selected_path
        updated.loc[row_idx, "included_in_dataset_version"] = INCLUDED_DATASET_VERSION
        updated.loc[row_idx, "python_bandpass_requested"] = "NO"
    rows = [
        {"metric": "ledger_rows_before", "value": len(ledger)},
        {"metric": "ledger_rows_after", "value": len(updated)},
        {"metric": "included_incremental_subjects", "value": len(included_ids)},
        {"metric": "included_incremental_missing_in_ledger", "value": len(missing_in_ledger)},
        {"metric": "ledger_imported_rows_after", "value": int(updated["processing_status"].eq("imported").sum())},
        {"metric": "ledger_excluded_rows_after", "value": int(updated["processing_status"].eq("excluded").sum())},
        {"metric": "python_bandpass_requested_non_no_after", "value": int((updated["python_bandpass_requested"] != "NO").sum())},
        {"metric": "missing_in_ledger_subjects", "value": "|".join(missing_in_ledger)},
    ]
    return updated, pd.DataFrame(rows)


def remaining_ledger_pending_subjects(ledger: pd.DataFrame) -> pd.DataFrame:
    out = ledger.copy()
    for col in [
        "SubjectID",
        "ledger_scope",
        "priority_batch",
        "dicom_series_ok",
        "roisignals_status",
        "processing_status",
        "action_needed",
        "included_in_dataset_version",
        "uploaded_batch",
        "uploaded_path",
        "notes_santiago",
    ]:
        if col not in out.columns:
            out[col] = ""
    status = out["processing_status"].map(lambda x: clean(x).lower())
    remaining = out[status.isin({"pending", "excluded"})].copy()
    cols = [
        "SubjectID",
        "ledger_scope",
        "priority_batch",
        "dicom_series_ok",
        "roisignals_status",
        "processing_status",
        "action_needed",
        "included_in_dataset_version",
        "uploaded_batch",
        "uploaded_path",
        "notes_santiago",
    ]
    return remaining[cols].sort_values(["processing_status", "SubjectID"]).reset_index(drop=True)


def write_readme(
    qc_dir: Path,
    output_root: Path,
    tensor_path: Path,
    symlink_status: str,
    base_n: int,
    incremental_n: int,
    included_incremental_n: int,
    duplicate_n: int,
    final_tensor: np.ndarray,
    metadata: pd.DataFrame,
    evolution: pd.DataFrame,
    added_summary: pd.DataFrame,
    remaining_ledger: pd.DataFrame,
) -> None:
    training_ready = ready_subset(metadata)
    cn_ge = training_ready[
        training_ready["ResearchGroup_Mapped"].eq("CN") & training_ready["Manufacturer"].eq("GE")
    ]
    ad_cn_ready = training_ready[training_ready["ResearchGroup_Mapped"].isin(["AD", "CN"])]
    missing_covars = ad_cn_ready[ad_cn_ready["Age"].map(clean).eq("") | ad_cn_ready["Sex"].map(clean).eq("")]
    baseline_ready = (
        tuple(final_tensor.shape[1:]) == (EXPECTED_CHANNELS, EXPECTED_ROIS, EXPECTED_ROIS)
        and int(np.isnan(final_tensor).sum()) == 0
        and missing_covars.empty
        and not bool(metadata["SubjectID"].duplicated().any())
    )
    added_20260513 = int(
        added_summary.loc[added_summary["step"].eq("batch20260513 added"), "n_added"].iloc[0]
    )
    added_20260514 = int(
        added_summary.loc[added_summary["step"].eq("batch20260514 added"), "n_added"].iloc[0]
    )
    added_20260514b = int(
        added_summary.loc[added_summary["step"].eq("batch20260514b added"), "n_added"].iloc[0]
    )
    final_ad = int(evolution.loc[evolution["version"].eq("v5_1_batch20260514b"), "AD"].iloc[0])
    final_cn = int(evolution.loc[evolution["version"].eq("v5_1_batch20260514b"), "CN"].iloc[0])
    final_mci = int(evolution.loc[evolution["version"].eq("v5_1_batch20260514b"), "MCI"].iloc[0])
    final_cn_ge = int(evolution.loc[evolution["version"].eq("v5_1_batch20260514b"), "CN_GE"].iloc[0])
    final_cn_siemens = int(evolution.loc[evolution["version"].eq("v5_1_batch20260514b"), "CN_Siemens"].iloc[0])
    final_cn_philips = int(evolution.loc[evolution["version"].eq("v5_1_batch20260514b"), "CN_Philips"].iloc[0])
    final_ad_ge = int(evolution.loc[evolution["version"].eq("v5_1_batch20260514b"), "AD_GE"].iloc[0])
    final_ad_siemens = int(evolution.loc[evolution["version"].eq("v5_1_batch20260514b"), "AD_Siemens"].iloc[0])
    final_ad_philips = int(evolution.loc[evolution["version"].eq("v5_1_batch20260514b"), "AD_Philips"].iloc[0])
    final_mci_ge = int(evolution.loc[evolution["version"].eq("v5_1_batch20260514b"), "MCI_GE"].iloc[0])
    final_mci_siemens = int(evolution.loc[evolution["version"].eq("v5_1_batch20260514b"), "MCI_Siemens"].iloc[0])
    final_mci_philips = int(evolution.loc[evolution["version"].eq("v5_1_batch20260514b"), "MCI_Philips"].iloc[0])
    remaining_pending = int(remaining_ledger["processing_status"].map(lambda x: clean(x).lower()).eq("pending").sum())
    remaining_excluded = int(remaining_ledger["processing_status"].map(lambda x: clean(x).lower()).eq("excluded").sum())
    paper_ad = int(PAPER_ORIGINAL["AD"])
    ad_new_net = final_ad - paper_ad
    evolution_text = "```text\n" + evolution.to_string(index=False) + "\n```"
    added_text = "```text\n" + added_summary.to_string(index=False) + "\n```"
    lines = [
        "# ADNI v5.1 Batch 20260514b No-Python-Bandpass Full Build QC",
        "",
        f"Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "Assembly-only candidate build. v5.1_batch20260514 was reused and Martin batch 20260514b was appended only for QC-pass, non-duplicate subjects.",
        "",
        "## Explicit Answers",
        "",
        f"- Paper original subjects: `{PAPER_ORIGINAL['N_tensor']}`.",
        f"- v5.1_batch20260514 subjects: `{base_n}`.",
        f"- batch20260513 added: `{added_20260513}`.",
        f"- batch20260514 added: `{added_20260514}`.",
        f"- batch20260514b incremental subjects: `{incremental_n}`.",
        f"- batch20260514b actually added: `{included_incremental_n}` from summary `{added_20260514b}`.",
        f"- Incremental duplicates excluded: `{duplicate_n}`.",
        f"- Final N: `{final_tensor.shape[0]}`.",
        f"- Final training-ready N: `{len(training_ready)}`.",
        f"- Final AD/CN/MCI: AD=`{final_ad}`, CN=`{final_cn}`, MCI=`{final_mci}`.",
        f"- Final CN by manufacturer: GE=`{final_cn_ge}`, Siemens=`{final_cn_siemens}`, Philips=`{final_cn_philips}`.",
        f"- Final AD by manufacturer: GE=`{final_ad_ge}`, Siemens=`{final_ad_siemens}`, Philips=`{final_ad_philips}`.",
        f"- Final MCI by manufacturer: GE=`{final_mci_ge}`, Siemens=`{final_mci_siemens}`, Philips=`{final_mci_philips}`.",
        f"- Final tensor shape: `{tuple(int(x) for x in final_tensor.shape)}`.",
        f"- Final tensor dtype: `{final_tensor.dtype}`.",
        f"- Final tensor NaNs: `{int(np.isnan(final_tensor).sum())}`.",
        f"- Output root: `{output_root}`.",
        f"- Tensor path: `{tensor_path}`.",
        f"- Local symlink status: `{symlink_status}`.",
        f"- Net new AD versus paper original: `{ad_new_net}`.",
        f"- CN-GE incorporated in final training-ready metadata: `{len(cn_ge)}`.",
        f"- AD/CN training-ready rows missing Age or Sex: `{len(missing_covars)}`.",
        f"- Duplicated subjects in final metadata: `{int(metadata['SubjectID'].duplicated().sum())}`.",
        f"- Ledger rows still pending: `{remaining_pending}`.",
        f"- Ledger rows excluded: `{remaining_excluded}`.",
        f"- Ready to train baseline `[1,0,2]`: `{baseline_ready}`.",
        "- Python bandpass applied/requested in final path: `False` / `NO`.",
        "- v5.1_gecn9 used: `False`; it remains quarantine.",
        "- Training run: `False`.",
        "",
        "## Dataset Evolution",
        "",
        evolution_text,
        "",
        "## Added Subjects By Step",
        "",
        added_text,
        "",
        "## Files",
        "",
        "- Output dataset:",
        f"  - `{output_root / 'subject_tensors' / FINAL_TENSOR_NAME}`",
        f"  - `{output_root / 'subject_metadata_v5_1_batch20260514b_no_pybandpass.csv'}`",
        f"  - `{output_root / 'training_ready_metadata_v5_1_batch20260514b_no_pybandpass.csv'}`",
        "- QC:",
        "  - `subject_alignment.csv`",
        "  - `duplicate_report.csv`",
        "  - `diagnosis_counts.csv`",
        "  - `diagnosis_x_manufacturer.csv`",
        "  - `added_subjects_batch20260514b.csv`",
        "  - `dataset_evolution_summary.csv`",
        "  - `added_subjects_by_step_summary.csv`",
        "  - `remaining_ledger_pending_subjects.csv`",
        "  - `ledger_update_summary.csv`",
        "  - `incremental_inclusion_decision.csv`",
        "",
        "## Note On Paper Original Manufacturer Cells",
        "",
        "The paper original row uses the provided diagnosis totals. Diagnosis-by-manufacturer cells are left blank because only total manufacturer counts were provided here, not per-diagnosis manufacturer counts.",
        "",
        "## Next Step",
        "",
        "Review this candidate build before launching any baseline training. No training was executed by this script.",
    ]
    (qc_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    prepare_dir(args.output_root, args.overwrite)
    prepare_dir(args.qc_dir, args.overwrite)
    ensure_output_file(args.ledger_imported, args.overwrite)

    base = load_tensor_npz(args.base_tensor, "v5_1_batch20260514")
    incremental = load_tensor_npz(args.incremental_tensor, "incremental_batch20260514b")
    validate_tensor_compatibility(base, incremental)
    v5_base_ids = load_subject_ids(args.v5_base_tensor)

    base_ids = base["subject_ids"]
    incremental_ids = incremental["subject_ids"]
    base_metadata = read_csv(args.base_metadata)
    base_training = read_csv(args.base_training_metadata)
    v5_base_metadata = read_csv(args.v5_base_metadata)
    v5_base_training = read_csv(args.v5_base_training_metadata)
    incremental_metadata = read_csv(args.incremental_metadata)
    incremental_qc = read_csv(args.incremental_qc, required=False)
    batch_decision = read_csv(args.batch_decision)
    ledger = read_csv(args.ledger)
    master_manifest = read_csv(args.master_manifest)

    if base_metadata["SubjectID"].astype(str).tolist() != base_ids:
        raise RuntimeError("Base metadata SubjectID order does not match v5.1_batch20260514 tensor subject_ids")
    v5_base_metadata = filter_metadata_to_subject_ids(v5_base_metadata, v5_base_ids, "v5_base")

    include_table = build_incremental_include_table(
        incremental_ids=incremental_ids,
        base_ids=base_ids,
        incremental_metadata=incremental_metadata,
        batch_decision=batch_decision,
        incremental_qc=incremental_qc,
    )
    duplicate_report = build_duplicate_report(include_table)
    included_incremental_ids = include_table.loc[include_table["include_in_final"], "SubjectID"].astype(str).tolist()
    included_incremental_indices = include_table.loc[include_table["include_in_final"], "incremental_tensor_index"].astype(int).tolist()

    final_subject_ids = list(base_ids) + included_incremental_ids
    final_tensor = np.concatenate(
        [base["tensor"], incremental["tensor"][included_incremental_indices]],
        axis=0,
    ).astype(np.float32, copy=False)
    if final_tensor.shape[0] != len(final_subject_ids):
        raise RuntimeError("Final tensor first dimension does not match final subject_ids")
    if len(final_subject_ids) != len(set(final_subject_ids)):
        duplicated = sorted({sid for sid in final_subject_ids if final_subject_ids.count(sid) > 1})
        raise RuntimeError(f"Final tensor would contain duplicated SubjectID values: {duplicated}")
    if np.isnan(final_tensor).any():
        raise RuntimeError("Final tensor contains NaNs after assembly")

    tensor_path = write_tensor_npz(args.output_root, base, incremental, final_tensor, final_subject_ids)

    base_metadata = base_metadata.copy()
    base_metadata["dataset_name"] = DATASET_NAME
    base_metadata["included_in_dataset_version"] = INCLUDED_DATASET_VERSION
    base_metadata["tensor_index"] = np.arange(len(base_metadata))
    incremental_final_metadata = build_incremental_metadata_rows(
        included_incremental_ids,
        start_index=len(base_ids),
        incremental_metadata=incremental_metadata,
        batch_decision=batch_decision,
        ledger=ledger,
        master_manifest=master_manifest,
    )
    metadata = pd.concat([base_metadata, incremental_final_metadata], ignore_index=True)
    if metadata["SubjectID"].tolist() != final_subject_ids:
        raise RuntimeError("Metadata SubjectID order does not match final tensor subject_ids")
    metadata["training_ready"] = metadata["training_ready"].map(is_true)
    training_ready = ready_subset(metadata)

    metadata_path = args.output_root / "subject_metadata_v5_1_batch20260514b_no_pybandpass.csv"
    training_path = args.output_root / "training_ready_metadata_v5_1_batch20260514b_no_pybandpass.csv"
    metadata.to_csv(metadata_path, index=False)
    training_ready.to_csv(training_path, index=False)

    ledger_imported, ledger_summary = update_ledger_candidate(ledger, incremental_final_metadata, batch_decision)
    ledger_imported.to_csv(args.ledger_imported, index=False)
    remaining_ledger = remaining_ledger_pending_subjects(ledger_imported)

    symlink_status = "skipped_by_user"
    if not args.no_symlink:
        symlink_status = create_symlink(args.local_symlink, args.output_root, args.overwrite_symlink)

    alignment = make_subject_alignment(final_subject_ids, base_ids, incremental_ids, include_table, metadata)
    dx_counts = diagnosis_counts(metadata)
    dx_man = diagnosis_x_manufacturer(metadata)
    added_subjects = incremental_final_metadata.copy()
    evolution = dataset_evolution_summary(
        v5_meta=v5_base_metadata,
        v5_train=v5_base_training,
        base_meta=base_metadata,
        base_train=base_training,
        final_meta=metadata,
        final_train=training_ready,
    )
    added_summary = added_subjects_by_step_summary(base_metadata, added_subjects)

    alignment.to_csv(args.qc_dir / "subject_alignment.csv", index=False)
    duplicate_report.to_csv(args.qc_dir / "duplicate_report.csv", index=False)
    dx_counts.to_csv(args.qc_dir / "diagnosis_counts.csv", index=False)
    dx_man.to_csv(args.qc_dir / "diagnosis_x_manufacturer.csv", index=False)
    added_subjects.to_csv(args.qc_dir / "added_subjects_batch20260514b.csv", index=False)
    evolution.to_csv(args.qc_dir / "dataset_evolution_summary.csv", index=False)
    added_summary.to_csv(args.qc_dir / "added_subjects_by_step_summary.csv", index=False)
    remaining_ledger.to_csv(args.qc_dir / "remaining_ledger_pending_subjects.csv", index=False)
    ledger_summary.to_csv(args.qc_dir / "ledger_update_summary.csv", index=False)
    include_table.to_csv(args.qc_dir / "incremental_inclusion_decision.csv", index=False)

    command_log = {
        "script": str(Path(__file__).resolve()),
        "generated": datetime.now().isoformat(timespec="seconds"),
        "base_tensor": str(args.base_tensor),
        "base_metadata": str(args.base_metadata),
        "base_training_metadata": str(args.base_training_metadata),
        "v5_base_tensor": str(args.v5_base_tensor),
        "v5_base_metadata": str(args.v5_base_metadata),
        "v5_base_training_metadata": str(args.v5_base_training_metadata),
        "incremental_tensor": str(args.incremental_tensor),
        "incremental_metadata": str(args.incremental_metadata),
        "incremental_qc": str(args.incremental_qc),
        "batch_decision": str(args.batch_decision),
        "ledger": str(args.ledger),
        "ledger_imported": str(args.ledger_imported),
        "master_manifest": str(args.master_manifest),
        "output_root": str(args.output_root),
        "local_symlink": str(args.local_symlink),
        "qc_dir": str(args.qc_dir),
        "base_subjects": len(base_ids),
        "incremental_subjects": len(incremental_ids),
        "incremental_added": len(included_incremental_ids),
        "incremental_duplicates_excluded": int(include_table["in_base_v5_1_batch20260514"].sum()),
        "remaining_ledger_pending_rows": int(
            remaining_ledger["processing_status"].map(lambda x: clean(x).lower()).eq("pending").sum()
        ),
        "remaining_ledger_excluded_rows": int(
            remaining_ledger["processing_status"].map(lambda x: clean(x).lower()).eq("excluded").sum()
        ),
        "final_shape": tuple(int(x) for x in final_tensor.shape),
        "final_nan_count": int(np.isnan(final_tensor).sum()),
        "python_bandpass_applied": False,
        "uses_v5_1_gecn9": False,
        "training_run": False,
        "previous_versions_overwritten": False,
    }
    (args.output_root / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.qc_dir / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    write_readme(
        qc_dir=args.qc_dir,
        output_root=args.output_root,
        tensor_path=tensor_path,
        symlink_status=symlink_status,
        base_n=len(base_ids),
        incremental_n=len(incremental_ids),
        included_incremental_n=len(included_incremental_ids),
        duplicate_n=int(include_table["in_base_v5_1_batch20260514"].sum()),
        final_tensor=final_tensor,
        metadata=metadata,
        evolution=evolution,
        added_summary=added_summary,
        remaining_ledger=remaining_ledger,
    )

    print(f"output_root={args.output_root}")
    print(f"tensor_path={tensor_path}")
    print(f"base_subjects={len(base_ids)}")
    print(f"incremental_subjects={len(incremental_ids)}")
    print(f"incremental_added={len(included_incremental_ids)}")
    print(f"incremental_duplicates_excluded={int(include_table['in_base_v5_1_batch20260514'].sum())}")
    print(f"final_shape={tuple(int(x) for x in final_tensor.shape)}")
    print(f"final_dtype={final_tensor.dtype}")
    print(f"final_nan_count={int(np.isnan(final_tensor).sum())}")
    print(f"training_ready_rows={len(training_ready)}")
    print(
        "training_ready_cn_ge="
        f"{len(training_ready[(training_ready['ResearchGroup_Mapped'].eq('CN')) & (training_ready['Manufacturer'].eq('GE'))])}"
    )
    print("python_bandpass_applied=False")
    print("uses_v5_1_gecn9=False")
    print("training_run=False")


if __name__ == "__main__":
    main()
