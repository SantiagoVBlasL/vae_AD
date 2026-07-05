#!/usr/bin/env python3
"""Assemble ADNI v5.1 batch-20260513 final candidate dataset.

This is an assembly-only step:
- reuse the conservative v5 tensor;
- append only QC-pass, non-duplicate subjects from Martin bandpass batch 20260513;
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

DATASET_NAME = "adni_expanded_v5_1_batch20260513_no_pybandpass"
INCLUDED_DATASET_VERSION = "v5.1_batch20260513"
SOURCE_BATCH = "20260513_bandpass_batch1"
EXPECTED_CHANNELS = 7
EXPECTED_ROIS = 131
PYTHON_BANDPASS_APPLIED = False

DEFAULT_V5_TENSOR = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_dparsf10000_no_pybandpass/subject_tensors/"
    "GLOBAL_TENSOR_ADNI_expanded_v5_dparsf10000_no_pybandpass.npz"
)
DEFAULT_V5_TRAINING_METADATA = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_dparsf10000_no_pybandpass/"
    "training_ready_metadata_v5_dparsf10000_no_pybandpass.csv"
)
DEFAULT_V5_SUBJECT_METADATA = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_dparsf10000_no_pybandpass/"
    "subject_metadata_v5_dparsf10000_no_pybandpass.csv"
)
DEFAULT_INCREMENTAL_TENSOR = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_v5_1_batch20260513_incremental_no_pybandpass/"
    "INCREMENTAL_TENSOR_MARTIN_BANDPASS_BATCH20260513.npz"
)
DEFAULT_INCREMENTAL_METADATA = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_v5_1_batch20260513_incremental_no_pybandpass/"
    "incremental_subject_metadata.csv"
)
DEFAULT_INCREMENTAL_QC = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_v5_1_batch20260513_incremental_no_pybandpass/"
    "incremental_qc_summary.csv"
)
DEFAULT_LEDGER = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "final_cn_ge_inventory_before_martin_request"
    / "ADNI_v5_1_DATA_LEDGER_CURRENT.csv"
)
DEFAULT_BATCH_DECISION = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "martin_bandpass_batch_20260513"
    / "batch_import_decision.csv"
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
    "adni_expanded_v5_1_batch20260513_no_pybandpass"
)
DEFAULT_LOCAL_SYMLINK = (
    PROJECT_ROOT
    / "data"
    / "revision_bspc_2026"
    / "adni_expanded_v5_1_batch20260513_no_pybandpass"
)
DEFAULT_LEDGER_CANDIDATE = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "martin_bandpass_batch_20260513"
    / "ADNI_v5_1_DATA_LEDGER_20260513_after_batch1_imported_candidate.csv"
)
DEFAULT_QC_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_batch20260513_full_build_qc"
)

FINAL_TENSOR_NAME = "GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260513_no_pybandpass.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ADNI v5.1 batch20260513 final candidate tensor and metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--v5-tensor", type=Path, default=DEFAULT_V5_TENSOR)
    parser.add_argument("--v5-training-metadata", type=Path, default=DEFAULT_V5_TRAINING_METADATA)
    parser.add_argument("--v5-subject-metadata", type=Path, default=DEFAULT_V5_SUBJECT_METADATA)
    parser.add_argument("--incremental-tensor", type=Path, default=DEFAULT_INCREMENTAL_TENSOR)
    parser.add_argument("--incremental-metadata", type=Path, default=DEFAULT_INCREMENTAL_METADATA)
    parser.add_argument("--incremental-qc", type=Path, default=DEFAULT_INCREMENTAL_QC)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--batch-decision", type=Path, default=DEFAULT_BATCH_DECISION)
    parser.add_argument("--master-manifest", type=Path, default=DEFAULT_MASTER_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--local-symlink", type=Path, default=DEFAULT_LOCAL_SYMLINK)
    parser.add_argument("--ledger-candidate", type=Path, default=DEFAULT_LEDGER_CANDIDATE)
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


def array_scalar_str(zf: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in zf.files:
        return default
    arr = zf[key]
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0])
    return "|".join(str(x) for x in arr.astype(str).reshape(-1).tolist())


def array_scalar_bool(zf: np.lib.npyio.NpzFile, key: str, default: bool = False) -> bool:
    if key not in zf.files:
        return default
    arr = zf[key]
    if arr.shape == ():
        return bool(arr.item())
    if arr.size == 1:
        return bool(arr.reshape(-1)[0])
    raise RuntimeError(f"Expected scalar bool array for {key}")


def load_tensor_npz(path: Path, label: str) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as zf:
        required = ["global_tensor_data", "subject_ids", "channel_names", "python_bandpass_applied"]
        missing = [key for key in required if key not in zf.files]
        if missing:
            raise RuntimeError(f"{label} tensor missing required keys {missing}: {path}")
        tensor = zf["global_tensor_data"].astype(np.float32, copy=False)
        payload: Dict[str, Any] = {
            "label": label,
            "path": path,
            "tensor": tensor,
            "subject_ids": [str(x) for x in zf["subject_ids"].astype(str).tolist()],
            "channel_names": [str(x) for x in zf["channel_names"].astype(str).tolist()],
            "python_bandpass_applied": array_scalar_bool(zf, "python_bandpass_applied"),
            "rois_count": int(np.asarray(zf["rois_count"]).reshape(-1)[0]) if "rois_count" in zf.files else tensor.shape[-1],
            "target_len_ts": int(np.asarray(zf["target_len_ts"]).reshape(-1)[0]) if "target_len_ts" in zf.files else 140,
            "tr_seconds": float(np.asarray(zf["tr_seconds"]).reshape(-1)[0]) if "tr_seconds" in zf.files else 3.0,
            "preprocessing_source": array_scalar_str(zf, "preprocessing_source", "DPARSF_ROISignals_AAL3_10000"),
            "roi_order_name": array_scalar_str(zf, "roi_order_name", ""),
            "roi_names_in_order": zf["roi_names_in_order"].astype(str) if "roi_names_in_order" in zf.files else np.asarray([], dtype="U1"),
            "network_labels_in_order": zf["network_labels_in_order"].astype(str)
            if "network_labels_in_order" in zf.files
            else np.asarray([], dtype="U1"),
        }
    return payload


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
        duplicates = sorted({sid for sid in ids if ids.count(sid) > 1})
        raise RuntimeError(f"{label} has duplicate SubjectID values: {duplicates}")


def validate_tensor_compatibility(v5: Dict[str, Any], incremental: Dict[str, Any]) -> None:
    validate_single_tensor(v5)
    validate_single_tensor(incremental)
    if v5["channel_names"] != incremental["channel_names"]:
        raise RuntimeError(
            "channel_names mismatch between v5 and incremental tensors: "
            f"{v5['channel_names']} vs {incremental['channel_names']}"
        )
    if v5["rois_count"] != EXPECTED_ROIS or incremental["rois_count"] != EXPECTED_ROIS:
        raise RuntimeError(f"rois_count mismatch: v5={v5['rois_count']} incremental={incremental['rois_count']}")
    if v5["target_len_ts"] != incremental["target_len_ts"]:
        raise RuntimeError(f"target_len_ts mismatch: v5={v5['target_len_ts']} incremental={incremental['target_len_ts']}")
    if abs(v5["tr_seconds"] - incremental["tr_seconds"]) > 1e-6:
        raise RuntimeError(f"TR mismatch: v5={v5['tr_seconds']} incremental={incremental['tr_seconds']}")


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
    v5_ids: Sequence[str],
    incremental_metadata: pd.DataFrame,
    batch_decision: pd.DataFrame,
    incremental_qc: pd.DataFrame,
) -> pd.DataFrame:
    meta = index_by_subject(incremental_metadata)
    decisions = index_by_subject(batch_decision)
    qc = index_by_subject(incremental_qc)
    v5_id_set = set(v5_ids)
    rows: List[Dict[str, Any]] = []
    for idx, sid in enumerate(incremental_ids):
        decision = decisions.get(sid, {})
        qc_row = qc.get(sid, {})
        import_ready = clean(decision.get("import_ready", "")).lower() == "yes"
        qc_status = clean(qc_row.get("status", "ok" if incremental_qc.empty else ""))
        qc_pass = status_ok(qc_status) if qc_status else False
        duplicate = sid in v5_id_set
        reason = ""
        if sid not in meta:
            reason = append_reason(reason, "missing_incremental_metadata")
        if sid not in decisions:
            reason = append_reason(reason, "missing_batch_decision")
        if not import_ready:
            reason = append_reason(reason, f"import_ready_not_yes:{clean(decision.get('import_ready', 'missing'))}")
        if not incremental_qc.empty and not qc_pass:
            reason = append_reason(reason, f"incremental_qc_not_ok:{qc_status or 'missing'}")
        if duplicate:
            reason = append_reason(reason, "duplicate_with_v5_base_excluded")
        include = sid in meta and sid in decisions and import_ready and (qc_pass or incremental_qc.empty) and not duplicate
        rows.append(
            {
                "SubjectID": sid,
                "incremental_tensor_index": idx,
                "in_v5_base": duplicate,
                "in_incremental_metadata": sid in meta,
                "in_batch_decision": sid in decisions,
                "import_ready": clean(decision.get("import_ready", "")),
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
        "in_v5_base",
        "in_incremental_metadata",
        "in_batch_decision",
        "import_ready",
        "incremental_qc_status",
        "include_in_final",
        "action",
        "reason",
        "selected_path",
    ]
    duplicates = include_table[include_table["in_v5_base"]].copy()
    if duplicates.empty:
        return pd.DataFrame(columns=columns)
    duplicates["action"] = "excluded_incremental_duplicate"
    duplicates["reason"] = duplicates["exclusion_reason"]
    return duplicates[columns]


def build_v5_metadata_rows(
    v5_ids: Sequence[str],
    v5_subject_metadata: pd.DataFrame,
    v5_training_metadata: pd.DataFrame,
) -> pd.DataFrame:
    meta = index_by_subject(v5_subject_metadata)
    training_ids = set(v5_training_metadata["SubjectID"].astype(str).tolist()) if "SubjectID" in v5_training_metadata.columns else set()
    rows: List[Dict[str, Any]] = []
    missing = [sid for sid in v5_ids if sid not in meta]
    if missing:
        raise RuntimeError(f"v5 subject_metadata missing tensor subjects: {missing[:20]}")
    for final_idx, sid in enumerate(v5_ids):
        src = meta[sid]
        dx = normalize_diagnosis(src.get("ResearchGroup_Mapped", ""))
        age = clean(src.get("Age", ""))
        sex = clean(src.get("Sex", ""))
        manufacturer = normalize_manufacturer(src.get("Manufacturer", ""))
        training_ready = sid in training_ids
        reason = clean(src.get("exclusion_reason", ""))
        if not training_ready:
            reason = append_reason(reason, "not_in_v5_training_ready_metadata")
            if not dx:
                reason = append_reason(reason, "missing_diagnosis")
            if dx in {"AD", "CN"} and (not age or not sex):
                reason = append_reason(reason, "missing_age_or_sex_for_ad_cn")
        rows.append(
            {
                "SubjectID": sid,
                "tensor_index": final_idx,
                "tensor_source": "v5_base_conservative",
                "dataset_name": DATASET_NAME,
                "included_in_dataset_version": INCLUDED_DATASET_VERSION,
                "ResearchGroup_Mapped": dx,
                "Diagnosis": dx,
                "Age": age,
                "Sex": sex,
                "Manufacturer": manufacturer,
                "Site3": clean(src.get("Site3", "")),
                "ImageID": clean(src.get("ImageID", "")),
                "Visit": clean(src.get("Visit", "")),
                "metadata_source": clean(src.get("metadata_source", "")),
                "source_label": clean(src.get("source_label", "")),
                "source_batch": "v5_dparsf10000_no_pybandpass",
                "roisignals_path": "",
                "stage_guess": "",
                "spectral_class": "",
                "priority_batch": "",
                "ledger_scope": "",
                "dicom_series_ok": "",
                "python_bandpass_requested": "NO",
                "python_bandpass_applied": False,
                "exclude_from_supervised": not training_ready,
                "supervised_exclusion_reason": reason,
                "training_ready": bool(training_ready),
            }
        )
    return pd.DataFrame(rows)


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
        dx = normalize_diagnosis(dx_raw)
        age, age_source = first_value(sid, "Age", sources)
        sex, sex_source = first_value(sid, "Sex", sources)
        manufacturer_raw, man_source = first_value(sid, "Manufacturer", sources)
        manufacturer = normalize_manufacturer(manufacturer_raw)
        image_id, image_source = first_value(sid, "ImageID", sources)
        visit, visit_source = first_value(sid, "Visit", sources)
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
                "tensor_source": "martin_bandpass_batch20260513_incremental",
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
                "source_label": "martin_bandpass_batch20260513",
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
    v5: Dict[str, Any],
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
        channel_names=np.asarray(v5["channel_names"], dtype="U64"),
        rois_count=np.asarray(EXPECTED_ROIS, dtype=np.int32),
        target_len_ts=np.asarray(v5["target_len_ts"], dtype=np.int32),
        tr_seconds=np.asarray(v5["tr_seconds"], dtype=np.float32),
        python_bandpass_applied=np.asarray(False, dtype=np.bool_),
        preprocessing_source=np.asarray(v5["preprocessing_source"], dtype="U96"),
        dataset_name=np.asarray(DATASET_NAME, dtype="U96"),
        source_global_tensors=np.asarray([str(v5["path"]), str(incremental["path"])], dtype="U256"),
        uses_v5_1_gecn9=np.asarray(False, dtype=np.bool_),
        assembled_with_v5=np.asarray(True, dtype=np.bool_),
        source_batch=np.asarray(SOURCE_BATCH, dtype="U32"),
        roi_order_name=np.asarray(v5["roi_order_name"], dtype="U96"),
        roi_names_in_order=np.asarray(v5["roi_names_in_order"], dtype="U128"),
        network_labels_in_order=np.asarray(v5["network_labels_in_order"], dtype="U64"),
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
    v5_ids: Sequence[str],
    incremental_ids: Sequence[str],
    include_table: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    meta = metadata.set_index("SubjectID", drop=False).to_dict(orient="index")
    v5_pos = {sid: idx for idx, sid in enumerate(v5_ids)}
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
                "v5_tensor_index": v5_pos.get(sid, ""),
                "incremental_tensor_index": inc_pos.get(sid, ""),
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
                "v5_tensor_index": v5_pos.get(sid, ""),
                "incremental_tensor_index": row.get("incremental_tensor_index", ""),
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
    for scope, sub in [("all_tensor_subjects", metadata), ("training_ready", metadata[metadata["training_ready"].astype(bool)])]:
        counts = sub["ResearchGroup_Mapped"].replace("", "UNKNOWN").value_counts(dropna=False).to_dict()
        for diagnosis in ["AD", "CN", "MCI", "UNKNOWN"]:
            rows.append({"scope": scope, "Diagnosis": diagnosis, "n": int(counts.get(diagnosis, 0))})
        rows.append({"scope": scope, "Diagnosis": "TOTAL", "n": int(len(sub))})
    return pd.DataFrame(rows)


def diagnosis_x_manufacturer(metadata: pd.DataFrame) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for scope, sub in [("all_tensor_subjects", metadata), ("training_ready", metadata[metadata["training_ready"].astype(bool)])]:
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


def update_ledger_candidate(
    ledger: pd.DataFrame,
    included_incremental_metadata: pd.DataFrame,
    batch_decision: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    updated = ledger.copy()
    decisions = index_by_subject(batch_decision)
    included_ids = included_incremental_metadata["SubjectID"].astype(str).tolist()
    if "SubjectID" not in updated.columns:
        raise RuntimeError("Ledger is missing SubjectID")
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
    updated_idx = updated.set_index("SubjectID", drop=False)
    missing_in_ledger = []
    for sid in included_ids:
        if sid not in updated_idx.index:
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


def write_readme(
    qc_dir: Path,
    output_root: Path,
    tensor_path: Path,
    symlink_status: str,
    v5_n: int,
    incremental_n: int,
    included_incremental_n: int,
    duplicate_n: int,
    excluded_incremental_n: int,
    final_tensor: np.ndarray,
    metadata: pd.DataFrame,
    dx_counts: pd.DataFrame,
    dx_man: pd.DataFrame,
) -> None:
    training_ready = metadata[metadata["training_ready"].astype(bool)].copy()
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
    counts_text = "```text\n" + dx_counts.to_string(index=False) + "\n```"
    dx_man_text = "```text\n" + dx_man.to_string(index=False) + "\n```"
    lines = [
        "# ADNI v5.1 Batch 20260513 No-Python-Bandpass Full Build QC",
        "",
        f"Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "Assembly-only candidate build. The conservative v5 tensor was reused and Martin batch 20260513 was appended only for QC-pass, non-duplicate subjects.",
        "",
        "## Explicit Answers",
        "",
        f"- v5 base subjects: `{v5_n}`.",
        f"- Incremental batch subjects: `{incremental_n}`.",
        f"- Incremental subjects actually added: `{included_incremental_n}`.",
        f"- Incremental duplicates excluded: `{duplicate_n}`.",
        f"- Other incremental exclusions: `{excluded_incremental_n - duplicate_n}`.",
        f"- Final tensor shape: `{tuple(int(x) for x in final_tensor.shape)}`.",
        f"- Final tensor dtype: `{final_tensor.dtype}`.",
        f"- Final tensor NaNs: `{int(np.isnan(final_tensor).sum())}`.",
        f"- Output root: `{output_root}`.",
        f"- Tensor path: `{tensor_path}`.",
        f"- Local symlink status: `{symlink_status}`.",
        "- Python bandpass applied/requested in final path: `False` / `NO`.",
        "- v5.1_gecn9 used: `False`; it remains quarantine.",
        "- v5 overwritten: `False`.",
        "- Training run: `False`.",
        f"- CN-GE training-ready count: `{len(cn_ge)}`.",
        f"- AD/CN training-ready rows missing Age or Sex: `{len(missing_covars)}`.",
        f"- Ready to train baseline `[1,0,2]`: `{baseline_ready}`.",
        "",
        "## Diagnosis Counts",
        "",
        counts_text,
        "",
        "## Diagnosis x Manufacturer",
        "",
        dx_man_text,
        "",
        "## Files",
        "",
        "- Output dataset:",
        f"  - `{output_root / 'subject_tensors' / FINAL_TENSOR_NAME}`",
        f"  - `{output_root / 'subject_metadata_v5_1_batch20260513_no_pybandpass.csv'}`",
        f"  - `{output_root / 'training_ready_metadata_v5_1_batch20260513_no_pybandpass.csv'}`",
        "- QC:",
        "  - `subject_alignment.csv`",
        "  - `diagnosis_counts.csv`",
        "  - `diagnosis_x_manufacturer.csv`",
        "  - `duplicate_report.csv`",
        "  - `ledger_update_summary.csv`",
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
    ensure_output_file(args.ledger_candidate, args.overwrite)

    v5 = load_tensor_npz(args.v5_tensor, "v5_base")
    incremental = load_tensor_npz(args.incremental_tensor, "incremental_batch20260513")
    validate_tensor_compatibility(v5, incremental)

    v5_ids = v5["subject_ids"]
    incremental_ids = incremental["subject_ids"]
    incremental_metadata = read_csv(args.incremental_metadata)
    incremental_qc = read_csv(args.incremental_qc, required=False)
    batch_decision = read_csv(args.batch_decision)
    ledger = read_csv(args.ledger)
    master_manifest = read_csv(args.master_manifest)
    v5_subject_metadata = read_csv(args.v5_subject_metadata)
    v5_training_metadata = read_csv(args.v5_training_metadata)

    include_table = build_incremental_include_table(
        incremental_ids=incremental_ids,
        v5_ids=v5_ids,
        incremental_metadata=incremental_metadata,
        batch_decision=batch_decision,
        incremental_qc=incremental_qc,
    )
    duplicate_report = build_duplicate_report(include_table)
    included_incremental_ids = include_table.loc[include_table["include_in_final"], "SubjectID"].astype(str).tolist()
    included_incremental_indices = include_table.loc[include_table["include_in_final"], "incremental_tensor_index"].astype(int).tolist()

    final_subject_ids = list(v5_ids) + included_incremental_ids
    final_tensor = np.concatenate([v5["tensor"], incremental["tensor"][included_incremental_indices]], axis=0).astype(
        np.float32, copy=False
    )
    if final_tensor.shape[0] != len(final_subject_ids):
        raise RuntimeError("Final tensor first dimension does not match final subject_ids")
    if len(final_subject_ids) != len(set(final_subject_ids)):
        duplicated = sorted({sid for sid in final_subject_ids if final_subject_ids.count(sid) > 1})
        raise RuntimeError(f"Final tensor would contain duplicated SubjectID values: {duplicated}")
    if np.isnan(final_tensor).any():
        raise RuntimeError("Final tensor contains NaNs after assembly")

    tensor_path = write_tensor_npz(args.output_root, v5, incremental, final_tensor, final_subject_ids)

    v5_metadata = build_v5_metadata_rows(v5_ids, v5_subject_metadata, v5_training_metadata)
    incremental_final_metadata = build_incremental_metadata_rows(
        included_incremental_ids,
        start_index=len(v5_ids),
        incremental_metadata=incremental_metadata,
        batch_decision=batch_decision,
        ledger=ledger,
        master_manifest=master_manifest,
    )
    metadata = pd.concat([v5_metadata, incremental_final_metadata], ignore_index=True)
    if metadata["SubjectID"].tolist() != final_subject_ids:
        raise RuntimeError("Metadata SubjectID order does not match final tensor subject_ids")
    training_ready = metadata[metadata["training_ready"].astype(bool)].copy()

    metadata_path = args.output_root / "subject_metadata_v5_1_batch20260513_no_pybandpass.csv"
    training_path = args.output_root / "training_ready_metadata_v5_1_batch20260513_no_pybandpass.csv"
    metadata.to_csv(metadata_path, index=False)
    training_ready.to_csv(training_path, index=False)

    ledger_candidate, ledger_summary = update_ledger_candidate(ledger, incremental_final_metadata, batch_decision)
    ledger_candidate.to_csv(args.ledger_candidate, index=False)

    symlink_status = "skipped_by_user"
    if not args.no_symlink:
        symlink_status = create_symlink(args.local_symlink, args.output_root, args.overwrite_symlink)

    alignment = make_subject_alignment(final_subject_ids, v5_ids, incremental_ids, include_table, metadata)
    dx_counts = diagnosis_counts(metadata)
    dx_man = diagnosis_x_manufacturer(metadata)
    alignment.to_csv(args.qc_dir / "subject_alignment.csv", index=False)
    dx_counts.to_csv(args.qc_dir / "diagnosis_counts.csv", index=False)
    dx_man.to_csv(args.qc_dir / "diagnosis_x_manufacturer.csv", index=False)
    duplicate_report.to_csv(args.qc_dir / "duplicate_report.csv", index=False)
    ledger_summary.to_csv(args.qc_dir / "ledger_update_summary.csv", index=False)
    include_table.to_csv(args.qc_dir / "incremental_inclusion_decision.csv", index=False)

    command_log = {
        "script": str(Path(__file__).resolve()),
        "generated": datetime.now().isoformat(timespec="seconds"),
        "v5_tensor": str(args.v5_tensor),
        "v5_training_metadata": str(args.v5_training_metadata),
        "v5_subject_metadata": str(args.v5_subject_metadata),
        "incremental_tensor": str(args.incremental_tensor),
        "incremental_metadata": str(args.incremental_metadata),
        "incremental_qc": str(args.incremental_qc),
        "ledger": str(args.ledger),
        "batch_decision": str(args.batch_decision),
        "master_manifest": str(args.master_manifest),
        "output_root": str(args.output_root),
        "local_symlink": str(args.local_symlink),
        "ledger_candidate": str(args.ledger_candidate),
        "qc_dir": str(args.qc_dir),
        "v5_subjects": len(v5_ids),
        "incremental_subjects": len(incremental_ids),
        "incremental_added": len(included_incremental_ids),
        "incremental_duplicates_excluded": int(include_table["in_v5_base"].sum()),
        "final_shape": tuple(int(x) for x in final_tensor.shape),
        "final_nan_count": int(np.isnan(final_tensor).sum()),
        "python_bandpass_applied": False,
        "uses_v5_1_gecn9": False,
        "training_run": False,
        "v5_overwritten": False,
    }
    (args.output_root / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.qc_dir / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    write_readme(
        qc_dir=args.qc_dir,
        output_root=args.output_root,
        tensor_path=tensor_path,
        symlink_status=symlink_status,
        v5_n=len(v5_ids),
        incremental_n=len(incremental_ids),
        included_incremental_n=len(included_incremental_ids),
        duplicate_n=int(include_table["in_v5_base"].sum()),
        excluded_incremental_n=int((~include_table["include_in_final"]).sum()),
        final_tensor=final_tensor,
        metadata=metadata,
        dx_counts=dx_counts,
        dx_man=dx_man,
    )

    print(f"output_root={args.output_root}")
    print(f"tensor_path={tensor_path}")
    print(f"v5_subjects={len(v5_ids)}")
    print(f"incremental_subjects={len(incremental_ids)}")
    print(f"incremental_added={len(included_incremental_ids)}")
    print(f"incremental_duplicates_excluded={int(include_table['in_v5_base'].sum())}")
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
