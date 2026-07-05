#!/usr/bin/env python3
"""Fuse current ADNI v5 tensor with the 9-subject CN-GE partial branch.

This is an assembly-only step:
- no connectivity is recalculated for the 495 current v5 subjects;
- no Python bandpass is applied;
- no training is run;
- existing v5 outputs are not modified.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_NAME = "adni_expanded_v5_1_gecn9_no_pybandpass"

DEFAULT_V5_TENSOR = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_dparsf10000_no_pybandpass/subject_tensors/"
    "GLOBAL_TENSOR_ADNI_expanded_v5_dparsf10000_no_pybandpass.npz"
)
DEFAULT_GECN9_TENSOR = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_gecn9_no_pybandpass_partial/"
    "GLOBAL_TENSOR_ADNI_expanded_v5_1_gecn9_no_pybandpass_partial.npz"
)
DEFAULT_V5_TRAINING_METADATA = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_dparsf10000_no_pybandpass/"
    "training_ready_metadata_v5_dparsf10000_no_pybandpass.csv"
)
DEFAULT_V5_MANIFEST = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_dparsf10000_no_pybandpass/"
    "subject_manifest_v5_dparsf10000_no_pybandpass.csv"
)
DEFAULT_AUGMENTED_MANIFEST = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_gecn_augmented_manifest"
    / "adni_v5_1_gecn_augmented_first_visit_manifest.csv"
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
    "adni_expanded_v5_1_gecn9_no_pybandpass"
)
DEFAULT_LOCAL_SYMLINK = (
    PROJECT_ROOT
    / "data"
    / "revision_bspc_2026"
    / "adni_expanded_v5_1_gecn9_no_pybandpass"
)
DEFAULT_QC_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_gecn9_full_build_qc"
)

MANUAL_EXCLUDE_FROM_SUPERVISED = {
    "128_S_2002": "invalid_or_missing_diagnosis_metadata",
    "114_S_6039": "invalid_nonfinite_signal_or_absent_from_tensor",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fused ADNI v5.1 GECN9 tensor and metadata without recalculating connectivity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--v5-tensor", type=Path, default=DEFAULT_V5_TENSOR)
    parser.add_argument("--gecn9-tensor", type=Path, default=DEFAULT_GECN9_TENSOR)
    parser.add_argument("--v5-training-metadata", type=Path, default=DEFAULT_V5_TRAINING_METADATA)
    parser.add_argument("--v5-manifest", type=Path, default=DEFAULT_V5_MANIFEST)
    parser.add_argument("--augmented-manifest", type=Path, default=DEFAULT_AUGMENTED_MANIFEST)
    parser.add_argument("--master-manifest", type=Path, default=DEFAULT_MASTER_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--local-symlink", type=Path, default=DEFAULT_LOCAL_SYMLINK)
    parser.add_argument("--qc-dir", type=Path, default=DEFAULT_QC_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-symlink", action="store_true")
    return parser.parse_args()


def clean_string(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def bool_like(value: Any) -> bool:
    return clean_string(value).lower() in {"true", "1", "yes"}


def canonical_manufacturer(value: Any) -> str:
    text = clean_string(value).upper()
    if not text:
        return "UNKNOWN"
    if "GE" in text:
        return "GE"
    if "PHILIPS" in text:
        return "Philips"
    if "SIEMENS" in text:
        return "SIEMENS"
    return clean_string(value)


def normalize_group(value: Any) -> str:
    text = clean_string(value).upper()
    if text in {"AD", "CN", "MCI"}:
        return text
    if "DEMENT" in text or "ALZ" in text:
        return "AD"
    if "CONTROL" in text or text == "NORMAL":
        return "CN"
    if "MCI" in text:
        return "MCI"
    return ""


def prepare_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(f"Output directory exists and is not empty; pass --overwrite: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_npz_tensor(path: Path, label: str) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as zf:
        required = ["global_tensor_data", "subject_ids", "channel_names", "python_bandpass_applied"]
        missing = [key for key in required if key not in zf.files]
        if missing:
            raise RuntimeError(f"{label} tensor missing keys {missing}: {path}")
        tensor = zf["global_tensor_data"].astype(np.float32, copy=False)
        subject_ids = [str(x) for x in zf["subject_ids"].astype(str)]
        channel_names = [str(x) for x in zf["channel_names"].astype(str)]
        payload: Dict[str, Any] = {
            "path": path,
            "label": label,
            "tensor": tensor,
            "subject_ids": subject_ids,
            "channel_names": channel_names,
            "python_bandpass_applied": bool(zf["python_bandpass_applied"]),
            "keys": list(zf.files),
            "rois_count": int(zf["rois_count"]) if "rois_count" in zf.files else tensor.shape[-1],
            "target_len_ts": int(zf["target_len_ts"]) if "target_len_ts" in zf.files else 140,
            "tr_seconds": float(zf["tr_seconds"]) if "tr_seconds" in zf.files else 3.0,
            "preprocessing_source": str(zf["preprocessing_source"]) if "preprocessing_source" in zf.files else "",
            "roi_order_name": str(zf["roi_order_name"]) if "roi_order_name" in zf.files else "",
            "roi_names_in_order": zf["roi_names_in_order"].astype(str) if "roi_names_in_order" in zf.files else np.asarray([]),
            "network_labels_in_order": zf["network_labels_in_order"].astype(str) if "network_labels_in_order" in zf.files else np.asarray([]),
        }
    return payload


def validate_tensors(v5: Dict[str, Any], gecn9: Dict[str, Any]) -> None:
    for payload in [v5, gecn9]:
        tensor = payload["tensor"]
        ids = payload["subject_ids"]
        if tensor.ndim != 4:
            raise RuntimeError(f"{payload['label']} tensor is not 4D: {tensor.shape}")
        if tensor.shape[0] != len(ids):
            raise RuntimeError(f"{payload['label']} tensor first dim does not match subject_ids")
        if len(ids) != len(set(ids)):
            duplicates = sorted({sid for sid in ids if ids.count(sid) > 1})
            raise RuntimeError(f"{payload['label']} duplicate subject IDs: {duplicates}")
        if np.isnan(tensor).any():
            raise RuntimeError(f"{payload['label']} tensor contains NaNs")
        if payload["python_bandpass_applied"]:
            raise RuntimeError(f"{payload['label']} has python_bandpass_applied=True")
    if set(v5["subject_ids"]) & set(gecn9["subject_ids"]):
        dup = sorted(set(v5["subject_ids"]) & set(gecn9["subject_ids"]))
        raise RuntimeError(f"SubjectID overlap between v5 and GECN9 tensors: {dup}")
    if v5["tensor"].shape[1:] != gecn9["tensor"].shape[1:]:
        raise RuntimeError(f"Tensor shape mismatch: {v5['tensor'].shape} vs {gecn9['tensor'].shape}")
    if v5["channel_names"] != gecn9["channel_names"]:
        raise RuntimeError(f"channel_names mismatch: {v5['channel_names']} vs {gecn9['channel_names']}")
    if v5["rois_count"] != gecn9["rois_count"]:
        raise RuntimeError("rois_count mismatch")
    if v5["target_len_ts"] != gecn9["target_len_ts"]:
        raise RuntimeError("target_len_ts mismatch")
    if abs(v5["tr_seconds"] - gecn9["tr_seconds"]) > 1e-9:
        raise RuntimeError("tr_seconds mismatch")


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


def read_csv_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() else pd.DataFrame()


def index_by_subject(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    if df.empty or "SubjectID" not in df.columns:
        return {}
    return df.drop_duplicates("SubjectID", keep="first").set_index("SubjectID", drop=False).to_dict(orient="index")


def value_from_sources(sid: str, field: str, sources: Sequence[Tuple[str, Dict[str, Dict[str, Any]]]]) -> Tuple[str, str]:
    for label, indexed in sources:
        if sid not in indexed:
            continue
        value = clean_string(indexed[sid].get(field, ""))
        if value:
            return value, label
    return "", ""


def build_metadata(
    subject_ids: Sequence[str],
    tensor_sources: Dict[str, str],
    v5_training: pd.DataFrame,
    v5_manifest: pd.DataFrame,
    master_manifest: pd.DataFrame,
    augmented_manifest: pd.DataFrame,
) -> pd.DataFrame:
    source_maps = [
        ("augmented_manifest", index_by_subject(augmented_manifest)),
        ("master_manifest", index_by_subject(master_manifest)),
        ("v5_training_metadata", index_by_subject(v5_training)),
        ("v5_manifest", index_by_subject(v5_manifest)),
    ]
    rows: List[Dict[str, Any]] = []
    for idx, sid in enumerate(subject_ids):
        row: Dict[str, Any] = {
            "SubjectID": sid,
            "tensor_index": idx,
            "tensor_source": tensor_sources[sid],
            "dataset_name": DATASET_NAME,
            "python_bandpass_applied": False,
        }
        metadata_sources_used: List[str] = []
        for field in [
            "ResearchGroup_Mapped",
            "Age",
            "Sex",
            "Manufacturer",
            "Site3",
            "ImageID",
            "Visit",
            "Description",
            "source_label",
            "roisignals_path",
            "stage_guess",
            "compatible_for_v5_1_direct",
            "preprocessing_status",
            "gecn_augmented_status",
            "gecn_augmented_warning",
            "gecn_source_from_audit",
        ]:
            value, source = value_from_sources(sid, field, source_maps)
            row[field] = value
            if source:
                metadata_sources_used.append(source)
        row["ResearchGroup_Mapped"] = normalize_group(row.get("ResearchGroup_Mapped", ""))
        row["Manufacturer"] = canonical_manufacturer(row.get("Manufacturer", ""))
        row["metadata_sources_used"] = "|".join(sorted(set(metadata_sources_used)))
        row["exclude_from_supervised"] = False
        row["supervised_exclusion_reason"] = ""
        if sid in MANUAL_EXCLUDE_FROM_SUPERVISED:
            row["exclude_from_supervised"] = True
            row["supervised_exclusion_reason"] = MANUAL_EXCLUDE_FROM_SUPERVISED[sid]
        if not row["ResearchGroup_Mapped"]:
            row["exclude_from_supervised"] = True
            row["supervised_exclusion_reason"] = append_reason(row["supervised_exclusion_reason"], "missing_diagnosis")
        if row["ResearchGroup_Mapped"] in {"AD", "CN"} and (not clean_string(row.get("Age")) or not clean_string(row.get("Sex"))):
            row["exclude_from_supervised"] = True
            row["supervised_exclusion_reason"] = append_reason(row["supervised_exclusion_reason"], "missing_age_or_sex_for_ad_cn")
        row["training_ready"] = not bool(row["exclude_from_supervised"])
        rows.append(row)
    return pd.DataFrame(rows)


def append_reason(current: str, reason: str) -> str:
    parts = [p for p in clean_string(current).split("|") if p]
    if reason not in parts:
        parts.append(reason)
    return "|".join(parts)


def write_npz(
    output_root: Path,
    v5: Dict[str, Any],
    gecn9: Dict[str, Any],
    final_tensor: np.ndarray,
    final_subject_ids: Sequence[str],
) -> Path:
    tensor_dir = output_root / "subject_tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    path = tensor_dir / "GLOBAL_TENSOR_ADNI_expanded_v5_1_gecn9_no_pybandpass.npz"
    np.savez_compressed(
        path,
        global_tensor_data=final_tensor.astype(np.float32, copy=False),
        subject_ids=np.asarray(final_subject_ids),
        channel_names=np.asarray(v5["channel_names"]),
        rois_count=np.asarray(v5["rois_count"]),
        target_len_ts=np.asarray(v5["target_len_ts"]),
        tr_seconds=np.asarray(v5["tr_seconds"]),
        python_bandpass_applied=np.asarray(False),
        preprocessing_source=np.asarray(v5["preprocessing_source"] or "DPARSF_ROISignals_AAL3_10000"),
        dataset_name=np.asarray(DATASET_NAME),
        source_global_tensors=np.asarray([str(v5["path"]), str(gecn9["path"])]),
        roi_order_name=np.asarray(v5["roi_order_name"]),
        roi_names_in_order=np.asarray(v5["roi_names_in_order"]),
        network_labels_in_order=np.asarray(v5["network_labels_in_order"]),
    )
    return path


def training_counts(metadata: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    rows.append({"summary_type": "all_tensor_subjects", "group1": "all", "n": len(metadata)})
    rows.append({"summary_type": "training_ready", "group1": "yes", "n": int(metadata["training_ready"].sum())})
    rows.append({"summary_type": "training_ready", "group1": "no", "n": int((~metadata["training_ready"]).sum())})
    for source, sub in metadata.groupby("tensor_source", dropna=False):
        rows.append({"summary_type": "tensor_source", "group1": source, "n": len(sub)})
    ready = metadata[metadata["training_ready"]].copy()
    for dx, sub in ready.groupby("ResearchGroup_Mapped", dropna=False):
        rows.append({"summary_type": "training_ready_diagnosis", "group1": dx or "UNKNOWN", "n": len(sub)})
    for sex, sub in ready.groupby("Sex", dropna=False):
        rows.append({"summary_type": "training_ready_sex", "group1": sex or "UNKNOWN", "n": len(sub)})
    cn_ge = ready[ready["ResearchGroup_Mapped"].eq("CN") & ready["Manufacturer"].eq("GE")]
    rows.append({"summary_type": "training_ready_cn_ge", "group1": "CN_GE", "n": len(cn_ge)})
    ad_cn = ready[ready["ResearchGroup_Mapped"].isin(["AD", "CN"])]
    missing_age_sex = ad_cn[ad_cn["Age"].map(clean_string).eq("") | ad_cn["Sex"].map(clean_string).eq("")]
    rows.append({"summary_type": "ad_cn_training_ready_missing_age_or_sex", "group1": "AD_CN", "n": len(missing_age_sex)})
    for sid in ["035_S_6927", "114_S_6039", "128_S_2002"]:
        rows.append(
            {
                "summary_type": "special_subject_training_ready",
                "group1": sid,
                "n": int(ready["SubjectID"].eq(sid).any()),
            }
        )
    return pd.DataFrame(rows)


def diagnosis_manufacturer_table(metadata: pd.DataFrame) -> pd.DataFrame:
    ready = metadata[metadata["training_ready"]].copy()
    if ready.empty:
        return pd.DataFrame(columns=["ResearchGroup_Mapped", "Manufacturer", "n"])
    return (
        ready.groupby(["ResearchGroup_Mapped", "Manufacturer"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["ResearchGroup_Mapped", "Manufacturer"])
    )


def build_alignment(
    final_subject_ids: Sequence[str],
    v5_ids: Sequence[str],
    gecn9_ids: Sequence[str],
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    meta = metadata.set_index("SubjectID", drop=False)
    rows = []
    for idx, sid in enumerate(final_subject_ids):
        row = {
            "SubjectID": sid,
            "final_tensor_index": idx,
            "in_v5_tensor": sid in set(v5_ids),
            "in_gecn9_partial_tensor": sid in set(gecn9_ids),
            "tensor_source": "gecn9_partial" if sid in set(gecn9_ids) else "v5_current",
        }
        if sid in meta.index:
            for col in [
                "ResearchGroup_Mapped",
                "Age",
                "Sex",
                "Manufacturer",
                "training_ready",
                "exclude_from_supervised",
                "supervised_exclusion_reason",
                "gecn_augmented_status",
                "gecn_augmented_warning",
            ]:
                row[col] = meta.loc[sid, col]
        rows.append(row)
    return pd.DataFrame(rows)


def write_readme(
    qc_dir: Path,
    output_root: Path,
    tensor_path: Path,
    symlink_status: str,
    final_tensor: np.ndarray,
    metadata: pd.DataFrame,
    counts: pd.DataFrame,
    dx_man: pd.DataFrame,
) -> None:
    ready = metadata[metadata["training_ready"]]
    cn_ge = ready[ready["ResearchGroup_Mapped"].eq("CN") & ready["Manufacturer"].eq("GE")]
    ad_cn = ready[ready["ResearchGroup_Mapped"].isin(["AD", "CN"])]
    missing_age_sex = ad_cn[ad_cn["Age"].map(clean_string).eq("") | ad_cn["Sex"].map(clean_string).eq("")]
    special = metadata[metadata["SubjectID"].isin(["035_S_6927", "114_S_6039", "128_S_2002"])][
        ["SubjectID", "training_ready", "ResearchGroup_Mapped", "Age", "Sex", "Manufacturer", "supervised_exclusion_reason"]
    ]
    dx_text = "```text\n" + dx_man.to_string(index=False) + "\n```" if not dx_man.empty else "No table."
    special_text = "```text\n" + special.to_string(index=False) + "\n```" if not special.empty else "No special subjects in final tensor metadata."
    lines = [
        "# ADNI v5.1 GECN9 No-Python-Bandpass Full Tensor Build QC",
        "",
        "Assembly-only build. Existing v5 connectivity matrices were reused; only the already-computed GECN9 partial branch was concatenated. No training was run.",
        "",
        "## Explicit Answers",
        "",
        f"- Output root: `{output_root}`.",
        f"- Local symlink status: `{symlink_status}`.",
        f"- Final tensor path: `{tensor_path}`.",
        f"- Final tensor shape: `{tuple(int(x) for x in final_tensor.shape)}`.",
        f"- Final tensor dtype: `{final_tensor.dtype}`.",
        f"- Final tensor NaNs: `{int(np.isnan(final_tensor).sum())}`.",
        "- Python bandpass applied: `False`.",
        f"- Subject IDs in tensor: `{len(metadata)}`.",
        f"- Training-ready metadata rows: `{len(ready)}`.",
        f"- CN-GE in training-ready metadata: `{len(cn_ge)}`.",
        f"- AD/CN training-ready rows missing Age or Sex: `{len(missing_age_sex)}`.",
        f"- `035_S_6927` included in supervised training-ready metadata: `{bool(ready['SubjectID'].eq('035_S_6927').any())}`.",
        f"- `114_S_6039` included in tensor/training-ready: `tensor={bool(metadata['SubjectID'].eq('114_S_6039').any())}`, `training_ready={bool(ready['SubjectID'].eq('114_S_6039').any())}`.",
        f"- `128_S_2002` included in tensor/training-ready: `tensor={bool(metadata['SubjectID'].eq('128_S_2002').any())}`, `training_ready={bool(ready['SubjectID'].eq('128_S_2002').any())}`.",
        "",
        "## Diagnosis x Manufacturer",
        "",
        dx_text,
        "",
        "## Special Subjects",
        "",
        special_text,
        "",
        "## Files",
        "",
        "- `subject_metadata_v5_1_gecn9_no_pybandpass.csv`",
        "- `training_ready_metadata_v5_1_gecn9_no_pybandpass.csv`",
        "- `subject_tensors/GLOBAL_TENSOR_ADNI_expanded_v5_1_gecn9_no_pybandpass.npz`",
        "- `v5_1_gecn9_subject_alignment.csv`",
        "- `v5_1_gecn9_training_ready_counts.csv`",
        "- `v5_1_gecn9_diagnosis_manufacturer_table.csv`",
        "",
        "## Next Step",
        "",
        "Review QC before any training command. This script did not train a model.",
    ]
    (qc_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    prepare_dir(args.output_root, args.overwrite)
    prepare_dir(args.qc_dir, args.overwrite)

    v5 = load_npz_tensor(args.v5_tensor, "v5_current")
    gecn9 = load_npz_tensor(args.gecn9_tensor, "gecn9_partial")
    validate_tensors(v5, gecn9)

    final_subject_ids = list(v5["subject_ids"]) + list(gecn9["subject_ids"])
    final_tensor = np.concatenate([v5["tensor"], gecn9["tensor"]], axis=0).astype(np.float32, copy=False)
    if final_tensor.shape[0] != len(final_subject_ids):
        raise RuntimeError("Final tensor shape does not match final subject count")
    if np.isnan(final_tensor).any():
        raise RuntimeError("Final tensor contains NaNs after concatenation")

    tensor_path = write_npz(args.output_root, v5, gecn9, final_tensor, final_subject_ids)

    v5_training = read_csv_optional(args.v5_training_metadata)
    v5_manifest = read_csv_optional(args.v5_manifest)
    augmented_manifest = read_csv_optional(args.augmented_manifest)
    master_manifest = read_csv_optional(args.master_manifest)
    tensor_sources = {sid: "v5_current" for sid in v5["subject_ids"]}
    tensor_sources.update({sid: "gecn9_partial" for sid in gecn9["subject_ids"]})
    metadata = build_metadata(final_subject_ids, tensor_sources, v5_training, v5_manifest, master_manifest, augmented_manifest)
    training_ready = metadata[metadata["training_ready"]].copy()

    metadata.to_csv(args.output_root / "subject_metadata_v5_1_gecn9_no_pybandpass.csv", index=False)
    training_ready.to_csv(args.output_root / "training_ready_metadata_v5_1_gecn9_no_pybandpass.csv", index=False)

    alignment = build_alignment(final_subject_ids, v5["subject_ids"], gecn9["subject_ids"], metadata)
    counts = training_counts(metadata)
    dx_man = diagnosis_manufacturer_table(metadata)
    alignment.to_csv(args.qc_dir / "v5_1_gecn9_subject_alignment.csv", index=False)
    counts.to_csv(args.qc_dir / "v5_1_gecn9_training_ready_counts.csv", index=False)
    dx_man.to_csv(args.qc_dir / "v5_1_gecn9_diagnosis_manufacturer_table.csv", index=False)

    symlink_status = "skipped_by_user"
    if not args.no_symlink:
        symlink_status = create_symlink(args.local_symlink, args.output_root, args.overwrite)

    command = {
        "script": str(Path(__file__).resolve()),
        "v5_tensor": str(args.v5_tensor),
        "gecn9_tensor": str(args.gecn9_tensor),
        "v5_training_metadata": str(args.v5_training_metadata),
        "v5_manifest": str(args.v5_manifest),
        "augmented_manifest": str(args.augmented_manifest),
        "master_manifest": str(args.master_manifest),
        "output_root": str(args.output_root),
        "local_symlink": str(args.local_symlink),
        "qc_dir": str(args.qc_dir),
        "overwrite": bool(args.overwrite),
        "python_bandpass_applied": False,
        "connectivity_recalculated_for_v5_current": False,
        "training_run": False,
    }
    (args.output_root / "command_log.json").write_text(json.dumps(command, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.qc_dir / "command_log.json").write_text(json.dumps(command, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(args.qc_dir, args.output_root, tensor_path, symlink_status, final_tensor, metadata, counts, dx_man)

    print(f"Wrote fused v5.1 GECN9 dataset to {args.output_root}")
    print(f"tensor_shape={tuple(final_tensor.shape)} nan_count={int(np.isnan(final_tensor).sum())}")
    print(f"training_ready={len(training_ready)} cn_ge={len(training_ready[(training_ready['ResearchGroup_Mapped'].eq('CN')) & (training_ready['Manufacturer'].eq('GE'))])}")
    print("No connectivity recalculation for existing v5 subjects. No training run.")


if __name__ == "__main__":
    main()
