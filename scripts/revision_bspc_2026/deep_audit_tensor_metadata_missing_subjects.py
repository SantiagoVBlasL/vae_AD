#!/usr/bin/env python3
"""Deep read-only trace for tensor subjects missing from final metadata.

This script does not modify tensor, metadata, ledger, configs, or training
outputs. It only writes audit artifacts under
results/revision_bspc_2026/tensor_metadata_missing_subjects_audit/.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MISSING_SUBJECTS = ("035_S_6927", "128_S_2002")

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
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "tensor_metadata_missing_subjects_audit"
)
DEFAULT_BASE_PRETRAINING_QC = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_expanded_v5_dparsf10000_no_pybandpass_pretraining_qc"
    / "training_ready_manifest.csv"
)

TARGETED_SOURCES = [
    "data/AD_fMRI_4_28_2026.csv",
    "data/AD_fMRI_4_28_2026_extended.csv",
    "data/idaSearch_4_03_2026.csv",
    "results/revision_bspc_2026/adni_v5_1_master_manifest/adni_v5_1_master_subject_manifest_first_visit.csv",
    "results/revision_bspc_2026/adni_v5_1_master_manifest/adni_v5_1_preprocessing_request_for_martin.csv",
    "results/revision_bspc_2026/new_adni_ad_timepoint_overlap_audit/new_ad_subject_summary.csv",
    "results/revision_bspc_2026/new_adni_ad_timepoint_overlap_audit/longitudinal_leakage_risk_report.csv",
    "results/revision_bspc_2026/adni_v5_1_batch20260514b_full_build_qc/subject_alignment.csv",
    "results/revision_bspc_2026/adni_expanded_v5_dparsf10000_no_pybandpass_pretraining_qc/training_ready_manifest.csv",
    "results/revision_bspc_2026/adni_expanded_v5_dparsf10000_no_pybandpass_pretraining_qc/missing_demographics_subjects.csv",
    "results/revision_bspc_2026/adni_expanded_v5_dparsf10000_no_pybandpass_pretraining_qc/unknown_diagnosis_subjects.csv",
    "results/revision_bspc_2026/adni_expanded_v5_dparsf10000_no_pybandpass_pretraining_qc/channel_fallback_subjects.csv",
    "results/revision_bspc_2026/adni_v5_dparsf_only_rebuild/pre_full_extraction_qc/unknown_subjects_resolution.csv",
    "results/revision_bspc_2026/adni_v5_dparsf_only_rebuild/pre_full_extraction_qc/manifest_final_qc.csv",
    "results/revision_bspc_2026/desde_cero_roisignals_aal3_audit/invalid_or_low_finite_signal_subjects.csv",
    "results/revision_bspc_2026/desde_cero_roisignals_aal3_audit/roisignals_inventory.csv",
    "results/revision_bspc_2026/martin_bandpass_batch_20260513/batch_import_decision.csv",
    "results/revision_bspc_2026/martin_bandpass_batch_20260513/batch_roisignals_qc.csv",
    "results/revision_bspc_2026/martin_bandpass_batch_20260514/batch_import_decision.csv",
    "results/revision_bspc_2026/martin_bandpass_batch_20260514/batch_roisignals_qc.csv",
    "results/revision_bspc_2026/martin_bandpass_batch_20260514b/batch_import_decision.csv",
    "results/revision_bspc_2026/martin_bandpass_batch_20260514b/batch_roisignals_qc.csv",
]

BASE_AND_VERSION_METADATA = [
    Path(
        "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
        "adni_expanded_v5_dparsf10000_no_pybandpass/"
        "subject_metadata_v5_dparsf10000_no_pybandpass.csv"
    ),
    Path(
        "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
        "adni_expanded_v5_dparsf10000_no_pybandpass/"
        "training_ready_metadata_v5_dparsf10000_no_pybandpass.csv"
    ),
    Path(
        "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
        "adni_expanded_v5_1_batch20260513_no_pybandpass/"
        "subject_metadata_v5_1_batch20260513_no_pybandpass.csv"
    ),
    Path(
        "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
        "adni_expanded_v5_1_batch20260513_no_pybandpass/"
        "training_ready_metadata_v5_1_batch20260513_no_pybandpass.csv"
    ),
    Path(
        "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
        "adni_expanded_v5_1_batch20260514_no_pybandpass/"
        "subject_metadata_v5_1_batch20260514_no_pybandpass.csv"
    ),
    Path(
        "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
        "adni_expanded_v5_1_batch20260514_no_pybandpass/"
        "training_ready_metadata_v5_1_batch20260514_no_pybandpass.csv"
    ),
    Path(
        "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
        "adni_expanded_v5_1_batch20260514b_no_pybandpass/"
        "subject_metadata_v5_1_batch20260514b_no_pybandpass.csv"
    ),
    Path(
        "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
        "adni_expanded_v5_1_batch20260514b_no_pybandpass/"
        "training_ready_metadata_v5_1_batch20260514b_no_pybandpass.csv"
    ),
]

SCAN_SUFFIXES = {".csv", ".tsv", ".json", ".md", ".txt"}
MAX_TEXT_FILE_BYTES = 25 * 1024 * 1024
GENERATED_OUTPUT_NAMES = {
    "missing_subjects_deep_trace.csv",
    "missing_subjects_deep_trace.md",
    "per_subject_decision.csv",
    "per_subject_decision.md",
    "final_recommendation.md",
    "command_log.json",
    "final_action_table.csv",
    "final_action_table.md",
    "final_manuscript_safe_note.md",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only deep audit for tensor subjects missing from final metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tensor", type=Path, default=DEFAULT_TENSOR)
    parser.add_argument("--final-metadata", type=Path, default=DEFAULT_FINAL_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subjects", nargs="+", default=list(MISSING_SUBJECTS))
    parser.add_argument("--max-scan-mb", type=float, default=75.0)
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(path)


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "<na>"} else text


def is_truthy(value: Any) -> bool:
    return clean(value).lower() in {"1", "true", "yes", "y"}


def short_json(row: pd.Series, max_chars: int = 2800) -> str:
    payload = {str(k): clean(v) for k, v in row.to_dict().items() if clean(v)}
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def read_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    return pd.read_csv(
        path,
        sep=sep,
        dtype=str,
        keep_default_na=False,
        on_bad_lines="skip",
        low_memory=False,
    )


def priority_rank(path: Path) -> int:
    relative = rel(path)
    for idx, source in enumerate(TARGETED_SOURCES, start=1):
        if relative == source:
            return idx
    for idx, source in enumerate(TARGETED_SOURCES, start=1):
        if source in relative:
            return idx
    if relative.startswith("data/"):
        return 100
    return 200


def iter_scan_files(max_scan_mb: float) -> Iterable[Path]:
    roots = [PROJECT_ROOT / "data", PROJECT_ROOT / "results" / "revision_bspc_2026"]
    emitted = set()
    for source in TARGETED_SOURCES:
        path = PROJECT_ROOT / source
        if path.exists():
            emitted.add(path.resolve())
            yield path
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in SCAN_SUFFIXES:
                continue
            if path.parent.resolve() == DEFAULT_OUTPUT_DIR.resolve() and path.name in GENERATED_OUTPUT_NAMES:
                continue
            resolved = path.resolve()
            if resolved in emitted:
                continue
            try:
                if path.stat().st_size > max_scan_mb * 1024 * 1024:
                    continue
            except OSError:
                continue
            emitted.add(resolved)
            yield path


def trace_subject_rows(subjects: Sequence[str], max_scan_mb: float) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    subjects = tuple(subjects)
    for path in iter_scan_files(max_scan_mb):
        suffix = path.suffix.lower()
        size_bytes = path.stat().st_size if path.exists() else 0
        if suffix in {".csv", ".tsv"}:
            try:
                df = read_table(path)
            except Exception as exc:  # pragma: no cover - audit resilience
                rows.append(
                    {
                        "SubjectID": "",
                        "file": rel(path),
                        "priority_rank": priority_rank(path),
                        "matched_rows_total": "",
                        "preview_index": "",
                        "match_type": "read_error",
                        "columns": "",
                        "row_preview": f"{type(exc).__name__}: {exc}",
                        "file_size_bytes": size_bytes,
                    }
                )
                continue
            if df.empty:
                continue
            row_text = df.astype(str).agg(" ".join, axis=1)
            for sid in subjects:
                mask = row_text.str.contains(sid, regex=False, na=False)
                matched = df.loc[mask]
                for idx, (_, row) in enumerate(matched.iterrows()):
                    rows.append(
                        {
                            "SubjectID": sid,
                            "file": rel(path),
                            "priority_rank": priority_rank(path),
                            "matched_rows_total": int(len(matched)),
                            "preview_index": idx,
                            "match_type": "table_row",
                            "columns": "|".join(map(str, df.columns)),
                            "row_preview": short_json(row),
                            "file_size_bytes": size_bytes,
                        }
                    )
        else:
            if size_bytes > MAX_TEXT_FILE_BYTES:
                continue
            try:
                text = path.read_text(errors="replace")
            except Exception:
                continue
            lines = text.splitlines()
            for sid in subjects:
                hit_lines = [i for i, line in enumerate(lines) if sid in line]
                for preview_index, line_idx in enumerate(hit_lines[:10]):
                    lo = max(0, line_idx - 1)
                    hi = min(len(lines), line_idx + 2)
                    context = "\\n".join(lines[lo:hi])
                    rows.append(
                        {
                            "SubjectID": sid,
                            "file": rel(path),
                            "priority_rank": priority_rank(path),
                            "matched_rows_total": len(hit_lines),
                            "preview_index": preview_index,
                            "match_type": "text_context",
                            "columns": "",
                            "row_preview": context[:2800],
                            "file_size_bytes": size_bytes,
                        }
                    )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "SubjectID",
                "file",
                "priority_rank",
                "matched_rows_total",
                "preview_index",
                "match_type",
                "columns",
                "row_preview",
                "file_size_bytes",
            ]
        )
    return out.sort_values(["SubjectID", "priority_rank", "file", "preview_index"]).reset_index(drop=True)


def tensor_presence(tensor_path: Path, final_metadata_path: Path, subjects: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    with np.load(tensor_path, allow_pickle=False) as npz:
        subject_ids = npz["subject_ids"].astype(str).tolist()
        tensor_shape = tuple(npz["global_tensor_data"].shape)
        python_bandpass = bool(npz["python_bandpass_applied"].item()) if "python_bandpass_applied" in npz.files else None
    metadata = read_table(final_metadata_path)
    meta_subjects = set(metadata["SubjectID"].astype(str)) if "SubjectID" in metadata.columns else set()
    info: Dict[str, Dict[str, Any]] = {}
    for sid in subjects:
        info[sid] = {
            "present_in_tensor": sid in subject_ids,
            "tensor_index": subject_ids.index(sid) if sid in subject_ids else "",
            "present_in_final_training_metadata": sid in meta_subjects,
            "tensor_shape": tensor_shape,
            "python_bandpass_applied": python_bandpass,
        }
    return info


def pool_membership(final_metadata_path: Path, subjects: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    final_metadata = read_table(final_metadata_path)
    base_pretraining = read_table(DEFAULT_BASE_PRETRAINING_QC) if DEFAULT_BASE_PRETRAINING_QC.exists() else pd.DataFrame()

    pool_info: Dict[str, Dict[str, Any]] = {}
    for sid in subjects:
        final_rows = (
            final_metadata[final_metadata["SubjectID"].astype(str) == sid]
            if "SubjectID" in final_metadata.columns
            else pd.DataFrame()
        )
        final_dx = ""
        final_in_vae = False
        final_in_classifier = False
        if not final_rows.empty:
            final_dx = clean(final_rows.iloc[0].get("ResearchGroup_Mapped", ""))
            final_in_vae = True
            final_in_classifier = final_dx in {"AD", "CN"}

        base_rows = (
            base_pretraining[base_pretraining["SubjectID"].astype(str) == sid]
            if "SubjectID" in base_pretraining.columns
            else pd.DataFrame()
        )
        base_use_for_vae = ""
        base_use_for_supervised = ""
        base_use_for_supervised_with_demo = ""
        base_exclude = ""
        base_research_group = ""
        if not base_rows.empty:
            row = base_rows.iloc[0]
            base_use_for_vae = clean(row.get("use_for_vae", ""))
            base_use_for_supervised = clean(row.get("use_for_supervised", ""))
            base_use_for_supervised_with_demo = clean(row.get("use_for_supervised_with_demo", ""))
            base_exclude = clean(row.get("exclude", ""))
            base_research_group = clean(row.get("ResearchGroup_Mapped", ""))

        pool_info[sid] = {
            "appears_in_final_classifier_cn_ad_pool": final_in_classifier,
            "appears_in_final_vae_pretraining_pool": final_in_vae,
            "final_training_ready_diagnosis": final_dx,
            "base_pretraining_qc_use_for_vae": base_use_for_vae,
            "base_pretraining_qc_use_for_supervised": base_use_for_supervised,
            "base_pretraining_qc_use_for_supervised_with_demo": base_use_for_supervised_with_demo,
            "base_pretraining_qc_exclude": base_exclude,
            "base_pretraining_qc_diagnosis": base_research_group,
        }
    return pool_info


def metadata_version_trace(subjects: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in BASE_AND_VERSION_METADATA:
        if not path.exists():
            continue
        try:
            df = read_table(path)
        except Exception as exc:
            rows.append(
                {
                    "SubjectID": "",
                    "file": str(path),
                    "dataset_stage": path.parent.name,
                    "row_status": "read_error",
                    "row_preview": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        if "SubjectID" not in df.columns:
            continue
        for sid in subjects:
            matched = df[df["SubjectID"].astype(str) == sid]
            if matched.empty:
                rows.append(
                    {
                        "SubjectID": sid,
                        "file": str(path),
                        "dataset_stage": path.parent.name,
                        "row_status": "absent",
                        "row_preview": "",
                    }
                )
            else:
                for _, row in matched.iterrows():
                    rows.append(
                        {
                            "SubjectID": sid,
                            "file": str(path),
                            "dataset_stage": path.parent.name,
                            "row_status": "present",
                            "ResearchGroup_Mapped": clean(row.get("ResearchGroup_Mapped", "")),
                            "Age": clean(row.get("Age", "")),
                            "Sex": clean(row.get("Sex", "")),
                            "Manufacturer": clean(row.get("Manufacturer", "")),
                            "Site3": clean(row.get("Site3", "")),
                            "ImageID": clean(row.get("ImageID", "")),
                            "Visit": clean(row.get("Visit", "")),
                            "training_ready": clean(row.get("training_ready", "")),
                            "exclude": clean(row.get("exclude", "")),
                            "exclude_from_supervised": clean(row.get("exclude_from_supervised", "")),
                            "supervised_exclusion_reason": clean(row.get("supervised_exclusion_reason", "")),
                            "metadata_source": clean(row.get("metadata_source", "")),
                            "source_label": clean(row.get("source_label", "")),
                            "row_preview": short_json(row),
                        }
                    )
    return pd.DataFrame(rows)


def build_decisions(
    tensor_info: Dict[str, Dict[str, Any]],
    pool_info: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    rows = [
        {
            "SubjectID": "035_S_6927",
            **tensor_info["035_S_6927"],
            **pool_info["035_S_6927"],
            "Diagnosis": "AD",
            "Age": "60 (AD_fMRI); 59.6 (idaSearch)",
            "Sex": "F",
            "Manufacturer": "SIEMENS in idaSearch imaging protocol; UNKNOWN in v5.1 master manifest",
            "Site3": "035",
            "ImageID": "1436478",
            "Visit": "sc / ADNI Screening",
            "has_valid_age_sex": "yes",
            "has_valid_manufacturer_site_image_visit": "yes_with_manifest_manufacturer_gap",
            "roisignals_usable": "yes",
            "roisignals_evidence": (
                "AAL3 170 ROIs; ARWSDCFN passband; finite_fraction=0.9764705882; "
                "scale_label=around_10000_global_scaled; spectral_class=bandpassed_like"
            ),
            "intentionally_excluded": "no",
            "exclusion_or_loss_reason": (
                "Not excluded for diagnosis/signal. It was absent from base v5 training-ready metadata "
                "because base v5 subject metadata carried missing Age/Sex/Manufacturer even though "
                "AD_fMRI_4_28_2026 and idaSearch_4_03_2026 contain usable demographics."
            ),
            "primary_source_files": (
                "data/AD_fMRI_4_28_2026.csv; data/idaSearch_4_03_2026.csv; "
                "results/revision_bspc_2026/adni_v5_1_master_manifest/adni_v5_1_preprocessing_request_for_martin.csv; "
                "results/revision_bspc_2026/adni_v5_1_batch20260514b_full_build_qc/subject_alignment.csv"
            ),
            "merge_or_build_stage_lost": (
                "base v5 training-ready construction: present in subject_metadata_v5_dparsf10000_no_pybandpass.csv "
                "but absent from training_ready_metadata_v5_dparsf10000_no_pybandpass.csv; v5.1 builders then "
                "propagated not_in_v5_training_ready_metadata|missing_age_or_sex_for_ad_cn."
            ),
            "current_dataset_decision": "keep_only_in_tensor_excluded_from_supervised_and_final_vae_pools",
            "future_rebuild_decision": "add_to_training_ready_metadata",
            "decision": "recover_in_future_metadata",
            "action_needed": (
                "In a future metadata-only rebuild, recover AD supervised metadata using ImageID=1436478, "
                "Visit=sc, Age=60 or 59.6, Sex=F, Manufacturer=SIEMENS, Site3=035. No tensor change is needed "
                "for the current read-only audit."
            ),
            "ask_martin": "no",
        },
        {
            "SubjectID": "128_S_2002",
            **tensor_info["128_S_2002"],
            **pool_info["128_S_2002"],
            "Diagnosis": "unknown",
            "Age": "",
            "Sex": "",
            "Manufacturer": "UNKNOWN",
            "Site3": "128 derived from SubjectID only",
            "ImageID": "",
            "Visit": "",
            "has_valid_age_sex": "no",
            "has_valid_manufacturer_site_image_visit": "no",
            "roisignals_usable": "no_for_final_v5_1",
            "roisignals_evidence": (
                "Historical desde_cero ROISignals exists with shape 200x170 and finite_fraction=0.9764705882, "
                "but scale_label=unknown, 96.9% near-zero/no negatives, max~168170, OMST fallback occurred, "
                "and no diagnosis/demographic metadata could be resolved."
            ),
            "intentionally_excluded": "yes",
            "exclusion_or_loss_reason": (
                "invalid_or_missing_diagnosis_metadata|missing_diagnosis plus suspicious historical signal/provenance. "
                "Unknown-subject resolution explicitly marked it unresolved and excluded."
            ),
            "primary_source_files": (
                "results/revision_bspc_2026/adni_expanded_v5_dparsf10000_no_pybandpass_pretraining_qc/unknown_diagnosis_subjects.csv; "
                "results/revision_bspc_2026/adni_v5_dparsf_only_rebuild/pre_full_extraction_qc/unknown_subjects_resolution.csv; "
                "results/revision_bspc_2026/desde_cero_roisignals_aal3_audit/invalid_or_low_finite_signal_subjects.csv; "
                "results/revision_bspc_2026/adni_v5_1_batch20260514b_full_build_qc/subject_alignment.csv"
            ),
            "merge_or_build_stage_lost": (
                "No later merge loss: it was already unresolved/excluded in base v5 QC and remained absent "
                "from all training-ready metadata."
            ),
            "current_dataset_decision": "keep_only_in_tensor_excluded_from_supervised_and_final_vae_pools",
            "future_rebuild_decision": "exclude_from_tensor_and_training",
            "decision": "remove_from_tensor_in_future_rebuild",
            "action_needed": (
                "Keep excluded from current supervised metadata. In a future clean rebuild, remove this subject "
                "from tensor/VAE candidate pool unless independent diagnosis, demographics, visit provenance, "
                "and compatible ROISignals are recovered."
            ),
            "ask_martin": "no",
        },
    ]
    return pd.DataFrame(rows)


def write_markdown_table(df: pd.DataFrame, path: Path, max_col_width: int = 120) -> None:
    display = df.copy()
    for col in display.columns:
        display[col] = display[col].map(lambda x: clean(x)[:max_col_width])
    path.write_text(display.to_markdown(index=False) + "\n")


def write_recommendation(decisions: pd.DataFrame, path: Path) -> None:
    by_sid = {row["SubjectID"]: row for _, row in decisions.iterrows()}
    s035 = by_sid["035_S_6927"]
    s128 = by_sid["128_S_2002"]
    text = f"""# Final Recommendation: Tensor Subjects Missing From Final Metadata

This audit is read-only. It does not modify the tensor, final metadata, ledger, configs, or model outputs.

## Executive Decision

- `035_S_6927`: **recover in future metadata**. The subject is a valid AD first-visit candidate with usable passband AAL3 ROISignals. It was lost from training-ready metadata because demographics/manufacturer were not carried into the base v5 metadata row, even though local ADNI exports contain usable Age/Sex and idaSearch contains Siemens scanner information.
- `128_S_2002`: **keep excluded now and remove from tensor in a future clean rebuild**. The subject has unresolved diagnosis/demographics and suspicious historical ROISignals/provenance. It should not be recovered without independent metadata and signal provenance.

Neither subject appears in the current final 646-row training-ready metadata. Therefore neither appears in the current final AD/CN classifier pool nor in the current final VAE pretraining pool used by the locked model. The current running model should not be modified as a result of this audit.

## 035_S_6927

- Diagnosis: {s035['Diagnosis']}
- Age/Sex: {s035['Age']}; {s035['Sex']}
- Manufacturer/Site/Image/Visit: {s035['Manufacturer']}; Site3={s035['Site3']}; ImageID={s035['ImageID']}; Visit={s035['Visit']}
- ROISignals: {s035['roisignals_evidence']}
- Exclusion status: {s035['intentionally_excluded']}
- Where it was lost: {s035['merge_or_build_stage_lost']}
- Action: {s035['action_needed']}
- Current final classifier AD/CN pool: {s035['appears_in_final_classifier_cn_ad_pool']}
- Current final VAE pretraining pool: {s035['appears_in_final_vae_pretraining_pool']}
- Base pretraining QC flags: use_for_vae={s035['base_pretraining_qc_use_for_vae']}; use_for_supervised={s035['base_pretraining_qc_use_for_supervised']}; use_for_supervised_with_demo={s035['base_pretraining_qc_use_for_supervised_with_demo']}

## 128_S_2002

- Diagnosis: {s128['Diagnosis']}
- Age/Sex: missing
- Manufacturer/Site/Image/Visit: {s128['Manufacturer']}; {s128['Site3']}; ImageID missing; Visit missing
- ROISignals: {s128['roisignals_evidence']}
- Exclusion status: {s128['intentionally_excluded']}
- Reason: {s128['exclusion_or_loss_reason']}
- Action: {s128['action_needed']}
- Current final classifier AD/CN pool: {s128['appears_in_final_classifier_cn_ad_pool']}
- Current final VAE pretraining pool: {s128['appears_in_final_vae_pretraining_pool']}
- Base pretraining QC flags: use_for_vae={s128['base_pretraining_qc_use_for_vae']}; use_for_supervised={s128['base_pretraining_qc_use_for_supervised']}; exclude={s128['base_pretraining_qc_exclude']}

## Build-Stage Interpretation

The tensor has 648 subjects while final training-ready metadata has 646. The mismatch is explained by two base-conservative tensor subjects that were not eligible for final supervised metadata:

1. `035_S_6927` is recoverable because external local metadata supports AD, Age, Sex, ImageID, Visit, and Siemens manufacturer. This is a metadata propagation issue, not a tensor or signal QC failure.
2. `128_S_2002` is not recoverable from current local evidence because it has missing diagnosis/demographics and questionable historical signal scaling/provenance.

No request to Martin is required for these two subjects as a direct result of this audit. `035_S_6927` can be corrected locally in a future metadata rebuild; `128_S_2002` should remain excluded unless independent source metadata and compatible ROISignals are later found.

For future builds, add a build-time assertion: every tensor subject must either have a corresponding metadata row or appear in an explicit exclusion ledger with a machine-readable reason and pool membership flag.
"""
    path.write_text(text)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    started = datetime.now().isoformat(timespec="seconds")
    subjects = tuple(args.subjects)

    tensor_info = tensor_presence(args.tensor, args.final_metadata, subjects)
    pools = pool_membership(args.final_metadata, subjects)
    trace = trace_subject_rows(subjects, args.max_scan_mb)
    version_trace = metadata_version_trace(subjects)
    decisions = build_decisions(tensor_info, pools)

    if not version_trace.empty:
        version_trace_as_trace = version_trace.copy()
        version_trace_as_trace["file"] = version_trace_as_trace["file"].map(str)
        version_trace_as_trace["priority_rank"] = 50
        version_trace_as_trace["matched_rows_total"] = version_trace_as_trace["row_status"]
        version_trace_as_trace["preview_index"] = ""
        version_trace_as_trace["match_type"] = "metadata_version_trace"
        version_trace_as_trace["columns"] = ""
        version_trace_as_trace["file_size_bytes"] = ""
        for col in trace.columns:
            if col not in version_trace_as_trace.columns:
                version_trace_as_trace[col] = ""
        version_trace_as_trace = version_trace_as_trace[trace.columns]
        trace = pd.concat([trace, version_trace_as_trace], ignore_index=True)
        trace = trace.sort_values(["SubjectID", "priority_rank", "file", "preview_index"]).reset_index(drop=True)

    trace_path = output_dir / "missing_subjects_deep_trace.csv"
    decisions_path = output_dir / "per_subject_decision.csv"
    trace.to_csv(trace_path, index=False)
    decisions.to_csv(decisions_path, index=False)
    write_markdown_table(trace, output_dir / "missing_subjects_deep_trace.md")
    write_markdown_table(decisions, output_dir / "per_subject_decision.md")
    write_recommendation(decisions, output_dir / "final_recommendation.md")

    command_log = {
        "script": rel(Path(__file__)),
        "started": started,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "cwd": str(PROJECT_ROOT),
        "subjects": list(subjects),
        "tensor": str(args.tensor),
        "final_metadata": str(args.final_metadata),
        "output_dir": rel(output_dir),
        "read_only_inputs": True,
        "modified_tensor_metadata_ledger_configs_or_training_outputs": False,
        "outputs": [
            rel(trace_path),
            rel(output_dir / "missing_subjects_deep_trace.md"),
            rel(decisions_path),
            rel(output_dir / "per_subject_decision.md"),
            rel(output_dir / "final_recommendation.md"),
            rel(output_dir / "command_log.json"),
        ],
        "trace_rows": int(len(trace)),
        "decision_rows": int(len(decisions)),
        "tensor_presence": tensor_info,
        "pool_membership": pools,
    }
    (output_dir / "command_log.json").write_text(json.dumps(command_log, indent=2, ensure_ascii=False))

    print(json.dumps(command_log, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
