#!/usr/bin/env python3
"""Build a manifest-first ADNI v5 DPARSF-bandpass-only subject table.

This script does not compute connectivity and does not train. It selects one
ROI signal source per subject from an inventory, reconciles metadata, and writes
blocking flags for the final v5 rebuild.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_NAME = "adni_expanded_v5_passband_dparsf_only"
DEFAULT_REBUILD_ROOT = PROJECT_ROOT / "results" / "revision_bspc_2026" / "adni_v5_dparsf_only_rebuild"
DEFAULT_INVENTORY = DEFAULT_REBUILD_ROOT / "roi_signal_inventory" / "roi_signal_subject_summary.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "revision_bspc_2026" / DATASET_NAME
DEFAULT_V4_METADATA = (
    PROJECT_ROOT
    / "data"
    / "revision_bspc_2026"
    / "adni_expanded_v4_all_available"
    / "subject_metadata_adni_expanded_v4_all_available.csv"
)
DEFAULT_ORIGINAL_METADATA = PROJECT_ROOT / "data" / "SubjectsData_AAL3_procesado2.csv"
DEFAULT_ADNI_DOWNLOAD_NOW = PROJECT_ROOT / "data" / "adni_download_now.csv"
DEFAULT_EXTRA_METADATA = [
    PROJECT_ROOT / "data" / "AD_fMRI_4_28_2026.csv",
    PROJECT_ROOT / "data" / "RevisionPaperfMRI_2026_04_4_06_2026.csv",
    PROJECT_ROOT / "data" / "RevisionPaperfMRI_2026_04_4_06_2026_SecondBatch.csv",
    PROJECT_ROOT / "data" / "idaSearch_4_03_2026.csv",
]
DEFAULT_PLAN_PATH = DEFAULT_REBUILD_ROOT / "connectivity_extraction_plan.md"
DEFAULT_TENSOR_OUTPUT_ROOT = Path(
    "/media/diego/My_Book_Diego/vae_AD_data/revision_bspc_2026/adni_expanded_v5_passband_dparsf_only"
)
DEFAULT_LOCAL_SYMLINK = PROJECT_ROOT / "data" / "revision_bspc_2026" / DATASET_NAME

SUBJECT_RE = re.compile(r"(?<!\d)(\d{3}_S_\d{4})(?!\d)", re.IGNORECASE)
SUBJECT_COLS = ["SubjectID", "Subject", "Subject ID", "PTID", "subject_id", "RID"]
DX_COLS = ["ResearchGroup_Mapped", "ResearchGroup", "Research Group", "Group", "DX", "Diagnosis"]
AGE_COLS = ["Age", "AGE"]
SEX_COLS = ["Sex", "PTGENDER", "Gender"]
MANUFACTURER_COLS = ["Manufacturer", "Mfr", "MFG"]
SITE_COLS = ["Site3", "Site", "SITEID", "RID_SITE"]
IMAGE_COLS = ["ImageID", "Image ID", "Image Data ID", "IMAGEUID"]
VISIT_COLS = ["Visit", "VISCODE", "VISCODE2", "VisitCode", "Visit Code"]
DATE_COLS = ["StudyDate", "Study Date", "Acq Date", "Acquisition Date", "EXAMDATE", "Archive Date"]
DESCRIPTION_COLS = ["Description", "Series Description", "ProtocolName"]
PROTOCOL_COLS = ["ImagingProtocol", "Imaging Protocol", "Protocol", "Sequence"]
REASON_COLS = ["reason", "Reason", "selection_reason"]

CHANNEL_NAMES_MASTER = [
    "Pearson_OMST_GCE_Signed_Weighted",
    "Pearson_Full_FisherZ_Signed",
    "MI_KNN_Symmetric",
    "dFC_AbsDiffMean",
    "dFC_StdDev",
    "DistanceCorr",
    "Granger_F_lag1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build v5 DPARSF-only ADNI subject manifest from ROI signal inventory and metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--roi-summary", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--v4-metadata", type=Path, default=DEFAULT_V4_METADATA)
    parser.add_argument("--original-metadata", type=Path, default=DEFAULT_ORIGINAL_METADATA)
    parser.add_argument("--adni-download-now", type=Path, default=DEFAULT_ADNI_DOWNLOAD_NOW)
    parser.add_argument("--extra-metadata", type=Path, nargs="*", default=DEFAULT_EXTRA_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plan-path", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument("--tensor-output-root", type=Path, default=DEFAULT_TENSOR_OUTPUT_ROOT)
    parser.add_argument("--local-symlink", type=Path, default=DEFAULT_LOCAL_SYMLINK)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(f"Output directory exists and is not empty; pass --overwrite: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def normalize_subject(value: Any) -> str:
    if pd.isna(value):
        return ""
    match = SUBJECT_RE.search(str(value).strip().upper())
    return match.group(1).upper() if match else ""


def normalize_scalar(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def normalize_diagnosis(value: Any) -> str:
    text = normalize_scalar(value).upper().replace(" ", "")
    if text in {"CN", "NL", "NORMAL", "CONTROL", "CONTROLS"}:
        return "CN"
    if text in {"AD", "DEMENTIA", "ALZHEIMER", "ALZHEIMERS", "ALZHEIMER'S"}:
        return "AD"
    if text in {"MCI", "EMCI", "LMCI", "SMC"} or "MCI" in text:
        return "MCI"
    return text


def normalize_image_id(value: Any) -> str:
    text = normalize_scalar(value).upper()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.lstrip("I")


def normalize_date(value: Any) -> str:
    text = normalize_scalar(value)
    if not text:
        return ""
    parsed = pd.to_datetime(text, errors="coerce")
    return "" if pd.isna(parsed) else parsed.strftime("%Y-%m-%d")


def normalize_site(value: Any, sid: str) -> str:
    text = normalize_scalar(value)
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if text:
        return text
    return sid.split("_", 1)[0].lstrip("0") or sid.split("_", 1)[0]


def normalize_manufacturer(value: Any) -> str:
    text = normalize_scalar(value)
    upper = text.upper()
    if "SIEMENS" in upper:
        return "SIEMENS"
    if "PHILIPS" in upper:
        return "Philips"
    if "GE" in upper:
        return "GE MEDICAL SYSTEMS"
    return text


def parse_manufacturer_from_protocol(value: Any) -> str:
    match = re.search(r"Manufacturer=([^;]+)", normalize_scalar(value), flags=re.IGNORECASE)
    return normalize_manufacturer(match.group(1)) if match else ""


def first_existing_col(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower = {str(col).strip().lower(): str(col) for col in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        hit = lower.get(candidate.lower())
        if hit:
            return hit
    return None


def read_csv(path: Path) -> pd.DataFrame:
    path = resolve(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def metadata_rows(path: Path, source_kind: str, priority: int) -> pd.DataFrame:
    df = read_csv(path)
    if df.empty:
        return pd.DataFrame()
    sid_col = first_existing_col(df.columns, SUBJECT_COLS)
    if sid_col is None:
        return pd.DataFrame()
    dx_col = first_existing_col(df.columns, DX_COLS)
    age_col = first_existing_col(df.columns, AGE_COLS)
    sex_col = first_existing_col(df.columns, SEX_COLS)
    mfr_col = first_existing_col(df.columns, MANUFACTURER_COLS)
    site_col = first_existing_col(df.columns, SITE_COLS)
    image_col = first_existing_col(df.columns, IMAGE_COLS)
    visit_col = first_existing_col(df.columns, VISIT_COLS)
    date_col = first_existing_col(df.columns, DATE_COLS)
    desc_col = first_existing_col(df.columns, DESCRIPTION_COLS)
    protocol_col = first_existing_col(df.columns, PROTOCOL_COLS)
    reason_col = first_existing_col(df.columns, REASON_COLS)
    out: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        sid = normalize_subject(row.get(sid_col, ""))
        if not sid:
            continue
        mfr = normalize_manufacturer(row.get(mfr_col, "")) if mfr_col else ""
        if not mfr and protocol_col:
            mfr = parse_manufacturer_from_protocol(row.get(protocol_col, ""))
        out.append(
            {
                "SubjectID": sid,
                "ResearchGroup_Mapped": normalize_diagnosis(row.get(dx_col, "")) if dx_col else "",
                "Age": normalize_scalar(row.get(age_col, "")) if age_col else "",
                "Sex": normalize_scalar(row.get(sex_col, "")) if sex_col else "",
                "Manufacturer": mfr,
                "Site3": normalize_site(row.get(site_col, ""), sid) if site_col else normalize_site("", sid),
                "ImageID": normalize_image_id(row.get(image_col, "")) if image_col else "",
                "Visit": normalize_scalar(row.get(visit_col, "")) if visit_col else "",
                "StudyDate": normalize_date(row.get(date_col, "")) if date_col else "",
                "Description": normalize_scalar(row.get(desc_col, "")) if desc_col else "",
                "reason": normalize_scalar(row.get(reason_col, "")) if reason_col else "",
                "metadata_source": str(resolve(path)),
                "source_kind": source_kind,
                "source_priority": priority,
                "source_row_index": int(idx),
            }
        )
    return pd.DataFrame(out)


def metadata_quality(row: Mapping[str, Any]) -> int:
    score = 0
    for col in ["ResearchGroup_Mapped", "Age", "Sex", "Manufacturer", "Site3"]:
        if normalize_scalar(row.get(col, "")):
            score += 2
    for col in ["ImageID", "Visit", "StudyDate", "Description"]:
        if normalize_scalar(row.get(col, "")):
            score += 1
    if "preferred" in normalize_scalar(row.get("reason", "")).lower():
        score += 4
    return score


def visit_rank(value: Any) -> int:
    text = normalize_scalar(value).lower()
    if "initial" in text or "init" in text:
        return 0
    if "baseline" in text or text == "bl":
        return 1
    if "screen" in text or text == "sc":
        return 2
    if text.startswith("v") and text[1:].isdigit():
        return 10 + int(text[1:])
    if text in {"y1", "year1"}:
        return 20
    return 50


def choose_metadata(sub: pd.DataFrame) -> Dict[str, Any]:
    if sub.empty:
        return {}
    ranked = sub.copy()
    ranked["_preferred"] = ranked["reason"].str.contains("preferred", case=False, na=False).map(lambda x: 0 if x else 1)
    ranked["_visit_rank"] = ranked["Visit"].map(visit_rank)
    ranked["_date"] = ranked["StudyDate"].replace("", "9999-99-99")
    ranked["_image"] = pd.to_numeric(ranked["ImageID"], errors="coerce").fillna(float("inf"))
    ranked["_quality"] = ranked.apply(metadata_quality, axis=1)
    ranked = ranked.sort_values(["_preferred", "source_priority", "_visit_rank", "_date", "_image", "_quality"], ascending=[True, True, True, True, True, False])
    return ranked.iloc[0].to_dict()


def combine_metadata(args: argparse.Namespace) -> Tuple[pd.DataFrame, set[str], set[str]]:
    v4_frame = metadata_rows(args.v4_metadata, "v4_metadata", 1)
    original_frame = metadata_rows(args.original_metadata, "original_metadata", 2)
    frames = [
        v4_frame,
        original_frame,
        metadata_rows(args.adni_download_now, "adni_download_now", 3),
    ]
    for path in args.extra_metadata:
        frames.append(metadata_rows(path, "extra_metadata", 4))
    frames = [frame for frame in frames if not frame.empty]
    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    v4_subjects = set(v4_frame["SubjectID"].astype(str)) if not v4_frame.empty else set()
    original_subjects = set(original_frame["SubjectID"].astype(str)) if not original_frame.empty else set()
    return combined, v4_subjects, original_subjects


def expected_subjects_from_download(metadata: pd.DataFrame) -> set[str]:
    if metadata.empty:
        return set()
    sub = metadata[metadata["source_kind"] == "adni_download_now"].copy()
    if sub.empty:
        return set()
    sub = sub[sub["ResearchGroup_Mapped"].map(normalize_diagnosis).eq("AD")].copy()
    if sub.empty:
        return set()
    preferred = sub[sub["reason"].str.contains("preferred", case=False, na=False)].copy()
    return set((preferred if not preferred.empty else sub)["SubjectID"].astype(str))


def build_manifest(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    roi = read_csv(args.roi_summary)
    metadata, v4_subjects, original_subjects = combine_metadata(args)
    signal_subjects = set(roi["SubjectID"].astype(str)) if not roi.empty else set()
    expected_subjects = expected_subjects_from_download(metadata)
    all_subjects = sorted(v4_subjects | signal_subjects | expected_subjects)

    rows: List[Dict[str, Any]] = []
    for sid in all_subjects:
        md_sub = metadata[metadata["SubjectID"] == sid].copy() if not metadata.empty else pd.DataFrame()
        md = choose_metadata(md_sub)
        roi_sub = roi[roi["SubjectID"] == sid].copy() if not roi.empty else pd.DataFrame()
        roi_row = roi_sub.iloc[0].to_dict() if not roi_sub.empty else {}
        txt_path = normalize_scalar(roi_row.get("preferred_txt_path", ""))
        mat_path = normalize_scalar(roi_row.get("preferred_mat_path", ""))
        has_dparsf = bool(str(roi_row.get("preferred_suspected_dparsf_bandpass", "")).lower() in {"true", "1", "yes"})
        missing_signal = not (txt_path and has_dparsf)
        missing_metadata = not (
            normalize_scalar(md.get("ResearchGroup_Mapped", ""))
            and normalize_scalar(md.get("Age", ""))
            and normalize_scalar(md.get("Sex", ""))
        )
        candidate = not missing_signal and not missing_metadata
        rows.append(
            {
                "SubjectID": sid,
                "ResearchGroup_Mapped": md.get("ResearchGroup_Mapped", ""),
                "Age": md.get("Age", ""),
                "Sex": md.get("Sex", ""),
                "Manufacturer": md.get("Manufacturer", ""),
                "Site3": md.get("Site3", sid.split("_", 1)[0].lstrip("0")),
                "ImageID": md.get("ImageID", ""),
                "Visit": md.get("Visit", ""),
                "StudyDate": md.get("StudyDate", ""),
                "metadata_source": md.get("metadata_source", "MISSING"),
                "metadata_hit_count": int(len(md_sub)),
                "already_in_v4": sid in v4_subjects,
                "already_in_original": sid in original_subjects,
                "expected_from_adni_download_now": sid in expected_subjects,
                "txt_path": txt_path,
                "mat_path": mat_path,
                "source_root": normalize_scalar(roi_row.get("preferred_source_root", "")),
                "source_batch": normalize_scalar(roi_row.get("preferred_source_batch", "")),
                "n_signal_sources": roi_row.get("n_signal_sources", 0),
                "has_dparsf_bandpass_signal": has_dparsf,
                "txt_shape": normalize_scalar(roi_row.get("preferred_txt_shape", "")),
                "txt_rows": roi_row.get("preferred_txt_rows", np.nan),
                "txt_cols": roi_row.get("preferred_txt_cols", np.nan),
                "preprocessing_tag": "DPARSF_bandpass_only",
                "python_bandpass_applied": False,
                "TR": 3.0,
                "target_len": 140,
                "roi_input_count": 170,
                "roi_output_count": 131,
                "candidate_for_training": candidate,
                "missing_metadata": missing_metadata,
                "missing_signal": missing_signal,
                "blocking_reason": blocking_reason(candidate, missing_metadata, missing_signal, has_dparsf, txt_path),
            }
        )
    manifest = pd.DataFrame(rows).sort_values("SubjectID")
    metadata_cols = [
        "SubjectID",
        "ResearchGroup_Mapped",
        "Age",
        "Sex",
        "Manufacturer",
        "Site3",
        "ImageID",
        "Visit",
        "StudyDate",
        "metadata_source",
        "preprocessing_tag",
        "python_bandpass_applied",
        "TR",
        "target_len",
        "roi_input_count",
        "roi_output_count",
        "candidate_for_training",
    ]
    return manifest, manifest[metadata_cols].copy(), metadata


def blocking_reason(candidate: bool, missing_metadata: bool, missing_signal: bool, has_dparsf: bool, txt_path: str) -> str:
    if candidate:
        return ""
    parts: List[str] = []
    if missing_metadata:
        parts.append("missing_metadata")
    if not txt_path:
        parts.append("missing_txt_signal")
    elif missing_signal and not has_dparsf:
        parts.append("txt_signal_not_confirmed_dparsf_bandpass")
    elif missing_signal:
        parts.append("missing_dparsf_bandpass_signal")
    return "|".join(parts)


def count_diag(df: pd.DataFrame, candidate_only: bool = False) -> str:
    sub = df[df["candidate_for_training"].astype(bool)] if candidate_only and "candidate_for_training" in df.columns else df
    counts = sub["ResearchGroup_Mapped"].map(normalize_diagnosis).value_counts(dropna=False).to_dict()
    return ", ".join(
        [
            f"CN={int(counts.get('CN', 0))}",
            f"AD={int(counts.get('AD', 0))}",
            f"MCI={int(counts.get('MCI', 0))}",
            f"UNKNOWN={int(counts.get('', 0))}",
        ]
    )


def write_readme(path: Path, manifest: pd.DataFrame) -> None:
    final_n = int(len(manifest))
    candidate_n = int(manifest["candidate_for_training"].astype(bool).sum())
    dparsf_n = int(manifest["has_dparsf_bandpass_signal"].astype(bool).sum())
    v4_missing_signal = int((manifest["already_in_v4"].astype(bool) & manifest["missing_signal"].astype(bool)).sum())
    blockers = manifest[~manifest["candidate_for_training"].astype(bool)].copy()
    missing_114 = bool((manifest["SubjectID"] == "114_S_6039").any() and manifest.loc[manifest["SubjectID"] == "114_S_6039", "missing_signal"].astype(bool).any())
    s301 = manifest[manifest["SubjectID"] == "301_S_6592"]
    s301_ad = bool(not s301.empty and normalize_diagnosis(s301.iloc[0]["ResearchGroup_Mapped"]) == "AD")
    lines = [
        "# ADNI v5 DPARSF-Only Subject Manifest",
        "",
        "Manifest-only rebuild step. No connectivity matrices were computed and no model training was run.",
        "",
        "## Explicit Answers",
        "",
        f"- Subjects in candidate manifest union: `{final_n}`",
        f"- Diagnosis counts in manifest: `{count_diag(manifest)}`",
        f"- Candidate-for-training subjects with complete metadata and DPARSF-bandpass txt: `{candidate_n}`",
        f"- Candidate-for-training diagnosis counts: `{count_diag(manifest, candidate_only=True)}`",
        f"- Subjects with confirmed DPARSF-bandpass signal: `{dparsf_n}`",
        f"- v4 subjects missing confirmed DPARSF-bandpass signal: `{v4_missing_signal}`",
        f"- Can reconstruct complete final dataset now? `{'YES' if len(blockers) == 0 else 'NO'}`",
        f"- Is `114_S_6039` missing? `{'YES' if missing_114 else 'NO'}`",
        f"- Does `301_S_6592` enter as AD? `{'YES' if s301_ad else 'NO'}`",
        f"- Subjects blocking training: `{len(blockers)}`",
        "",
        "## Blocking Rule",
        "",
        "A subject is blocked if metadata is incomplete or if a DPARSF-bandpass ROI `.txt` signal is missing. Existing tensors/matrices are intentionally ignored.",
        "",
        "## Top Blocking Reasons",
        "",
    ]
    if blockers.empty:
        lines.append("- None.")
    else:
        reason_counts = blockers["blocking_reason"].value_counts().head(20)
        lines.extend(f"- `{reason}`: `{count}`" for reason, count in reason_counts.items())
    lines.extend(
        [
            "",
            "## Methodological Status",
            "",
            "The final v5 rebuild is not yet cleared for connectivity extraction unless every retained final-training subject has a confirmed DPARSF-bandpass ROI signal source. Python bandpass must remain OFF.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plan(path: Path, manifest: pd.DataFrame, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blockers = manifest[~manifest["candidate_for_training"].astype(bool)]
    lines = [
        "# Connectivity Extraction Plan: adni_expanded_v5_passband_dparsf_only",
        "",
        "Prepared only. Do not run until the subject manifest is reviewed and confirmed.",
        "",
        "## Contract",
        "",
        "- Python bandpass: `OFF`",
        "- Input signals: DPARSF-bandpass ROI time series (`ROISignals_*.txt`) selected by manifest.",
        "- Recompute connectivity for every subject in the confirmed final training dataset.",
        "- Do not mix with old tensors or connectivity matrices generated with Python bandpass ON.",
        "- ROI reduction: `170 -> 131` using `data/ROI_MNI_V7_vol.txt`.",
        "- ROI reorder: Yeo-17/manual network order using `data/aal3_131_manual_network_order.csv`.",
        "- target_len: `140`",
        "- TR: `3.0`",
        "- Channels: all 7 historical channels:",
    ]
    lines.extend(f"  - `{idx}`: `{name}`" for idx, name in enumerate(CHANNEL_NAMES_MASTER))
    lines.extend(
        [
            "",
            "## Output",
            "",
            f"- Big-disk output root: `{args.tensor_output_root}`",
            f"- Local symlink: `{args.local_symlink}`",
            "",
            "## Gate",
            "",
            f"- Manifest rows: `{len(manifest)}`",
            f"- Candidate-for-training rows: `{int(manifest['candidate_for_training'].astype(bool).sum())}`",
            f"- Blocking rows: `{len(blockers)}`",
            "- Connectivity extraction status: `BLOCKED_UNTIL_MANIFEST_CONFIRMED`",
            "",
            "## Future Dry-Run Command Placeholder",
            "",
            "```bash",
            "/home/diego/anaconda3/envs/vae_ad/bin/python scripts/revision_bspc_2026/extract_adni_v5_dparsf_only_connectivity.py --manifest data/revision_bspc_2026/adni_expanded_v5_passband_dparsf_only/subject_manifest_adni_expanded_v5_passband_dparsf_only.csv --output-root /media/diego/My_Book_Diego/vae_AD_data/revision_bspc_2026/adni_expanded_v5_passband_dparsf_only --python-bandpass off --dry-run",
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.roi_summary = resolve(args.roi_summary)
    args.v4_metadata = resolve(args.v4_metadata)
    args.original_metadata = resolve(args.original_metadata)
    args.adni_download_now = resolve(args.adni_download_now)
    args.extra_metadata = [resolve(path) for path in args.extra_metadata]
    args.output_dir = resolve(args.output_dir)
    args.plan_path = resolve(args.plan_path)
    args.local_symlink = resolve(args.local_symlink)

    prepare_output_dir(args.output_dir, args.overwrite)
    manifest, metadata, raw_metadata_hits = build_manifest(args)
    manifest_path = args.output_dir / f"subject_manifest_{DATASET_NAME}.csv"
    metadata_path = args.output_dir / f"subject_metadata_{DATASET_NAME}.csv"
    manifest.to_csv(manifest_path, index=False)
    metadata.to_csv(metadata_path, index=False)
    raw_metadata_hits.to_csv(args.output_dir / "metadata_raw_hits_for_manifest.csv", index=False)
    write_readme(args.output_dir / "README.md", manifest)
    write_plan(args.plan_path, manifest, args)

    print(f"Wrote manifest to {manifest_path}")
    print(f"Wrote metadata to {metadata_path}")
    print(f"Wrote plan to {args.plan_path}")
    print(f"Manifest rows: {len(manifest)}")
    print(f"Candidate-for-training rows: {int(manifest['candidate_for_training'].astype(bool).sum())}")
    print(f"Blocking rows: {int((~manifest['candidate_for_training'].astype(bool)).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
