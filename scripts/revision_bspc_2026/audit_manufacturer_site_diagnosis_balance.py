#!/usr/bin/env python3
"""
Audit diagnostic balance by manufacturer and site for the BSPC 2026 ADNI expansion.

The script is read-only for input data and writes derived audit tables to:
results/revision_bspc_2026/dataset_balance_audit/
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "results" / "revision_bspc_2026" / "dataset_balance_audit"

HISTORICAL_METADATA = PROJECT_ROOT / "data" / "SubjectsData_AAL3_procesado2.csv"
EXPANDED_V1_METADATA = (
    PROJECT_ROOT
    / "data"
    / "revision_bspc_2026"
    / "adni_expanded_v1"
    / "subject_metadata_adni_expanded_v1.csv"
)
EXPANDED_V2_METADATA = (
    PROJECT_ROOT
    / "data"
    / "revision_bspc_2026"
    / "adni_expanded_v2"
    / "subject_metadata_adni_expanded_v2.csv"
)

CANDIDATE_PATHS = [
    PROJECT_ROOT / "data" / "RevisionPaperfMRI_2026_04_4_06_2026.csv",
    PROJECT_ROOT / "data" / "RevisionPaperfMRI_2026_04_4_06_2026_SecondBatch.csv",
    PROJECT_ROOT / "data" / "adni_download_now.csv",
]
BATCH_DIR = PROJECT_ROOT / "data" / "RevisionPaperfMRI_2026_04_pilot" / "batches"
OUTPUT_ROOTS = [
    Path("/media/diego/Datos/adni_expansion/MARTIN59"),
    Path("/media/diego/Datos/adni_expansion/SIEMENS_available"),
    Path("/media/diego/Datos/adni_expansion/GE_smoketest3"),
    Path("/media/diego/Datos/adni_expansion/GE_batch7"),
]

REQUIRED_COLUMNS = [
    "SubjectID",
    "ResearchGroup_Mapped",
    "Manufacturer",
    "Site3",
    "Age",
    "Sex",
    "ImageID",
]
DIAGNOSES = ["CN", "AD", "MCI"]
CN_AD = ["CN", "AD"]
SUBJECT_ID_RE = re.compile(r"\d{3}_S_\d{4}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit ADNI diagnostic balance by Manufacturer, Site3, and candidate availability.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--historical-metadata", type=Path, default=HISTORICAL_METADATA)
    parser.add_argument("--expanded-v1-metadata", type=Path, default=EXPANDED_V1_METADATA)
    parser.add_argument("--expanded-v2-metadata", type=Path, default=EXPANDED_V2_METADATA)
    parser.add_argument("--candidate-csv", type=Path, action="append", default=None)
    parser.add_argument("--batch-dir", type=Path, default=BATCH_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--reference-dataset",
        choices=["historical", "expanded_v1", "expanded_v2"],
        default="expanded_v2",
        help="Dataset used to score candidate priorities.",
    )
    parser.add_argument(
        "--manufacturer-min-cn-ad",
        type=int,
        default=5,
        help="Minimum CN and AD count target per manufacturer.",
    )
    return parser.parse_args()


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def normalize_subject_id(value: object) -> str:
    return str(value).strip()


def normalize_image_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.startswith("I") and text[1:].isdigit():
        text = text[1:]
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def normalize_site3(value: object, subject_id: Optional[str] = None) -> str:
    if pd.notna(value) and str(value).strip() != "":
        text = str(value).strip()
    elif subject_id:
        text = str(subject_id).split("_", 1)[0]
    else:
        return ""
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


def manufacturer_token(value: str) -> str:
    value = normalize_manufacturer(value)
    if value == "GE MEDICAL SYSTEMS":
        return "GE"
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").upper()


def validate_columns(df: pd.DataFrame, label: str, columns: Sequence[str] = REQUIRED_COLUMNS) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise RuntimeError(f"{label} missing required columns: {missing}")
    empty_required = [
        c
        for c in ["SubjectID", "ResearchGroup_Mapped", "Manufacturer", "Site3"]
        if df[c].isna().any() or (df[c].astype(str).str.strip() == "").any()
    ]
    if empty_required:
        raise RuntimeError(f"{label} has empty values in required columns: {empty_required}")


def standardize_metadata(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rename_map = {
        "Subject": "SubjectID",
        "Subject ID": "SubjectID",
        "Group": "ResearchGroup_Mapped",
        "Research Group": "ResearchGroup_Mapped",
        "Image Data ID": "ImageID",
        "Image ID": "ImageID",
        "FieldStrength": "Field Strength",
        "SliceThickness": "Slice Thickness",
    }
    out = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}).copy()
    if "ResearchGroup_Mapped" not in out.columns and "ResearchGroup" in out.columns:
        out["ResearchGroup_Mapped"] = out["ResearchGroup"]
    if "Site3" not in out.columns and "SubjectID" in out.columns:
        out["Site3"] = out["SubjectID"].map(lambda sid: normalize_site3(None, normalize_subject_id(sid)))
    if "ImageID" not in out.columns:
        out["ImageID"] = ""

    validate_columns(out, label)

    out["SubjectID"] = out["SubjectID"].map(normalize_subject_id)
    out["ResearchGroup_Mapped"] = out["ResearchGroup_Mapped"].map(normalize_diagnosis)
    out["Manufacturer"] = out["Manufacturer"].map(normalize_manufacturer)
    out["Site3"] = [
        normalize_site3(site, sid) for site, sid in zip(out["Site3"], out["SubjectID"])
    ]
    out["ImageID"] = out["ImageID"].map(normalize_image_id)
    out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
    out["Sex"] = out["Sex"].astype(str).str.strip()

    dup_mask = out["SubjectID"].duplicated(keep=False)
    if dup_mask.any():
        out = out.drop_duplicates("SubjectID", keep="first").reset_index(drop=True)

    return out


def load_dataset(path: Path, label: str) -> pd.DataFrame:
    ensure_exists(path, label)
    df = pd.read_csv(path)
    out = standardize_metadata(df, label)
    out["dataset"] = label
    return out


def pivot_counts(df: pd.DataFrame, index_cols: Sequence[str], diagnoses: Sequence[str] = DIAGNOSES) -> pd.DataFrame:
    filtered = df[df["ResearchGroup_Mapped"].isin(diagnoses)].copy()
    counts = (
        filtered.groupby(list(index_cols) + ["ResearchGroup_Mapped"], dropna=False)
        .size()
        .unstack(fill_value=0)
    )
    for diag in diagnoses:
        if diag not in counts.columns:
            counts[diag] = 0
    counts = counts[list(diagnoses)].reset_index()
    counts.columns.name = None
    return counts


def build_manufacturer_counts(datasets: Dict[str, pd.DataFrame], min_target: int) -> pd.DataFrame:
    frames = []
    for name, df in datasets.items():
        table = pivot_counts(df, ["Manufacturer"], DIAGNOSES)
        table.insert(0, "dataset", name)
        table["total_cn_ad_mci"] = table[DIAGNOSES].sum(axis=1)
        table["total_cn_ad"] = table["CN"] + table["AD"]
        table["CN_AD_abs_diff"] = (table["CN"] - table["AD"]).abs()
        table["CN_deficit_to_min_target"] = (min_target - table["CN"]).clip(lower=0)
        table["AD_deficit_to_min_target"] = (min_target - table["AD"]).clip(lower=0)
        frames.append(table)
    return pd.concat(frames, ignore_index=True).sort_values(["dataset", "Manufacturer"])


def build_manufacturer_proportions(counts: pd.DataFrame) -> pd.DataFrame:
    out = counts.copy()
    denom_all = out["total_cn_ad_mci"].replace(0, np.nan)
    denom_cn_ad = out["total_cn_ad"].replace(0, np.nan)
    for diag in DIAGNOSES:
        out[f"prop_{diag}_of_cn_ad_mci"] = out[diag] / denom_all
    for diag in CN_AD:
        out[f"prop_{diag}_of_cn_ad"] = out[diag] / denom_cn_ad
    keep = [
        "dataset",
        "Manufacturer",
        "total_cn_ad_mci",
        "total_cn_ad",
        "prop_CN_of_cn_ad_mci",
        "prop_AD_of_cn_ad_mci",
        "prop_MCI_of_cn_ad_mci",
        "prop_CN_of_cn_ad",
        "prop_AD_of_cn_ad",
    ]
    return out[keep].sort_values(["dataset", "Manufacturer"])


def build_site_counts(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for name, df in datasets.items():
        table = pivot_counts(df, ["Manufacturer", "Site3"], DIAGNOSES)
        site = pivot_counts(df, ["Site3"], CN_AD)
        site = site.rename(columns={"CN": "site_CN", "AD": "site_AD"})
        table = table.merge(site, on="Site3", how="left", validate="many_to_one")
        table.insert(0, "dataset", name)
        table["total_cn_ad_mci"] = table[DIAGNOSES].sum(axis=1)
        table["total_cn_ad"] = table["CN"] + table["AD"]
        table["manufacturer_site_valid_loso_cn_ad_min2"] = (table["CN"] >= 2) & (table["AD"] >= 2)
        table["manufacturer_site_valid_loso_cn_ad_min1"] = (table["CN"] >= 1) & (table["AD"] >= 1)
        table["site_valid_loso_cn_ad_min2"] = (table["site_CN"] >= 2) & (table["site_AD"] >= 2)
        table["site_valid_loso_cn_ad_min1"] = (table["site_CN"] >= 1) & (table["site_AD"] >= 1)
        frames.append(table)
    return pd.concat(frames, ignore_index=True).sort_values(
        ["dataset", "Manufacturer", "Site3"], kind="stable"
    )


def read_subject_ids_from_npz(path: Path) -> Set[str]:
    try:
        with np.load(path, allow_pickle=True) as npz:
            if "subject_ids" not in npz:
                return set()
            return {normalize_subject_id(x) for x in np.asarray(npz["subject_ids"]).reshape(-1)}
    except Exception:
        return set()


def collect_processed_output_ids(output_roots: Sequence[Path]) -> Dict[str, Set[str]]:
    result: Dict[str, Set[str]] = {}
    for root in output_roots:
        ids: Set[str] = set()
        if root.exists():
            for npz_path in root.glob("**/GLOBAL_TENSOR*.npz"):
                ids.update(read_subject_ids_from_npz(npz_path))
            for tensor_path in root.glob("**/tensor_7ch_131rois_*.npz"):
                ids.update(SUBJECT_ID_RE.findall(tensor_path.name))
        result[root.name] = ids
    return result


def is_assignment_batch_file(path: Path) -> bool:
    name = path.name.lower()
    if name in {
        "adni_expansion_assignment_report.csv",
        "adni_expansion_assignment_report.json",
        "siemens_covregressed_available_now.txt",
        "siemens_done_covregressed.txt",
        "siemens_ready_for_dparsf.txt",
        "philips_done_covregressed.txt",
        "ge_done_covregressed.txt",
    }:
        return True
    prefixes = (
        "martin_secondbatch",
        "santiago_",
        "ge_batch7",
        "siemens_batch_001",
    )
    return name.startswith(prefixes)


def collect_assigned_batch_ids(batch_dir: Path) -> Dict[str, Set[str]]:
    result: Dict[str, Set[str]] = {}
    if not batch_dir.exists():
        return result
    for path in sorted(batch_dir.glob("*")):
        if not path.is_file() or not is_assignment_batch_file(path):
            continue
        text = path.read_text(errors="ignore")
        ids = set(SUBJECT_ID_RE.findall(text))
        if ids:
            result[path.name] = ids
    return result


def make_lookup(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    keep = ["SubjectID", *[c for c in columns if c in df.columns]]
    return df[keep].drop_duplicates("SubjectID", keep="first")


def load_candidate_csv(path: Path) -> pd.DataFrame:
    ensure_exists(path, f"candidate CSV {path.name}")
    raw = pd.read_csv(path)
    source_name = path.name

    rename_map = {
        "Subject": "SubjectID",
        "Image Data ID": "ImageID",
        "Group": "ResearchGroup_Mapped",
        "ResearchGroup": "ResearchGroup_Mapped",
        "FieldStrength": "Field Strength",
        "SliceThickness": "Slice Thickness",
    }
    df = raw.rename(columns={k: v for k, v in rename_map.items() if k in raw.columns}).copy()
    if "SubjectID" not in df.columns:
        raise RuntimeError(f"{path} cannot be normalized: missing Subject/SubjectID")
    if "ResearchGroup_Mapped" not in df.columns:
        raise RuntimeError(f"{path} cannot be normalized: missing Group/ResearchGroup")
    if "ImageID" not in df.columns:
        df["ImageID"] = ""
    if "Manufacturer" not in df.columns:
        df["Manufacturer"] = np.nan
    if "Site3" not in df.columns:
        df["Site3"] = df["SubjectID"].map(lambda sid: normalize_site3(None, normalize_subject_id(sid)))
    if "Age" not in df.columns:
        df["Age"] = np.nan
    if "Sex" not in df.columns:
        df["Sex"] = ""

    df["SubjectID"] = df["SubjectID"].map(normalize_subject_id)
    df["ResearchGroup_Mapped"] = df["ResearchGroup_Mapped"].map(normalize_diagnosis)
    df["Manufacturer"] = df["Manufacturer"].map(
        lambda x: normalize_manufacturer(x) if pd.notna(x) and str(x).strip() else np.nan
    )
    df["Site3"] = [normalize_site3(site, sid) for site, sid in zip(df["Site3"], df["SubjectID"])]
    df["ImageID"] = df["ImageID"].map(normalize_image_id)
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Sex"] = df["Sex"].astype(str).str.strip()
    df["source_file"] = source_name
    return df


def combine_candidates(candidate_paths: Sequence[Path], batch_dir: Path) -> pd.DataFrame:
    frames = [load_candidate_csv(path) for path in candidate_paths]
    combined = pd.concat(frames, ignore_index=True, sort=False)

    enrichment_frames = []
    for path in candidate_paths:
        if path.name == "adni_download_now.csv":
            enrichment_frames.append(load_candidate_csv(path))

    assignment = batch_dir / "adni_expansion_assignment_report.csv"
    if assignment.exists():
        assignment_df = pd.read_csv(assignment)
        assignment_df = assignment_df.rename(columns={"Group": "ResearchGroup_Mapped"}).copy()
        for col in ["SubjectID", "ImageID", "ResearchGroup_Mapped", "Manufacturer", "Age_master", "Sex_master"]:
            if col not in assignment_df.columns:
                assignment_df[col] = np.nan
        assignment_df["Age"] = assignment_df["Age_master"]
        assignment_df["Sex"] = assignment_df["Sex_master"]
        assignment_df["Site3"] = assignment_df["SubjectID"].map(lambda sid: normalize_site3(None, sid))
        assignment_df["source_file"] = assignment.name
        assignment_df = assignment_df[
            ["SubjectID", "ImageID", "ResearchGroup_Mapped", "Manufacturer", "Site3", "Age", "Sex", "source_file"]
        ]
        assignment_df["SubjectID"] = assignment_df["SubjectID"].map(normalize_subject_id)
        assignment_df["ResearchGroup_Mapped"] = assignment_df["ResearchGroup_Mapped"].map(normalize_diagnosis)
        assignment_df["Manufacturer"] = assignment_df["Manufacturer"].map(normalize_manufacturer)
        assignment_df["ImageID"] = assignment_df["ImageID"].map(normalize_image_id)
        assignment_df["Age"] = pd.to_numeric(assignment_df["Age"], errors="coerce")
        enrichment_frames.append(assignment_df)

    disk_status = batch_dir / "adni_expansion_disk_status_all.csv"
    if disk_status.exists():
        disk_df = pd.read_csv(disk_status)
        disk_df = disk_df.rename(
            columns={
                "ResearchGroup": "ResearchGroup_Mapped",
                "Manufacturer_raw": "Manufacturer",
            }
        ).copy()
        disk_df["Site3"] = disk_df["SubjectID"].map(lambda sid: normalize_site3(None, sid))
        disk_df["Age"] = np.nan
        disk_df["Sex"] = ""
        disk_df["source_file"] = disk_status.name
        disk_df = disk_df[
            ["SubjectID", "ImageID", "ResearchGroup_Mapped", "Manufacturer", "Site3", "Age", "Sex", "source_file"]
        ]
        disk_df["SubjectID"] = disk_df["SubjectID"].map(normalize_subject_id)
        disk_df["ResearchGroup_Mapped"] = disk_df["ResearchGroup_Mapped"].map(normalize_diagnosis)
        disk_df["Manufacturer"] = disk_df["Manufacturer"].map(normalize_manufacturer)
        disk_df["ImageID"] = disk_df["ImageID"].map(normalize_image_id)
        enrichment_frames.append(disk_df)

    if enrichment_frames:
        enrichment = pd.concat(enrichment_frames, ignore_index=True, sort=False)
        enrichment = enrichment.drop_duplicates("SubjectID", keep="first")
        combined = combined.merge(
            enrichment.add_suffix("_enrich"),
            left_on="SubjectID",
            right_on="SubjectID_enrich",
            how="left",
            validate="many_to_one",
        )
        for col in ["ImageID", "ResearchGroup_Mapped", "Manufacturer", "Site3", "Age", "Sex"]:
            enrich_col = f"{col}_enrich"
            if enrich_col in combined.columns:
                missing = combined[col].isna() | (combined[col].astype(str).str.strip() == "")
                combined.loc[missing, col] = combined.loc[missing, enrich_col]
        if "SubjectID_enrich" in combined.columns:
            combined = combined.drop(columns=[c for c in combined.columns if c.endswith("_enrich")])

    combined["Manufacturer"] = combined["Manufacturer"].map(normalize_manufacturer)
    combined["ResearchGroup_Mapped"] = combined["ResearchGroup_Mapped"].map(normalize_diagnosis)
    combined["Site3"] = [normalize_site3(site, sid) for site, sid in zip(combined["Site3"], combined["SubjectID"])]
    combined["ImageID"] = combined["ImageID"].map(normalize_image_id)

    validate_columns(combined, "combined candidates")

    aggregation: Dict[str, object] = {
        "ImageID": lambda s: first_non_empty(s),
        "ResearchGroup_Mapped": lambda s: first_non_empty(s),
        "Manufacturer": lambda s: first_non_empty(s),
        "Site3": lambda s: first_non_empty(s),
        "Age": lambda s: first_non_empty(s),
        "Sex": lambda s: first_non_empty(s),
        "source_file": lambda s: ";".join(sorted(set(str(x) for x in s if str(x).strip()))),
    }
    out = combined.groupby("SubjectID", as_index=False).agg(aggregation)
    out["ResearchGroup_Mapped"] = out["ResearchGroup_Mapped"].map(normalize_diagnosis)
    out["Manufacturer"] = out["Manufacturer"].map(normalize_manufacturer)
    out["Site3"] = [normalize_site3(site, sid) for site, sid in zip(out["Site3"], out["SubjectID"])]
    out["ImageID"] = out["ImageID"].map(normalize_image_id)
    validate_columns(out, "deduplicated candidates")
    return out


def first_non_empty(values: Iterable[object]) -> object:
    for value in values:
        if pd.notna(value) and str(value).strip() != "":
            return value
    return np.nan


def build_exclusion_sources(
    datasets: Dict[str, pd.DataFrame],
    processed_output_ids: Dict[str, Set[str]],
    assigned_batch_ids: Dict[str, Set[str]],
) -> Dict[str, Set[str]]:
    sources: Dict[str, Set[str]] = {}
    sources["historical"] = set(datasets["historical"]["SubjectID"])
    sources["expanded_v1"] = set(datasets["expanded_v1"]["SubjectID"])
    sources["expanded_v2"] = set(datasets["expanded_v2"]["SubjectID"])
    for name, ids in processed_output_ids.items():
        sources[f"processed_output:{name}"] = ids
    martin_assigned: Set[str] = set()
    other_assigned: Set[str] = set()
    for name, ids in assigned_batch_ids.items():
        if name.lower().startswith("martin_secondbatch"):
            martin_assigned.update(ids)
        else:
            other_assigned.update(ids)
    sources["assigned_batch:martin_secondbatch"] = martin_assigned
    sources["assigned_batch:other_used_batches"] = other_assigned
    return sources


def annotate_exclusions(candidates: pd.DataFrame, exclusion_sources: Dict[str, Set[str]]) -> pd.DataFrame:
    rows = []
    for _, row in candidates.iterrows():
        sid = row["SubjectID"]
        reasons = [name for name, ids in exclusion_sources.items() if sid in ids]
        out = row.to_dict()
        out["is_excluded"] = bool(reasons)
        out["exclusion_reason"] = ";".join(reasons)
        rows.append(out)
    return pd.DataFrame(rows)


def current_counts_maps(reference: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    man_counts = pivot_counts(reference, ["Manufacturer"], CN_AD)
    site_counts = pivot_counts(reference, ["Manufacturer", "Site3"], CN_AD)
    return man_counts, site_counts


def score_candidates(
    candidates: pd.DataFrame,
    reference: pd.DataFrame,
    min_target: int,
) -> pd.DataFrame:
    man_counts, site_counts = current_counts_maps(reference)
    man_lookup = man_counts.set_index("Manufacturer")[CN_AD].to_dict("index")
    site_lookup = site_counts.set_index(["Manufacturer", "Site3"])[CN_AD].to_dict("index")

    other_counts = man_counts[man_counts["Manufacturer"].isin(["Philips", "SIEMENS"])]
    ge_reference = {
        diag: float(other_counts[diag].median()) if not other_counts.empty else 0.0
        for diag in CN_AD
    }

    scored = []
    for _, row in candidates.iterrows():
        diag = row["ResearchGroup_Mapped"]
        man = row["Manufacturer"]
        site = row["Site3"]
        token = manufacturer_token(man)
        reasons: List[str] = []
        score = 0

        m_counts = man_lookup.get(man, {"CN": 0, "AD": 0})
        s_counts = site_lookup.get((man, site), {"CN": 0, "AD": 0})
        current_cn = int(s_counts.get("CN", 0))
        current_ad = int(s_counts.get("AD", 0))

        if diag == "AD" and current_cn > 0 and current_ad == 0:
            score += 100
            reasons.extend(["LOSO_cell_completion", f"{token}_AD_needed"])
        if diag == "CN" and current_ad > 0 and current_cn == 0:
            score += 100
            reasons.extend(["LOSO_cell_completion", f"{token}_CN_needed"])
        if diag in CN_AD:
            if (diag == "CN" and current_ad >= 1 and current_cn < 2) or (
                diag == "AD" and current_cn >= 1 and current_ad < 2
            ):
                score += 35
                reasons.append("LOSO_min2_strengthening")

            man_diag_count = int(m_counts.get(diag, 0))
            if man_diag_count < min_target:
                score += 70 + (min_target - man_diag_count)
                reasons.append(f"{token}_{diag}_needed")

            if man == "GE MEDICAL SYSTEMS":
                ge_current = man_diag_count
                if ge_current < ge_reference.get(diag, 0):
                    score += 30
                    reasons.append(f"GE_{diag}_underrepresented")
                    if f"GE_{diag}_needed" not in reasons:
                        reasons.append(f"GE_{diag}_needed")

        if not reasons and diag in CN_AD:
            score += 5
            reasons.append("general_CN_AD_pool")

        out = row.to_dict()
        out["current_cell_CN"] = current_cn
        out["current_cell_AD"] = current_ad
        out["current_manufacturer_CN"] = int(m_counts.get("CN", 0))
        out["current_manufacturer_AD"] = int(m_counts.get("AD", 0))
        out["priority_score"] = int(score)
        out["reason"] = ";".join(dict.fromkeys(reasons))
        scored.append(out)

    result = pd.DataFrame(scored)
    return result.sort_values(
        ["priority_score", "Manufacturer", "Site3", "ResearchGroup_Mapped", "SubjectID"],
        ascending=[False, True, True, True, True],
    ).reset_index(drop=True)


def build_missing_recommendations(
    reference: pd.DataFrame,
    scored_candidates: pd.DataFrame,
    min_target: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    _, site_counts = current_counts_maps(reference)

    for _, row in site_counts.iterrows():
        cn = int(row["CN"])
        ad = int(row["AD"])
        if cn > 0 and ad == 0:
            need_diag = "AD"
        elif ad > 0 and cn == 0:
            need_diag = "CN"
        else:
            continue
        cand = scored_candidates[
            (scored_candidates["Manufacturer"] == row["Manufacturer"])
            & (scored_candidates["Site3"] == str(row["Site3"]))
            & (scored_candidates["ResearchGroup_Mapped"] == need_diag)
        ].copy()
        rows.append(
            {
                "recommendation_type": "LOSO_cell_completion",
                "Manufacturer": row["Manufacturer"],
                "Site3": row["Site3"],
                "needed_diagnosis": need_diag,
                "current_CN": cn,
                "current_AD": ad,
                "target_CN": None,
                "target_AD": None,
                "available_candidate_count": int(len(cand)),
                "top_candidate_subjects": ";".join(cand.head(10)["SubjectID"].astype(str).tolist()),
                "reason": f"{manufacturer_token(row['Manufacturer'])}_{need_diag}_needed",
            }
        )

    man_counts, _ = current_counts_maps(reference)
    for _, row in man_counts.iterrows():
        for diag in CN_AD:
            deficit = max(0, int(min_target - row[diag]))
            if deficit <= 0:
                continue
            cand = scored_candidates[
                (scored_candidates["Manufacturer"] == row["Manufacturer"])
                & (scored_candidates["ResearchGroup_Mapped"] == diag)
            ]
            rows.append(
                {
                    "recommendation_type": "manufacturer_min_cn_ad",
                    "Manufacturer": row["Manufacturer"],
                    "Site3": "",
                    "needed_diagnosis": diag,
                    "current_CN": int(row["CN"]),
                    "current_AD": int(row["AD"]),
                    "target_CN": min_target,
                    "target_AD": min_target,
                    "available_candidate_count": int(len(cand)),
                    "top_candidate_subjects": ";".join(cand.head(10)["SubjectID"].astype(str).tolist()),
                    "reason": f"{manufacturer_token(row['Manufacturer'])}_{diag}_needed",
                }
            )

    ge = man_counts[man_counts["Manufacturer"] == "GE MEDICAL SYSTEMS"]
    others = man_counts[man_counts["Manufacturer"].isin(["Philips", "SIEMENS"])]
    if not ge.empty and not others.empty:
        ge_row = ge.iloc[0]
        for diag in CN_AD:
            target = int(np.ceil(float(others[diag].median())))
            deficit = max(0, target - int(ge_row[diag]))
            if deficit <= 0:
                continue
            cand = scored_candidates[
                (scored_candidates["Manufacturer"] == "GE MEDICAL SYSTEMS")
                & (scored_candidates["ResearchGroup_Mapped"] == diag)
            ]
            rows.append(
                {
                    "recommendation_type": "GE_vs_philips_siemens_balance",
                    "Manufacturer": "GE MEDICAL SYSTEMS",
                    "Site3": "",
                    "needed_diagnosis": diag,
                    "current_CN": int(ge_row["CN"]),
                    "current_AD": int(ge_row["AD"]),
                    "target_CN": target if diag == "CN" else None,
                    "target_AD": target if diag == "AD" else None,
                    "available_candidate_count": int(len(cand)),
                    "top_candidate_subjects": ";".join(cand.head(10)["SubjectID"].astype(str).tolist()),
                    "reason": f"GE_{diag}_needed",
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "recommendation_type",
                "Manufacturer",
                "Site3",
                "needed_diagnosis",
                "current_CN",
                "current_AD",
                "target_CN",
                "target_AD",
                "available_candidate_count",
                "top_candidate_subjects",
                "reason",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["recommendation_type", "available_candidate_count", "Manufacturer", "Site3"],
        ascending=[True, False, True, True],
    )


def loso_site_summary(site_counts: pd.DataFrame, dataset: str) -> Dict[str, object]:
    sub = site_counts[site_counts["dataset"] == dataset]
    site = (
        sub[["Site3", "site_CN", "site_AD"]]
        .drop_duplicates("Site3")
        .sort_values("Site3", key=lambda s: s.astype(str))
    )
    min2 = site[(site["site_CN"] >= 2) & (site["site_AD"] >= 2)]
    min1 = site[(site["site_CN"] >= 1) & (site["site_AD"] >= 1)]
    return {
        "n_valid_sites_min2": int(len(min2)),
        "valid_sites_min2": ",".join(min2["Site3"].astype(str).tolist()),
        "n_valid_sites_min1": int(len(min1)),
        "valid_sites_min1": ",".join(min1["Site3"].astype(str).tolist()),
    }


def build_readme(
    args: argparse.Namespace,
    datasets: Dict[str, pd.DataFrame],
    manufacturer_counts: pd.DataFrame,
    site_counts: pd.DataFrame,
    candidates_all: pd.DataFrame,
    candidates_scored: pd.DataFrame,
    exclusion_sources: Dict[str, Set[str]],
    missing_recs: pd.DataFrame,
) -> str:
    ref = args.reference_dataset
    ref_counts = manufacturer_counts[manufacturer_counts["dataset"] == ref].copy()
    ref_counts["diagnostic_imbalance_abs_CN_minus_AD"] = (ref_counts["CN"] - ref_counts["AD"]).abs()
    most_diag = ref_counts.sort_values("diagnostic_imbalance_abs_CN_minus_AD", ascending=False).iloc[0]
    ref_counts["cn_ad_total"] = ref_counts["CN"] + ref_counts["AD"]
    most_under = ref_counts.sort_values("cn_ad_total", ascending=True).iloc[0]

    min_target = args.manufacturer_min_cn_ad
    deficit_lines = []
    for _, row in ref_counts.sort_values("Manufacturer").iterrows():
        deficit_lines.append(
            f"- {row['Manufacturer']}: CN +{int(max(0, min_target - row['CN']))}, "
            f"AD +{int(max(0, min_target - row['AD']))} to reach CN>={min_target} and AD>={min_target}"
        )

    loso_lines = []
    for dataset in datasets:
        s = loso_site_summary(site_counts, dataset)
        loso_lines.append(
            f"- {dataset}: min2 sites={s['n_valid_sites_min2']} ({s['valid_sites_min2']}), "
            f"min1 sites={s['n_valid_sites_min1']} ({s['valid_sites_min1']})"
        )

    exclusion_lines = [
        f"- {name}: {len(ids)} IDs" for name, ids in sorted(exclusion_sources.items())
    ]

    top = candidates_scored.head(20)
    top_lines = [
        f"- {r.SubjectID} | {r.ResearchGroup_Mapped} | {r.Manufacturer} | Site3={r.Site3} "
        f"| score={r.priority_score} | {r.reason}"
        for r in top.itertuples(index=False)
    ]

    rec_summary = missing_recs.head(20).to_string(index=False) if not missing_recs.empty else "No missing cells."

    return "\n".join(
        [
            "# Dataset Balance Audit",
            "",
            f"Created UTC: {datetime.now(timezone.utc).isoformat()}",
            f"Reference dataset for candidate priority: {ref}",
            "",
            "## Main Findings",
            "",
            f"- Most diagnosis-imbalanced manufacturer in {ref}: {most_diag['Manufacturer']} "
            f"(CN={int(most_diag['CN'])}, AD={int(most_diag['AD'])}, "
            f"|CN-AD|={int(most_diag['diagnostic_imbalance_abs_CN_minus_AD'])}).",
            f"- Most underrepresented manufacturer by CN+AD total in {ref}: {most_under['Manufacturer']} "
            f"(CN+AD={int(most_under['cn_ad_total'])}).",
            "",
            "## Deficit To Minimum Per Manufacturer",
            "",
            *deficit_lines,
            "",
            "## LOSO Valid Sites",
            "",
            *loso_lines,
            "",
            "## Candidate Filtering",
            "",
            f"- Raw deduplicated candidates: {len(candidates_all)}",
            f"- Candidate rows after exclusions: {len(candidates_scored)}",
            "",
            "Exclusion sources:",
            *exclusion_lines,
            "",
            "## Top 20 Recommended Subjects",
            "",
            *(top_lines if top_lines else ["No candidate subjects available after exclusions."]),
            "",
            "## Missing Cell Recommendation Preview",
            "",
            "```",
            rec_summary,
            "```",
            "",
            "## Output Files",
            "",
            "- manufacturer_diagnosis_counts.csv",
            "- manufacturer_diagnosis_proportions.csv",
            "- site_manufacturer_diagnosis_counts.csv",
            "- candidate_priority_subjects.csv",
            "- missing_cells_recommendation.csv",
            "",
        ]
    )


def print_final_summary(
    args: argparse.Namespace,
    manufacturer_counts: pd.DataFrame,
    candidates_scored: pd.DataFrame,
) -> None:
    ref_counts = manufacturer_counts[manufacturer_counts["dataset"] == args.reference_dataset].copy()
    ref_counts["diagnostic_imbalance_abs_CN_minus_AD"] = (ref_counts["CN"] - ref_counts["AD"]).abs()
    most_diag = ref_counts.sort_values("diagnostic_imbalance_abs_CN_minus_AD", ascending=False).iloc[0]

    print("\nDataset balance audit complete")
    print(f"Output dir: {args.output_dir}")
    print(
        "Most diagnosis-imbalanced manufacturer "
        f"({args.reference_dataset}): {most_diag['Manufacturer']} "
        f"CN={int(most_diag['CN'])}, AD={int(most_diag['AD'])}, "
        f"abs_diff={int(most_diag['diagnostic_imbalance_abs_CN_minus_AD'])}"
    )
    print(f"\nDeficit to CN>={args.manufacturer_min_cn_ad} and AD>={args.manufacturer_min_cn_ad}:")
    for _, row in ref_counts.sort_values("Manufacturer").iterrows():
        print(
            f"- {row['Manufacturer']}: "
            f"CN +{int(max(0, args.manufacturer_min_cn_ad - row['CN']))}, "
            f"AD +{int(max(0, args.manufacturer_min_cn_ad - row['AD']))}"
        )

    print("\nTop 20 recommended subjects:")
    cols = ["SubjectID", "ResearchGroup_Mapped", "Manufacturer", "Site3", "priority_score", "reason"]
    if candidates_scored.empty:
        print("No candidate subjects available after exclusions.")
    else:
        print(candidates_scored.head(20)[cols].to_string(index=False))


def main() -> int:
    args = parse_args()
    candidate_paths = args.candidate_csv if args.candidate_csv else CANDIDATE_PATHS

    datasets = {
        "historical": load_dataset(args.historical_metadata, "historical"),
        "expanded_v1": load_dataset(args.expanded_v1_metadata, "expanded_v1"),
        "expanded_v2": load_dataset(args.expanded_v2_metadata, "expanded_v2"),
    }

    manufacturer_counts = build_manufacturer_counts(datasets, args.manufacturer_min_cn_ad)
    manufacturer_props = build_manufacturer_proportions(manufacturer_counts)
    site_counts = build_site_counts(datasets)

    processed_output_ids = collect_processed_output_ids(OUTPUT_ROOTS)
    assigned_batch_ids = collect_assigned_batch_ids(args.batch_dir)
    exclusion_sources = build_exclusion_sources(datasets, processed_output_ids, assigned_batch_ids)

    candidates_all = combine_candidates(candidate_paths, args.batch_dir)
    candidates_with_exclusion = annotate_exclusions(candidates_all, exclusion_sources)
    candidates_available = candidates_with_exclusion[~candidates_with_exclusion["is_excluded"]].copy()
    candidates_scored = score_candidates(
        candidates_available,
        datasets[args.reference_dataset],
        args.manufacturer_min_cn_ad,
    )
    missing_recs = build_missing_recommendations(
        datasets[args.reference_dataset],
        candidates_scored,
        args.manufacturer_min_cn_ad,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manufacturer_counts.to_csv(args.output_dir / "manufacturer_diagnosis_counts.csv", index=False)
    manufacturer_props.to_csv(args.output_dir / "manufacturer_diagnosis_proportions.csv", index=False)
    site_counts.to_csv(args.output_dir / "site_manufacturer_diagnosis_counts.csv", index=False)
    candidates_scored.to_csv(args.output_dir / "candidate_priority_subjects.csv", index=False)
    missing_recs.to_csv(args.output_dir / "missing_cells_recommendation.csv", index=False)

    readme = build_readme(
        args,
        datasets,
        manufacturer_counts,
        site_counts,
        candidates_all,
        candidates_scored,
        exclusion_sources,
        missing_recs,
    )
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reference_dataset": args.reference_dataset,
        "input_paths": {
            "historical": str(args.historical_metadata),
            "expanded_v1": str(args.expanded_v1_metadata),
            "expanded_v2": str(args.expanded_v2_metadata),
            "candidates": [str(p) for p in candidate_paths],
            "batch_dir": str(args.batch_dir),
            "output_roots": [str(p) for p in OUTPUT_ROOTS],
        },
        "n_candidates_raw_deduplicated": int(len(candidates_all)),
        "n_candidates_after_exclusion": int(len(candidates_scored)),
        "exclusion_source_counts": {k: len(v) for k, v in exclusion_sources.items()},
        "outputs": [
            "manufacturer_diagnosis_counts.csv",
            "manufacturer_diagnosis_proportions.csv",
            "site_manufacturer_diagnosis_counts.csv",
            "candidate_priority_subjects.csv",
            "missing_cells_recommendation.csv",
            "README.md",
        ],
    }
    with (args.output_dir / "audit_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    print_final_summary(args, manufacturer_counts, candidates_scored)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
