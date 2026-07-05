#!/usr/bin/env python3
"""Build ADNI v5.1 first-visit master manifest.

Read-only with respect to source data:
- no connectivity computation;
- no model training;
- no source files moved/copied/deleted.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.io as scipy_io


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "revision_bspc_2026" / "adni_v5_1_master_manifest"

PRIMARY_INPUTS = [
    DATA_DIR / "SubjctsDataAndTestsAAL3.csv",
    DATA_DIR / "RevisionPaperfMRI_2026_04_4_06_2026_extended.csv",
    DATA_DIR / "AD_fMRI_4_28_2026_extended.csv",
]

V5_MANIFEST = (
    DATA_DIR
    / "revision_bspc_2026"
    / "adni_expanded_v5_dparsf10000_no_pybandpass"
    / "subject_manifest_v5_dparsf10000_no_pybandpass.csv"
)

V5_COMPOSITION_DIR = PROJECT_ROOT / "results" / "revision_bspc_2026" / "v5_dataset_composition_and_split_risk"
GE_CN_TRACE_DIR = PROJECT_ROOT / "results" / "revision_bspc_2026" / "ge_cn_physical_file_trace"
V5_GE_FLOW_DIR = PROJECT_ROOT / "results" / "revision_bspc_2026" / "v5_ge_subject_flow_audit"

ROISIGNAL_ROOTS = [
    (Path("/media/diego/Datos/desde_cero/ROISignalsAAL3"), "desde_cero_historical_10000"),
    (
        Path("/media/diego/My_Book_Diego/vae_AD_data/adni_passband_20260510/ResultsAAL3/ROISignals_AAL3_FunImgARWSDCFN"),
        "new_passband_20260510_ARWSDCFN_10000",
    ),
    (
        Path("/media/diego/Datos/adni_expansion/MARTIN_20260429_PHILIPS10/OneDrive_2_29-4-2026/ResultsAAL3/ROISignals_AAL3_FunImgARWSDCF"),
        "martin_20260429_ARWSDCF",
    ),
    (
        Path("/media/diego/Datos/adni_expansion/GE_batch7/ROISignals_AAL3_from_CovRegressed_GE_batch7"),
        "ge_batch7_covregressed",
    ),
    (
        Path("/media/diego/Datos/adni_expansion/GE_smoketest3/ROISignals_AAL3_from_CovRegressed_GE_smoketest3"),
        "ge_smoketest3_covregressed",
    ),
    (Path("/media/diego/Datos/adni_bridge_expansion/dparsf_single/GE/FunImgARWSDCovs"), "ge_dparsf_single_covs"),
    (
        Path("/media/diego/Datos/adni_bridge_expansion/dparsf_single/GE/Results/ROISignals_FunImgARWSDC"),
        "ge_dparsf_single_ARWSDC",
    ),
]

SUBJECT_RE = re.compile(r"(?<!\d)(\d{3}_S_\d{4})(?!\d)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ADNI v5.1 master manifest with first-visit primary set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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


def clean_string(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def image_id_numeric(value: Any) -> float:
    text = clean_string(value).upper().strip()
    if text.startswith("I"):
        text = text[1:]
    digits = re.sub(r"\D", "", text)
    return float(digits) if digits else np.inf


def normalize_image_id(value: Any) -> str:
    text = clean_string(value).upper().strip()
    return text[1:] if text.startswith("I") and text[1:].isdigit() else text


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


def manufacturer_from_protocol(value: Any) -> str:
    text = clean_string(value)
    if not text:
        return ""
    match = re.search(r"Manufacturer\s*=\s*([^;]+)", text, flags=re.IGNORECASE)
    return canonical_manufacturer(match.group(1)) if match else ""


def first_existing_col(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower = {str(col).strip().lower(): str(col) for col in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        hit = lower.get(candidate.lower())
        if hit:
            return hit
    return None


def standardize_primary_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    sid_col = first_existing_col(df.columns, ["SubjectID", "Subject", "PTID", "subject_id"])
    img_col = first_existing_col(df.columns, ["ImageID", "ImageDataID", "IMAGEUID", "Image Data ID"])
    group_col = first_existing_col(df.columns, ["ResearchGroup", "ResearchGroup_Mapped", "Group", "DX"])
    study_col = first_existing_col(df.columns, ["StudyDate", "AcqDate", "ScanDate"])
    protocol_col = first_existing_col(df.columns, ["ImagingProtocol", "Protocol"])
    manufacturer_col = first_existing_col(df.columns, ["Manufacturer", "MANUFACTURER"])
    out = pd.DataFrame()
    out["SubjectID"] = df[sid_col].map(normalize_subject) if sid_col else ""
    out["ImageID"] = df[img_col].map(normalize_image_id) if img_col else ""
    out["ResearchGroup"] = df[group_col].map(clean_string) if group_col else ""
    out["ResearchGroup_Mapped"] = out["ResearchGroup"].map(normalize_group)
    for target, candidates in {
        "Visit": ["Visit", "VISCODE", "VisitCode"],
        "Age": ["Age", "AGE"],
        "Sex": ["Sex", "PTGENDER", "Gender"],
        "Phase": ["Phase"],
        "Description": ["Description", "SeriesDescription"],
    }.items():
        col = first_existing_col(df.columns, candidates)
        out[target] = df[col].map(clean_string) if col else ""
    out["StudyDate"] = df[study_col].map(clean_string) if study_col else ""
    out["AcqDate"] = out["StudyDate"]
    out["ImagingProtocol"] = df[protocol_col].map(clean_string) if protocol_col else ""
    manufacturer = df[manufacturer_col].map(clean_string).map(canonical_manufacturer) if manufacturer_col else pd.Series([""] * len(df))
    from_protocol = out["ImagingProtocol"].map(manufacturer_from_protocol)
    out["Manufacturer"] = [m if m and m != "UNKNOWN" else p for m, p in zip(manufacturer, from_protocol)]
    out["Manufacturer"] = out["Manufacturer"].replace("", "UNKNOWN")
    out["Manufacturer_Source"] = np.where(from_protocol.astype(bool), "imaging_protocol", np.where(out["Manufacturer"].ne("UNKNOWN"), "csv_column", ""))
    out["source_csv"] = path.name
    out["source_row"] = np.arange(len(out))
    out = out[out["SubjectID"].astype(bool)].copy()
    return out


def load_primary_images() -> pd.DataFrame:
    frames = [standardize_primary_csv(path) for path in PRIMARY_INPUTS if path.exists()]
    all_images = pd.concat(frames, ignore_index=True, sort=False)
    all_images["_dedupe_key"] = (
        all_images["SubjectID"].astype(str) + "|" + all_images["ImageID"].astype(str) + "|" + all_images["Visit"].astype(str)
    )
    all_images = all_images.drop_duplicates("_dedupe_key", keep="first").drop(columns=["_dedupe_key"])
    return all_images


def load_aux_metadata() -> pd.DataFrame:
    paths = [
        DATA_DIR / "SubjectsData_AAL3_procesado2.csv",
        DATA_DIR / "adni_download_now.csv",
        V5_MANIFEST,
    ]
    rows: List[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        sid_col = first_existing_col(df.columns, ["SubjectID", "Subject", "PTID"])
        if sid_col is None:
            continue
        img_col = first_existing_col(df.columns, ["ImageID", "ImageDataID", "IMAGEUID"])
        man_col = first_existing_col(df.columns, ["Manufacturer", "MANUFACTURER"])
        group_col = first_existing_col(df.columns, ["ResearchGroup_Mapped", "ResearchGroup", "Group"])
        out = pd.DataFrame({"SubjectID": df[sid_col].map(normalize_subject)})
        out["ImageID"] = df[img_col].map(normalize_image_id) if img_col else ""
        out["Manufacturer_aux"] = df[man_col].map(canonical_manufacturer) if man_col else ""
        out["ResearchGroup_Mapped_aux"] = df[group_col].map(normalize_group) if group_col else ""
        for target, candidates in {"Age_aux": ["Age"], "Sex_aux": ["Sex"], "Visit_aux": ["Visit"], "Site3_aux": ["Site3"]}.items():
            col = first_existing_col(df.columns, candidates)
            out[target] = df[col].map(clean_string) if col else ""
        out["aux_source"] = path.name
        rows.append(out[out["SubjectID"].astype(bool)])
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame(columns=["SubjectID"])


def enrich_metadata(all_images: pd.DataFrame, aux: pd.DataFrame) -> pd.DataFrame:
    df = all_images.copy()
    if aux.empty:
        df["Site3"] = ""
        return df
    aux_by_img = aux[aux["ImageID"].astype(bool)].drop_duplicates(["SubjectID", "ImageID"], keep="first")
    df = df.merge(aux_by_img, on=["SubjectID", "ImageID"], how="left", suffixes=("", "_img"))
    aux_by_sid = aux.drop_duplicates("SubjectID", keep="first").set_index("SubjectID")
    for idx, row in df.iterrows():
        sid = row["SubjectID"]
        sid_meta = aux_by_sid.loc[sid] if sid in aux_by_sid.index else None
        if row.get("Manufacturer", "UNKNOWN") == "UNKNOWN":
            value = clean_string(row.get("Manufacturer_aux", ""))
            if not value and sid_meta is not None:
                value = clean_string(sid_meta.get("Manufacturer_aux", ""))
            if value:
                df.at[idx, "Manufacturer"] = canonical_manufacturer(value)
                df.at[idx, "Manufacturer_Source"] = "aux_metadata"
        for field, aux_field in [
            ("ResearchGroup_Mapped", "ResearchGroup_Mapped_aux"),
            ("Age", "Age_aux"),
            ("Sex", "Sex_aux"),
            ("Visit", "Visit_aux"),
            ("Site3", "Site3_aux"),
        ]:
            if not clean_string(row.get(field, "")):
                value = clean_string(row.get(aux_field, ""))
                if not value and sid_meta is not None:
                    value = clean_string(sid_meta.get(aux_field, ""))
                if value:
                    df.at[idx, field] = value
    if "Site3" not in df.columns:
        df["Site3"] = ""
    df["Manufacturer"] = df["Manufacturer"].replace("", "UNKNOWN")
    return df


def visit_priority(visit: Any, description: Any, image_id: Any, study_date: Any) -> Tuple[int, float, str, float]:
    text = f"{clean_string(visit)} {clean_string(description)}".lower()
    if re.search(r"\b(sc|screen|screening|baseline|initial|init)\b", text):
        rank = 0
    elif re.search(r"\b(m\d+|y\d+|v\d+|year|month|follow|fu)\b", text):
        rank = 10
    else:
        rank = 5
    date = pd.to_datetime(clean_string(study_date), errors="coerce")
    date_rank = float(date.value) if not pd.isna(date) else np.inf
    return rank, image_id_numeric(image_id), clean_string(study_date), date_rank


def select_first_visits(all_images: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = all_images.copy()
    priorities = df.apply(lambda r: visit_priority(r.get("Visit"), r.get("Description"), r.get("ImageID"), r.get("StudyDate")), axis=1)
    df["visit_priority_rank"] = [p[0] for p in priorities]
    df["image_id_numeric"] = [p[1] for p in priorities]
    df["study_date_rank"] = [p[3] for p in priorities]
    df = df.sort_values(["SubjectID", "visit_priority_rank", "image_id_numeric", "study_date_rank", "source_csv", "source_row"])
    df["is_first_visit_selected"] = False
    first_idx = df.groupby("SubjectID", sort=False).head(1).index
    df.loc[first_idx, "is_first_visit_selected"] = True
    first = df.loc[first_idx].sort_values("SubjectID").copy()
    extra = df.loc[~df.index.isin(first_idx)].sort_values(["SubjectID", "visit_priority_rank", "image_id_numeric"]).copy()
    return first, extra


def discover_roisignals() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for root, label in ROISIGNAL_ROOTS:
        if not root.exists():
            continue
        for suffix in [".mat", ".txt"]:
            for path in sorted(root.glob(f"ROISignals_*{suffix}")):
                sid = normalize_subject(path.name)
                if sid:
                    rows.append({"SubjectID": sid, "roisignals_path": str(path), "roisignals_root": str(root), "roisignals_label": label, "suffix": suffix})
    return pd.DataFrame(rows)


def stage_guess(path: str, label: str) -> str:
    low = f"{path} {label}".lower()
    if "roisignalsaAL3".lower() in low or "desde_cero" in low:
        return "historical_10000"
    if "arwsdcfn" in low:
        return "ARWSDCFN_passband_10000"
    if "covregressed" in low or "covs" in low:
        return "CovRegressed_needs_stage_confirmation"
    if "arwsdcf" in low:
        return "ARWSDCF_needs_stage_confirmation"
    if "arwsdc" in low:
        return "ARWSDC_needs_stage_confirmation"
    return "unknown"


def roisignal_priority(row: pd.Series) -> int:
    sg = row["stage_guess"]
    if sg == "historical_10000":
        return 0
    if sg == "ARWSDCFN_passband_10000":
        return 1
    if "ARWSDCF" in sg:
        return 5
    if "ARWSDC" in sg or "CovRegressed" in sg:
        return 8
    return 20


def choose_roisignals(rois: pd.DataFrame) -> pd.DataFrame:
    if rois.empty:
        return rois
    df = rois.copy()
    df["stage_guess"] = df.apply(lambda r: stage_guess(r["roisignals_path"], r["roisignals_label"]), axis=1)
    df["roisignals_priority"] = df.apply(roisignal_priority, axis=1)
    df["suffix_priority"] = df["suffix"].map({".mat": 0, ".txt": 1}).fillna(9)
    df = df.sort_values(["SubjectID", "roisignals_priority", "suffix_priority", "roisignals_path"])
    chosen = df.groupby("SubjectID", sort=False).head(1).copy()
    grouped = df.groupby("SubjectID").agg(
        n_roisignals_files_found=("roisignals_path", "count"),
        all_roisignals_paths=("roisignals_path", lambda s: "|".join(s.astype(str))),
        all_stage_guesses=("stage_guess", lambda s: "|".join(sorted(set(s.astype(str))))),
    )
    return chosen.merge(grouped, on="SubjectID", how="left")


def load_signal_array(path: Path) -> Tuple[Optional[np.ndarray], str, str]:
    try:
        if path.suffix.lower() == ".txt":
            try:
                arr = np.loadtxt(path, delimiter=",", dtype=np.float64)
                return arr, str(tuple(int(x) for x in arr.shape)), "ok_txt_comma"
            except Exception:
                arr = np.loadtxt(path, dtype=np.float64)
                return arr, str(tuple(int(x) for x in arr.shape)), "ok_txt_whitespace"
        entries = [(name, tuple(int(x) for x in shape), klass) for name, shape, klass in scipy_io.whosmat(path)]
        candidates = [(1 if "signal" in name.lower() else 0, int(np.prod(shape)), name, shape) for name, shape, klass in entries if len(shape) == 2]
        if not candidates:
            return None, "", f"no_2d_signal:{entries}"
        _score, _size, name, _shape = sorted(candidates, reverse=True)[0]
        arr = np.asarray(scipy_io.loadmat(path, variable_names=[name])[name], dtype=np.float64)
        return arr, str(tuple(int(x) for x in arr.shape)), f"ok_mat:{name}"
    except Exception as exc:
        return None, "", f"failed:{exc}"


def scale_label(values: np.ndarray) -> str:
    if values.size == 0:
        return "other"
    mean = float(np.nanmean(values))
    median = float(np.nanmedian(values))
    frac_neg = float(np.mean(values < 0))
    typical = np.nanmedian(np.abs([mean, median]))
    if np.isfinite(typical) and 1000 <= typical <= 50000 and frac_neg < 0.01:
        return "around_10000_global_scaled"
    if abs(mean) < max(1e-6, 0.1 * float(np.nanstd(values))) and frac_neg > 0.01:
        return "zero_centered"
    return "other"


def roisignal_qc(chosen_rois: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in chosen_rois.iterrows():
        path = Path(row["roisignals_path"])
        arr, shape, status = load_signal_array(path)
        out = row.to_dict()
        out.update({"shape": shape, "load_status": status, "finite_fraction": 0.0, "scale_label": "other"})
        if arr is not None and arr.ndim == 2 and arr.size:
            oriented = arr.T if arr.shape[0] == 170 and arr.shape[1] != 170 else arr
            values = oriented.ravel()
            finite = values[np.isfinite(values)]
            out.update(
                {
                    "n_timepoints": int(oriented.shape[0]),
                    "n_rois": int(oriented.shape[1]),
                    "finite_fraction": float(finite.size / values.size),
                    "mean": float(np.nanmean(finite)) if finite.size else np.nan,
                    "std": float(np.nanstd(finite)) if finite.size else np.nan,
                    "median": float(np.nanmedian(finite)) if finite.size else np.nan,
                    "min": float(np.nanmin(finite)) if finite.size else np.nan,
                    "max": float(np.nanmax(finite)) if finite.size else np.nan,
                    "scale_label": scale_label(finite),
                }
            )
        rows.append(out)
    return pd.DataFrame(rows)


def load_ge_reference() -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    priority = GE_CN_TRACE_DIR / "ge_cn_priority19_physical_trace.csv"
    if priority.exists():
        df = pd.read_csv(priority, dtype=str, keep_default_na=False)
        df["is_v4_priority19_ge_cn"] = df["priority_tier"].eq("tier1_v4_19")
        rows.append(df[["SubjectID", "priority_tier", "action_needed", "file_stage_guess", "is_v4_priority19_ge_cn"]])
    candidates = V5_GE_FLOW_DIR / "possible_ge_cn_candidates.csv"
    if candidates.exists():
        df = pd.read_csv(candidates, dtype=str, keep_default_na=False)
        df["is_possible_ge_cn_candidate"] = True
        keep = [c for c in ["SubjectID", "note", "is_possible_ge_cn_candidate"] if c in df.columns]
        rows.append(df[keep])
    if not rows:
        return pd.DataFrame(columns=["SubjectID"])
    out = rows[0]
    for df in rows[1:]:
        out = out.merge(df, on="SubjectID", how="outer")
    return out.drop_duplicates("SubjectID", keep="first")


def compatibility(row: pd.Series) -> Tuple[str, str, str]:
    if not bool(row.get("has_roisignals", False)):
        return "no", "needs_preprocessing", "needs_preprocessing"
    if clean_string(row.get("load_status", "")).startswith("failed") or float(row.get("finite_fraction", 0.0) or 0.0) < 0.95:
        return "no", "exclude", "invalid_or_low_finite_signal"
    if int(row.get("n_rois", 0) or 0) != 170:
        return "no", "exclude", "roi_count_not_170"
    stage = clean_string(row.get("stage_guess", ""))
    scale = clean_string(row.get("scale_label", ""))
    if stage in {"historical_10000", "ARWSDCFN_passband_10000"} and scale == "around_10000_global_scaled":
        return "yes", "ready_for_v5_1_direct", "ready_for_v5_1_direct"
    if "CovRegressed" in stage or "ARWSDC" in stage or "ARWSDCF" in stage:
        return "no", "needs_stage_confirmation", "needs_stage_confirmation"
    return "no", "needs_stage_confirmation", "unknown_stage_or_scale"


def append_action(current: Any, tag: str) -> str:
    parts = [p for p in clean_string(current).split("|") if p and p != "nan"]
    if not parts:
        parts = ["ready_for_v5_1_direct"]
    if parts == ["ready_for_v5_1_direct"]:
        parts = []
    if tag not in parts:
        parts.append(tag)
    return "|".join(parts) if parts else "ready_for_v5_1_direct"


def add_roisignals_to_first_visit(first: pd.DataFrame, qc: pd.DataFrame, ge_ref: pd.DataFrame) -> pd.DataFrame:
    df = first.merge(qc, on="SubjectID", how="left", suffixes=("", "_roi"))
    df["has_roisignals"] = df["roisignals_path"].notna()
    for col in ["n_roisignals_files_found", "n_timepoints", "n_rois", "finite_fraction", "mean", "std", "median", "min", "max"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if not ge_ref.empty:
        df = df.merge(ge_ref, on="SubjectID", how="left", suffixes=("", "_ge"))
    else:
        df["is_v4_priority19_ge_cn"] = False
    compat = df.apply(compatibility, axis=1)
    df["compatible_for_v5_1_direct"] = [x[0] for x in compat]
    df["preprocessing_status"] = [x[1] for x in compat]
    df["action_needed"] = [x[2] for x in compat]
    needs_manufacturer = df["Manufacturer"].fillna("UNKNOWN").eq("UNKNOWN")
    df.loc[needs_manufacturer, "action_needed"] = df.loc[needs_manufacturer, "action_needed"].map(
        lambda value: append_action(value, "needs_manufacturer_metadata")
    )
    unknown_dx = df["ResearchGroup_Mapped"].fillna("").eq("")
    df.loc[unknown_dx, "action_needed"] = df.loc[unknown_dx, "action_needed"].map(
        lambda value: append_action(value, "needs_diagnosis_metadata")
    )
    return df


def balance_summary(first: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    def add_counts(summary_type: str, group_cols: List[str]) -> None:
        grouped = first.copy()
        for col in group_cols:
            grouped[col] = grouped[col].fillna("UNKNOWN").replace("", "UNKNOWN")
        for keys, sub in grouped.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {"summary_type": summary_type, "n": len(sub)}
            for idx, key in enumerate(keys, start=1):
                row[f"group{idx}"] = key
            rows.append(row)
    add_counts("Diagnosis", ["ResearchGroup_Mapped"])
    add_counts("Manufacturer", ["Manufacturer"])
    add_counts("Diagnosis_x_Manufacturer", ["ResearchGroup_Mapped", "Manufacturer"])
    add_counts("Sex", ["Sex"])
    add_counts("preprocessed_status", ["preprocessing_status"])
    add_counts("preprocessed_yes_no", ["compatible_for_v5_1_direct"])
    age_df = first.copy()
    age_df["Age_num"] = pd.to_numeric(age_df["Age"], errors="coerce")
    for dx, sub in age_df.groupby(age_df["ResearchGroup_Mapped"].fillna("UNKNOWN").replace("", "UNKNOWN")):
        rows.append(
            {
                "summary_type": "Age_by_Diagnosis",
                "group1": dx,
                "n": int(sub["Age_num"].notna().sum()),
                "age_mean": float(sub["Age_num"].mean()) if sub["Age_num"].notna().any() else np.nan,
                "age_median": float(sub["Age_num"].median()) if sub["Age_num"].notna().any() else np.nan,
                "age_min": float(sub["Age_num"].min()) if sub["Age_num"].notna().any() else np.nan,
                "age_max": float(sub["Age_num"].max()) if sub["Age_num"].notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def preprocessing_request(first: pd.DataFrame) -> pd.DataFrame:
    mask = first["action_needed"].fillna("").ne("ready_for_v5_1_direct")
    cols = [
        "SubjectID",
        "ImageID",
        "ResearchGroup_Mapped",
        "Visit",
        "Age",
        "Sex",
        "Manufacturer",
        "Description",
        "has_roisignals",
        "roisignals_path",
        "stage_guess",
        "shape",
        "finite_fraction",
        "scale_label",
        "compatible_for_v5_1_direct",
        "preprocessing_status",
        "action_needed",
    ]
    return first.loc[mask, [c for c in cols if c in first.columns]].copy()


def write_readme(output_dir: Path, all_images: pd.DataFrame, first: pd.DataFrame, extra: pd.DataFrame, request: pd.DataFrame) -> None:
    counts = first["ResearchGroup_Mapped"].fillna("UNKNOWN").replace("", "UNKNOWN").value_counts().to_dict()
    ready_n = int(first["compatible_for_v5_1_direct"].eq("yes").sum())
    missing_n = int(first["compatible_for_v5_1_direct"].ne("yes").sum())
    if "is_v4_priority19_ge_cn" in first.columns:
        ge19_mask = first["is_v4_priority19_ge_cn"].astype(str).str.lower().isin({"true", "1", "yes"})
        ge19 = first[ge19_mask]
    else:
        ge19 = pd.DataFrame()
    ge19_found = int(ge19["has_roisignals"].sum()) if not ge19.empty else 0
    ge19_direct = int(ge19["compatible_for_v5_1_direct"].eq("yes").sum()) if not ge19.empty else 0
    metadata_only_n = int(
        request["compatible_for_v5_1_direct"].eq("yes").sum()
    ) if "compatible_for_v5_1_direct" in request.columns else 0
    s035 = first[first["SubjectID"].eq("035_S_6927")]
    s114 = first[first["SubjectID"].eq("114_S_6039")]
    ready_for_tensor = missing_n == 0
    next_cmd = (
        "/home/diego/anaconda3/envs/vae_ad/bin/python "
        "scripts/revision_bspc_2026/build_v5_dparsf10000_no_pybandpass_manifest_and_extract.py "
        "--smoke-n-subjects 5 --overwrite"
    )
    lines = [
        "# ADNI v5.1 Master Manifest First Visit",
        "",
        "Read-only audit/manifest build. No connectivity and no training were run.",
        "",
        "## Explicit Answers",
        "",
        f"- Total subjects wanted in first-visit primary set: `{first['SubjectID'].nunique()}`.",
        f"- Total image rows in unified all-images table: `{len(all_images)}`.",
        f"- Longitudinal/additional-visit rows separated: `{len(extra)}`.",
        f"- First-visit diagnosis counts: `{json.dumps(counts, sort_keys=True)}`.",
        f"- Preprocessed/direct-compatible subjects: `{ready_n}`.",
        f"- Missing/not-direct subjects: `{missing_n}`.",
        f"- v4 priority CN-GE subjects recovered with any ROISignals: `{ge19_found}/{len(ge19)}`.",
        f"- v4 priority CN-GE subjects direct-compatible for v5.1: `{ge19_direct}/{len(ge19)}`.",
        f"- `035_S_6927` Age/Sex recovered: `{not s035.empty and bool(clean_string(s035.iloc[0].get('Age')) and clean_string(s035.iloc[0].get('Sex')))}`.",
        f"- `114_S_6039` status: `{s114.iloc[0]['action_needed'] if not s114.empty else 'not_in_primary_set'}`.",
        f"- Subjects Martin must review/process or complete metadata: `{len(request)}` rows in `adni_v5_1_preprocessing_request_for_martin.csv`.",
        f"- Metadata-only rows among those requests: `{metadata_only_n}`.",
        f"- Ready to build tensor v5.1 now? `{'YES' if ready_for_tensor else 'NO'}`.",
        f"- Recommended next command after resolving request CSV: `{next_cmd}`.",
        "",
        "## Outputs",
        "",
        "- `adni_v5_1_master_subject_manifest_all_images.csv`",
        "- `adni_v5_1_master_subject_manifest_first_visit.csv`",
        "- `adni_v5_1_longitudinal_extra_visits.csv`",
        "- `adni_v5_1_preprocessing_request_for_martin.csv`",
        "- `adni_v5_1_balance_summary.csv`",
        "- `adni_v5_1_roisignals_qc_summary.csv`",
    ]
    output_dir.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    prepare_output_dir(output_dir, args.overwrite)

    all_images = enrich_metadata(load_primary_images(), load_aux_metadata())
    first, extra = select_first_visits(all_images)
    rois = discover_roisignals()
    chosen_rois = choose_roisignals(rois)
    qc = roisignal_qc(chosen_rois) if not chosen_rois.empty else pd.DataFrame(columns=["SubjectID"])
    first_aug = add_roisignals_to_first_visit(first, qc, load_ge_reference())
    all_aug = all_images.merge(
        first_aug[["SubjectID", "ImageID", "is_first_visit_selected", "has_roisignals", "compatible_for_v5_1_direct", "preprocessing_status", "action_needed"]],
        on=["SubjectID", "ImageID"],
        how="left",
    )
    request = preprocessing_request(first_aug)
    summary = balance_summary(first_aug)

    all_aug.to_csv(output_dir / "adni_v5_1_master_subject_manifest_all_images.csv", index=False)
    first_aug.to_csv(output_dir / "adni_v5_1_master_subject_manifest_first_visit.csv", index=False)
    extra.to_csv(output_dir / "adni_v5_1_longitudinal_extra_visits.csv", index=False)
    request.to_csv(output_dir / "adni_v5_1_preprocessing_request_for_martin.csv", index=False)
    summary.to_csv(output_dir / "adni_v5_1_balance_summary.csv", index=False)
    qc.to_csv(output_dir / "adni_v5_1_roisignals_qc_summary.csv", index=False)
    write_readme(output_dir, all_aug, first_aug, extra, request)

    print(f"Wrote ADNI v5.1 master manifest to {output_dir}")
    print(f"all_image_rows={len(all_aug)} first_visit_subjects={first_aug['SubjectID'].nunique()} extra_visit_rows={len(extra)}")
    print(f"diagnosis_counts={first_aug['ResearchGroup_Mapped'].fillna('UNKNOWN').replace('', 'UNKNOWN').value_counts().to_dict()}")
    print(f"direct_compatible={int(first_aug['compatible_for_v5_1_direct'].eq('yes').sum())} not_direct={int(first_aug['compatible_for_v5_1_direct'].ne('yes').sum())}")
    print("No connectivity computed. No training run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
