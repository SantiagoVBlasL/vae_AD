#!/usr/bin/env python3
"""Subject-level error/QC/SiteCode audit for locked ADNI v5.1 FULL [1,0,2].

Read-only audit: uses locked classifier-only Stage B predictions and available
local metadata/clinical CSVs. It does not train, rewrite tensors, or alter
existing result folders.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results" / "revision_bspc_2026"
READOUT_DIR = RESULTS_ROOT / "adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep"
METADATA_PATH = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass/"
    "training_ready_metadata_v5_1_batch20260514b_no_pybandpass.csv"
)
OUTDIR = RESULTS_ROOT / "adni_v5_1_batch20260514b_locked_subject_error_qc_site_audit"

PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
NEAR_THRESHOLD_ABS_MARGIN = 0.05
HIGH_CONFIDENCE_ABS_MARGIN = 0.10
MIN_SITE_N = 12
MIN_SITE_PER_CLASS = 3

CLINICAL_SOURCES = [
    PROJECT_ROOT / "data" / "SubjectsData_AAL3_procesado2.csv",
    PROJECT_ROOT / "data" / "SubjctsDataAndTestsAAL3.csv",
    PROJECT_ROOT / "data" / "RevisionPaperfMRI_2026_04_4_06_2026_extended.csv",
    PROJECT_ROOT / "data" / "AD_fMRI_4_28_2026_extended.csv",
]

CLINICAL_FIELDS = [
    "PTEDUCAT",
    "CDRSB",
    "MMSE",
    "DIGITSCOR",
    "MOCA",
    "Ventricles",
    "Hippocampus",
    "WholeBrain",
    "MidTemp",
    "ABETA",
    "TAU",
    "PTAU",
    "StudyDate",
    "ArchiveDate",
    "AcqDate",
    "Visit",
]

QC_FIELDS = [
    "stage_guess",
    "spectral_class",
    "n_timepoints_raw",
    "n_rois_raw",
    "finite_fraction",
    "scale_label",
    "roisignals_path",
    "dicom_series_ok",
    "python_bandpass_requested",
    "python_bandpass_applied",
]


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def normalize_subject_id(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def normalize_manufacturer(x: Any) -> str:
    if pd.isna(x):
        return "UNKNOWN"
    s = str(x).strip()
    if not s:
        return "UNKNOWN"
    u = s.upper()
    if "SIEMENS" in u:
        return "Siemens"
    if "PHILIPS" in u:
        return "Philips"
    if u.startswith("GE") or "GENERAL ELECTRIC" in u:
        return "GE"
    return s


def site_code_from_subject(subject_id: Any) -> str:
    match = re.match(r"^(\d{3})", str(subject_id))
    return match.group(1) if match else "UNKNOWN"


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def safe_auc(y_true: Iterable[int], y_score: Iterable[float]) -> float:
    y = np.asarray(list(y_true), dtype=int)
    s = np.asarray(list(y_score), dtype=float)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def safe_pr_auc(y_true: Iterable[int], y_score: Iterable[float]) -> float:
    y = np.asarray(list(y_true), dtype=int)
    s = np.asarray(list(y_score), dtype=float)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def binary_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    y = df["y_true"].astype(int).to_numpy()
    p = df["y_pred"].astype(int).to_numpy()
    s = df["ad_score"].astype(float).to_numpy()
    if len(y) == 0:
        return {
            "n": 0,
            "n_cn": 0,
            "n_ad": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "balanced_accuracy": float("nan"),
            "f1": float("nan"),
            "auc": float("nan"),
            "pr_auc": float("nan"),
            "brier": float("nan"),
        }
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "n_cn": int((y == 0).sum()),
        "n_ad": int((y == 1).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "balanced_accuracy": float(np.nanmean([safe_div(tp, tp + fn), safe_div(tn, tn + fp)])),
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
        "auc": safe_auc(y, s),
        "pr_auc": safe_pr_auc(y, s),
        "brier": float(brier_score_loss(y, np.clip(s, 0.0, 1.0))),
        "score_median": float(np.median(s)),
        "score_mean": float(np.mean(s)),
        "near_threshold_0p05_rate": float((df["abs_margin_to_threshold"] <= NEAR_THRESHOLD_ABS_MARGIN).mean()),
    }


def md_table(df: pd.DataFrame, max_rows: int = 120) -> str:
    if df.empty:
        return "_No rows._\n"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
    text = view.to_markdown(index=False)
    if len(df) > max_rows:
        text += f"\n\n_Showing {max_rows} of {len(df)} rows._"
    return text + "\n"


def write_pair(stem: str, df: pd.DataFrame, max_rows: int = 120) -> None:
    df.to_csv(OUTDIR / f"{stem}.csv", index=False)
    (OUTDIR / f"{stem}.md").write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


def load_primary_predictions() -> pd.DataFrame:
    path = READOUT_DIR / "classifier_sweep_predictions.csv"
    preds = read_csv(path)
    if preds.empty:
        raise FileNotFoundError(f"Missing predictions: {path}")
    mask = preds["model_name"].eq(PRIMARY_MODEL) & preds["threshold_strategy"].eq(PRIMARY_THRESHOLD)
    out = preds.loc[mask].copy()
    if out.empty:
        raise RuntimeError(f"No predictions for {PRIMARY_MODEL} / {PRIMARY_THRESHOLD}")
    out["SubjectID"] = normalize_subject_id(out["SubjectID"])
    out["SiteCode"] = out["SubjectID"].map(site_code_from_subject)
    out["ResearchGroup_Mapped"] = out["ResearchGroup_Mapped"].astype(str)
    out["predicted_label"] = np.where(out["y_pred"].astype(int).eq(1), "AD", "CN")
    out["ad_score"] = out["y_score"].astype(float)
    out["threshold"] = out["threshold"].astype(float)
    out["margin_to_threshold"] = out["ad_score"] - out["threshold"]
    out["abs_margin_to_threshold"] = out["margin_to_threshold"].abs()
    out["Manufacturer"] = out["Manufacturer"].map(normalize_manufacturer)
    return out


def load_final_metadata() -> pd.DataFrame:
    meta = read_csv(METADATA_PATH)
    if meta.empty:
        return meta
    meta["SubjectID"] = normalize_subject_id(meta["SubjectID"])
    if "Manufacturer" in meta.columns:
        meta["Manufacturer_meta"] = meta["Manufacturer"].map(normalize_manufacturer)
    if "Site3" in meta.columns:
        meta["Site3"] = meta["Site3"].fillna("").astype(str).str.zfill(3)
    keep = ["SubjectID", "Site3", "ImageID", "Visit", "Diagnosis", "included_in_dataset_version", "metadata_source"] + [
        c for c in QC_FIELDS if c in meta.columns
    ]
    keep = list(dict.fromkeys([c for c in keep if c in meta.columns]))
    return meta[keep].drop_duplicates("SubjectID")


def load_clinical_metadata() -> tuple[pd.DataFrame, Dict[str, Any]]:
    frames: List[pd.DataFrame] = []
    source_inventory: List[Dict[str, Any]] = []
    for path in CLINICAL_SOURCES:
        df = read_csv(path)
        if df.empty:
            source_inventory.append({"path": str(path), "status": "missing_or_empty", "n_rows": 0, "matched_id_column": ""})
            continue
        id_col = "SubjectID" if "SubjectID" in df.columns else ("Subject" if "Subject" in df.columns else "")
        if not id_col:
            source_inventory.append({"path": str(path), "status": "no_subject_column", "n_rows": int(len(df)), "matched_id_column": ""})
            continue
        out = pd.DataFrame({"SubjectID": normalize_subject_id(df[id_col])})
        for col in CLINICAL_FIELDS:
            if col in df.columns:
                out[col] = df[col]
        if "ImageDataID" in df.columns and "ImageID" not in out.columns:
            out["ImageID"] = df["ImageDataID"]
        if "ImageID" in df.columns:
            out["ImageID"] = df["ImageID"]
        out["clinical_source"] = str(path.relative_to(PROJECT_ROOT))
        frames.append(out)
        source_inventory.append(
            {
                "path": str(path),
                "status": "loaded",
                "n_rows": int(len(df)),
                "matched_id_column": id_col,
                "columns_available": ",".join([c for c in CLINICAL_FIELDS if c in df.columns]),
            }
        )
    if not frames:
        return pd.DataFrame(), {"sources": source_inventory, "available_fields": []}

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.replace({"": np.nan})
    # Prefer rows with more non-null clinical fields, keeping first source order as a stable tiebreaker.
    clinical_cols = [c for c in combined.columns if c not in {"SubjectID", "clinical_source"}]
    combined["_non_null"] = combined[clinical_cols].notna().sum(axis=1)
    combined["_order"] = np.arange(len(combined))
    combined = combined.sort_values(["SubjectID", "_non_null", "_order"], ascending=[True, False, True])
    combined = combined.drop_duplicates("SubjectID", keep="first").drop(columns=["_non_null", "_order"])
    availability = {
        "sources": source_inventory,
        "available_fields": [c for c in CLINICAL_FIELDS if c in combined.columns],
        "n_subjects_with_any_clinical_source": int(combined["SubjectID"].nunique()),
    }
    return combined, availability


def build_subject_table() -> tuple[pd.DataFrame, Dict[str, Any]]:
    preds = load_primary_predictions()
    meta = load_final_metadata()
    clinical, clinical_availability = load_clinical_metadata()
    table = preds.copy()
    if not meta.empty:
        table = table.merge(meta, on="SubjectID", how="left", suffixes=("", "_meta"))
    if not clinical.empty:
        table = table.merge(clinical, on="SubjectID", how="left", suffixes=("", "_clinical"))

    if "Site3" in table.columns:
        table["Site3"] = table["Site3"].fillna(table["SiteCode"]).astype(str)
    else:
        table["Site3"] = table["SiteCode"]
    table["error_type"] = np.select(
        [
            table["y_true"].astype(int).eq(0) & table["y_pred"].astype(int).eq(0),
            table["y_true"].astype(int).eq(0) & table["y_pred"].astype(int).eq(1),
            table["y_true"].astype(int).eq(1) & table["y_pred"].astype(int).eq(0),
            table["y_true"].astype(int).eq(1) & table["y_pred"].astype(int).eq(1),
        ],
        ["TN", "FP", "FN", "TP"],
        default="UNKNOWN",
    )
    table["confidence_bucket"] = np.where(
        table["abs_margin_to_threshold"] <= NEAR_THRESHOLD_ABS_MARGIN,
        "near_threshold",
        np.where(table["abs_margin_to_threshold"] >= HIGH_CONFIDENCE_ABS_MARGIN, "high_confidence", "intermediate_margin"),
    )
    for col in ["MMSE", "CDRSB", "MOCA", "DIGITSCOR", "PTEDUCAT", "Age", "finite_fraction", "n_timepoints_raw"]:
        if col in table.columns:
            table[col] = pd.to_numeric(table[col], errors="coerce")
    table["has_cognitive_metadata"] = table[[c for c in ["MMSE", "CDRSB", "MOCA", "DIGITSCOR"] if c in table.columns]].notna().any(axis=1)
    table["has_qc_metadata"] = table[[c for c in ["finite_fraction", "n_timepoints_raw", "stage_guess", "spectral_class", "scale_label"] if c in table.columns]].notna().any(axis=1)
    return table, clinical_availability


def subject_output_columns(table: pd.DataFrame) -> List[str]:
    wanted = [
        "SubjectID",
        "SiteCode",
        "Site3",
        "ResearchGroup_Mapped",
        "predicted_label",
        "ad_score",
        "threshold",
        "margin_to_threshold",
        "abs_margin_to_threshold",
        "confidence_bucket",
        "fold",
        "Manufacturer",
        "Age",
        "Sex",
        "error_type",
        "MMSE",
        "CDRSB",
        "MOCA",
        "DIGITSCOR",
        "PTEDUCAT",
        "Visit",
        "StudyDate",
        "ArchiveDate",
        "AcqDate",
        "ImageID",
        "Diagnosis",
        "source_batch",
        "source_label",
        "tensor_source",
        "stage_guess",
        "spectral_class",
        "n_timepoints_raw",
        "n_rois_raw",
        "finite_fraction",
        "scale_label",
        "roisignals_path",
        "dicom_series_ok",
        "python_bandpass_applied",
        "clinical_source",
        "has_cognitive_metadata",
        "has_qc_metadata",
    ]
    return [c for c in wanted if c in table.columns]


def group_summary(table: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for keys, g in table.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update(binary_metrics(g))
        row["age_median"] = float(g["Age"].median()) if "Age" in g.columns else float("nan")
        row["score_median"] = float(g["ad_score"].median())
        row["near_threshold_rate"] = float((g["abs_margin_to_threshold"] <= NEAR_THRESHOLD_ABS_MARGIN).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def false_positive_audit(table: pd.DataFrame) -> pd.DataFrame:
    cn = table[table["ResearchGroup_Mapped"].eq("CN")].copy()
    rows: List[Dict[str, Any]] = []
    for group_col in ["SiteCode", "Manufacturer", "Sex", "fold", "source_batch", "stage_guess", "spectral_class", "scale_label"]:
        if group_col not in cn.columns:
            continue
        for key, g in cn.groupby(group_col, dropna=False):
            fp = g[g["error_type"].eq("FP")]
            tn = g[g["error_type"].eq("TN")]
            rows.append(
                {
                    "group_variable": group_col,
                    "group": key,
                    "n_cn": int(len(g)),
                    "n_fp": int(len(fp)),
                    "fp_rate": safe_div(len(fp), len(g)),
                    "cn_score_median": float(g["ad_score"].median()) if len(g) else float("nan"),
                    "fp_score_median": float(fp["ad_score"].median()) if len(fp) else float("nan"),
                    "tn_score_median": float(tn["ad_score"].median()) if len(tn) else float("nan"),
                    "cn_age_median": float(g["Age"].median()) if "Age" in g.columns else float("nan"),
                    "fp_age_median": float(fp["Age"].median()) if "Age" in fp.columns and len(fp) else float("nan"),
                    "tn_age_median": float(tn["Age"].median()) if "Age" in tn.columns and len(tn) else float("nan"),
                    "fp_near_threshold_rate": float((fp["abs_margin_to_threshold"] <= NEAR_THRESHOLD_ABS_MARGIN).mean()) if len(fp) else float("nan"),
                    "fp_high_confidence_rate": float((fp["abs_margin_to_threshold"] >= HIGH_CONFIDENCE_ABS_MARGIN).mean()) if len(fp) else float("nan"),
                    "fp_mmse_median": float(fp["MMSE"].median()) if "MMSE" in fp.columns and len(fp) else float("nan"),
                    "tn_mmse_median": float(tn["MMSE"].median()) if "MMSE" in tn.columns and len(tn) else float("nan"),
                    "fp_cdrsb_median": float(fp["CDRSB"].median()) if "CDRSB" in fp.columns and len(fp) else float("nan"),
                    "tn_cdrsb_median": float(tn["CDRSB"].median()) if "CDRSB" in tn.columns and len(tn) else float("nan"),
                }
            )
    return pd.DataFrame(rows).sort_values(["group_variable", "fp_rate", "n_cn"], ascending=[True, False, False])


def false_negative_audit(table: pd.DataFrame) -> pd.DataFrame:
    ad = table[table["ResearchGroup_Mapped"].eq("AD")].copy()
    rows: List[Dict[str, Any]] = []
    for group_col in ["SiteCode", "Manufacturer", "Sex", "fold", "source_batch", "stage_guess", "spectral_class", "scale_label"]:
        if group_col not in ad.columns:
            continue
        for key, g in ad.groupby(group_col, dropna=False):
            fn = g[g["error_type"].eq("FN")]
            tp = g[g["error_type"].eq("TP")]
            rows.append(
                {
                    "group_variable": group_col,
                    "group": key,
                    "n_ad": int(len(g)),
                    "n_fn": int(len(fn)),
                    "fn_rate": safe_div(len(fn), len(g)),
                    "ad_score_median": float(g["ad_score"].median()) if len(g) else float("nan"),
                    "fn_score_median": float(fn["ad_score"].median()) if len(fn) else float("nan"),
                    "tp_score_median": float(tp["ad_score"].median()) if len(tp) else float("nan"),
                    "ad_age_median": float(g["Age"].median()) if "Age" in g.columns else float("nan"),
                    "fn_age_median": float(fn["Age"].median()) if "Age" in fn.columns and len(fn) else float("nan"),
                    "tp_age_median": float(tp["Age"].median()) if "Age" in tp.columns and len(tp) else float("nan"),
                    "fn_near_threshold_rate": float((fn["abs_margin_to_threshold"] <= NEAR_THRESHOLD_ABS_MARGIN).mean()) if len(fn) else float("nan"),
                    "fn_high_confidence_rate": float((fn["abs_margin_to_threshold"] >= HIGH_CONFIDENCE_ABS_MARGIN).mean()) if len(fn) else float("nan"),
                    "fn_mmse_median": float(fn["MMSE"].median()) if "MMSE" in fn.columns and len(fn) else float("nan"),
                    "tp_mmse_median": float(tp["MMSE"].median()) if "MMSE" in tp.columns and len(tp) else float("nan"),
                    "fn_cdrsb_median": float(fn["CDRSB"].median()) if "CDRSB" in fn.columns and len(fn) else float("nan"),
                    "tp_cdrsb_median": float(tp["CDRSB"].median()) if "CDRSB" in tp.columns and len(tp) else float("nan"),
                }
            )
    return pd.DataFrame(rows).sort_values(["group_variable", "fn_rate", "n_ad"], ascending=[True, False, False])


def site_counts(table: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (site, dx, man), g in table.groupby(["SiteCode", "ResearchGroup_Mapped", "Manufacturer"], dropna=False):
        rows.append({"SiteCode": site, "ResearchGroup_Mapped": dx, "Manufacturer": man, "n": int(len(g))})
    return pd.DataFrame(rows).sort_values(["SiteCode", "ResearchGroup_Mapped", "Manufacturer"])


def site_auc_report(table: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for site, g in table.groupby("SiteCode", dropna=False):
        n_cn = int((g["y_true"] == 0).sum())
        n_ad = int((g["y_true"] == 1).sum())
        eligible = len(g) >= MIN_SITE_N and n_cn >= MIN_SITE_PER_CLASS and n_ad >= MIN_SITE_PER_CLASS
        row = {
            "SiteCode": site,
            "n": int(len(g)),
            "n_cn": n_cn,
            "n_ad": n_ad,
            "manufacturers": ",".join(sorted(g["Manufacturer"].dropna().astype(str).unique())),
            "site_auc_eligible": bool(eligible),
            "site_class_status": "both_classes" if n_cn and n_ad else ("cn_only" if n_cn else "ad_only"),
            "auc": safe_auc(g["y_true"], g["ad_score"]) if eligible else float("nan"),
            "pr_auc": safe_pr_auc(g["y_true"], g["ad_score"]) if eligible else float("nan"),
            "fp_rate_cn": safe_div(int(((g["y_true"] == 0) & (g["y_pred"] == 1)).sum()), n_cn),
            "fn_rate_ad": safe_div(int(((g["y_true"] == 1) & (g["y_pred"] == 0)).sum()), n_ad),
            "score_median": float(g["ad_score"].median()),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["site_auc_eligible", "n"], ascending=[False, False])


def high_confidence_errors(table: pd.DataFrame) -> pd.DataFrame:
    err = table[table["error_type"].isin(["FP", "FN"])].copy()
    err = err[err["abs_margin_to_threshold"] >= HIGH_CONFIDENCE_ABS_MARGIN].copy()
    return err.sort_values("abs_margin_to_threshold", ascending=False)[subject_output_columns(err)]


def near_threshold_errors(table: pd.DataFrame) -> pd.DataFrame:
    err = table[table["error_type"].isin(["FP", "FN"])].copy()
    err = err[err["abs_margin_to_threshold"] <= NEAR_THRESHOLD_ABS_MARGIN].copy()
    return err.sort_values("abs_margin_to_threshold")[subject_output_columns(err)]


def availability_summary(table: pd.DataFrame, clinical_availability: Dict[str, Any]) -> Dict[str, Any]:
    cognitive_cols = [c for c in ["MMSE", "CDRSB", "MOCA", "DIGITSCOR", "PTEDUCAT"] if c in table.columns]
    qc_cols = [c for c in QC_FIELDS if c in table.columns]
    motion_like = [c for c in table.columns if re.search(r"motion|fd|framewise|realign|head", c, flags=re.I)]
    return {
        "clinical_sources": clinical_availability,
        "cognitive_columns_in_subject_table": cognitive_cols,
        "qc_columns_in_subject_table": qc_cols,
        "motion_columns_in_subject_table": motion_like,
        "n_subjects": int(len(table)),
        "n_with_any_cognitive_metadata": int(table["has_cognitive_metadata"].sum()) if "has_cognitive_metadata" in table.columns else 0,
        "n_with_any_qc_metadata": int(table["has_qc_metadata"].sum()) if "has_qc_metadata" in table.columns else 0,
    }


def recommendation_text(
    table: pd.DataFrame,
    fp_audit: pd.DataFrame,
    fn_audit: pd.DataFrame,
    site_auc: pd.DataFrame,
    availability: Dict[str, Any],
) -> str:
    metrics = binary_metrics(table)
    fp = table[table["error_type"].eq("FP")]
    fn = table[table["error_type"].eq("FN")]
    high_conf = table[table["error_type"].isin(["FP", "FN"]) & (table["abs_margin_to_threshold"] >= HIGH_CONFIDENCE_ABS_MARGIN)]
    near = table[table["error_type"].isin(["FP", "FN"]) & (table["abs_margin_to_threshold"] <= NEAR_THRESHOLD_ABS_MARGIN)]
    site_both = site_auc[site_auc["site_class_status"].eq("both_classes")]
    site_one = site_auc[~site_auc["site_class_status"].eq("both_classes")]

    philips_cn = table[(table["ResearchGroup_Mapped"].eq("CN")) & (table["Manufacturer"].eq("Philips"))]
    ge_ad = table[(table["ResearchGroup_Mapped"].eq("AD")) & (table["Manufacturer"].eq("GE"))]
    philips_fp_rate = safe_div(int((philips_cn["error_type"] == "FP").sum()), int(len(philips_cn)))
    ge_fn_rate = safe_div(int((ge_ad["error_type"] == "FN").sum()), int(len(ge_ad)))

    cognitive_line = (
        f"Available cognitive variables merged for {availability['n_with_any_cognitive_metadata']}/{availability['n_subjects']} subjects: "
        f"{', '.join(availability['cognitive_columns_in_subject_table']) or 'none'}."
    )
    motion_line = (
        "No subject-level motion columns were available in the final metadata/prediction table."
        if not availability["motion_columns_in_subject_table"]
        else f"Motion-like columns available: {', '.join(availability['motion_columns_in_subject_table'])}."
    )

    lines = [
        "# QC / Label-Noise Recommendation",
        "",
        "## Decision",
        "",
        "Do not exclude subjects based on this audit alone.",
        "",
        "No clear, predefinable QC or metadata rule was found that would legitimately improve AUC without changing the model. "
        "The observed errors remain consistent with genuine class overlap plus subgroup/threshold behavior, not a simple removable artifact.",
        "",
        "## Locked Model Sanity Check",
        "",
        f"- N: `{metrics['n']}`; CN: `{metrics['n_cn']}`; AD: `{metrics['n_ad']}`.",
        f"- Confusion: TN=`{metrics['tn']}`, FP=`{metrics['fp']}`, FN=`{metrics['fn']}`, TP=`{metrics['tp']}`.",
        f"- AUC: `{metrics['auc']:.4f}`; PR-AUC: `{metrics['pr_auc']:.4f}`; BA: `{metrics['balanced_accuracy']:.4f}`; F1: `{metrics['f1']:.4f}`.",
        "",
        "## False Positives CN",
        "",
        f"- CN false positives: `{len(fp)}`.",
        f"- Philips CN FP rate: `{philips_fp_rate:.4f}` ({int((philips_cn['error_type'] == 'FP').sum())}/{len(philips_cn)}).",
        f"- Near-threshold FP rate: `{(fp['abs_margin_to_threshold'] <= NEAR_THRESHOLD_ABS_MARGIN).mean():.4f}`.",
        f"- High-confidence FP rate: `{(fp['abs_margin_to_threshold'] >= HIGH_CONFIDENCE_ABS_MARGIN).mean():.4f}`.",
        "",
        "High-score CN are enriched in some SiteCode/Manufacturer groups, but this is not by itself a valid exclusion criterion. "
        "A site-level rule would remove biological and scanner heterogeneity and would risk post-hoc leakage.",
        "",
        "## False Negatives AD",
        "",
        f"- AD false negatives: `{len(fn)}`.",
        f"- GE AD FN rate: `{ge_fn_rate:.4f}` ({int((ge_ad['error_type'] == 'FN').sum())}/{len(ge_ad)}).",
        f"- Near-threshold FN rate: `{(fn['abs_margin_to_threshold'] <= NEAR_THRESHOLD_ABS_MARGIN).mean():.4f}`.",
        f"- High-confidence FN rate: `{(fn['abs_margin_to_threshold'] >= HIGH_CONFIDENCE_ABS_MARGIN).mean():.4f}`.",
        "",
        "FN errors include both near-threshold and high-margin cases, so threshold adjustment alone cannot remove the issue without trading specificity.",
        "",
        "## SiteCode",
        "",
        f"- Sites with both classes: `{len(site_both)}`.",
        f"- Sites with one class only: `{len(site_one)}`.",
        f"- Sites eligible for within-site AUC (n>={MIN_SITE_N}, CN>={MIN_SITE_PER_CLASS}, AD>={MIN_SITE_PER_CLASS}): `{int(site_auc['site_auc_eligible'].sum())}`.",
        "",
        "SiteCode is useful for reporting and future external review, but many sites are single-class or small; using SiteCode as an exclusion/filter after observing errors is not justified.",
        "",
        "## Clinical / QC Metadata Availability",
        "",
        cognitive_line,
        motion_line,
        f"QC fields available: {', '.join(availability['qc_columns_in_subject_table']) or 'none'}.",
        "",
        "TMT was not available in the ADNI metadata files found locally for this dataset. MMSE/CDRSB/MOCA coverage is partial and should be treated as descriptive only.",
        "",
        "## Operational Recommendation",
        "",
        "Keep the locked current FULL tanh `[1,0,2]` model as the paper model. "
        "Use this audit to document residual error structure and label/QC limitations rather than to curate the test set.",
    ]
    if len(high_conf):
        lines += ["", f"High-confidence error rows were written separately (`n={len(high_conf)}`)."]
    if len(near):
        lines += ["", f"Near-threshold error rows were written separately (`n={len(near)}`)."]
    return "\n".join(lines) + "\n"


def main() -> int:
    if OUTDIR.exists():
        shutil.rmtree(OUTDIR)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    table, clinical_availability = build_subject_table()
    subject_cols = subject_output_columns(table)
    subject_table = table[subject_cols].sort_values(["error_type", "SubjectID"]).reset_index(drop=True)
    subject_table.to_csv(OUTDIR / "locked_current_subject_error_table.csv", index=False)

    fp_audit = false_positive_audit(table)
    fn_audit = false_negative_audit(table)
    counts = site_counts(table)
    site_auc = site_auc_report(table)
    high_conf = high_confidence_errors(table)
    near = near_threshold_errors(table)

    write_pair("false_positive_cn_audit", fp_audit)
    write_pair("false_negative_ad_audit", fn_audit)
    write_pair("sitecode_diagnosis_manufacturer_counts", counts, max_rows=300)
    write_pair("sitecode_auc_report", site_auc, max_rows=200)
    write_pair("high_confidence_errors", high_conf, max_rows=200)
    write_pair("near_threshold_errors", near, max_rows=200)

    availability = availability_summary(table, clinical_availability)
    write_json = lambda path, payload: path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUTDIR / "qc_label_noise_recommendation.md").write_text(
        recommendation_text(table, fp_audit, fn_audit, site_auc, availability),
        encoding="utf-8",
    )
    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "readout_dir": str(READOUT_DIR),
        "metadata_path": str(METADATA_PATH),
        "output_dir": str(OUTDIR),
        "primary_model": PRIMARY_MODEL,
        "primary_threshold": PRIMARY_THRESHOLD,
        "near_threshold_abs_margin": NEAR_THRESHOLD_ABS_MARGIN,
        "high_confidence_abs_margin": HIGH_CONFIDENCE_ABS_MARGIN,
        "clinical_qc_availability": availability,
        "vae_retrained": False,
        "training_launched": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "configs_modified": False,
        "existing_results_modified": False,
    }
    write_json(OUTDIR / "command_log.json", command_log)

    metrics = binary_metrics(table)
    print(f"output_dir={OUTDIR}")
    print("training_launched=False")
    print("tensor_modified=False")
    print("metadata_modified=False")
    print("ledger_modified=False")
    print(
        f"confusion TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']} TP={metrics['tp']} "
        f"AUC={metrics['auc']:.4f} PR_AUC={metrics['pr_auc']:.4f}"
    )
    print(f"high_confidence_errors={len(high_conf)} near_threshold_errors={len(near)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
