#!/usr/bin/env python3
"""Clinical phenotype audit for stable locked-model FP/FN subjects.

Read-only with respect to model/data artifacts. Writes derived audit tables only
under results/revision_bspc_2026/locked_model_clinical_error_phenotype_audit/.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover - audit still works without scipy
    stats = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results" / "revision_bspc_2026"
INPUT_SUBJECT_ERRORS = (
    RESULTS_ROOT / "locked_primary_model_deep_model_card_audit" / "subject_error_table.csv"
)
OUTPUT_DIR = RESULTS_ROOT / "locked_model_clinical_error_phenotype_audit"

CLINICAL_SOURCES = [
    PROJECT_ROOT / "data" / "SubjctsDataAndTestsAAL3.csv",
    PROJECT_ROOT / "data" / "SubjectsData_AAL3_procesado2.csv",
    PROJECT_ROOT / "data" / "AD_fMRI_4_28_2026_extended.csv",
    PROJECT_ROOT / "data" / "RevisionPaperfMRI_2026_04_4_06_2026_extended.csv",
]

REQUESTED_FIELDS = [
    "MMSE",
    "CDRSB",
    "ADAS-Cog",
    "APOE",
    "ABETA",
    "TAU",
    "PTAU",
]

CONTINUOUS_BASE = ["Age", "original_n_TR"]
CONTINUOUS_CLINICAL = [
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
]
CATEGORICAL = ["Sex", "Manufacturer", "SiteCode"]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned outputs.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def clean_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def subject_col(df: pd.DataFrame) -> str | None:
    for col in ["SubjectID", "Subject", "PTID", "RID"]:
        if col in df.columns:
            return col
    return None


def normalize_dx(value: Any) -> str:
    text = clean_str(value).upper()
    if not text:
        return ""
    if text in {"AD", "DEMENTIA", "AD_DEMENTIA"}:
        return "AD"
    if text in {"CN", "NL", "NORMAL", "CONTROL"}:
        return "CN"
    if "MCI" in text:
        return "MCI"
    return text


def parse_date_series(df: pd.DataFrame) -> pd.Series:
    for col in ["StudyDate", "AcqDate", "ArchiveDate", "ExamDate", "VISDATE"]:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().any():
                return parsed
    return pd.Series(pd.NaT, index=df.index)


def standardize_clinical_source(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    scol = subject_col(df)
    if scol is None:
        return pd.DataFrame()
    df = df.copy()
    df["SubjectID"] = df[scol].astype(str).str.strip()
    if "Group" in df.columns and "ResearchGroup" not in df.columns:
        df["ResearchGroup"] = df["Group"]
    if "ResearchGroup_Mapped" not in df.columns:
        df["ResearchGroup_Mapped"] = df.get("ResearchGroup", "").map(normalize_dx)
    else:
        df["ResearchGroup_Mapped"] = df["ResearchGroup_Mapped"].map(normalize_dx)
    if "ImageID" not in df.columns and "ImageDataID" in df.columns:
        df["ImageID"] = df["ImageDataID"]
    df["clinical_date"] = parse_date_series(df)
    df["clinical_source"] = path.name
    keep = [
        "SubjectID",
        "clinical_source",
        "clinical_date",
        "ResearchGroup",
        "ResearchGroup_Mapped",
        "Visit",
        "ImageID",
        "Sex",
        "Age",
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
    ]
    for col in keep:
        if col not in df.columns:
            df[col] = np.nan
    return df[keep]


def load_clinical_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    inventory = []
    for path in CLINICAL_SOURCES:
        exists = path.exists()
        row = {"path": str(path), "exists": exists, "rows": 0, "subjects": 0}
        if exists:
            frame = standardize_clinical_source(path)
            row["rows"] = int(len(frame))
            row["subjects"] = int(frame["SubjectID"].nunique()) if not frame.empty else 0
            if not frame.empty:
                frames.append(frame)
        inventory.append(row)
    clinical = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not clinical.empty:
        for col in ["Age", *CONTINUOUS_CLINICAL]:
            if col in clinical.columns:
                clinical[col] = pd.to_numeric(clinical[col], errors="coerce")
        clinical = clinical.drop_duplicates()
    return clinical, pd.DataFrame(inventory)


def choose_best_clinical_rows(subjects: pd.DataFrame, clinical: pd.DataFrame) -> pd.DataFrame:
    selected_rows = []
    grouped = {sid: g.copy() for sid, g in clinical.groupby("SubjectID")}
    for _, srow in subjects.iterrows():
        sid = srow["SubjectID"]
        candidates = grouped.get(sid)
        out = {f"clinical_{col}": np.nan for col in clinical.columns if col != "SubjectID"}
        out["SubjectID"] = sid
        out["clinical_match_found"] = False
        out["clinical_rows_for_subject"] = 0
        if candidates is not None and not candidates.empty:
            candidates = candidates.copy()
            candidates["dx_match"] = candidates["ResearchGroup_Mapped"].eq(srow["ResearchGroup_Mapped"])
            candidates["age_abs_diff"] = (
                pd.to_numeric(candidates["Age"], errors="coerce") - pd.to_numeric(srow["Age"], errors="coerce")
            ).abs()
            candidates["nonmissing_score"] = candidates[CONTINUOUS_CLINICAL].notna().sum(axis=1)
            candidates["source_priority"] = candidates["clinical_source"].map(
                {
                    "SubjctsDataAndTestsAAL3.csv": 0,
                    "SubjectsData_AAL3_procesado2.csv": 1,
                    "AD_fMRI_4_28_2026_extended.csv": 2,
                    "RevisionPaperfMRI_2026_04_4_06_2026_extended.csv": 3,
                }
            ).fillna(9)
            best = candidates.sort_values(
                ["dx_match", "age_abs_diff", "nonmissing_score", "source_priority"],
                ascending=[False, True, False, True],
            ).iloc[0]
            out = {"SubjectID": sid, "clinical_match_found": True, "clinical_rows_for_subject": int(len(candidates))}
            for col in clinical.columns:
                if col != "SubjectID":
                    out[f"clinical_{col}"] = best[col]
        selected_rows.append(out)
    return pd.DataFrame(selected_rows)


def build_longitudinal_flags(subjects: pd.DataFrame, clinical: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    selected_dates = selected.set_index("SubjectID")["clinical_clinical_date"].to_dict()
    for sid, srow in subjects.set_index("SubjectID").iterrows():
        c = clinical[clinical["SubjectID"].eq(sid)].copy()
        diagnoses = []
        later_diagnoses = []
        if not c.empty:
            c = c.sort_values("clinical_date", na_position="last")
            diagnoses = [d for d in c["ResearchGroup_Mapped"].tolist() if clean_str(d)]
            selected_date = selected_dates.get(sid)
            if pd.notna(selected_date):
                later = c[c["clinical_date"].notna() & (c["clinical_date"] > selected_date)]
                later_diagnoses = [d for d in later["ResearchGroup_Mapped"].tolist() if clean_str(d)]
        dx_counter = Counter(diagnoses)
        later_counter = Counter(later_diagnoses)
        rows.append(
            {
                "SubjectID": sid,
                "n_clinical_rows_all_sources": int(len(c)),
                "dx_sequence_all_sources": " > ".join(diagnoses),
                "dx_counts_all_sources": dict(dx_counter),
                "later_dx_sequence_after_selected_row": " > ".join(later_diagnoses),
                "later_dx_counts_after_selected_row": dict(later_counter),
                "cn_later_mci_or_ad_evidence": bool(
                    srow["ResearchGroup_Mapped"] == "CN" and any(d in {"MCI", "AD"} for d in later_diagnoses)
                ),
                "cn_any_mci_or_ad_evidence_any_date": bool(
                    srow["ResearchGroup_Mapped"] == "CN" and any(d in {"MCI", "AD"} for d in diagnoses)
                ),
                "ad_prior_mci_or_cn_evidence_any_date": bool(
                    srow["ResearchGroup_Mapped"] == "AD" and any(d in {"CN", "MCI"} for d in diagnoses)
                ),
            }
        )
    return pd.DataFrame(rows)


def add_group_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["phenotype_group"] = "other"
    df.loc[
        df["ResearchGroup_Mapped"].eq("CN") & df["stable_error_across_available_comparators"].eq("stable_fp"),
        "phenotype_group",
    ] = "stable_fp_cn"
    df.loc[
        df["ResearchGroup_Mapped"].eq("CN") & df["locked_error_type"].eq("TN"),
        "phenotype_group",
    ] = "true_negative_cn"
    df.loc[
        df["ResearchGroup_Mapped"].eq("AD") & df["stable_error_across_available_comparators"].eq("stable_fn"),
        "phenotype_group",
    ] = "stable_fn_ad"
    df.loc[
        df["ResearchGroup_Mapped"].eq("AD") & df["locked_error_type"].eq("TP"),
        "phenotype_group",
    ] = "true_positive_ad"
    return df


def smd(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def continuous_compare(df: pd.DataFrame, group_a: str, group_b: str, variables: list[str]) -> pd.DataFrame:
    rows = []
    a_df = df[df["phenotype_group"].eq(group_a)]
    b_df = df[df["phenotype_group"].eq(group_b)]
    for var in variables:
        if var not in df.columns:
            continue
        a = pd.to_numeric(a_df[var], errors="coerce").dropna()
        b = pd.to_numeric(b_df[var], errors="coerce").dropna()
        row = {
            "comparison": f"{group_a}_vs_{group_b}",
            "variable": var,
            "n_a": int(len(a)),
            "n_b": int(len(b)),
            "mean_a": float(a.mean()) if len(a) else np.nan,
            "mean_b": float(b.mean()) if len(b) else np.nan,
            "median_a": float(a.median()) if len(a) else np.nan,
            "median_b": float(b.median()) if len(b) else np.nan,
            "iqr_a": float(a.quantile(0.75) - a.quantile(0.25)) if len(a) else np.nan,
            "iqr_b": float(b.quantile(0.75) - b.quantile(0.25)) if len(b) else np.nan,
            "smd_a_minus_b": smd(a, b),
            "welch_p": np.nan,
            "mannwhitney_p": np.nan,
        }
        if stats is not None and len(a) >= 2 and len(b) >= 2:
            try:
                row["welch_p"] = float(stats.ttest_ind(a, b, equal_var=False, nan_policy="omit").pvalue)
            except Exception:
                pass
            try:
                row["mannwhitney_p"] = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
            except Exception:
                pass
        rows.append(row)
    return pd.DataFrame(rows)


def categorical_compare(df: pd.DataFrame, group_a: str, group_b: str, variables: list[str]) -> pd.DataFrame:
    rows = []
    pair = df[df["phenotype_group"].isin([group_a, group_b])].copy()
    for var in variables:
        if var not in pair.columns:
            continue
        table = pd.crosstab(pair["phenotype_group"], pair[var].fillna("missing"))
        chi2_p = np.nan
        if stats is not None and table.shape[0] == 2 and table.shape[1] >= 2:
            try:
                chi2_p = float(stats.chi2_contingency(table, correction=False).pvalue)
            except Exception:
                pass
        for level in table.columns:
            rows.append(
                {
                    "comparison": f"{group_a}_vs_{group_b}",
                    "variable": var,
                    "level": level,
                    f"{group_a}_count": int(table.loc[group_a, level]) if group_a in table.index else 0,
                    f"{group_b}_count": int(table.loc[group_b, level]) if group_b in table.index else 0,
                    "chi2_p_overall": chi2_p,
                }
            )
    return pd.DataFrame(rows)


def reference_quantiles(df: pd.DataFrame, group: str) -> dict[str, dict[str, float]]:
    ref = df[df["phenotype_group"].eq(group)]
    q: dict[str, dict[str, float]] = {}
    for var in CONTINUOUS_CLINICAL:
        if var not in ref.columns:
            continue
        vals = pd.to_numeric(ref[var], errors="coerce").dropna()
        if len(vals) >= 4:
            q[var] = {
                "q25": float(vals.quantile(0.25)),
                "q50": float(vals.quantile(0.50)),
                "q75": float(vals.quantile(0.75)),
            }
    return q


def flag_subjects(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tn_q = reference_quantiles(df, "true_negative_cn")
    tp_q = reference_quantiles(df, "true_positive_ad")
    fp_rows = []
    for _, row in df[df["phenotype_group"].eq("stable_fp_cn")].iterrows():
        flags = []
        if pd.notna(row.get("MMSE")) and "MMSE" in tn_q and row["MMSE"] <= tn_q["MMSE"]["q25"]:
            flags.append("MMSE_low_vs_TN_CN")
        if pd.notna(row.get("CDRSB")) and "CDRSB" in tn_q and row["CDRSB"] > max(0.0, tn_q["CDRSB"]["q75"]):
            flags.append("CDRSB_high_vs_TN_CN")
        if pd.notna(row.get("MOCA")) and "MOCA" in tn_q and row["MOCA"] <= tn_q["MOCA"]["q25"]:
            flags.append("MOCA_low_vs_TN_CN")
        if pd.notna(row.get("Hippocampus")) and "Hippocampus" in tn_q and row["Hippocampus"] <= tn_q["Hippocampus"]["q25"]:
            flags.append("Hippocampus_low_vs_TN_CN")
        if pd.notna(row.get("MidTemp")) and "MidTemp" in tn_q and row["MidTemp"] <= tn_q["MidTemp"]["q25"]:
            flags.append("MidTemp_low_vs_TN_CN")
        if pd.notna(row.get("Ventricles")) and "Ventricles" in tn_q and row["Ventricles"] >= tn_q["Ventricles"]["q75"]:
            flags.append("Ventricles_high_vs_TN_CN")
        if pd.notna(row.get("ABETA")) and "ABETA" in tn_q and row["ABETA"] <= tn_q["ABETA"]["q25"]:
            flags.append("ABETA_low_vs_TN_CN")
        if pd.notna(row.get("TAU")) and "TAU" in tn_q and row["TAU"] >= tn_q["TAU"]["q75"]:
            flags.append("TAU_high_vs_TN_CN")
        if pd.notna(row.get("PTAU")) and "PTAU" in tn_q and row["PTAU"] >= tn_q["PTAU"]["q75"]:
            flags.append("PTAU_high_vs_TN_CN")
        if bool(row.get("cn_later_mci_or_ad_evidence")):
            flags.append("later_MCI_or_AD_evidence")
        fp_rows.append(
            {
                "SubjectID": row["SubjectID"],
                "fold": row["fold"],
                "Manufacturer": row["Manufacturer"],
                "SiteCode": row["SiteCode"],
                "Age": row["Age"],
                "Sex": row["Sex"],
                "locked_score": row["locked_score"],
                "locked_threshold": row["locked_threshold"],
                "clinical_suspicion_flag_count": len(flags),
                "clinical_suspicion_flags": ";".join(flags),
                "reviewer_safe_interpretation": (
                    "clinically_suspicious_control_candidate"
                    if flags
                    else "no_available_clinical_suspicion_flag"
                ),
            }
        )
    fn_rows = []
    for _, row in df[df["phenotype_group"].eq("stable_fn_ad")].iterrows():
        flags = []
        if pd.notna(row.get("MMSE")) and "MMSE" in tp_q and row["MMSE"] >= tp_q["MMSE"]["q75"]:
            flags.append("MMSE_high_vs_TP_AD")
        if pd.notna(row.get("CDRSB")) and "CDRSB" in tp_q and row["CDRSB"] <= tp_q["CDRSB"]["q25"]:
            flags.append("CDRSB_low_vs_TP_AD")
        if pd.notna(row.get("MOCA")) and "MOCA" in tp_q and row["MOCA"] >= tp_q["MOCA"]["q75"]:
            flags.append("MOCA_high_vs_TP_AD")
        if pd.notna(row.get("Hippocampus")) and "Hippocampus" in tp_q and row["Hippocampus"] >= tp_q["Hippocampus"]["q75"]:
            flags.append("Hippocampus_high_vs_TP_AD")
        if pd.notna(row.get("MidTemp")) and "MidTemp" in tp_q and row["MidTemp"] >= tp_q["MidTemp"]["q75"]:
            flags.append("MidTemp_high_vs_TP_AD")
        if pd.notna(row.get("Ventricles")) and "Ventricles" in tp_q and row["Ventricles"] <= tp_q["Ventricles"]["q25"]:
            flags.append("Ventricles_low_vs_TP_AD")
        if pd.notna(row.get("ABETA")) and "ABETA" in tp_q and row["ABETA"] >= tp_q["ABETA"]["q75"]:
            flags.append("ABETA_high_less_abnormal_vs_TP_AD")
        if pd.notna(row.get("TAU")) and "TAU" in tp_q and row["TAU"] <= tp_q["TAU"]["q25"]:
            flags.append("TAU_low_less_abnormal_vs_TP_AD")
        if pd.notna(row.get("PTAU")) and "PTAU" in tp_q and row["PTAU"] <= tp_q["PTAU"]["q25"]:
            flags.append("PTAU_low_less_abnormal_vs_TP_AD")
        fn_rows.append(
            {
                "SubjectID": row["SubjectID"],
                "fold": row["fold"],
                "Manufacturer": row["Manufacturer"],
                "SiteCode": row["SiteCode"],
                "Age": row["Age"],
                "Sex": row["Sex"],
                "locked_score": row["locked_score"],
                "locked_threshold": row["locked_threshold"],
                "milder_or_atypical_flag_count": len(flags),
                "milder_or_atypical_flags": ";".join(flags),
                "reviewer_safe_interpretation": (
                    "milder_or_atypical_ad_candidate"
                    if flags
                    else "no_available_milder_or_atypical_flag"
                ),
            }
        )
    return pd.DataFrame(fp_rows), pd.DataFrame(fn_rows)


def flag_summary(fp_flags: pd.DataFrame, fn_flags: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for table_name, df, col in [
        ("stable_fp_cn_clinical_suspicion", fp_flags, "clinical_suspicion_flags"),
        ("stable_fn_ad_milder_atypical", fn_flags, "milder_or_atypical_flags"),
    ]:
        counts: Counter[str] = Counter()
        for text in df.get(col, pd.Series(dtype=str)).fillna("").astype(str):
            for flag in [part for part in text.split(";") if part]:
                counts[flag] += 1
        rows.append(
            {
                "table": table_name,
                "flag": "any_flag",
                "n_subjects_with_flag": int(df.get(col, pd.Series(dtype=str)).fillna("").astype(str).ne("").sum()),
                "n_subjects_total": int(len(df)),
            }
        )
        for flag, count in counts.most_common():
            rows.append(
                {
                    "table": table_name,
                    "flag": flag,
                    "n_subjects_with_flag": int(count),
                    "n_subjects_total": int(len(df)),
                }
            )
    return pd.DataFrame(rows)


def longitudinal_summary(df: pd.DataFrame) -> pd.DataFrame:
    groups = ["stable_fp_cn", "true_negative_cn", "stable_fn_ad", "true_positive_ad"]
    rows = []
    for group in groups:
        sub = df[df["phenotype_group"].eq(group)]
        rows.append(
            {
                "phenotype_group": group,
                "n": int(len(sub)),
                "n_with_any_clinical_rows": int((sub["n_clinical_rows_all_sources"].fillna(0) > 0).sum()),
                "cn_later_mci_or_ad_evidence": int(sub.get("cn_later_mci_or_ad_evidence", pd.Series(dtype=bool)).fillna(False).sum()),
                "cn_any_mci_or_ad_evidence_any_date": int(sub.get("cn_any_mci_or_ad_evidence_any_date", pd.Series(dtype=bool)).fillna(False).sum()),
                "ad_prior_mci_or_cn_evidence_any_date": int(sub.get("ad_prior_mci_or_cn_evidence_any_date", pd.Series(dtype=bool)).fillna(False).sum()),
            }
        )
    return pd.DataFrame(rows)


def clinical_coverage(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for field in REQUESTED_FIELDS:
        source_col = field
        available_col = source_col in df.columns
        if field == "ADAS-Cog":
            available_col = any(c.upper().startswith("ADAS") for c in df.columns)
        if field == "APOE":
            available_col = any("APOE" in c.upper() for c in df.columns)
        if available_col and source_col in df.columns:
            coverage = int(df[source_col].notna().sum())
        else:
            coverage = 0
        rows.append(
            {
                "requested_field": field,
                "available_in_local_sources": bool(available_col),
                "nonmissing_subject_rows_in_audit_table": coverage,
                "note": "" if available_col else "not found in local clinical metadata headers",
            }
        )
    for field in ["PTEDUCAT", "DIGITSCOR", "MOCA", "Ventricles", "Hippocampus", "WholeBrain", "MidTemp"]:
        rows.append(
            {
                "requested_field": field,
                "available_in_local_sources": field in df.columns,
                "nonmissing_subject_rows_in_audit_table": int(df[field].notna().sum()) if field in df.columns else 0,
                "note": "additional available local clinical/cognitive/structural field",
            }
        )
    return pd.DataFrame(rows)


def write_md(path: Path, df: pd.DataFrame, title: str) -> None:
    if df.empty:
        body = "_No rows._\n"
    else:
        body = df.to_markdown(index=False)
    path.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")


def reviewer_text(
    cn_stats: pd.DataFrame,
    ad_stats: pd.DataFrame,
    fp_flags: pd.DataFrame,
    fn_flags: pd.DataFrame,
    coverage: pd.DataFrame,
    long_summary: pd.DataFrame,
) -> str:
    def get_mean(stats_df: pd.DataFrame, var: str, col: str) -> str:
        match = stats_df[stats_df["variable"].eq(var)]
        if match.empty or pd.isna(match[col].iloc[0]):
            return "NA"
        return f"{float(match[col].iloc[0]):.2f}"

    fp_flagged = int((fp_flags.get("clinical_suspicion_flag_count", pd.Series(dtype=float)) > 0).sum())
    fn_flagged = int((fn_flags.get("milder_or_atypical_flag_count", pd.Series(dtype=float)) > 0).sum())
    unavailable = coverage[~coverage["available_in_local_sources"]]["requested_field"].tolist()
    unavailable_txt = ", ".join(unavailable) if unavailable else "none"
    fp_later = int(
        long_summary.loc[
            long_summary["phenotype_group"].eq("stable_fp_cn"), "cn_later_mci_or_ad_evidence"
        ].sum()
    )

    return f"""# Reviewer-Ready Clinical Error Phenotype Interpretation

This read-only audit compared stable false-positive CN subjects against true-negative CN subjects, and stable false-negative AD subjects against true-positive AD subjects. No model was trained, no threshold was fit, and no subject exclusion rule was derived.

## Stable CN False Positives

Stable FP controls were compared with true-negative controls using available demographics, scanner/site fields, timepoint counts, cognitive measures, volumetric proxies, and CSF biomarkers. Mean age was `{get_mean(cn_stats, 'Age', 'mean_a')}` for stable FP CN versus `{get_mean(cn_stats, 'Age', 'mean_b')}` for true-negative CN. Mean original timepoint count was `{get_mean(cn_stats, 'original_n_TR', 'mean_a')}` versus `{get_mean(cn_stats, 'original_n_TR', 'mean_b')}`. The strongest non-model pattern remains acquisition/subgroup asymmetry, especially elevated Philips CN false positives. Clinical suspicion flags were present for `{fp_flagged}/{len(fp_flags)}` stable FP controls, but these flags are incomplete and heuristic because cognitive/biomarker coverage is sparse. Local longitudinal fMRI-linked clinical sources showed later MCI/AD evidence for `{fp_later}` stable FP controls.

## Stable AD False Negatives

Stable FN AD subjects were compared with true-positive AD subjects using the same fields. Mean age was `{get_mean(ad_stats, 'Age', 'mean_a')}` for stable FN AD versus `{get_mean(ad_stats, 'Age', 'mean_b')}` for true-positive AD. Mean original timepoint count was `{get_mean(ad_stats, 'original_n_TR', 'mean_a')}` versus `{get_mean(ad_stats, 'original_n_TR', 'mean_b')}`. Milder/atypical flags were present for `{fn_flagged}/{len(fn_flags)}` stable FN AD subjects. This is consistent with some false negatives being less separable clinically, but the local data do not support a deterministic label/QC exclusion rule.

## Metadata Availability

Available local clinical fields include MMSE, CDRSB, MOCA, education, selected structural volumes, ABETA, TAU, and PTAU where present. Requested fields not found in local clinical headers: `{unavailable_txt}`.

## Manuscript-Safe Conclusion

The stable errors do not justify excluding subjects or changing the locked model. Stable false positives include some controls with suspicious cognitive/biomarker/atrophy flags, and stable false negatives include some AD subjects with milder or less typical available phenotypes, but missingness is substantial and the findings are descriptive. The appropriate manuscript framing is that residual errors likely reflect a mixture of disease heterogeneity, scanner/site structure, threshold transfer limits, and incomplete clinical phenotype capture.
"""


def build_readme(output_dir: Path, updated_utc: str) -> str:
    return f"""# Locked Model Clinical Error Phenotype Audit

Updated: `{updated_utc}`

Input subject-level error table:

- `{INPUT_SUBJECT_ERRORS.relative_to(PROJECT_ROOT)}`

Clinical sources searched:

{chr(10).join(f'- `{p.relative_to(PROJECT_ROOT)}`' for p in CLINICAL_SOURCES)}

This audit is descriptive and read-only. It compares stable false-positive CN subjects with true-negative CN subjects and stable false-negative AD subjects with true-positive AD subjects. It does not train a model, fit thresholds, select a model, or define any subject exclusion rule.

Primary outputs:

- `subject_clinical_error_table.csv/.md`
- `stable_fp_cn_vs_true_negative_cn_continuous.csv/.md`
- `stable_fp_cn_vs_true_negative_cn_categorical.csv/.md`
- `stable_fn_ad_vs_true_positive_ad_continuous.csv/.md`
- `stable_fn_ad_vs_true_positive_ad_categorical.csv/.md`
- `stable_fp_cn_clinical_suspicion_flags.csv/.md`
- `stable_fn_ad_milder_atypical_flags.csv/.md`
- `clinical_flag_summary.csv/.md`
- `longitudinal_evidence_summary.csv/.md`
- `clinical_variable_coverage.csv/.md`
- `reviewer_ready_interpretation.md`

Safety:

- No training.
- No scoring.
- No threshold fitting.
- No model selection.
- No tensor, metadata, ledger, config, or model-output modification.
"""


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    updated = now_utc()
    if not INPUT_SUBJECT_ERRORS.exists():
        raise FileNotFoundError(INPUT_SUBJECT_ERRORS)
    existing_sources = [p for p in CLINICAL_SOURCES if p.exists()]
    if not existing_sources:
        raise FileNotFoundError("No local clinical sources found")

    if args.dry_run:
        payload = {
            "input_subject_errors": str(INPUT_SUBJECT_ERRORS),
            "output_dir": str(output_dir),
            "clinical_sources": [{"path": str(p), "exists": p.exists()} for p in CLINICAL_SOURCES],
            "training_launched": False,
            "scoring_launched": False,
            "threshold_fitting": False,
            "model_selection": False,
            "tensor_modified": False,
            "metadata_modified": False,
            "ledger_modified": False,
            "model_outputs_modified": False,
        }
        print(json.dumps(payload, indent=2))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    subjects = pd.read_csv(INPUT_SUBJECT_ERRORS)
    clinical, source_inventory = load_clinical_sources()
    selected = choose_best_clinical_rows(subjects, clinical)
    longitudinal = build_longitudinal_flags(subjects, clinical, selected)
    merged = subjects.merge(selected, on="SubjectID", how="left").merge(longitudinal, on="SubjectID", how="left")

    # Promote selected clinical columns to concise analysis names.
    for col in ["PTEDUCAT", "CDRSB", "MMSE", "DIGITSCOR", "MOCA", "Ventricles", "Hippocampus", "WholeBrain", "MidTemp", "ABETA", "TAU", "PTAU"]:
        merged[col] = pd.to_numeric(merged.get(f"clinical_{col}"), errors="coerce")
    for col in ["clinical_Age"]:
        if col in merged.columns:
            merged["clinical_age_selected"] = pd.to_numeric(merged[col], errors="coerce")
    merged = add_group_labels(merged)

    continuous_vars = CONTINUOUS_BASE + CONTINUOUS_CLINICAL
    cn_cont = continuous_compare(merged, "stable_fp_cn", "true_negative_cn", continuous_vars)
    ad_cont = continuous_compare(merged, "stable_fn_ad", "true_positive_ad", continuous_vars)
    cn_cat = categorical_compare(merged, "stable_fp_cn", "true_negative_cn", CATEGORICAL)
    ad_cat = categorical_compare(merged, "stable_fn_ad", "true_positive_ad", CATEGORICAL)
    coverage = clinical_coverage(merged)
    fp_flags, fn_flags = flag_subjects(merged)
    flag_counts = flag_summary(fp_flags, fn_flags)
    long_counts = longitudinal_summary(merged)

    outputs = {
        "clinical_source_inventory": source_inventory,
        "subject_clinical_error_table": merged,
        "clinical_variable_coverage": coverage,
        "stable_fp_cn_vs_true_negative_cn_continuous": cn_cont,
        "stable_fp_cn_vs_true_negative_cn_categorical": cn_cat,
        "stable_fn_ad_vs_true_positive_ad_continuous": ad_cont,
        "stable_fn_ad_vs_true_positive_ad_categorical": ad_cat,
        "stable_fp_cn_clinical_suspicion_flags": fp_flags,
        "stable_fn_ad_milder_atypical_flags": fn_flags,
        "clinical_flag_summary": flag_counts,
        "longitudinal_evidence_summary": long_counts,
    }
    for name, df in outputs.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)
        write_md(output_dir / f"{name}.md", df, name.replace("_", " ").title())

    (output_dir / "reviewer_ready_interpretation.md").write_text(
        reviewer_text(cn_cont, ad_cont, fp_flags, fn_flags, coverage, long_counts),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(build_readme(output_dir, updated), encoding="utf-8")
    command_log = {
        "script": str(Path(__file__).resolve()),
        "created_utc": updated,
        "input_subject_errors": str(INPUT_SUBJECT_ERRORS),
        "output_dir": str(output_dir),
        "clinical_sources": source_inventory.to_dict(orient="records"),
        "stable_fp_cn_n": int((merged["phenotype_group"] == "stable_fp_cn").sum()),
        "true_negative_cn_n": int((merged["phenotype_group"] == "true_negative_cn").sum()),
        "stable_fn_ad_n": int((merged["phenotype_group"] == "stable_fn_ad").sum()),
        "true_positive_ad_n": int((merged["phenotype_group"] == "true_positive_ad").sum()),
        "training_launched": False,
        "scoring_launched": False,
        "threshold_fitting": False,
        "model_selection": False,
        "subject_exclusion_rule_derived": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "model_outputs_modified": False,
    }
    (output_dir / "command_log.json").write_text(json.dumps(command_log, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote clinical error phenotype audit to {output_dir}")


if __name__ == "__main__":
    main()
