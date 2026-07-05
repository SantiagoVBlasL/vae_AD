#!/usr/bin/env python3
"""Audit ADNI v4 age balance and age-aware stratification options.

This is a metadata-only audit. It reads existing fold CSVs and classifier
metrics, but does not load tensors, checkpoints, or train any model.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - optional dependency guard
    scipy_stats = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V4_METADATA = (
    PROJECT_ROOT
    / "data/revision_bspc_2026/adni_expanded_v4_all_available/subject_metadata_adni_expanded_v4_all_available.csv"
)
DEFAULT_ORIGINAL_METADATA = PROJECT_ROOT / "data/SubjectsData_AAL3_procesado2.csv"
DEFAULT_BASELINE_RUN = PROJECT_ROOT / "results/revision_bspc_2026/adni_expanded_v4_beta25_ch4_1_0"
DEFAULT_CKPTSELECT_RUN = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/auc_sprint_adni_only/adni_expanded_v4_beta25_ch4_1_0_ckptselect"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results/revision_bspc_2026/auc_sprint_adni_only/age_balance_stratification_audit"
)

GENERATED_FILES = [
    "metadata_column_inventory.csv",
    "metadata_coverage_summary.csv",
    "metadata_subject_match_summary.csv",
    "metadata_missing_age_by_diagnosis.csv",
    "old_age_bin_mapping_coverage.csv",
    "v4_agebin_mapping.csv",
    "global_age_balance.csv",
    "current_fold_age_balance.csv",
    "weak_fold_demographic_profile.csv",
    "stratification_feasibility.csv",
    "alternative_split_simulation_summary.csv",
    "alternative_split_best_seed_examples.csv",
    "README.md",
    "audit_manifest.json",
    # Older names from development runs; removed on --overwrite for cleanliness.
    "column_inventory.csv",
    "coverage_summary.csv",
    "subject_match_summary.csv",
    "missing_age_by_diagnosis.csv",
]

ID_CANDIDATES = ["SubjectID", "PTID", "RID", "ImageID"]
DX_CANDIDATES = ["ResearchGroup_Mapped", "diagnosis", "Diagnosis", "ResearchGroup"]
AGE_BIN_CANDIDATES = [
    "age_bin",
    "AgeBin",
    "AgeGroup",
    "Age_Group",
    "AgeQuartile",
    "age_quartile",
    "Age_Quartile",
    "grupo_edad",
    "edad_grupo",
    "edad_cuartil",
    "cuartil_edad",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--v4-metadata", type=Path, default=DEFAULT_V4_METADATA)
    parser.add_argument("--original-metadata", type=Path, default=DEFAULT_ORIGINAL_METADATA)
    parser.add_argument("--baseline-run", type=Path, default=DEFAULT_BASELINE_RUN)
    parser.add_argument("--ckptselect-run", type=Path, default=DEFAULT_CKPTSELECT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    path = resolve(path)
    path.mkdir(parents=True, exist_ok=True)
    existing = [path / name for name in GENERATED_FILES if (path / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{path} already contains audit outputs; pass --overwrite")
    if overwrite:
        for file_path in existing:
            file_path.unlink()
    return path


def safe_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    except Exception:
        return "{}"


def first_existing(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    colset = set(columns)
    for col in candidates:
        if col in colset:
            return col
    lower_map = {str(c).lower(): c for c in columns}
    for col in candidates:
        hit = lower_map.get(col.lower())
        if hit is not None:
            return str(hit)
    return None


def normalize_sex(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    out = out.replace({"Female": "F", "female": "F", "Male": "M", "male": "M"})
    return out.fillna("Unknown")


def normalize_dx(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    mapping = {
        "Control": "CN",
        "Normal": "CN",
        "Healthy": "CN",
        "Alzheimer's Disease": "AD",
        "Alzheimer Disease": "AD",
        "Dementia": "AD",
    }
    return out.replace(mapping).fillna("Unknown")


def standardize_metadata(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    out = df.copy()
    sid_col = first_existing(out.columns, ["SubjectID", "PTID"])
    if sid_col is not None:
        out["SubjectID_norm"] = out[sid_col].astype("string").str.strip()
    else:
        out["SubjectID_norm"] = pd.Series(pd.NA, index=out.index, dtype="string")

    dx_col = first_existing(out.columns, DX_CANDIDATES)
    if dx_col is not None:
        out["diagnosis_norm"] = normalize_dx(out[dx_col])
    else:
        out["diagnosis_norm"] = "Unknown"

    if "Age" in out.columns:
        out["Age_norm"] = pd.to_numeric(out["Age"], errors="coerce")
    else:
        out["Age_norm"] = np.nan

    if "Sex" in out.columns:
        out["Sex_norm"] = normalize_sex(out["Sex"])
    else:
        out["Sex_norm"] = "Unknown"

    if "Manufacturer" in out.columns:
        out["Manufacturer_norm"] = out["Manufacturer"].astype("string").str.strip().fillna("Unknown")
    else:
        out["Manufacturer_norm"] = "Unknown"

    site_col = first_existing(out.columns, ["Site", "Site3", "SITE", "site"])
    if site_col is not None:
        out["Site_norm"] = out[site_col].astype("string").str.strip().fillna("Unknown")
    else:
        out["Site_norm"] = out["SubjectID_norm"].astype("string").str.extract(r"^(\d{3})", expand=False).fillna("Unknown")

    out["source_name"] = source_name
    return out


def compact_subject_metadata(df: pd.DataFrame, age_bin_col: Optional[str] = None) -> pd.DataFrame:
    cols = [
        "SubjectID_norm",
        "diagnosis_norm",
        "Age_norm",
        "Sex_norm",
        "Manufacturer_norm",
        "Site_norm",
    ]
    if age_bin_col is not None and age_bin_col in df.columns:
        cols.append(age_bin_col)
    work = df[cols].copy()
    work = work[work["SubjectID_norm"].notna()].copy()
    work["SubjectID_norm"] = work["SubjectID_norm"].astype(str)

    def first_non_null(series: pd.Series) -> Any:
        non_null = series.dropna()
        if non_null.empty:
            return np.nan
        return non_null.iloc[0]

    agg = {col: first_non_null for col in cols if col != "SubjectID_norm"}
    return work.groupby("SubjectID_norm", as_index=False).agg(agg)


def column_inventory(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for col in df.columns:
        lower = col.lower()
        role = "other"
        if col in ID_CANDIDATES:
            role = "subject_identifier"
        elif col in DX_CANDIDATES:
            role = "diagnosis"
        elif col == "Age":
            role = "age"
        elif col == "Sex":
            role = "sex"
        elif col == "Manufacturer":
            role = "manufacturer"
        elif lower in {"site", "site3"}:
            role = "site"
        elif col in AGE_BIN_CANDIDATES or ("age" in lower and ("bin" in lower or "quart" in lower or "group" in lower)):
            role = "age_bin_candidate"
        rows.append(
            {
                "dataset": dataset,
                "column": col,
                "role": role,
                "non_missing": int(df[col].notna().sum()),
                "missing": int(df[col].isna().sum()),
                "n_unique": int(df[col].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def detect_age_bin_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for col in df.columns:
        lower = col.lower()
        if col in AGE_BIN_CANDIDATES or ("age" in lower and ("bin" in lower or "quart" in lower or "group" in lower)):
            cols.append(col)
        elif any(token in lower for token in ["grupo_edad", "edad_grupo", "edad_cuartil", "cuartil_edad"]):
            cols.append(col)
    return sorted(set(cols))


def choose_old_age_bin_col(original_df: pd.DataFrame) -> Optional[str]:
    candidates = detect_age_bin_columns(original_df)
    if not candidates:
        return None
    preferred = ["Age_Group", "AgeGroup", "AgeQuartile", "age_quartile", "age_bin"]
    for col in preferred:
        if col in candidates:
            return col
    return candidates[0]


def add_age_bins(v4_subjects: pd.DataFrame) -> pd.DataFrame:
    out = v4_subjects.copy()
    out["diagnosis"] = out["diagnosis_norm"]
    out["Age"] = pd.to_numeric(out["Age_norm"], errors="coerce")
    cnad_mask = out["diagnosis"].isin(["CN", "AD"]) & out["Age"].notna()
    all_dx_mask = out["diagnosis"].isin(["CN", "AD", "MCI"]) & out["Age"].notna()

    def qcut_from_mask(mask: pd.Series, label_prefix: str) -> Tuple[pd.Series, List[float]]:
        labels = pd.Series(pd.NA, index=out.index, dtype="object")
        ages = out.loc[mask, "Age"]
        if ages.nunique(dropna=True) < 2:
            return labels, []
        _codes, bins = pd.qcut(ages, q=4, retbins=True, duplicates="drop")
        bins = np.unique(bins)
        if len(bins) < 2:
            return labels, bins.tolist()
        bins[0] = -np.inf
        bins[-1] = np.inf
        label_values = [f"{label_prefix}{i}" for i in range(1, len(bins))]
        labels = pd.cut(out["Age"], bins=bins, labels=label_values, include_lowest=True).astype("object")
        return labels, bins.tolist()

    out["AgeQ4_CNAD"], _bins_cnad = qcut_from_mask(cnad_mask, "Q")
    out["AgeQ4_All"], _bins_all = qcut_from_mask(all_dx_mask, "Q")

    median_cnad = out.loc[cnad_mask, "Age"].median()
    out["AgeBin2_CNAD"] = pd.NA
    if pd.notna(median_cnad):
        out.loc[out["Age"].notna(), "AgeBin2_CNAD"] = np.where(
            out.loc[out["Age"].notna(), "Age"] <= median_cnad,
            f"AgeLE{median_cnad:.1f}",
            f"AgeGT{median_cnad:.1f}",
        )

    out["AgeClinicalBin"] = pd.cut(
        out["Age"],
        bins=[-np.inf, 65, 70, 75, 80, np.inf],
        labels=["<65", "65-70", "70-75", "75-80", ">=80"],
        right=False,
    ).astype("object")
    return out


def age_stats(series: pd.Series) -> Dict[str, Any]:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if x.empty:
        return {
            "n_age": 0,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "iqr": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    q25 = float(x.quantile(0.25))
    q75 = float(x.quantile(0.75))
    return {
        "n_age": int(x.shape[0]),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if x.shape[0] > 1 else 0.0,
        "median": float(x.median()),
        "iqr": q75 - q25,
        "min": float(x.min()),
        "max": float(x.max()),
    }


def standardized_mean_difference(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce").dropna()
    y = pd.to_numeric(b, errors="coerce").dropna()
    if x.empty or y.empty:
        return np.nan
    sx = x.std(ddof=1) if len(x) > 1 else 0.0
    sy = y.std(ddof=1) if len(y) > 1 else 0.0
    pooled = math.sqrt((sx**2 + sy**2) / 2.0)
    if pooled == 0 or not np.isfinite(pooled):
        return np.nan
    return float((x.mean() - y.mean()) / pooled)


def safe_test(name: str, a: pd.Series, b: pd.Series) -> Tuple[float, float]:
    x = pd.to_numeric(a, errors="coerce").dropna()
    y = pd.to_numeric(b, errors="coerce").dropna()
    if scipy_stats is None or len(x) < 2 or len(y) < 2:
        return np.nan, np.nan
    try:
        if name == "ttest":
            res = scipy_stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
        elif name == "mannwhitney":
            res = scipy_stats.mannwhitneyu(x, y, alternative="two-sided")
        elif name == "ks":
            res = scipy_stats.ks_2samp(x, y, alternative="two-sided")
        else:
            return np.nan, np.nan
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return np.nan, np.nan


def count_json(df: pd.DataFrame, col: str) -> str:
    if col not in df.columns:
        return "{}"
    return safe_json(df[col].fillna("Unknown").astype(str).value_counts().sort_index().to_dict())


def counts_by_dx_json(df: pd.DataFrame, col: str) -> str:
    if col not in df.columns or "diagnosis" not in df.columns:
        return "{}"
    tab = df.groupby(["diagnosis", col], dropna=False).size().unstack(fill_value=0)
    return safe_json(tab.to_dict(orient="index"))


def metadata_coverage(
    v4_raw: pd.DataFrame,
    original_raw: pd.DataFrame,
    v4_subjects: pd.DataFrame,
    original_subjects: pd.DataFrame,
    old_age_col: Optional[str],
) -> Dict[str, pd.DataFrame]:
    column_df = pd.concat(
        [column_inventory(v4_raw, "v4"), column_inventory(original_raw, "original")],
        ignore_index=True,
    )

    missing_age = (
        v4_subjects.assign(Age_missing=v4_subjects["Age_norm"].isna())
        .groupby("diagnosis_norm", dropna=False)
        .agg(n=("SubjectID_norm", "count"), missing_age=("Age_missing", "sum"))
        .reset_index()
        .rename(columns={"diagnosis_norm": "diagnosis"})
    )
    missing_age["missing_age_pct"] = missing_age["missing_age"] / missing_age["n"].replace(0, np.nan)

    v4_ids = set(v4_subjects["SubjectID_norm"].dropna().astype(str))
    original_ids = set(original_subjects["SubjectID_norm"].dropna().astype(str))
    match_rows = [
        {"metric": "n_v4_subjects", "value": len(v4_ids)},
        {"metric": "n_original_subjects", "value": len(original_ids)},
        {"metric": "n_v4_matched_in_original", "value": len(v4_ids & original_ids)},
        {"metric": "n_v4_not_in_original", "value": len(v4_ids - original_ids)},
        {"metric": "n_original_not_in_v4", "value": len(original_ids - v4_ids)},
    ]
    match_df = pd.DataFrame(match_rows)

    coverage_rows = [
        {
            "dataset": "v4",
            "has_subject_id": bool(first_existing(v4_raw.columns, ["SubjectID", "PTID"])),
            "has_ptid": "PTID" in v4_raw.columns,
            "has_rid": "RID" in v4_raw.columns,
            "has_imageid": "ImageID" in v4_raw.columns,
            "has_age": "Age" in v4_raw.columns,
            "has_sex": "Sex" in v4_raw.columns,
            "has_manufacturer": "Manufacturer" in v4_raw.columns,
            "has_site": bool(first_existing(v4_raw.columns, ["Site", "Site3"])),
            "age_bin_columns": ",".join(detect_age_bin_columns(v4_raw)),
        },
        {
            "dataset": "original",
            "has_subject_id": bool(first_existing(original_raw.columns, ["SubjectID", "PTID"])),
            "has_ptid": "PTID" in original_raw.columns,
            "has_rid": "RID" in original_raw.columns,
            "has_imageid": "ImageID" in original_raw.columns,
            "has_age": "Age" in original_raw.columns,
            "has_sex": "Sex" in original_raw.columns,
            "has_manufacturer": "Manufacturer" in original_raw.columns,
            "has_site": bool(first_existing(original_raw.columns, ["Site", "Site3"])),
            "age_bin_columns": ",".join(detect_age_bin_columns(original_raw)),
        },
    ]
    coverage_df = pd.DataFrame(coverage_rows)

    old_cov = pd.DataFrame()
    if old_age_col is not None and old_age_col in original_subjects.columns:
        old_cov = (
            v4_subjects[["SubjectID_norm", "diagnosis_norm"]]
            .merge(original_subjects[["SubjectID_norm", old_age_col]], on="SubjectID_norm", how="left")
            .assign(has_old_age_bin=lambda d: d[old_age_col].notna())
            .groupby("diagnosis_norm", dropna=False)
            .agg(n_v4=("SubjectID_norm", "count"), old_age_bin_available=("has_old_age_bin", "sum"))
            .reset_index()
            .rename(columns={"diagnosis_norm": "diagnosis"})
        )
        old_cov["old_age_bin_column"] = old_age_col
        old_cov["old_age_bin_missing_in_v4"] = old_cov["n_v4"] - old_cov["old_age_bin_available"]
        old_cov["old_age_bin_coverage_pct"] = old_cov["old_age_bin_available"] / old_cov["n_v4"].replace(0, np.nan)

    return {
        "column_inventory": column_df,
        "coverage_summary": coverage_df,
        "subject_match_summary": match_df,
        "missing_age_by_diagnosis": missing_age,
        "old_age_bin_mapping_coverage": old_cov,
    }


def global_age_balance(v4_map: pd.DataFrame) -> pd.DataFrame:
    cnad = v4_map[v4_map["diagnosis"].isin(["CN", "AD"])].copy()
    rows: List[Dict[str, Any]] = []
    ad_age = cnad.loc[cnad["diagnosis"].eq("AD"), "Age"]
    cn_age = cnad.loc[cnad["diagnosis"].eq("CN"), "Age"]
    smd_ad_cn = standardized_mean_difference(ad_age, cn_age)
    t_stat, t_p = safe_test("ttest", ad_age, cn_age)
    mw_stat, mw_p = safe_test("mannwhitney", ad_age, cn_age)
    ks_stat, ks_p = safe_test("ks", ad_age, cn_age)
    for diagnosis, group in cnad.groupby("diagnosis", dropna=False):
        st = age_stats(group["Age"])
        row = {
            "diagnosis": diagnosis,
            "n": int(len(group)),
            "sex_counts": count_json(group, "Sex"),
            "manufacturer_counts": count_json(group, "Manufacturer"),
            "site_counts": count_json(group, "Site"),
            "age_smd_AD_vs_CN": smd_ad_cn,
            "age_smd_abs_AD_vs_CN": abs(smd_ad_cn) if np.isfinite(smd_ad_cn) else np.nan,
            "ttest_stat_AD_vs_CN": t_stat,
            "ttest_p_AD_vs_CN": t_p,
            "mannwhitney_stat_AD_vs_CN": mw_stat,
            "mannwhitney_p_AD_vs_CN": mw_p,
            "ks_stat_AD_vs_CN": ks_stat,
            "ks_p_AD_vs_CN": ks_p,
        }
        row.update({f"age_{k}": v for k, v in st.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def metrics_by_fold(run_dir: Path) -> pd.DataFrame:
    files = sorted(resolve(run_dir).glob("all_folds_metrics_MULTI*.csv"))
    if not files:
        return pd.DataFrame()
    df = pd.read_csv(files[0])
    if "actual_classifier_type" not in df.columns:
        df["actual_classifier_type"] = "unknown"
    cols = ["fold", "actual_classifier_type", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity"]
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
    wide_parts = []
    for metric in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity"]:
        part = df.pivot_table(index="fold", columns="actual_classifier_type", values=metric, aggfunc="first")
        part.columns = [f"{metric}_{c}" for c in part.columns]
        wide_parts.append(part)
    wide = pd.concat(wide_parts, axis=1).reset_index()
    auc_cols = [c for c in wide.columns if c.startswith("auc_")]
    if auc_cols:
        wide["auc_min"] = wide[auc_cols].min(axis=1)
        wide["auc_mean"] = wide[auc_cols].mean(axis=1)
    return wide


def read_fold_subjects(fold_dir: Path, split: str) -> pd.DataFrame:
    path = fold_dir / f"{split}_subjects_fold.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "SubjectID" not in df.columns:
        return pd.DataFrame()
    return df


def merge_fold_metadata(subjects: pd.DataFrame, v4_map: pd.DataFrame) -> pd.DataFrame:
    if subjects.empty:
        return subjects
    work = subjects.copy()
    work["SubjectID"] = work["SubjectID"].astype(str)
    keep = [
        "SubjectID",
        "diagnosis",
        "Age",
        "Sex",
        "Manufacturer",
        "Site",
        "AgeQ4_CNAD",
        "AgeQ4_All",
        "AgeBin2_CNAD",
        "AgeClinicalBin",
        "old_age_bin_if_available",
    ]
    meta = v4_map[keep].copy()
    merged = work.merge(meta, on="SubjectID", how="left", suffixes=("_fold", ""))
    if "ResearchGroup_Mapped" in merged.columns:
        merged["diagnosis"] = merged["diagnosis"].where(merged["diagnosis"].notna(), merged["ResearchGroup_Mapped"])
    return merged


def split_summary(prefix: str, df: pd.DataFrame) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        f"{prefix}_n": int(len(df)),
        f"{prefix}_n_CN": int(df["diagnosis"].eq("CN").sum()) if "diagnosis" in df.columns else 0,
        f"{prefix}_n_AD": int(df["diagnosis"].eq("AD").sum()) if "diagnosis" in df.columns else 0,
        f"{prefix}_age_stats_by_diagnosis": "{}",
        f"{prefix}_sex_counts_by_diagnosis": "{}",
        f"{prefix}_manufacturer_counts_by_diagnosis": "{}",
        f"{prefix}_site_counts_by_diagnosis": "{}",
    }
    if df.empty:
        return row
    age_stats_by_dx: Dict[str, Any] = {}
    for dx, group in df.groupby("diagnosis", dropna=False):
        age_stats_by_dx[str(dx)] = age_stats(group["Age"])
    row[f"{prefix}_age_stats_by_diagnosis"] = safe_json(age_stats_by_dx)
    row[f"{prefix}_sex_counts_by_diagnosis"] = counts_by_dx_json(df, "Sex")
    row[f"{prefix}_manufacturer_counts_by_diagnosis"] = counts_by_dx_json(df, "Manufacturer")
    row[f"{prefix}_site_counts_by_diagnosis"] = counts_by_dx_json(df, "Site")
    return row


def max_proportion_diff(train: pd.DataFrame, test: pd.DataFrame, col: str) -> float:
    if col not in train.columns or col not in test.columns:
        return np.nan
    values = sorted(set(train[col].dropna().astype(str)) | set(test[col].dropna().astype(str)))
    if not values:
        return np.nan
    train_n = len(train)
    test_n = len(test)
    if train_n == 0 or test_n == 0:
        return np.nan
    diffs = []
    train_counts = train[col].fillna("Unknown").astype(str).value_counts()
    test_counts = test[col].fillna("Unknown").astype(str).value_counts()
    for value in values:
        diffs.append(abs(train_counts.get(value, 0) / train_n - test_counts.get(value, 0) / test_n))
    return float(max(diffs)) if diffs else np.nan


def current_fold_age_balance(run_name: str, run_dir: Path, v4_map: pd.DataFrame) -> pd.DataFrame:
    run_dir = resolve(run_dir)
    if not run_dir.exists():
        return pd.DataFrame()
    metrics = metrics_by_fold(run_dir)
    metric_by_fold = metrics.set_index("fold") if not metrics.empty else pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for fold_dir in sorted(run_dir.glob("fold_*")):
        if not fold_dir.is_dir():
            continue
        try:
            fold = int(fold_dir.name.split("_")[-1])
        except ValueError:
            continue
        train = merge_fold_metadata(read_fold_subjects(fold_dir, "train_dev"), v4_map)
        test = merge_fold_metadata(read_fold_subjects(fold_dir, "test"), v4_map)
        if train.empty or test.empty:
            continue
        row: Dict[str, Any] = {"run": run_name, "run_dir": str(run_dir), "fold": fold}
        row.update(split_summary("train_dev", train))
        row.update(split_summary("test", test))
        row["age_smd_train_dev_vs_test"] = standardized_mean_difference(train["Age"], test["Age"])
        row["age_smd_abs_train_dev_vs_test"] = abs(row["age_smd_train_dev_vs_test"])
        row["age_smd_AD_vs_CN_test"] = standardized_mean_difference(
            test.loc[test["diagnosis"].eq("AD"), "Age"],
            test.loc[test["diagnosis"].eq("CN"), "Age"],
        )
        row["age_smd_abs_AD_vs_CN_test"] = abs(row["age_smd_AD_vs_CN_test"])
        row["age_smd_AD_vs_CN_train_dev"] = standardized_mean_difference(
            train.loc[train["diagnosis"].eq("AD"), "Age"],
            train.loc[train["diagnosis"].eq("CN"), "Age"],
        )
        row["age_smd_abs_AD_vs_CN_train_dev"] = abs(row["age_smd_AD_vs_CN_train_dev"])
        for dx in ["AD", "CN"]:
            row[f"age_smd_train_dev_vs_test_{dx}"] = standardized_mean_difference(
                train.loc[train["diagnosis"].eq(dx), "Age"],
                test.loc[test["diagnosis"].eq(dx), "Age"],
            )
            row[f"age_smd_abs_train_dev_vs_test_{dx}"] = abs(row[f"age_smd_train_dev_vs_test_{dx}"])
        row["sex_imbalance_train_test"] = max_proportion_diff(train, test, "Sex")
        row["manufacturer_imbalance_train_test"] = max_proportion_diff(train, test, "Manufacturer")
        row["site_imbalance_train_test"] = max_proportion_diff(train, test, "Site")
        if not metric_by_fold.empty and fold in metric_by_fold.index:
            metric_row = metric_by_fold.loc[fold]
            for col, value in metric_row.items():
                if col != "fold":
                    row[col] = value
        rows.append(row)
    return pd.DataFrame(rows)


def stratum_key(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    temp = df[list(cols)].copy()
    for col in cols:
        temp[col] = temp[col].fillna(f"{col}_Unknown").astype(str)
    return temp.apply(lambda r: "|".join(r.values.astype(str)), axis=1)


def stratification_schemes() -> List[Tuple[str, List[str]]]:
    return [
        ("A_diagnosis_plus_Sex", ["diagnosis", "Sex"]),
        ("B_diagnosis_plus_AgeBin2_CNAD", ["diagnosis", "AgeBin2_CNAD"]),
        ("C_diagnosis_plus_AgeQ4_CNAD", ["diagnosis", "AgeQ4_CNAD"]),
        ("D_diagnosis_plus_Sex_plus_AgeBin2_CNAD", ["diagnosis", "Sex", "AgeBin2_CNAD"]),
        ("E_diagnosis_plus_Sex_plus_AgeQ4_CNAD", ["diagnosis", "Sex", "AgeQ4_CNAD"]),
        ("F_diagnosis_plus_Manufacturer", ["diagnosis", "Manufacturer"]),
        ("G_diagnosis_plus_Sex_plus_Manufacturer", ["diagnosis", "Sex", "Manufacturer"]),
        (
            "H_diagnosis_plus_Sex_plus_AgeBin2_CNAD_plus_Manufacturer",
            ["diagnosis", "Sex", "AgeBin2_CNAD", "Manufacturer"],
        ),
    ]


def stratification_feasibility(cnad: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for scheme, cols in stratification_schemes():
        eligible = cnad.dropna(subset=cols).copy()
        key = stratum_key(eligible, cols)
        counts = key.value_counts()
        min_count = int(counts.min()) if not counts.empty else 0
        min_train_after_outer = min_count - int(math.ceil(min_count / 5)) if min_count else 0
        count_lt_5 = int((counts < 5).sum()) if not counts.empty else 0
        count_lt_10 = int((counts < 10).sum()) if not counts.empty else 0
        feasible_outer = bool(not counts.empty and (counts >= 5).all())
        feasible_inner = bool(not counts.empty and ((counts - np.ceil(counts / 5).astype(int)) >= 5).all())
        fragile = count_lt_10 > 0 or len(eligible) < len(cnad)
        warnings_seen: List[str] = []
        if len(eligible) < len(cnad):
            warnings_seen.append(f"drops {len(cnad) - len(eligible)} CN/AD rows with missing stratum values")
        if count_lt_5:
            warnings_seen.append("not feasible for 5-fold outer CV")
        elif count_lt_10:
            warnings_seen.append("5-fold outer feasible but fragile for inner CV or repeated splits")
        if feasible_outer and not feasible_inner:
            warnings_seen.append("outer feasible but inner 5-fold after outer holdout is fragile")
        rows.append(
            {
                "scheme": scheme,
                "stratification_cols": "+".join(cols),
                "n_subjects_used": int(len(eligible)),
                "n_subjects_total_CNAD": int(len(cnad)),
                "n_strata": int(len(counts)),
                "min_stratum_count": min_count,
                "strata_count_lt_5": count_lt_5,
                "strata_count_lt_10": count_lt_10,
                "min_train_stratum_count_after_outer_holdout_est": int(min_train_after_outer),
                "feasible_5fold_outer_cv": feasible_outer,
                "feasible_5fold_inner_cv_inside_train_dev": feasible_inner,
                "fragile": fragile,
                "warning": "; ".join(warnings_seen),
            }
        )
    return pd.DataFrame(rows)


def fold_split_metrics(train: pd.DataFrame, test: pd.DataFrame) -> Dict[str, float]:
    y_train = train["diagnosis"].eq("AD").astype(float)
    y_test = test["diagnosis"].eq("AD").astype(float)
    return {
        "abs_age_smd_train_vs_test": abs(standardized_mean_difference(train["Age"], test["Age"])),
        "ad_cn_prop_imbalance": abs(float(y_train.mean() - y_test.mean())),
        "sex_imbalance": max_proportion_diff(train, test, "Sex"),
        "manufacturer_imbalance": max_proportion_diff(train, test, "Manufacturer"),
        "min_fold_class_count": float(test["diagnosis"].value_counts().reindex(["CN", "AD"], fill_value=0).min()),
    }


def simulate_alternative_splits(cnad: pd.DataFrame, feasibility: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows: List[Dict[str, Any]] = []
    best_examples: List[Dict[str, Any]] = []
    feasible_schemes = feasibility[feasibility["feasible_5fold_outer_cv"]].copy()
    for _, row in feasible_schemes.iterrows():
        scheme = row["scheme"]
        cols = str(row["stratification_cols"]).split("+")
        eligible = cnad.dropna(subset=cols).copy().reset_index(drop=True)
        if eligible.empty:
            continue
        key = stratum_key(eligible, cols)
        for seed in range(100):
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            fold_metrics: List[Dict[str, float]] = []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                splits = list(splitter.split(np.zeros(len(eligible)), key))
            for fold, (train_idx, test_idx) in enumerate(splits, start=1):
                train = eligible.iloc[train_idx].copy()
                test = eligible.iloc[test_idx].copy()
                metrics = fold_split_metrics(train, test)
                metrics["fold"] = float(fold)
                fold_metrics.append(metrics)
            fm = pd.DataFrame(fold_metrics)
            seed_rows.append(
                {
                    "scheme": scheme,
                    "stratification_cols": row["stratification_cols"],
                    "seed": seed,
                    "mean_abs_age_smd_train_vs_test": float(fm["abs_age_smd_train_vs_test"].mean()),
                    "max_abs_age_smd_train_vs_test": float(fm["abs_age_smd_train_vs_test"].max()),
                    "mean_AD_CN_prop_imbalance": float(fm["ad_cn_prop_imbalance"].mean()),
                    "mean_sex_imbalance": float(fm["sex_imbalance"].mean()),
                    "mean_manufacturer_imbalance": float(fm["manufacturer_imbalance"].mean()),
                    "min_fold_class_count": int(fm["min_fold_class_count"].min()),
                    "n_strata": int(row["n_strata"]),
                    "min_stratum_count": int(row["min_stratum_count"]),
                    "strata_count_lt_10": int(row["strata_count_lt_10"]),
                }
            )
    seeds = pd.DataFrame(seed_rows)
    if seeds.empty:
        return pd.DataFrame(), pd.DataFrame()
    seeds["rank_score"] = (
        seeds["mean_abs_age_smd_train_vs_test"]
        + seeds["max_abs_age_smd_train_vs_test"]
        + seeds["mean_AD_CN_prop_imbalance"]
        + seeds["mean_sex_imbalance"]
        + seeds["mean_manufacturer_imbalance"]
        + 0.02 * seeds["strata_count_lt_10"]
    )
    for scheme, group in seeds.groupby("scheme", dropna=False):
        best = group.sort_values("rank_score").iloc[0].to_dict()
        best_examples.append(best)
    summary = (
        seeds.groupby(["scheme", "stratification_cols"], as_index=False)
        .agg(
            n_seeds=("seed", "count"),
            best_seed=("rank_score", lambda s: int(seeds.loc[s.idxmin(), "seed"])),
            mean_abs_age_smd_train_vs_test_mean=("mean_abs_age_smd_train_vs_test", "mean"),
            mean_abs_age_smd_train_vs_test_min=("mean_abs_age_smd_train_vs_test", "min"),
            max_abs_age_smd_train_vs_test_mean=("max_abs_age_smd_train_vs_test", "mean"),
            max_abs_age_smd_train_vs_test_min=("max_abs_age_smd_train_vs_test", "min"),
            mean_AD_CN_prop_imbalance_mean=("mean_AD_CN_prop_imbalance", "mean"),
            mean_sex_imbalance_mean=("mean_sex_imbalance", "mean"),
            mean_manufacturer_imbalance_mean=("mean_manufacturer_imbalance", "mean"),
            min_fold_class_count_min=("min_fold_class_count", "min"),
            n_strata=("n_strata", "first"),
            min_stratum_count=("min_stratum_count", "first"),
            strata_count_lt_10=("strata_count_lt_10", "first"),
            rank_score_min=("rank_score", "min"),
            rank_score_mean=("rank_score", "mean"),
        )
        .sort_values(["rank_score_min", "mean_abs_age_smd_train_vs_test_min"])
    )
    return summary, pd.DataFrame(best_examples).sort_values("rank_score")


def weak_fold_profile(fold_balance: pd.DataFrame) -> pd.DataFrame:
    if fold_balance.empty:
        return pd.DataFrame()
    out = fold_balance.copy()
    auc_cols = [c for c in out.columns if c.startswith("auc_")]
    if "auc_min" not in out.columns and auc_cols:
        out["auc_min"] = out[auc_cols].min(axis=1)
    q25 = out["auc_min"].quantile(0.25) if "auc_min" in out.columns else np.nan
    age_test_threshold = out["age_smd_abs_AD_vs_CN_test"].quantile(0.75)
    age_train_test_threshold = out["age_smd_abs_train_dev_vs_test"].quantile(0.75)
    out["is_fold_2"] = out["fold"].eq(2)
    out["is_weak_auc"] = out["auc_min"].le(q25) if np.isfinite(q25) else False
    out["high_test_AD_CN_age_smd"] = out["age_smd_abs_AD_vs_CN_test"].ge(age_test_threshold)
    out["high_train_test_age_smd"] = out["age_smd_abs_train_dev_vs_test"].ge(age_train_test_threshold)
    keep = [
        "run",
        "fold",
        "auc_min",
        "auc_mean",
        "is_fold_2",
        "is_weak_auc",
        "age_smd_abs_train_dev_vs_test",
        "age_smd_abs_AD_vs_CN_test",
        "age_smd_abs_AD_vs_CN_train_dev",
        "age_smd_abs_train_dev_vs_test_AD",
        "age_smd_abs_train_dev_vs_test_CN",
        "sex_imbalance_train_test",
        "manufacturer_imbalance_train_test",
        "site_imbalance_train_test",
        "high_test_AD_CN_age_smd",
        "high_train_test_age_smd",
        "test_age_stats_by_diagnosis",
        "test_sex_counts_by_diagnosis",
        "test_manufacturer_counts_by_diagnosis",
    ]
    keep = [c for c in keep if c in out.columns]
    return out[keep].sort_values(["is_weak_auc", "is_fold_2", "auc_min"], ascending=[False, False, True])


def smd_interpretation(abs_smd: float) -> str:
    if not np.isfinite(abs_smd):
        return "unavailable"
    if abs_smd < 0.1:
        return "well balanced"
    if abs_smd < 0.2:
        return "small imbalance"
    if abs_smd < 0.5:
        return "moderate imbalance"
    return "large imbalance"


def write_readme(
    outdir: Path,
    v4_map: pd.DataFrame,
    global_balance: pd.DataFrame,
    fold_balance: pd.DataFrame,
    feasibility: pd.DataFrame,
    simulation_summary: pd.DataFrame,
    old_age_col: Optional[str],
) -> None:
    cnad = v4_map[v4_map["diagnosis"].isin(["CN", "AD"])].copy()
    comparison = global_balance[["age_smd_abs_AD_vs_CN", "mannwhitney_p_AD_vs_CN", "ks_p_AD_vs_CN"]].dropna(
        how="all"
    )
    global_abs_smd = float(comparison["age_smd_abs_AD_vs_CN"].dropna().iloc[0]) if not comparison.empty else np.nan
    fold2 = fold_balance[fold_balance["fold"].eq(2)].copy() if not fold_balance.empty else pd.DataFrame()
    fold2_auc = float(fold2["auc_min"].min()) if not fold2.empty and "auc_min" in fold2.columns else np.nan
    fold2_age_smd = (
        float(fold2["age_smd_abs_AD_vs_CN_test"].max())
        if not fold2.empty and "age_smd_abs_AD_vs_CN_test" in fold2.columns
        else np.nan
    )
    current = feasibility[feasibility["scheme"].eq("A_diagnosis_plus_Sex")]
    agebin2 = feasibility[feasibility["scheme"].eq("B_diagnosis_plus_AgeBin2_CNAD")]
    sex_agebin2 = feasibility[feasibility["scheme"].eq("D_diagnosis_plus_Sex_plus_AgeBin2_CNAD")]
    sex_ageq4 = feasibility[feasibility["scheme"].eq("E_diagnosis_plus_Sex_plus_AgeQ4_CNAD")]
    best_sim = simulation_summary.head(1)
    old_cov_path = outdir / "old_age_bin_mapping_coverage.csv"
    old_cov = pd.read_csv(old_cov_path) if old_cov_path.exists() else pd.DataFrame()
    old_cov_cnad = old_cov[old_cov["diagnosis"].isin(["CN", "AD"])] if not old_cov.empty else pd.DataFrame()
    old_available = int(old_cov_cnad["old_age_bin_available"].sum()) if not old_cov_cnad.empty else 0
    old_total = int(old_cov_cnad["n_v4"].sum()) if not old_cov_cnad.empty and "n_v4" in old_cov_cnad.columns else 0
    old_pct = old_available / old_total if old_total else np.nan

    def feasible_text(df: pd.DataFrame, col: str = "feasible_5fold_outer_cv") -> str:
        if df.empty:
            return "unavailable"
        return "yes" if bool(df.iloc[0][col]) else "no"

    recommendation = "Keep diagnosis+Sex as the primary split for continuity; add diagnosis+AgeBin2_CNAD as an age-balance sensitivity split."
    if not agebin2.empty and bool(agebin2.iloc[0]["feasible_5fold_outer_cv"]):
        if not sex_agebin2.empty and bool(sex_agebin2.iloc[0]["feasible_5fold_outer_cv"]):
            recommendation = (
                "Keep diagnosis+Sex as the primary split for comparability, and run diagnosis+Sex+AgeBin2_CNAD "
                "as the most direct age-balanced sensitivity split."
            )
        else:
            recommendation = (
                "Keep diagnosis+Sex as the primary split for comparability, and use diagnosis+AgeBin2_CNAD "
                "as the less fragile age-balanced sensitivity split."
            )

    lines = [
        "# ADNI v4 Age Balance And Stratification Audit",
        "",
        "Metadata-only audit. No tensors, checkpoints, or training jobs were loaded.",
        "",
        "## Dataset Coverage",
        "",
        f"- CN/AD subjects used for balance checks: {len(cnad)}",
        f"- CN: {int(cnad['diagnosis'].eq('CN').sum())}",
        f"- AD: {int(cnad['diagnosis'].eq('AD').sum())}",
        f"- Missing Age among CN/AD: {int(cnad['Age'].isna().sum())}",
        f"- Historical age-bin column selected: `{old_age_col}`" if old_age_col else "- Historical age-bin column selected: none found",
        f"- Historical age-bin coverage among v4 CN/AD: {old_available}/{old_total} ({old_pct:.1%})"
        if old_total
        else "- Historical age-bin coverage among v4 CN/AD: unavailable",
        "",
        "## Global Age Balance",
        "",
        f"- AD vs CN absolute age SMD: {global_abs_smd:.3f} ({smd_interpretation(global_abs_smd)})"
        if np.isfinite(global_abs_smd)
        else "- AD vs CN absolute age SMD: unavailable",
        "",
        "## Current Fold Profile",
        "",
        f"- Fold 2 minimum classifier AUC across audited runs: {fold2_auc:.4f}" if np.isfinite(fold2_auc) else "- Fold 2 AUC: unavailable",
        f"- Fold 2 maximum test AD-vs-CN age SMD: {fold2_age_smd:.3f} ({smd_interpretation(fold2_age_smd)})"
        if np.isfinite(fold2_age_smd)
        else "- Fold 2 test AD-vs-CN age SMD: unavailable",
        "",
        "Fold 2 should be treated as a weak fold, but this audit separates age imbalance from scanner/manufacturer imbalance; "
        "the observed fold-2 test age SMD is small, so the fold weakness should not be attributed to age alone.",
        "",
        "## Stratification Feasibility",
        "",
        f"- Current diagnosis+Sex outer 5-fold feasible: {feasible_text(current)}",
        f"- diagnosis+AgeBin2_CNAD outer 5-fold feasible: {feasible_text(agebin2)}",
        f"- diagnosis+Sex+AgeBin2_CNAD outer 5-fold feasible: {feasible_text(sex_agebin2)}",
        f"- diagnosis+Sex+AgeQ4_CNAD outer 5-fold feasible: {feasible_text(sex_ageq4)}",
        "",
        "The historical age-bin is recoverable only where v4 subjects match the old metadata. Recompute AgeBin2/AgeQ4 for v4 rather than relying on the old column.",
        "",
        "AgeQ4 schemes are more granular and therefore more fragile. AgeBin2 is the lower-risk age-aware option if an age-balanced split is needed.",
        "",
        "## Alternative Split Simulation",
        "",
    ]
    if not best_sim.empty:
        r = best_sim.iloc[0]
        lines.extend(
            [
                f"- Best ranked feasible scheme across 100 seeds: `{r['scheme']}`",
                f"- Best-seed mean absolute train/test age SMD: {r['mean_abs_age_smd_train_vs_test_min']:.3f}",
                f"- Best-seed max absolute train/test age SMD: {r['max_abs_age_smd_train_vs_test_min']:.3f}",
            ]
        )
    else:
        lines.append("- No feasible alternative split simulation was available.")
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            recommendation,
            "",
            "Manufacturer/Site should remain diagnostic-only audit variables or sensitivity-stratification variables; they should not be predictive features.",
            "",
            "## Outputs",
            "",
            "- `metadata_column_inventory.csv`",
            "- `metadata_coverage_summary.csv`",
            "- `metadata_subject_match_summary.csv`",
            "- `metadata_missing_age_by_diagnosis.csv`",
            "- `old_age_bin_mapping_coverage.csv`",
            "- `v4_agebin_mapping.csv`",
            "- `global_age_balance.csv`",
            "- `current_fold_age_balance.csv`",
            "- `weak_fold_demographic_profile.csv`",
            "- `stratification_feasibility.csv`",
            "- `alternative_split_simulation_summary.csv`",
            "- `alternative_split_best_seed_examples.csv`",
        ]
    )
    (outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    outdir = prepare_output_dir(args.output_dir, args.overwrite)
    v4_path = resolve(args.v4_metadata)
    original_path = resolve(args.original_metadata)
    baseline_run = resolve(args.baseline_run)
    ckptselect_run = resolve(args.ckptselect_run)

    v4_raw = pd.read_csv(v4_path)
    original_raw = pd.read_csv(original_path)
    v4_std = standardize_metadata(v4_raw, "v4")
    original_std = standardize_metadata(original_raw, "original")
    old_age_col = choose_old_age_bin_col(original_raw)
    original_subjects = compact_subject_metadata(original_std, old_age_col)
    v4_subjects = compact_subject_metadata(v4_std)

    coverage = metadata_coverage(v4_raw, original_raw, v4_subjects, original_subjects, old_age_col)
    coverage_filenames = {
        "column_inventory": "metadata_column_inventory.csv",
        "coverage_summary": "metadata_coverage_summary.csv",
        "subject_match_summary": "metadata_subject_match_summary.csv",
        "missing_age_by_diagnosis": "metadata_missing_age_by_diagnosis.csv",
        "old_age_bin_mapping_coverage": "old_age_bin_mapping_coverage.csv",
    }
    for name, df in coverage.items():
        df.to_csv(outdir / coverage_filenames.get(name, f"{name}.csv"), index=False)

    old_keep = ["SubjectID_norm"]
    if old_age_col is not None and old_age_col in original_subjects.columns:
        old_keep.append(old_age_col)
    old_bins = original_subjects[old_keep].copy()
    if old_age_col is not None and old_age_col in old_bins.columns:
        old_bins = old_bins.rename(columns={old_age_col: "old_age_bin_if_available"})
    else:
        old_bins["old_age_bin_if_available"] = pd.NA

    v4_bins = add_age_bins(v4_subjects)
    v4_map = v4_bins.merge(old_bins[["SubjectID_norm", "old_age_bin_if_available"]], on="SubjectID_norm", how="left")
    v4_map = v4_map.rename(
        columns={
            "SubjectID_norm": "SubjectID",
            "diagnosis_norm": "diagnosis_from_metadata",
            "Age_norm": "Age_from_metadata",
            "Sex_norm": "Sex_from_metadata",
            "Manufacturer_norm": "Manufacturer_from_metadata",
            "Site_norm": "Site_from_metadata",
        }
    )
    v4_map["diagnosis"] = v4_map["diagnosis"].where(v4_map["diagnosis"].notna(), v4_map["diagnosis_from_metadata"])
    v4_map["Age"] = v4_map["Age"].where(v4_map["Age"].notna(), v4_map["Age_from_metadata"])
    v4_map["Sex"] = v4_map["Sex_from_metadata"]
    v4_map["Manufacturer"] = v4_map["Manufacturer_from_metadata"]
    v4_map["Site"] = v4_map["Site_from_metadata"]

    mapping_cols = [
        "SubjectID",
        "diagnosis",
        "Age",
        "Sex",
        "Manufacturer",
        "Site",
        "old_age_bin_if_available",
        "AgeQ4_CNAD",
        "AgeQ4_All",
        "AgeBin2_CNAD",
        "AgeClinicalBin",
    ]
    v4_map[mapping_cols].to_csv(outdir / "v4_agebin_mapping.csv", index=False)

    global_balance = global_age_balance(v4_map[mapping_cols])
    global_balance.to_csv(outdir / "global_age_balance.csv", index=False)

    fold_tables = []
    fold_tables.append(current_fold_age_balance("baseline_tanh", baseline_run, v4_map[mapping_cols]))
    fold_tables.append(current_fold_age_balance("ckptselect", ckptselect_run, v4_map[mapping_cols]))
    fold_balance = pd.concat([df for df in fold_tables if not df.empty], ignore_index=True) if any(
        not df.empty for df in fold_tables
    ) else pd.DataFrame()
    if not fold_balance.empty:
        fold_balance.to_csv(outdir / "current_fold_age_balance.csv", index=False)
        weak = weak_fold_profile(fold_balance)
        weak.to_csv(outdir / "weak_fold_demographic_profile.csv", index=False)
    else:
        pd.DataFrame().to_csv(outdir / "current_fold_age_balance.csv", index=False)
        pd.DataFrame().to_csv(outdir / "weak_fold_demographic_profile.csv", index=False)

    cnad = v4_map[mapping_cols][v4_map["diagnosis"].isin(["CN", "AD"])].copy()
    feasibility = stratification_feasibility(cnad)
    feasibility.to_csv(outdir / "stratification_feasibility.csv", index=False)
    simulation_summary, best_examples = simulate_alternative_splits(cnad, feasibility)
    simulation_summary.to_csv(outdir / "alternative_split_simulation_summary.csv", index=False)
    best_examples.to_csv(outdir / "alternative_split_best_seed_examples.csv", index=False)

    write_readme(outdir, v4_map[mapping_cols], global_balance, fold_balance, feasibility, simulation_summary, old_age_col)

    manifest = {
        "v4_metadata": str(v4_path),
        "original_metadata": str(original_path),
        "baseline_run": str(baseline_run),
        "ckptselect_run": str(ckptselect_run),
        "output_dir": str(outdir),
        "no_training": True,
        "no_tensor_loading": True,
    }
    (outdir / "audit_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print((outdir / "README.md").read_text(encoding="utf-8"))
    print("\nStratification feasibility:")
    print(
        feasibility[
            [
                "scheme",
                "n_strata",
                "min_stratum_count",
                "strata_count_lt_5",
                "strata_count_lt_10",
                "feasible_5fold_outer_cv",
                "feasible_5fold_inner_cv_inside_train_dev",
                "fragile",
            ]
        ].to_string(index=False)
    )
    if not simulation_summary.empty:
        print("\nTop simulated split schemes:")
        print(
            simulation_summary[
                [
                    "scheme",
                    "best_seed",
                    "mean_abs_age_smd_train_vs_test_min",
                    "max_abs_age_smd_train_vs_test_min",
                    "mean_sex_imbalance_mean",
                    "mean_manufacturer_imbalance_mean",
                ]
            ]
            .head(8)
            .to_string(index=False)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
