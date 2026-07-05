#!/usr/bin/env python3
"""Metadata-only OASIS-3 TR=3 compatibility audit.

This script reads only existing metadata/audit CSVs. It does not download
images, preprocess, train, or load tensors/checkpoints/joblibs/large arrays.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ELIGIBILITY = PROJECT_ROOT / "results/revision_bspc_2026/oasis3_external_validation_cohort_sizing/oasis3_external_validation_eligibility_subjects.csv"
DEFAULT_STRICT_SELECTED = PROJECT_ROOT / "results/revision_bspc_2026/oasis3_external_validation_cohort_sizing/oasis3_external_validation_selected_subjects_strict.csv"
DEFAULT_CLINICAL_AUDIT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/oasis3_clinical_mapping_audit"
DEFAULT_FEASIBILITY_DIR = PROJECT_ROOT / "results/revision_bspc_2026/oasis3_feasibility_audit"
DEFAULT_METADATA_ROOT = Path("/media/diego/Datos/vae_AD_data/OASIS3/metadata_raw")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/oasis3_tr3_compatibility_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Metadata-only OASIS-3 TR=3 compatibility audit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--eligibility-path", type=Path, default=DEFAULT_ELIGIBILITY)
    parser.add_argument("--strict-selected-path", type=Path, default=DEFAULT_STRICT_SELECTED)
    parser.add_argument("--clinical-audit-dir", type=Path, default=DEFAULT_CLINICAL_AUDIT_DIR)
    parser.add_argument("--feasibility-dir", type=Path, default=DEFAULT_FEASIBILITY_DIR)
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def git_hash() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            text=True,
            capture_output=True,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else ""
    except Exception:
        return ""


def safe_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def parse_float(value: Any) -> float | None:
    text = safe_text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    path = resolve(path)
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {path}. Pass --overwrite.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_file(path: Path, label: str) -> Path:
    path = resolve(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, low_memory=False)


def tr_seconds(value: Any) -> float | None:
    num = parse_float(value)
    if num is None:
        return None
    if 2950 <= num <= 3050:
        return num / 1000.0
    return num


def is_tr3(value: Any) -> bool:
    sec = tr_seconds(value)
    return sec is not None and 2.95 <= sec <= 3.05


def is_tr22(value: Any) -> bool:
    sec = tr_seconds(value)
    return sec is not None and 2.15 <= sec <= 2.25


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def normalize_bool(value: Any) -> bool:
    return safe_text(value).lower() in {"true", "1", "yes"}


def age_bin(value: Any) -> str:
    num = parse_float(value)
    if num is None:
        return "unknown"
    if num < 65:
        return "<65"
    if num < 70:
        return "65-70"
    if num < 75:
        return "70-75"
    if num < 80:
        return "75-80"
    if num < 85:
        return "80-85"
    return ">=85"


def join_counts(df: pd.DataFrame, cols: List[str]) -> str:
    if df.empty:
        return "{}"
    grouped = df.groupby(cols, dropna=False).size().reset_index(name="n")
    parts = []
    for _, row in grouped.iterrows():
        key = "/".join(safe_text(row[c]) or "missing" for c in cols)
        parts.append(f"{key}:{int(row['n'])}")
    return "; ".join(parts)


def distribution_by_label(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["TR_seconds"] = work["TR"].map(tr_seconds)
    work["TR_or_RepetitionTime"] = work["TR"].map(safe_text)
    grouped = (
        work.groupby(
            ["TR_or_RepetitionTime", "TR_seconds", "provisional_label", "label_confidence", "manufacturer", "scanner_model"],
            dropna=False,
        )
        .agg(n_subjects=("subject_id", "nunique"), n_sessions=("session_id", "nunique"))
        .reset_index()
        .sort_values(["TR_seconds", "provisional_label", "label_confidence", "manufacturer", "scanner_model"], na_position="last")
    )
    return grouped


def tr3_candidates(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["TR"].map(is_tr3)].copy()
    out["TR_seconds"] = out["TR"].map(tr_seconds)
    out["TR_unit_interpretation"] = out["TR"].map(lambda v: "milliseconds_normalized_to_seconds" if (parse_float(v) is not None and parse_float(v) >= 100) else "seconds")
    out["eligibility_strict"] = out["eligible_strict"].map(normalize_bool)
    out["selection_status"] = out.apply(selection_status, axis=1)
    cols = [
        "subject_id",
        "session_id",
        "experiment_id",
        "provisional_label",
        "label_confidence",
        "CDRTOT",
        "CDRSUM",
        "abs_delta_clinical_to_MR_days",
        "TR",
        "TR_seconds",
        "TR_unit_interpretation",
        "manufacturer",
        "scanner_model",
        "bold_scan_id",
        "age_at_MR",
        "sex",
        "eligibility_strict",
        "selection_status",
    ]
    return out[cols].sort_values(["provisional_label", "label_confidence", "subject_id", "session_id"])


def selection_status(row: pd.Series) -> str:
    reasons = []
    if row.get("provisional_label") not in {"CN", "AD_DEMENTIA"}:
        reasons.append("label_not_CN_or_AD_DEMENTIA")
    if row.get("label_confidence") != "high":
        reasons.append("label_confidence_not_high")
    if not safe_text(row.get("manufacturer", "")):
        reasons.append("missing_manufacturer")
    if not safe_text(row.get("scanner_model", "")):
        reasons.append("missing_scanner_model")
    if not normalize_bool(row.get("has_task_rest_bold", "")):
        reasons.append("missing_task_rest_bold")
    if not reasons:
        return "TR3_candidate_for_balanced_selection"
    return "not_selectable:" + "|".join(reasons)


def candidate_filter(df: pd.DataFrame, *, tr_predicate, strict: bool) -> pd.DataFrame:
    work = df[df["TR"].map(tr_predicate)].copy()
    if strict:
        work = work[
            work["provisional_label"].isin(["CN", "AD_DEMENTIA"])
            & work["label_confidence"].eq("high")
            & bool_series(work["has_task_rest_bold"])
            & work["TR"].map(lambda x: tr_seconds(x) is not None)
            & work["manufacturer"].map(lambda x: bool(safe_text(x)))
            & work["scanner_model"].map(lambda x: bool(safe_text(x)))
        ].copy()
        delta = pd.to_numeric(work["abs_delta_clinical_to_MR_days"], errors="coerce")
        work = work[delta.isna() | (delta <= 90)].copy()
    else:
        work = work[
            work["provisional_label"].isin(["CN", "AD_DEMENTIA"])
            & work["label_confidence"].isin(["high", "medium"])
            & bool_series(work["has_task_rest_bold"])
            & work["TR"].map(lambda x: tr_seconds(x) is not None)
            & work["manufacturer"].map(lambda x: bool(safe_text(x)))
            & work["scanner_model"].map(lambda x: bool(safe_text(x)))
        ].copy()
        delta = pd.to_numeric(work["abs_delta_clinical_to_MR_days"], errors="coerce")
        work = work[delta.isna() | (delta <= 365)].copy()
    return select_one_session_per_subject(work)


def select_one_session_per_subject(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    work = df.copy()
    work["_conf_rank"] = work["label_confidence"].map({"high": 2, "medium": 1, "low": 0}).fillna(0)
    work["_delta"] = pd.to_numeric(work["abs_delta_clinical_to_MR_days"], errors="coerce").fillna(10**9)
    work["_session_day"] = pd.to_numeric(work["MR_session_day"], errors="coerce").fillna(10**9)
    work["_age"] = pd.to_numeric(work["age_at_MR"], errors="coerce").fillna(10**9)
    work = work.sort_values(["subject_id", "_conf_rank", "_delta", "_session_day", "_age", "session_id"], ascending=[True, False, True, True, True, True])
    return work.drop_duplicates("subject_id", keep="first").copy()


def select_balanced(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    counts = df["provisional_label"].value_counts()
    n = int(min(counts.get("CN", 0), counts.get("AD_DEMENTIA", 0)))
    if n == 0:
        return df.iloc[0:0].copy()
    selected_parts = []
    for label in ["CN", "AD_DEMENTIA"]:
        sub = df[df["provisional_label"].eq(label)].copy()
        other = df[df["provisional_label"].ne(label)].copy()
        other_median_age = pd.to_numeric(other["age_at_MR"], errors="coerce").median()
        sub["_age"] = pd.to_numeric(sub["age_at_MR"], errors="coerce")
        sub["_age_distance_to_other_median"] = (sub["_age"] - other_median_age).abs().fillna(10**9)
        sub["_sex_sort"] = sub["sex"].map(safe_text)
        sub["_delta"] = pd.to_numeric(sub["abs_delta_clinical_to_MR_days"], errors="coerce").fillna(10**9)
        sub["_session_day"] = pd.to_numeric(sub["MR_session_day"], errors="coerce").fillna(10**9)
        sub = sub.sort_values(["_age_distance_to_other_median", "_sex_sort", "_delta", "_session_day", "subject_id", "session_id"])
        selected_parts.append(sub.head(n))
    selected = pd.concat(selected_parts, ignore_index=True)
    drop_cols = [c for c in selected.columns if c.startswith("_")]
    return selected.drop(columns=drop_cols, errors="ignore").sort_values(["provisional_label", "subject_id", "session_id"])


def selected_output(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "subject_id",
        "session_id",
        "experiment_id",
        "provisional_label",
        "label_confidence",
        "CDRTOT",
        "CDRSUM",
        "abs_delta_clinical_to_MR_days",
        "TR",
        "manufacturer",
        "scanner_model",
        "bold_scan_id",
        "age_at_MR",
        "sex",
    ]
    return df[[c for c in cols if c in df.columns]].copy()


def cohort_summary(name: str, df: pd.DataFrame) -> Dict[str, Any]:
    cn = int(df[df["provisional_label"].eq("CN")]["subject_id"].nunique()) if not df.empty else 0
    ad = int(df[df["provisional_label"].eq("AD_DEMENTIA")]["subject_id"].nunique()) if not df.empty else 0
    return {
        "cohort": name,
        "n_CN": cn,
        "n_AD_DEMENTIA": ad,
        "max_balanced_N_per_class": min(cn, ad),
        "manufacturer_scanner_distribution": join_counts(df, ["manufacturer", "scanner_model"]) if not df.empty else "",
        "TR_distribution": join_counts(df, ["TR"]) if not df.empty else "",
        "sex_distribution": join_counts(df, ["sex"]) if not df.empty else "",
        "age_bin_distribution": join_counts(df.assign(age_bin=df["age_at_MR"].map(age_bin)) if not df.empty else df, ["age_bin"]) if not df.empty else "",
        "recommendation": recommendation_for_cohort(name, cn, ad),
    }


def recommendation_for_cohort(name: str, cn: int, ad: int) -> str:
    if min(cn, ad) >= 30:
        return "feasible_for_external_validation_after_smoke_test_and_qc"
    if min(cn, ad) >= 5:
        return "feasible_only_as_small_stress_test"
    if "TR=3" in name:
        return "not_feasible_for_CN_AD_external_validation"
    return "usable_as_stress_test_not_direct_TR_matched_validation"


def build_comparison(df: pd.DataFrame, strict_selected: pd.DataFrame) -> pd.DataFrame:
    tr3_strict = candidate_filter(df, tr_predicate=is_tr3, strict=True)
    tr3_relaxed = candidate_filter(df, tr_predicate=is_tr3, strict=False)
    tr22_strict = candidate_filter(df, tr_predicate=is_tr22, strict=True)
    all_tr_strict = strict_selected.copy()
    rows = [
        cohort_summary("TR=3 strict", tr3_strict),
        cohort_summary("TR=3 relaxed", tr3_relaxed),
        cohort_summary("TR=2.2 strict", tr22_strict),
        cohort_summary("all-TR strict", all_tr_strict),
    ]
    return pd.DataFrame(rows)


def readme_text(distribution: pd.DataFrame, tr3_all: pd.DataFrame, tr3_strict: pd.DataFrame, tr3_relaxed: pd.DataFrame, comparison: pd.DataFrame, warnings: List[str]) -> str:
    tr3_label_counts = tr3_all.groupby("provisional_label")["subject_id"].nunique().to_dict() if not tr3_all.empty else {}
    tr3_cn = int(tr3_all[tr3_all["provisional_label"].eq("CN")]["subject_id"].nunique()) if not tr3_all.empty else 0
    tr3_ad = int(tr3_all[tr3_all["provisional_label"].eq("AD_DEMENTIA")]["subject_id"].nunique()) if not tr3_all.empty else 0
    tr3_strict_cn = int(tr3_strict[tr3_strict["provisional_label"].eq("CN")]["subject_id"].nunique()) if not tr3_strict.empty else 0
    tr3_strict_ad = int(tr3_strict[tr3_strict["provisional_label"].eq("AD_DEMENTIA")]["subject_id"].nunique()) if not tr3_strict.empty else 0
    tr22 = comparison[comparison["cohort"].eq("TR=2.2 strict")]
    tr22_line = ""
    if not tr22.empty:
        tr22_line = f"- TR=2.2 strict: CN={int(tr22.iloc[0]['n_CN'])}, AD_DEMENTIA={int(tr22.iloc[0]['n_AD_DEMENTIA'])}, max balanced per class={int(tr22.iloc[0]['max_balanced_N_per_class'])}"
    recommendation = (
        "TR=3 OASIS CN/AD external validation is not feasible from current metadata because TR=3 task-rest BOLD rows are low-confidence UNKNOWN only. "
        "Proceed with the already selected TR=2.2 OASIS smoke-test as a feasibility/stress-test, not as a direct TR-matched validation."
        if min(tr3_strict_cn, tr3_strict_ad) == 0
        else "Proceed with a TR=3 smoke-test after manual review and image QC."
    )
    warning_block = "\n".join(f"- {w}" for w in warnings) if warnings else "- None."
    lines = [
        "# OASIS-3 TR=3 Compatibility Audit",
        "",
        "Metadata-only audit. No images were downloaded, no preprocessing or training was run, and no tensors/checkpoints/joblibs/large arrays were loaded.",
        "",
        "## Summary",
        "",
        f"- TR≈3.0 task-rest subjects by label: {json.dumps(tr3_label_counts, sort_keys=True)}",
        f"- TR≈3.0 CN subjects: {tr3_cn}",
        f"- TR≈3.0 AD_DEMENTIA subjects: {tr3_ad}",
        f"- TR≈3.0 strict CN/AD_DEMENTIA subjects: CN={tr3_strict_cn}, AD_DEMENTIA={tr3_strict_ad}",
        f"- TR≈3.0 max balanced strict N per class: {min(tr3_strict_cn, tr3_strict_ad)}",
        tr22_line,
        "",
        "## Feasibility",
        "",
        recommendation,
        "",
        "## Methodological Implications",
        "",
        "- ADNI paper pipeline used TR=3.0 s. OASIS TR=2.2 changes the physical duration represented by `target_len=140` from 420 s to 308 s.",
        "- Any dFC window length specified in TRs represents a shorter time window at TR=2.2 than at TR=3.0.",
        "- Bandpass filtering at 0.01-0.08 Hz remains definable for TR=2.2, but sampling rate and filter behavior differ from ADNI TR=3.",
        "- Granger lag 1 corresponds to 2.2 s in OASIS TR=2.2, not 3.0 s as in ADNI.",
        "",
        "## Recommended Next Action",
        "",
        "- Do not select a TR=3 CN/AD OASIS cohort from current metadata.",
        "- Use TR=2.2 OASIS only as an external feasibility/stress-test unless a separate TR/domain harmonization design is defined.",
        "- Avoid mixing OASIS into ADNI training unless the experiment is explicitly domain-harmonized and no longer framed as external validation.",
        "",
        "## Warnings",
        "",
        warning_block,
    ]
    return "\n".join([line for line in lines if line is not None]) + "\n"


def main() -> int:
    args = parse_args()
    eligibility_path = require_file(args.eligibility_path, "eligibility subjects CSV")
    strict_selected_path = require_file(args.strict_selected_path, "strict selected CSV")
    output_dir = prepare_output_dir(args.output_dir, args.overwrite)
    warnings: List[str] = []
    if not resolve(args.clinical_audit_dir).exists():
        warnings.append(f"clinical_audit_dir_missing:{resolve(args.clinical_audit_dir)}")
    if not resolve(args.feasibility_dir).exists():
        warnings.append(f"feasibility_dir_missing:{resolve(args.feasibility_dir)}")
    if not resolve(args.metadata_root).exists():
        warnings.append(f"metadata_root_missing:{resolve(args.metadata_root)}")

    eligibility = load_csv(eligibility_path)
    strict_selected = load_csv(strict_selected_path)
    distribution = distribution_by_label(eligibility)
    tr3_all = tr3_candidates(eligibility)
    tr3_strict_pool = candidate_filter(eligibility, tr_predicate=is_tr3, strict=True)
    tr3_relaxed_pool = candidate_filter(eligibility, tr_predicate=is_tr3, strict=False)
    tr3_strict_balanced = select_balanced(tr3_strict_pool)
    tr3_relaxed_balanced = select_balanced(tr3_relaxed_pool)
    comparison = build_comparison(eligibility, strict_selected)

    distribution.to_csv(output_dir / "oasis3_tr_distribution_by_label.csv", index=False)
    tr3_all.to_csv(output_dir / "oasis3_tr3_candidate_subjects.csv", index=False)
    selected_output(tr3_strict_balanced).to_csv(output_dir / "oasis3_tr3_strict_selected_balanced.csv", index=False)
    selected_output(tr3_relaxed_balanced).to_csv(output_dir / "oasis3_tr3_relaxed_selected_balanced.csv", index=False)
    comparison.to_csv(output_dir / "oasis3_tr3_vs_tr22_feasibility.csv", index=False)
    (output_dir / "README.md").write_text(readme_text(distribution, tr3_all, tr3_strict_balanced, tr3_relaxed_balanced, comparison, warnings), encoding="utf-8")

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_hash": git_hash(),
        "project_root": str(PROJECT_ROOT),
        "inputs": {
            "eligibility_subjects": str(eligibility_path),
            "strict_selected": str(strict_selected_path),
            "clinical_audit_dir": str(resolve(args.clinical_audit_dir)),
            "feasibility_dir": str(resolve(args.feasibility_dir)),
            "metadata_root": str(resolve(args.metadata_root)),
        },
        "outputs": {
            "tr_distribution": str(output_dir / "oasis3_tr_distribution_by_label.csv"),
            "tr3_candidates": str(output_dir / "oasis3_tr3_candidate_subjects.csv"),
            "tr3_strict_balanced": str(output_dir / "oasis3_tr3_strict_selected_balanced.csv"),
            "tr3_relaxed_balanced": str(output_dir / "oasis3_tr3_relaxed_selected_balanced.csv"),
            "comparison": str(output_dir / "oasis3_tr3_vs_tr22_feasibility.csv"),
            "readme": str(output_dir / "README.md"),
        },
        "tr3_rule": "TR seconds in [2.95, 3.05] or millisecond values in [2950, 3050].",
        "no_image_download": True,
        "no_preprocessing": True,
        "no_training": True,
        "no_large_array_loading": True,
        "warnings": warnings,
    }
    (output_dir / "audit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote outputs to {output_dir}")
    print(f"TR=3 candidate rows: {len(tr3_all)}")
    print(f"TR=3 strict balanced rows: {len(tr3_strict_balanced)}")
    print(f"TR=3 relaxed balanced rows: {len(tr3_relaxed_balanced)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
