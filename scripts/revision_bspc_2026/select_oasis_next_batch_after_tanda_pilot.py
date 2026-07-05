#!/usr/bin/env python3
"""Read-only OASIS next-batch selection audit after the Tanda 2026-05-25 pilot.

This script selects non-overlapping OASIS3 CN/AD_DEMENTIA candidate sessions for a
future calibration/test external-validation batch. It does not download,
preprocess, train, or modify source data.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results" / "revision_bspc_2026"
OUTPUT_DIR = RESULTS_ROOT / "oasis_next_batch_selection_audit"

MAIN_CANDIDATES = (
    RESULTS_ROOT
    / "oasis3_external_validation_tr2_candidate_manifest"
    / "oasis3_tr2_main_candidates_for_martin.csv"
)
INVENTORY = (
    RESULTS_ROOT
    / "oasis3_external_validation_tr2_candidate_manifest"
    / "oasis3_tr2_subject_session_inventory.csv"
)
PILOT_TANDA = RESULTS_ROOT / "oasis_tanda_2026_05_25_audit" / "subject_session_manifest.csv"
PILOT_DOWNLOAD = (
    RESULTS_ROOT
    / "oasis3_pilot_30cn_30ad_download_manifest"
    / "oasis3_pilot_30cn_30ad_subjects.csv"
)

TARGET_DIAGNOSES = ("CN", "AD_DEMENTIA")
TARGET_SCANNER = "TrioTim"
MINIMUM_N_PER_CLASS = 30
IDEAL_N_PER_CLASS = 60


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean_string(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "."}:
        return ""
    return text


def boolish(value: Any) -> bool:
    text = clean_string(value).lower()
    return text in {"true", "1", "yes", "y"}


def safe_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    view = df if max_rows is None else df.head(max_rows)
    if view.empty:
        return "_No rows._\n"
    header = "| " + " | ".join(str(c) for c in view.columns) + " |"
    sep = "| " + " | ".join("---" for _ in view.columns) + " |"
    rows = []
    for _, row in view.iterrows():
        rows.append("| " + " | ".join(clean_string(v) for v in row.tolist()) + " |")
    extra = ""
    if max_rows is not None and len(df) > max_rows:
        extra = f"\n\n_Showing first {max_rows} of {len(df)} rows._\n"
    return "\n".join([header, sep] + rows) + extra


def write_table(df: pd.DataFrame, csv_path: Path, md_path: Path, md_max_rows: int | None = None) -> None:
    df.to_csv(csv_path, index=False)
    md_path.write_text(safe_markdown(df, max_rows=md_max_rows), encoding="utf-8")


def load_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def normalize_session_sets(df: pd.DataFrame) -> tuple[set[str], set[str], set[str]]:
    subjects: set[str] = set()
    sessions: set[str] = set()
    experiments: set[str] = set()
    if "subject_id" in df.columns:
        subjects = {clean_string(v) for v in df["subject_id"] if clean_string(v)}
    for col in ("session_id", "bids_session"):
        if col in df.columns:
            sessions.update(clean_string(v) for v in df[col] if clean_string(v))
    if "experiment_id" in df.columns:
        experiments = {clean_string(v) for v in df["experiment_id"] if clean_string(v)}
        sessions.update(experiments)
    return subjects, sessions, experiments


def add_expected_run_counts(candidates: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    inv = inventory.copy()
    inv["_is_target_rest"] = (
        inv.get("is_resting_state", False).map(boolish)
        & inv.get("TR_class", "").astype(str).eq("target_2p15_2p25")
        & inv.get("diagnosis", "").isin(TARGET_DIAGNOSES)
        & inv.get("diagnosis_confidence", "").astype(str).eq("high")
    )
    run_counts = (
        inv[inv["_is_target_rest"]]
        .groupby(["subject_id", "experiment_id"], dropna=False)
        .agg(
            expected_tr2_rest_runs=("bold_identifier", "nunique"),
            expected_tr2_rest_run_labels=("run", lambda s: ";".join(sorted({clean_string(x) for x in s if clean_string(x)}))),
            target_tr2_series_descriptions=("SeriesDescription", lambda s: ";".join(sorted({clean_string(x) for x in s if clean_string(x)}))),
        )
        .reset_index()
    )
    out = candidates.merge(run_counts, on=["subject_id", "experiment_id"], how="left")
    out["expected_tr2_rest_runs"] = out["expected_tr2_rest_runs"].fillna(1).astype(int)
    out["has_two_or_more_expected_runs"] = out["expected_tr2_rest_runs"] >= 2
    return out


def build_candidate_pool() -> tuple[pd.DataFrame, pd.DataFrame]:
    main = load_required_csv(MAIN_CANDIDATES)
    inventory = load_required_csv(INVENTORY)
    pilot_tanda = load_required_csv(PILOT_TANDA)
    pilot_download = load_required_csv(PILOT_DOWNLOAD)

    pilot_subjects_1, pilot_sessions_1, pilot_experiments_1 = normalize_session_sets(pilot_tanda)
    pilot_subjects_2, pilot_sessions_2, pilot_experiments_2 = normalize_session_sets(pilot_download)
    pilot_subjects = pilot_subjects_1 | pilot_subjects_2
    pilot_sessions = pilot_sessions_1 | pilot_sessions_2
    pilot_experiments = pilot_experiments_1 | pilot_experiments_2

    pool = main.copy()
    pool = add_expected_run_counts(pool, inventory)
    pool["source_manifest"] = str(MAIN_CANDIDATES.relative_to(PROJECT_ROOT))
    pool["pilot_subject_overlap"] = pool["subject_id"].astype(str).isin(pilot_subjects)
    pool["pilot_session_overlap"] = pool["session_id"].astype(str).isin(pilot_sessions)
    pool["pilot_experiment_overlap"] = pool["experiment_id"].astype(str).isin(pilot_experiments | pilot_sessions)
    pool["excluded_due_to_pilot_overlap"] = (
        pool["pilot_subject_overlap"] | pool["pilot_session_overlap"] | pool["pilot_experiment_overlap"]
    )

    criteria_mask = (
        pool["diagnosis"].isin(TARGET_DIAGNOSES)
        & pool["diagnosis_confidence"].astype(str).eq("high")
        & pool["TR_seconds"].between(2.15, 2.25, inclusive="both")
        & pool["Manufacturer"].astype(str).str.lower().eq("siemens")
        & pool["subject_id"].notna()
        & pool["session_id"].notna()
        & pool["experiment_id"].notna()
    )
    after = pool[criteria_mask & ~pool["excluded_due_to_pilot_overlap"]].copy()
    after["_scanner_priority"] = (after["ScannerModel"].astype(str) != TARGET_SCANNER).astype(int)
    after["_run_priority"] = -after["expected_tr2_rest_runs"].astype(float)
    after["_session_rank_priority"] = pd.to_numeric(after["session_rank_per_subject"], errors="coerce").fillna(999)
    after = after.sort_values(
        [
            "diagnosis",
            "_scanner_priority",
            "_run_priority",
            "_session_rank_priority",
            "subject_id",
        ]
    ).reset_index(drop=True)

    overlap_audit_rows = [
        {
            "source": "tanda_processed_subject_session_manifest",
            "subjects": len(pilot_subjects_1),
            "sessions_or_session_aliases": len(pilot_sessions_1),
            "experiment_ids": len(pilot_experiments_1),
        },
        {
            "source": "pilot_download_manifest",
            "subjects": len(pilot_subjects_2),
            "sessions_or_session_aliases": len(pilot_sessions_2),
            "experiment_ids": len(pilot_experiments_2),
        },
        {
            "source": "pilot_union",
            "subjects": len(pilot_subjects),
            "sessions_or_session_aliases": len(pilot_sessions),
            "experiment_ids": len(pilot_experiments),
        },
        {
            "source": "main_candidates_before_exclusion",
            "subjects": int(pool["subject_id"].nunique()),
            "sessions_or_session_aliases": int(pool["session_id"].nunique()),
            "experiment_ids": int(pool["experiment_id"].nunique()),
        },
        {
            "source": "main_candidates_excluded_by_pilot_overlap",
            "subjects": int(pool.loc[pool["excluded_due_to_pilot_overlap"], "subject_id"].nunique()),
            "sessions_or_session_aliases": int(pool.loc[pool["excluded_due_to_pilot_overlap"], "session_id"].nunique()),
            "experiment_ids": int(pool.loc[pool["excluded_due_to_pilot_overlap"], "experiment_id"].nunique()),
        },
        {
            "source": "candidate_pool_after_excluding_pilot",
            "subjects": int(after["subject_id"].nunique()),
            "sessions_or_session_aliases": int(after["session_id"].nunique()),
            "experiment_ids": int(after["experiment_id"].nunique()),
        },
    ]
    overlap = pd.DataFrame(overlap_audit_rows)
    return after, overlap


def greedy_age_sex_pairs(pool: pd.DataFrame, n_pairs: int) -> pd.DataFrame:
    ad = pool[pool["diagnosis"] == "AD_DEMENTIA"].copy()
    cn = pool[pool["diagnosis"] == "CN"].copy()
    if len(ad) < n_pairs or len(cn) < n_pairs:
        raise ValueError(f"Not enough candidates for {n_pairs} pairs: CN={len(cn)} AD={len(ad)}")

    pair_rows = []
    for _, ad_row in ad.iterrows():
        for _, cn_row in cn.iterrows():
            ad_age = float(ad_row["age_at_MR"])
            cn_age = float(cn_row["age_at_MR"])
            sex_match = clean_string(ad_row["sex"]) == clean_string(cn_row["sex"])
            age_diff = abs(ad_age - cn_age)
            both_trio = int(ad_row["ScannerModel"] == TARGET_SCANNER) + int(cn_row["ScannerModel"] == TARGET_SCANNER)
            expected_runs_total = int(ad_row["expected_tr2_rest_runs"]) + int(cn_row["expected_tr2_rest_runs"])
            both_multi = int(ad_row["has_two_or_more_expected_runs"]) + int(cn_row["has_two_or_more_expected_runs"])
            session_rank_sum = (
                pd.to_numeric(pd.Series([ad_row["session_rank_per_subject"], cn_row["session_rank_per_subject"]]), errors="coerce")
                .fillna(999)
                .sum()
            )
            score = (
                (0 if sex_match else 1000)
                + age_diff
                - 0.20 * expected_runs_total
                - 0.50 * both_multi
                - 0.10 * both_trio
                + 0.01 * float(session_rank_sum)
            )
            pair_rows.append(
                {
                    "ad_subject_id": ad_row["subject_id"],
                    "cn_subject_id": cn_row["subject_id"],
                    "sex_match": sex_match,
                    "age_abs_diff": age_diff,
                    "expected_runs_total": expected_runs_total,
                    "both_multi_run_proxy": both_multi == 2,
                    "both_triotim": both_trio == 2,
                    "match_score": score,
                }
            )
    pairs = pd.DataFrame(pair_rows).sort_values(
        ["match_score", "sex_match", "age_abs_diff", "expected_runs_total", "ad_subject_id", "cn_subject_id"],
        ascending=[True, False, True, False, True, True],
    )

    used_ad: set[str] = set()
    used_cn: set[str] = set()
    selected_pairs = []
    for _, row in pairs.iterrows():
        ad_subject = str(row["ad_subject_id"])
        cn_subject = str(row["cn_subject_id"])
        if ad_subject in used_ad or cn_subject in used_cn:
            continue
        used_ad.add(ad_subject)
        used_cn.add(cn_subject)
        selected_pairs.append(row.to_dict())
        if len(selected_pairs) == n_pairs:
            break

    if len(selected_pairs) != n_pairs:
        raise RuntimeError(f"Greedy matching selected {len(selected_pairs)} pairs, expected {n_pairs}")
    selected = pd.DataFrame(selected_pairs)
    selected.insert(0, "match_pair_id", np.arange(1, len(selected) + 1))
    return selected


def rows_for_plan(pool: pd.DataFrame, pairs: pd.DataFrame, plan_name: str) -> pd.DataFrame:
    records = []
    by_subject = pool.set_index("subject_id", drop=False)
    for _, pair in pairs.iterrows():
        for diagnosis, subject_col, counterpart_col in [
            ("AD_DEMENTIA", "ad_subject_id", "cn_subject_id"),
            ("CN", "cn_subject_id", "ad_subject_id"),
        ]:
            row = by_subject.loc[pair[subject_col]].to_dict()
            row["selection_plan"] = plan_name
            row["match_pair_id"] = int(pair["match_pair_id"])
            row["matched_counterpart_subject_id"] = pair[counterpart_col]
            row["matched_counterpart_diagnosis"] = "CN" if diagnosis == "AD_DEMENTIA" else "AD_DEMENTIA"
            row["age_abs_diff_to_match"] = round(float(pair["age_abs_diff"]), 3)
            row["sex_match_to_pair"] = bool(pair["sex_match"])
            row["reason_selected"] = (
                "non_overlapping_pilot;high_confidence_CN_or_AD_DEMENTIA;rest_TR2p2;"
                f"scanner={row.get('Manufacturer','')}_{row.get('ScannerModel','')};"
                f"expected_tr2_rest_runs={row.get('expected_tr2_rest_runs','')};"
                "greedy_age_sex_matched_pair"
            )
            records.append(row)
    selected = pd.DataFrame(records)
    sort_cols = ["match_pair_id", "diagnosis", "subject_id"]
    selected = selected.sort_values(sort_cols).reset_index(drop=True)
    front_cols = [
        "selection_plan",
        "match_pair_id",
        "subject_id",
        "session_id",
        "experiment_id",
        "diagnosis",
        "age_at_MR",
        "sex",
        "Manufacturer",
        "ScannerModel",
        "TR_seconds",
        "expected_tr2_rest_runs",
        "has_two_or_more_expected_runs",
        "matched_counterpart_subject_id",
        "matched_counterpart_diagnosis",
        "age_abs_diff_to_match",
        "sex_match_to_pair",
        "reason_selected",
    ]
    remaining = [c for c in selected.columns if c not in front_cols and not c.startswith("_")]
    return selected[front_cols + remaining]


def summarize_balance(name: str, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dx in TARGET_DIAGNOSES:
        sub = df[df["diagnosis"] == dx]
        ages = pd.to_numeric(sub["age_at_MR"], errors="coerce")
        rows.append(
            {
                "cohort": name,
                "diagnosis": dx,
                "n": len(sub),
                "age_mean": round(float(ages.mean()), 3),
                "age_sd": round(float(ages.std(ddof=1)), 3) if len(sub) > 1 else 0.0,
                "age_median": round(float(ages.median()), 3),
                "age_iqr": round(float(ages.quantile(0.75) - ages.quantile(0.25)), 3),
                "sex_1_count": int((sub["sex"].astype(str) == "1").sum()),
                "sex_2_count": int((sub["sex"].astype(str) == "2").sum()),
                "trioTim_count": int((sub["ScannerModel"].astype(str) == TARGET_SCANNER).sum()),
                "biograph_mMR_count": int((sub["ScannerModel"].astype(str) == "Biograph_mMR").sum()),
                "expected_runs_mean": round(float(pd.to_numeric(sub["expected_tr2_rest_runs"], errors="coerce").mean()), 3),
                "two_or_more_expected_runs": int(sub["has_two_or_more_expected_runs"].sum()),
            }
        )
    summary = pd.DataFrame(rows)
    cn = summary[summary["diagnosis"] == "CN"].iloc[0]
    ad = summary[summary["diagnosis"] == "AD_DEMENTIA"].iloc[0]
    delta = pd.DataFrame(
        [
            {
                "cohort": name,
                "diagnosis": "CN_minus_AD_DEMENTIA_delta",
                "n": int(cn["n"] - ad["n"]),
                "age_mean": round(float(cn["age_mean"] - ad["age_mean"]), 3),
                "age_sd": "",
                "age_median": round(float(cn["age_median"] - ad["age_median"]), 3),
                "age_iqr": "",
                "sex_1_count": int(cn["sex_1_count"] - ad["sex_1_count"]),
                "sex_2_count": int(cn["sex_2_count"] - ad["sex_2_count"]),
                "trioTim_count": int(cn["trioTim_count"] - ad["trioTim_count"]),
                "biograph_mMR_count": int(cn["biograph_mMR_count"] - ad["biograph_mMR_count"]),
                "expected_runs_mean": round(float(cn["expected_runs_mean"] - ad["expected_runs_mean"]), 3),
                "two_or_more_expected_runs": int(cn["two_or_more_expected_runs"] - ad["two_or_more_expected_runs"]),
            }
        ]
    )
    return pd.concat([summary, delta], ignore_index=True)


def add_selection_overlap_rows(overlap: pd.DataFrame, selected_min: pd.DataFrame, selected_ideal: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, df in [("selected_minimum_30CN_30AD", selected_min), ("selected_ideal_60CN_60AD", selected_ideal)]:
        rows.append(
            {
                "source": name,
                "subjects": int(df["subject_id"].nunique()),
                "sessions_or_session_aliases": int(df["session_id"].nunique()),
                "experiment_ids": int(df["experiment_id"].nunique()),
                "pilot_subject_overlaps": int(df.get("pilot_subject_overlap", pd.Series(False, index=df.index)).sum()),
                "pilot_session_overlaps": int(df.get("pilot_session_overlap", pd.Series(False, index=df.index)).sum()),
                "pilot_experiment_overlaps": int(df.get("pilot_experiment_overlap", pd.Series(False, index=df.index)).sum()),
                "duplicate_subjects": int(len(df) - df["subject_id"].nunique()),
                "duplicate_experiment_ids": int(len(df) - df["experiment_id"].nunique()),
            }
        )
    return pd.concat([overlap, pd.DataFrame(rows)], ignore_index=True)


def download_manifest(selected_ideal: pd.DataFrame, selected_min: pd.DataFrame) -> pd.DataFrame:
    min_subjects = set(selected_min["subject_id"].astype(str))
    cols = [
        "subject_id",
        "session_id",
        "experiment_id",
        "diagnosis",
        "age_at_MR",
        "sex",
        "Manufacturer",
        "ScannerModel",
        "TR_seconds",
        "expected_tr2_rest_runs",
        "has_two_or_more_expected_runs",
        "session_rank_per_subject",
        "diagnosis_confidence",
        "clinical_delta_days",
        "CDRTOT",
        "CDRSUM",
        "reason_selected",
    ]
    manifest = selected_ideal[cols].copy()
    manifest.insert(0, "included_in_ideal_60CN_60AD", True)
    manifest.insert(0, "included_in_minimum_30CN_30AD", manifest["subject_id"].astype(str).isin(min_subjects))
    manifest["expected_runs"] = manifest["expected_tr2_rest_runs"]
    manifest["scanner_manufacturer_model"] = manifest["Manufacturer"].astype(str) + " " + manifest["ScannerModel"].astype(str)
    ordered = [
        "included_in_minimum_30CN_30AD",
        "included_in_ideal_60CN_60AD",
        "subject_id",
        "session_id",
        "experiment_id",
        "diagnosis",
        "age_at_MR",
        "sex",
        "scanner_manufacturer_model",
        "Manufacturer",
        "ScannerModel",
        "TR_seconds",
        "expected_runs",
        "expected_tr2_rest_runs",
        "has_two_or_more_expected_runs",
        "reason_selected",
    ]
    remaining = [c for c in manifest.columns if c not in ordered]
    return manifest[ordered + remaining]


def main() -> None:
    t0 = now_utc()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pool, overlap = build_candidate_pool()
    pairs_ideal = greedy_age_sex_pairs(pool, IDEAL_N_PER_CLASS)
    pairs_min = pairs_ideal.head(MINIMUM_N_PER_CLASS).copy()

    selected_ideal = rows_for_plan(pool, pairs_ideal, "ideal_60CN_60AD")
    selected_min = rows_for_plan(pool, pairs_min, "minimum_30CN_30AD")
    overlap = add_selection_overlap_rows(overlap, selected_min, selected_ideal)

    balance = pd.concat(
        [
            summarize_balance("candidate_pool_after_excluding_pilot", pool),
            summarize_balance("selected_minimum_30CN_30AD", selected_min),
            summarize_balance("selected_ideal_60CN_60AD", selected_ideal),
        ],
        ignore_index=True,
    )
    martin_manifest = download_manifest(selected_ideal, selected_min)

    candidate_columns = [
        "subject_id",
        "session_id",
        "experiment_id",
        "diagnosis",
        "diagnosis_confidence",
        "age_at_MR",
        "sex",
        "Manufacturer",
        "ScannerModel",
        "TR_seconds",
        "expected_tr2_rest_runs",
        "has_two_or_more_expected_runs",
        "session_rank_per_subject",
        "clinical_delta_days",
        "CDRTOT",
        "CDRSUM",
        "selection_reason",
        "expected_tr2_rest_run_labels",
        "target_tr2_series_descriptions",
    ]
    candidate_pool = pool[[c for c in candidate_columns if c in pool.columns]].copy()

    write_table(
        candidate_pool,
        OUTPUT_DIR / "candidate_pool_after_excluding_pilot.csv",
        OUTPUT_DIR / "candidate_pool_after_excluding_pilot.md",
        md_max_rows=120,
    )
    write_table(selected_min, OUTPUT_DIR / "selected_minimum_30CN_30AD.csv", OUTPUT_DIR / "selected_minimum_30CN_30AD.md")
    write_table(selected_ideal, OUTPUT_DIR / "selected_ideal_60CN_60AD.csv", OUTPUT_DIR / "selected_ideal_60CN_60AD.md")
    write_table(balance, OUTPUT_DIR / "age_sex_balance_summary.csv", OUTPUT_DIR / "age_sex_balance_summary.md")
    write_table(overlap, OUTPUT_DIR / "overlap_audit.csv", OUTPUT_DIR / "overlap_audit.md")
    write_table(martin_manifest, OUTPUT_DIR / "download_manifest_for_martin.csv", OUTPUT_DIR / "download_manifest_for_martin.md")

    final_recommendation = f"""# OASIS Next-Batch Selection Audit

Generated: `{t0}`

## Decision

**Recommended next request to Martin:** use the `selected_ideal_60CN_60AD` plan if feasible.
The `selected_minimum_30CN_30AD` plan is nested within the ideal selection and can be
used as a smaller calibration/test batch if storage or processing time is constrained.

## Source and exclusions

- Source candidate table: `{MAIN_CANDIDATES.relative_to(PROJECT_ROOT)}`
- Candidate criteria: high-confidence `CN` or `AD_DEMENTIA`, resting-state fMRI,
  TR approximately 2.2 s, Siemens scanner, one first usable TR2 visit per subject.
- Pilot exclusion sources:
  - `{PILOT_TANDA.relative_to(PROJECT_ROOT)}`
  - `{PILOT_DOWNLOAD.relative_to(PROJECT_ROOT)}`
- Non-overlap result: selected minimum and ideal plans have zero pilot subject,
  session, or experiment overlaps.

## Candidate pool after excluding pilot

- Total candidates: `{len(pool)}`
- CN: `{int((pool['diagnosis'] == 'CN').sum())}`
- AD_DEMENTIA: `{int((pool['diagnosis'] == 'AD_DEMENTIA').sum())}`
- TrioTim candidates: `{int((pool['ScannerModel'] == TARGET_SCANNER).sum())}`
- Candidates with two or more expected TR2 rest runs: `{int(pool['has_two_or_more_expected_runs'].sum())}`

## Matching strategy

The selection uses deterministic greedy AD-CN pairing. Pair priority is sex match,
then age proximity, with small preference for two-or-more expected TR2 rest runs,
TrioTim scanner, and earlier session rank. Diagnosis reliability was not traded off:
all selected rows remain high-confidence CN or AD_DEMENTIA.

The expected-run count is a metadata availability proxy, not post-download QC. Final
QC-usable run counts must be verified after Martin provides the next batch.

## Recommendation for downstream use

Use this next OASIS batch only for external calibration/test validation. Do not merge
it with ADNI training and do not use OASIS labels for ADNI model selection. If a
calibration design is needed, split a larger OASIS request into a fixed calibration
set and an untouched locked test set before any threshold or calibration adjustment.
"""
    (OUTPUT_DIR / "final_recommendation.md").write_text(final_recommendation, encoding="utf-8")

    command_log = {
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "timestamp_utc": t0,
        "output_dir": str(OUTPUT_DIR.relative_to(PROJECT_ROOT)),
        "inputs": {
            "main_candidates": str(MAIN_CANDIDATES.relative_to(PROJECT_ROOT)),
            "inventory": str(INVENTORY.relative_to(PROJECT_ROOT)),
            "pilot_tanda_manifest": str(PILOT_TANDA.relative_to(PROJECT_ROOT)),
            "pilot_download_manifest": str(PILOT_DOWNLOAD.relative_to(PROJECT_ROOT)),
        },
        "actions": [
            "loaded_existing_metadata_manifests",
            "excluded_pilot_subject_session_experiment_ids",
            "selected_nested_minimum_and_ideal_age_sex_matched_batches",
            "wrote_read_only_audit_outputs",
        ],
        "counts": {
            "candidate_pool_after_excluding_pilot": len(pool),
            "selected_minimum_rows": len(selected_min),
            "selected_ideal_rows": len(selected_ideal),
        },
        "safety": {
            "downloaded_data": False,
            "trained_models": False,
            "modified_input_data": False,
            "modified_existing_oasis_pilot_outputs": False,
        },
    }
    (OUTPUT_DIR / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")

    print(f"Wrote OASIS next-batch selection audit to {OUTPUT_DIR}")
    print(f"Candidate pool after pilot exclusion: {len(pool)}")
    print(f"Minimum selection: {len(selected_min)} rows")
    print(f"Ideal selection: {len(selected_ideal)} rows")


if __name__ == "__main__":
    main()
