#!/usr/bin/env python3
"""Prepare a read-only OASIS external calibration/test protocol package.

The package uses the already selected non-overlapping OASIS next-batch manifests
to define a locked calibration/test design before any model scores are seen.
It does not score models, train models, download data, or modify source outputs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results" / "revision_bspc_2026"
SELECTION_DIR = RESULTS_ROOT / "oasis_next_batch_selection_audit"
OUTPUT_DIR = RESULTS_ROOT / "oasis_external_calibration_test_protocol"

IDEAL_SELECTION = SELECTION_DIR / "selected_ideal_60CN_60AD.csv"
MINIMUM_SELECTION = SELECTION_DIR / "selected_minimum_30CN_30AD.csv"
PILOT_DIR = RESULTS_ROOT / "oasis_tanda_2026_05_25_external_scoring"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null", "."} else text


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._\n"
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(clean(v) for v in row.tolist()) + " |" for _, row in df.iterrows()]
    return "\n".join([header, sep] + rows) + "\n"


def write_csv_md(df: pd.DataFrame, stem: str) -> None:
    df.to_csv(OUTPUT_DIR / f"{stem}.csv", index=False)
    (OUTPUT_DIR / f"{stem}.md").write_text(md_table(df), encoding="utf-8")


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def build_pair_table(ideal: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pair_id, grp in ideal.groupby("match_pair_id", sort=True):
        cn = grp[grp["diagnosis"] == "CN"]
        ad = grp[grp["diagnosis"] == "AD_DEMENTIA"]
        if len(cn) != 1 or len(ad) != 1:
            raise ValueError(f"Expected one CN and one AD_DEMENTIA in match pair {pair_id}")
        cn_row = cn.iloc[0]
        ad_row = ad.iloc[0]
        sex_match = clean(cn_row["sex"]) == clean(ad_row["sex"])
        pair_age_mean = float(pd.to_numeric(grp["age_at_MR"], errors="coerce").mean())
        rows.append(
            {
                "match_pair_id": int(pair_id),
                "pair_sex": clean(cn_row["sex"]) if sex_match else "mixed",
                "pair_age_mean": round(pair_age_mean, 3),
                "age_abs_diff_within_pair": round(abs(float(cn_row["age_at_MR"]) - float(ad_row["age_at_MR"])), 3),
                "cn_subject_id": clean(cn_row["subject_id"]),
                "ad_subject_id": clean(ad_row["subject_id"]),
                "cn_age": float(cn_row["age_at_MR"]),
                "ad_age": float(ad_row["age_at_MR"]),
                "cn_expected_runs": int(cn_row["expected_tr2_rest_runs"]),
                "ad_expected_runs": int(ad_row["expected_tr2_rest_runs"]),
                "min_expected_runs": min(int(cn_row["expected_tr2_rest_runs"]), int(ad_row["expected_tr2_rest_runs"])),
            }
        )
    return pd.DataFrame(rows).sort_values(["pair_sex", "pair_age_mean", "match_pair_id"]).reset_index(drop=True)


def split_ideal_pairs(pair_table: pd.DataFrame) -> pd.DataFrame:
    """Assign 60 matched pairs into two 30-pair sets before scoring.

    We stratify by pair sex and sorted pair age. Alternating within each sex
    stratum keeps calibration/test age and sex distributions close without
    looking at model scores or labels beyond the planned diagnosis balance.
    """

    assigned = []
    for sex, sub in pair_table.groupby("pair_sex", sort=True):
        sub = sub.sort_values(["pair_age_mean", "match_pair_id"]).reset_index(drop=True)
        for i, (_, row) in enumerate(sub.iterrows()):
            subset = "calibration" if i % 2 == 0 else "locked_test"
            item = row.to_dict()
            item["ideal_60CN_60AD_subset"] = subset
            item["split_rule"] = "alternate_pairs_within_pair_sex_sorted_by_pair_age"
            assigned.append(item)
    out = pd.DataFrame(assigned)

    # If sex strata are odd, alternating can produce 31/29. Move the least
    # disruptive boundary pair by age until each subset has exactly 30 pairs.
    while (out["ideal_60CN_60AD_subset"] == "calibration").sum() > 30:
        candidates = out[out["ideal_60CN_60AD_subset"] == "calibration"].copy()
        candidates["_distance_to_median_age"] = (
            candidates["pair_age_mean"] - candidates["pair_age_mean"].median()
        ).abs()
        idx = candidates.sort_values(["_distance_to_median_age", "match_pair_id"], ascending=[False, False]).index[0]
        out.loc[idx, "ideal_60CN_60AD_subset"] = "locked_test"
        out.loc[idx, "split_rule"] += ";size_balance_boundary_adjustment"
    while (out["ideal_60CN_60AD_subset"] == "calibration").sum() < 30:
        candidates = out[out["ideal_60CN_60AD_subset"] == "locked_test"].copy()
        candidates["_distance_to_median_age"] = (
            candidates["pair_age_mean"] - candidates["pair_age_mean"].median()
        ).abs()
        idx = candidates.sort_values(["_distance_to_median_age", "match_pair_id"], ascending=[False, False]).index[0]
        out.loc[idx, "ideal_60CN_60AD_subset"] = "calibration"
        out.loc[idx, "split_rule"] += ";size_balance_boundary_adjustment"

    if (out["ideal_60CN_60AD_subset"] == "calibration").sum() != 30:
        raise RuntimeError("Calibration split is not 30 pairs")
    if (out["ideal_60CN_60AD_subset"] == "locked_test").sum() != 30:
        raise RuntimeError("Locked-test split is not 30 pairs")
    return out.sort_values(["ideal_60CN_60AD_subset", "pair_sex", "pair_age_mean", "match_pair_id"])


def rows_for_protocol(ideal: pd.DataFrame, minimum: pd.DataFrame, pair_split: pd.DataFrame) -> pd.DataFrame:
    split_map = pair_split.set_index("match_pair_id")["ideal_60CN_60AD_subset"].to_dict()
    split_rule = pair_split.set_index("match_pair_id")["split_rule"].to_dict()
    records = []

    for _, row in ideal.iterrows():
        pair_id = int(row["match_pair_id"])
        records.append(
            {
                "scenario": "ideal_60CN_60AD_processed",
                "analysis_subset": split_map[pair_id],
                "subject_id": row["subject_id"],
                "session_id": row["session_id"],
                "experiment_id": row["experiment_id"],
                "diagnosis": row["diagnosis"],
                "age_at_MR": row["age_at_MR"],
                "sex": row["sex"],
                "Manufacturer": row["Manufacturer"],
                "ScannerModel": row["ScannerModel"],
                "TR_seconds": row["TR_seconds"],
                "expected_tr2_rest_runs": row["expected_tr2_rest_runs"],
                "match_pair_id": pair_id,
                "split_rule": split_rule[pair_id],
                "allowed_use": "fit_threshold_or_calibration_only" if split_map[pair_id] == "calibration" else "locked_final_test_only",
                "may_use_labels_for_threshold_or_calibration": split_map[pair_id] == "calibration",
                "may_use_labels_for_final_external_metric": True,
                "notes": "split defined before model scoring; no OASIS VAE retraining",
            }
        )

    for _, row in minimum.iterrows():
        records.append(
            {
                "scenario": "minimum_30CN_30AD_processed",
                "analysis_subset": "locked_test",
                "subject_id": row["subject_id"],
                "session_id": row["session_id"],
                "experiment_id": row["experiment_id"],
                "diagnosis": row["diagnosis"],
                "age_at_MR": row["age_at_MR"],
                "sex": row["sex"],
                "Manufacturer": row["Manufacturer"],
                "ScannerModel": row["ScannerModel"],
                "TR_seconds": row["TR_seconds"],
                "expected_tr2_rest_runs": row["expected_tr2_rest_runs"],
                "match_pair_id": row["match_pair_id"],
                "split_rule": "fallback_minimum_batch_all_subjects_locked_test",
                "allowed_use": "locked_final_test_only",
                "may_use_labels_for_threshold_or_calibration": False,
                "may_use_labels_for_final_external_metric": True,
                "notes": "pilot may be used only for descriptive threshold-shift context",
            }
        )
    return pd.DataFrame(records)


def summarize_split(plan: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario, subset, diagnosis), sub in plan.groupby(["scenario", "analysis_subset", "diagnosis"], sort=True):
        ages = pd.to_numeric(sub["age_at_MR"], errors="coerce")
        rows.append(
            {
                "scenario": scenario,
                "analysis_subset": subset,
                "diagnosis": diagnosis,
                "n": len(sub),
                "age_mean": round(float(ages.mean()), 3),
                "age_sd": round(float(ages.std(ddof=1)), 3) if len(sub) > 1 else 0.0,
                "sex_1_count": int((sub["sex"].astype(str) == "1").sum()),
                "sex_2_count": int((sub["sex"].astype(str) == "2").sum()),
                "trioTim_count": int((sub["ScannerModel"].astype(str) == "TrioTim").sum()),
                "expected_runs_mean": round(float(pd.to_numeric(sub["expected_tr2_rest_runs"], errors="coerce").mean()), 3),
            }
        )
    return pd.DataFrame(rows)


def write_protocol_docs(t0: str, split_summary: pd.DataFrame) -> None:
    readme = f"""# OASIS External Calibration/Test Protocol

Generated: `{t0}`

## Purpose

This package defines a rigorous OASIS external calibration/test protocol for the
locked ADNI-trained models before any scores are inspected for the new OASIS
batch. It uses the non-overlapping next-batch selection prepared in
`results/revision_bspc_2026/oasis_next_batch_selection_audit/`.

## Non-negotiable rules

- Do not merge OASIS with ADNI training.
- Do not retrain the VAE on OASIS.
- Do not use OASIS locked-test labels for model selection, threshold tuning, or
  calibration.
- Treat the `Tanda_2026_05_25` pilot only as exploratory background and
  threshold-transfer context, not as the final locked test.
- Split the new `60 CN + 60 AD_DEMENTIA` batch before seeing model scores.

## Preferred design

If the ideal `60 CN + 60 AD_DEMENTIA` batch is processed, use:

- `30 CN + 30 AD_DEMENTIA` as OASIS calibration.
- `30 CN + 30 AD_DEMENTIA` as locked external test.

The split is done at the matched-pair level using sex-stratified age ordering.
No model scores are used.

## Fallback design

If only `30 CN + 30 AD_DEMENTIA` are processed, use the full new batch as the
locked external test. In that case, no OASIS-based threshold or intercept
calibration is permitted for the primary result. The pilot may be used only for
descriptive threshold-shift analysis.

## Split summary

{md_table(split_summary)}
"""
    (OUTPUT_DIR / "README.md").write_text(readme, encoding="utf-8")

    threshold_plan = """# Threshold Calibration Plan

## Primary baseline

Report the ADNI fixed-threshold result first for each model. This is the cleanest
external transfer estimate because no OASIS labels are used to choose the
operating point.

## Calibration-subset threshold-only recalibration

If the ideal `60 CN + 60 AD_DEMENTIA` batch is available:

1. Score the OASIS calibration subset with the frozen ADNI model.
2. Use only calibration labels to select a scalar threshold with the same rule
   used internally in ADNI: sensitivity >= 0.70, then maximize specificity.
3. Freeze that threshold.
4. Apply it once to the locked OASIS test subset.

No OASIS test labels may influence the threshold.

## Fallback minimum-batch case

If only `30 CN + 30 AD_DEMENTIA` are available, do not recalibrate thresholds for
the primary result. Report ADNI fixed-threshold performance only. Pilot-derived
thresholds may be shown as descriptive threshold-shift context, explicitly
labelled as non-confirmatory.
"""
    (OUTPUT_DIR / "threshold_calibration_plan.md").write_text(threshold_plan, encoding="utf-8")

    intercept_plan = """# Intercept-Only Recalibration Plan

The model weights and VAE remain fixed. Intercept-only recalibration adjusts only
the decision offset on the OASIS calibration subset.

## Procedure for the ideal batch

1. Compute the frozen model score/logit for OASIS calibration subjects.
2. Fit a one-parameter intercept shift using calibration labels only. The slope
   and all feature weights remain fixed.
3. Freeze the shifted intercept.
4. Evaluate once on the locked OASIS test subset.

This estimates cohort-level threshold shift without changing representation
learning or classifier feature weighting.

## Platt scaling

Platt scaling is optional and should only be reported as secondary if the
calibration subset is large enough and class-balanced. With 30 CN + 30 AD in
calibration, it is acceptable as a sensitivity analysis but should not replace
the fixed-threshold primary external result.

## Prohibited actions

- Do not refit classifier weights on OASIS.
- Do not refit the VAE on OASIS.
- Do not choose between calibration methods using locked-test labels.
"""
    (OUTPUT_DIR / "intercept_recalibration_plan.md").write_text(intercept_plan, encoding="utf-8")

    locked_test_plan = """# Locked Test Evaluation Plan

## Primary metrics

For each model and run-combination strategy, report:

- ROC-AUC
- PR-AUC
- balanced accuracy
- sensitivity
- specificity
- F1
- confusion matrix
- Brier score/calibration summary if scores are probabilistic

## Evaluation hierarchy

1. ADNI fixed threshold on locked OASIS test.
2. If the ideal batch is available, threshold-only OASIS calibration applied to
   locked OASIS test.
3. If reported, intercept-only recalibration applied to locked OASIS test.
4. Platt scaling only as secondary sensitivity, not as the primary claim.

## Data handling

OASIS labels are used only for the final metric calculation on the pre-declared
test subset. The test subset must not be inspected for model selection,
threshold selection, calibration method selection, or channel/model promotion.
"""
    (OUTPUT_DIR / "locked_test_evaluation_plan.md").write_text(locked_test_plan, encoding="utf-8")

    model_roles = """# Model Roles

## Primary model

`v5.1b [1,0,2] horizon4480/cycles56`

This remains the primary manuscript model because it is the best overall ADNI
internal model when ROC-AUC, PR-AUC, balanced accuracy, sensitivity, F1, subgroup
behavior, and robustness audits are considered together.

## Secondary simplified model

`v5.1b [1] offdiag_channelmean`

This is a simplified/AUROC-oriented sensitivity model. It may be scored on OASIS
but should not replace the primary model unless a pre-specified future validation
study supports that change.

## Secondary deconfounding model

Manufacturer-conditioned decoder-only beta-VAE.

This model is a deconfounding sensitivity analysis. It should be reported as
secondary even if OASIS performance is favorable, because internal ADNI
performance did not surpass the primary locked model and OASIS pilot advantages
were not stable across run handling.
"""
    (OUTPUT_DIR / "model_roles.md").write_text(model_roles, encoding="utf-8")

    reviewer_text = """# Reviewer-Response OASIS External Validation Text

We agree that external validation is essential for interpreting model
generalization. We therefore separated OASIS from all ADNI model development.
The initial OASIS Tanda_2026_05_25 batch was used only as an exploratory pilot
to diagnose threshold transfer and acquisition differences. For confirmatory
external validation, we prepared a new non-overlapping OASIS batch selection.
If the full 60 CN + 60 AD_DEMENTIA batch is processed, we will split it before
any model scoring into a 30 CN + 30 AD calibration subset and a 30 CN + 30 AD
locked test subset, matched as closely as possible for age and sex. Threshold or
intercept recalibration, if reported, will use only the calibration subset; the
locked test subset will be evaluated once. If only 30 CN + 30 AD_DEMENTIA are
processed, that batch will be treated as a locked external test and no
OASIS-based recalibration will be used for the primary result. OASIS data will
not be merged with ADNI, and no VAE or classifier weights will be retrained on
OASIS.
"""
    (OUTPUT_DIR / "reviewer_response_oasis_external_validation_text.md").write_text(
        reviewer_text, encoding="utf-8"
    )


def main() -> None:
    t0 = now_utc()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ideal = load_table(IDEAL_SELECTION)
    minimum = load_table(MINIMUM_SELECTION)
    pair_table = build_pair_table(ideal)
    pair_split = split_ideal_pairs(pair_table)
    protocol_plan = rows_for_protocol(ideal, minimum, pair_split)
    split_summary = summarize_split(protocol_plan)

    write_csv_md(protocol_plan, "calibration_test_split_plan")
    write_protocol_docs(t0, split_summary)

    command_log = {
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "timestamp_utc": t0,
        "output_dir": str(OUTPUT_DIR.relative_to(PROJECT_ROOT)),
        "inputs": {
            "ideal_selection": str(IDEAL_SELECTION.relative_to(PROJECT_ROOT)),
            "minimum_selection": str(MINIMUM_SELECTION.relative_to(PROJECT_ROOT)),
            "pilot_external_scoring_dir": str(PILOT_DIR.relative_to(PROJECT_ROOT)),
        },
        "actions": [
            "loaded_non_overlapping_oasis_next_batch_selection",
            "created_pre_score_pair_level_calibration_test_split_for_ideal_batch",
            "defined_minimum_batch_locked_test_fallback",
            "wrote_protocol_documents",
        ],
        "counts": {
            "ideal_rows": len(ideal),
            "minimum_rows": len(minimum),
            "ideal_calibration_rows": int(
                ((protocol_plan["scenario"] == "ideal_60CN_60AD_processed")
                & (protocol_plan["analysis_subset"] == "calibration")).sum()
            ),
            "ideal_locked_test_rows": int(
                ((protocol_plan["scenario"] == "ideal_60CN_60AD_processed")
                & (protocol_plan["analysis_subset"] == "locked_test")).sum()
            ),
            "minimum_locked_test_rows": int((protocol_plan["scenario"] == "minimum_30CN_30AD_processed").sum()),
        },
        "safety": {
            "trained_models": False,
            "scored_models": False,
            "modified_existing_outputs": False,
            "modified_input_data": False,
            "used_oasis_test_labels_for_model_selection": False,
        },
    }
    (OUTPUT_DIR / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")

    print(f"Wrote OASIS calibration/test protocol to {OUTPUT_DIR}")
    print(split_summary.to_string(index=False))


if __name__ == "__main__":
    main()
