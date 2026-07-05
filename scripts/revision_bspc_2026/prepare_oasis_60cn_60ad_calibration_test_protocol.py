#!/usr/bin/env python3
"""Prepare a locked OASIS 60CN/60AD calibration/test protocol.

This is a protocol-only artifact. It reads the selected OASIS candidate manifest
and writes a pre-score calibration/test split plus analysis plans. It does not
download data, build tensors, score models, train models, or modify inputs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results" / "revision_bspc_2026"
INPUT_CSV = RESULTS_ROOT / "oasis_next_batch_selection_audit" / "selected_ideal_60CN_60AD.csv"
OUTPUT_DIR = RESULTS_ROOT / "oasis_60cn_60ad_calibration_test_protocol"


def utc_now() -> str:
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


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    view = df if max_rows is None else df.head(max_rows)
    if view.empty:
        return "_No rows._\n"
    header = "| " + " | ".join(str(c) for c in view.columns) + " |"
    sep = "| " + " | ".join("---" for _ in view.columns) + " |"
    rows = ["| " + " | ".join(clean(v) for v in row.tolist()) + " |" for _, row in view.iterrows()]
    suffix = ""
    if max_rows is not None and len(df) > max_rows:
        suffix = f"\n\n_Showing first {max_rows} of {len(df)} rows._\n"
    return "\n".join([header, sep] + rows) + suffix


def write_table(df: pd.DataFrame, stem: str, max_md_rows: int | None = None) -> None:
    df.to_csv(OUTPUT_DIR / f"{stem}.csv", index=False)
    (OUTPUT_DIR / f"{stem}.md").write_text(md_table(df, max_rows=max_md_rows), encoding="utf-8")


def load_selection() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)
    df = pd.read_csv(INPUT_CSV)
    required = {
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
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {INPUT_CSV}: {missing}")
    counts = df["diagnosis"].value_counts().to_dict()
    if counts.get("CN", 0) != 60 or counts.get("AD_DEMENTIA", 0) != 60:
        raise ValueError(f"Expected 60 CN and 60 AD_DEMENTIA, got {counts}")
    if df["subject_id"].nunique() != 120:
        raise ValueError("Expected 120 unique subjects in selected_ideal_60CN_60AD.csv")
    if df["experiment_id"].nunique() != 120:
        raise ValueError("Expected 120 unique experiment_id values")
    return df


def build_pair_table(selection: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pair_id, grp in selection.groupby("match_pair_id", sort=True):
        cn = grp[grp["diagnosis"] == "CN"]
        ad = grp[grp["diagnosis"] == "AD_DEMENTIA"]
        if len(cn) != 1 or len(ad) != 1:
            raise ValueError(f"Pair {pair_id} does not contain exactly one CN and one AD_DEMENTIA")
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
                "min_expected_runs": min(int(cn_row["expected_tr2_rest_runs"]), int(ad_row["expected_tr2_rest_runs"])),
            }
        )
    pairs = pd.DataFrame(rows).sort_values(["pair_sex", "pair_age_mean", "match_pair_id"]).reset_index(drop=True)
    if len(pairs) != 60:
        raise ValueError(f"Expected 60 matched pairs, found {len(pairs)}")
    return pairs


def assign_pair_split(pairs: pd.DataFrame) -> pd.DataFrame:
    assigned = []
    for pair_sex, sub in pairs.groupby("pair_sex", sort=True):
        sub = sub.sort_values(["pair_age_mean", "match_pair_id"]).reset_index(drop=True)
        for i, (_, row) in enumerate(sub.iterrows()):
            item = row.to_dict()
            item["protocol_subset"] = "calibration" if i % 2 == 0 else "locked_test"
            item["split_rule"] = "alternate_matched_pairs_within_sex_sorted_by_pair_age"
            assigned.append(item)
    out = pd.DataFrame(assigned)

    # Guarantee exactly 30 pairs per subset. If a sex stratum has odd length,
    # move one age-boundary pair from the larger subset to the smaller subset.
    while int((out["protocol_subset"] == "calibration").sum()) > 30:
        cal = out[out["protocol_subset"] == "calibration"].copy()
        cal["_age_distance"] = (cal["pair_age_mean"] - cal["pair_age_mean"].median()).abs()
        idx = cal.sort_values(["_age_distance", "match_pair_id"], ascending=[False, False]).index[0]
        out.loc[idx, "protocol_subset"] = "locked_test"
        out.loc[idx, "split_rule"] += ";size_balance_boundary_adjustment"
    while int((out["protocol_subset"] == "calibration").sum()) < 30:
        test = out[out["protocol_subset"] == "locked_test"].copy()
        test["_age_distance"] = (test["pair_age_mean"] - test["pair_age_mean"].median()).abs()
        idx = test.sort_values(["_age_distance", "match_pair_id"], ascending=[False, False]).index[0]
        out.loc[idx, "protocol_subset"] = "calibration"
        out.loc[idx, "split_rule"] += ";size_balance_boundary_adjustment"

    if int((out["protocol_subset"] == "calibration").sum()) != 30:
        raise RuntimeError("Calibration pair count is not 30")
    if int((out["protocol_subset"] == "locked_test").sum()) != 30:
        raise RuntimeError("Locked-test pair count is not 30")
    return out


def build_split(selection: pd.DataFrame, pair_split: pd.DataFrame) -> pd.DataFrame:
    split_map = pair_split.set_index("match_pair_id")["protocol_subset"].to_dict()
    rule_map = pair_split.set_index("match_pair_id")["split_rule"].to_dict()
    age_pair_map = pair_split.set_index("match_pair_id")["age_abs_diff_within_pair"].to_dict()
    rows = []
    for _, row in selection.iterrows():
        pair_id = int(row["match_pair_id"])
        subset = split_map[pair_id]
        rows.append(
            {
                "protocol_subset": subset,
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
                "age_abs_diff_within_pair": age_pair_map[pair_id],
                "split_rule": rule_map[pair_id],
                "allowed_use": "threshold_or_intercept_calibration_only" if subset == "calibration" else "locked_external_test_only",
                "may_use_labels_for_threshold_selection": subset == "calibration",
                "may_use_labels_for_final_metric": True,
                "no_oasis_training": True,
                "notes": "split fixed before scoring; OASIS not merged with ADNI",
            }
        )
    out = pd.DataFrame(rows).sort_values(["protocol_subset", "match_pair_id", "diagnosis"]).reset_index(drop=True)
    return out


def split_summary(split: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (subset, diagnosis), sub in split.groupby(["protocol_subset", "diagnosis"], sort=True):
        ages = pd.to_numeric(sub["age_at_MR"], errors="coerce")
        rows.append(
            {
                "protocol_subset": subset,
                "diagnosis": diagnosis,
                "n": len(sub),
                "age_mean": round(float(ages.mean()), 3),
                "age_sd": round(float(ages.std(ddof=1)), 3),
                "age_median": round(float(ages.median()), 3),
                "sex_1_count": int((sub["sex"].astype(str) == "1").sum()),
                "sex_2_count": int((sub["sex"].astype(str) == "2").sum()),
                "trioTim_count": int((sub["ScannerModel"].astype(str) == "TrioTim").sum()),
                "expected_runs_mean": round(float(pd.to_numeric(sub["expected_tr2_rest_runs"], errors="coerce").mean()), 3),
            }
        )
    return pd.DataFrame(rows)


def write_docs(summary: pd.DataFrame, generated_utc: str) -> None:
    calibration_plan = f"""# OASIS 60CN/60AD Calibration Plan

Generated: `{generated_utc}`

## Locked Split

The new OASIS `60 CN + 60 AD_DEMENTIA` selection is split before any scoring:

{md_table(summary)}

## Rules

- OASIS is not merged with ADNI.
- The VAE is not retrained on OASIS.
- Classifiers are not retrained on OASIS.
- OASIS locked-test labels are never used for threshold selection, calibration,
  model choice, or channel choice.
- The pilot OASIS batch remains exploratory evidence of threshold shift only.

## Threshold Strategies

1. **ADNI fixed threshold baseline.** Apply the ADNI-derived fold/ensemble
   threshold directly. This is the primary transfer baseline.
2. **Threshold-only external recalibration.** On the OASIS calibration subset
   only, select a scalar threshold with the same operating rule as ADNI:
   sensitivity >= 0.70, then maximize specificity. Freeze this threshold and
   evaluate once on the locked test subset.
3. **Intercept-only recalibration.** As a secondary calibration analysis, keep
   all classifier weights fixed and fit only an intercept/offset on the
   calibration subset. Freeze the intercept and evaluate once on locked test.
4. **Platt/isotonic.** Do not run Platt or isotonic calibration unless sample
   size is explicitly judged sufficient in a later protocol amendment. With
   30 CN + 30 AD calibration subjects, these are not primary analyses.

## Prohibited Uses

- No OASIS VAE training.
- No OASIS classifier-weight training.
- No threshold fitting on locked test.
- No selecting between models using locked-test labels.
"""
    (OUTPUT_DIR / "calibration_plan.md").write_text(calibration_plan, encoding="utf-8")

    locked_plan = """# Locked Test Evaluation Plan

## Primary Report

For the locked OASIS test subset, report for each model and run-handling variant:

- ROC-AUC
- PR-AUC
- sensitivity
- specificity
- balanced accuracy
- F1
- confusion matrix
- Brier score
- ECE / reliability bins
- score distributions by diagnosis
- threshold-transfer table or plot

## Evaluation Order

1. Report ADNI fixed-threshold baseline.
2. Report threshold-only recalibration trained only on OASIS calibration.
3. Report intercept-only recalibration as secondary.
4. Report run-handling sensitivity.

The locked-test subset is evaluated once per pre-declared model/build/threshold
strategy. Locked-test labels must not be used to choose the model, build variant,
or threshold strategy.
"""
    (OUTPUT_DIR / "locked_test_evaluation_plan.md").write_text(locked_plan, encoding="utf-8")

    model_roles = """# Model Roles

## Primary

`v5.1b [1,0,2] horizon4480/cycles56`

This remains the primary locked ADNI model and the main OASIS external validation
target.

## Secondary Simplified / AUROC-Oriented

`v5.1b [1] offdiag_channelmean`

This model is scored as a secondary simplified/channel-ablation sensitivity.

## Secondary Deconfounding

Manufacturer-conditioned decoder-only beta-VAE.

This model is scored as a secondary deconfounding sensitivity. Manufacturer is
not passed to the classifier. Favorable OASIS results alone do not promote it
over the primary model.
"""
    (OUTPUT_DIR / "model_roles.md").write_text(model_roles, encoding="utf-8")

    run_handling = """# Run-Handling Plan

## Primary Build Variant

`concatenated_timeseries`

This is primary because Martin recommended combining QC-usable runs by
concatenating time series before computing one connectome per subject/session.

## ADNI-Like Sensitivity

`runwise_140TR_connectome_average`

This is the ADNI temporal-homogenization sensitivity analysis. Each QC-usable
run is cropped/truncated to 140 TR before connectome construction, and run-level
connectomes are averaged per subject/session.

## Supplementary

`runwise164_connectome_average`

This may be reported as supplementary because it keeps the full OASIS run length
while avoiding concatenation. It is not the primary protocol result.
"""
    (OUTPUT_DIR / "run_handling_plan.md").write_text(run_handling, encoding="utf-8")

    reviewer_text = """# Reviewer-Ready OASIS Protocol Text

We separated OASIS from all ADNI model development. The initial OASIS pilot was
used only to identify threshold-transfer issues and was not used as a final
locked test. For the next non-overlapping OASIS batch, we fixed the
calibration/test split before any model scoring: 30 CN and 30 AD_DEMENTIA are
assigned to an external calibration subset, and 30 CN and 30 AD_DEMENTIA are
assigned to an untouched locked external test subset, with age and sex matched
as closely as possible by matched-pair splitting.

The ADNI-trained VAE and classifiers will not be retrained on OASIS. The primary
external result will first report the ADNI fixed-threshold baseline. Any
threshold-only recalibration will be performed only on the OASIS calibration
subset using the pre-specified rule of sensitivity >= 0.70 with maximal
specificity, then evaluated once on the locked OASIS test subset. Intercept-only
recalibration is secondary. Locked-test labels will not be used for threshold
selection, model selection, or channel selection.
"""
    (OUTPUT_DIR / "reviewer_ready_oasis_protocol_text.md").write_text(reviewer_text, encoding="utf-8")


def main() -> None:
    generated_utc = utc_now()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selection = load_selection()
    pairs = build_pair_table(selection)
    pair_split = assign_pair_split(pairs)
    split = build_split(selection, pair_split)
    summary = split_summary(split)

    # Validation guards.
    counts = split.groupby(["protocol_subset", "diagnosis"]).size().to_dict()
    expected = {
        ("calibration", "CN"): 30,
        ("calibration", "AD_DEMENTIA"): 30,
        ("locked_test", "CN"): 30,
        ("locked_test", "AD_DEMENTIA"): 30,
    }
    if counts != expected:
        raise RuntimeError(f"Unexpected split counts: {counts}")
    cal_subjects = set(split.loc[split["protocol_subset"] == "calibration", "subject_id"])
    test_subjects = set(split.loc[split["protocol_subset"] == "locked_test", "subject_id"])
    if cal_subjects & test_subjects:
        raise RuntimeError("Calibration/test subject overlap detected")

    write_table(split, "split_calibration_test")
    write_docs(summary, generated_utc)

    command_log = {
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "timestamp_utc": generated_utc,
        "input": str(INPUT_CSV.relative_to(PROJECT_ROOT)),
        "output_dir": str(OUTPUT_DIR.relative_to(PROJECT_ROOT)),
        "split_counts": {f"{k[0]}_{k[1]}": int(v) for k, v in counts.items()},
        "calibration_test_subject_overlap": 0,
        "models": [
            "primary_locked_v5_1b_ch1_0_2",
            "secondary_ch1_only",
            "secondary_manufacturer_conditioned",
        ],
        "build_variants": {
            "primary": "concatenated_timeseries",
            "sensitivity": "runwise_140TR_connectome_average",
            "supplementary": "runwise164_connectome_average",
        },
        "threshold_strategies": [
            "ADNI_fixed_threshold_baseline",
            "threshold_only_recalibration_on_calibration_subset_sens_ge_0p70_max_spec",
            "intercept_only_recalibration_secondary",
        ],
        "safety": {
            "training_performed": False,
            "scoring_performed": False,
            "input_data_modified": False,
            "oasis_merged_with_adni": False,
            "locked_test_labels_used_for_threshold_selection": False,
        },
    }
    (OUTPUT_DIR / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")

    print(f"Wrote locked OASIS 60CN/60AD protocol to {OUTPUT_DIR}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
