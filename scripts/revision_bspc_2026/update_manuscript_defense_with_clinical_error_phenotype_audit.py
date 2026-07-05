#!/usr/bin/env python3
"""Update manuscript defense package with clinical error phenotype audit.

This modifies only manuscript-defense package files. It does not train, score,
fit thresholds, select models, derive subject exclusions, or modify
tensor/metadata/ledger/model-output artifacts.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results" / "revision_bspc_2026"
DEFENSE_DIR = RESULTS_ROOT / "adni_v5_1_batch20260514b_manuscript_defense_locked_current_model"
AUDIT_DIR = RESULTS_ROOT / "locked_model_clinical_error_phenotype_audit"

README = DEFENSE_DIR / "README.md"
FINAL_RECOMMENDATION = DEFENSE_DIR / "final_recommendation.md"
REVIEWER_TEXT = DEFENSE_DIR / "reviewer_response_ready_text.md"
COMMAND_LOG = DEFENSE_DIR / "command_log.json"
RESULT_MD = DEFENSE_DIR / "locked_model_clinical_error_phenotype_result.md"


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
    return str(value).strip()


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._\n"
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(clean(v) for v in row.tolist()) + " |")
    return "\n".join([header, sep] + rows) + "\n"


def marker_block(name: str, body: str) -> str:
    return f"<!-- BEGIN {name} -->\n{body.rstrip()}\n<!-- END {name} -->\n"


def upsert_marker(path: Path, name: str, body: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    begin = f"<!-- BEGIN {name} -->"
    end = f"<!-- END {name} -->"
    block = marker_block(name, body)
    if begin in text and end in text:
        pre = text.split(begin, 1)[0].rstrip()
        rest = text.split(begin, 1)[1].split(end, 1)[1].lstrip()
        text = f"{pre}\n\n{block}\n{rest}".rstrip() + "\n"
    else:
        text = text.rstrip() + "\n\n" + block
    path.write_text(text, encoding="utf-8")


def load_csv(name: str) -> pd.DataFrame:
    path = AUDIT_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def selected_continuous_summary() -> pd.DataFrame:
    cn = load_csv("stable_fp_cn_vs_true_negative_cn_continuous.csv")
    ad = load_csv("stable_fn_ad_vs_true_positive_ad_continuous.csv")
    rows = []
    for df, group_a, group_b in [
        (cn, "stable_fp_cn", "true_negative_cn"),
        (ad, "stable_fn_ad", "true_positive_ad"),
    ]:
        for var in ["Age", "original_n_TR", "MMSE", "CDRSB", "MOCA", "ABETA", "TAU", "PTAU"]:
            m = df[df["variable"].eq(var)]
            if m.empty:
                continue
            row = m.iloc[0]
            rows.append(
                {
                    "comparison": f"{group_a}_vs_{group_b}",
                    "variable": var,
                    "group_a": group_a,
                    "group_b": group_b,
                    "group_a_n": int(row["n_a"]) if pd.notna(row["n_a"]) else "",
                    "group_b_n": int(row["n_b"]) if pd.notna(row["n_b"]) else "",
                    "group_a_mean": round(float(row["mean_a"]), 3) if pd.notna(row["mean_a"]) else "",
                    "group_b_mean": round(float(row["mean_b"]), 3) if pd.notna(row["mean_b"]) else "",
                    "smd": round(float(row["smd_a_minus_b"]), 3) if pd.notna(row["smd_a_minus_b"]) else "",
                    "welch_p": float(row["welch_p"]) if pd.notna(row["welch_p"]) else "",
                    "mannwhitney_p": float(row["mannwhitney_p"]) if pd.notna(row["mannwhitney_p"]) else "",
                }
            )
    return pd.DataFrame(rows)


def selected_categorical_summary() -> pd.DataFrame:
    cn = load_csv("stable_fp_cn_vs_true_negative_cn_categorical.csv")
    ad = load_csv("stable_fn_ad_vs_true_positive_ad_categorical.csv")
    rows = []
    for source, group_a, group_b in [
        (cn, "stable_fp_cn", "true_negative_cn"),
        (ad, "stable_fn_ad", "true_positive_ad"),
    ]:
        for var in ["Sex", "Manufacturer"]:
            sub = source[source["variable"].eq(var)].copy()
            if sub.empty:
                continue
            for _, row in sub.iterrows():
                rows.append(
                    {
                        "comparison": f"{group_a}_vs_{group_b}",
                        "variable": var,
                        "level": row["level"],
                        "group_a": group_a,
                        "group_b": group_b,
                        "group_a_count": int(row[f"{group_a}_count"]),
                        "group_b_count": int(row[f"{group_b}_count"]),
                        "chi2_p_overall": float(row["chi2_p_overall"]) if pd.notna(row["chi2_p_overall"]) else "",
                    }
                )
    return pd.DataFrame(rows)


def build_result_doc(updated_utc: str) -> str:
    coverage = load_csv("clinical_variable_coverage.csv")
    flag_summary = load_csv("clinical_flag_summary.csv")
    longitudinal = load_csv("longitudinal_evidence_summary.csv")
    continuous = selected_continuous_summary()
    categorical = selected_categorical_summary()

    requested_coverage = coverage[coverage["requested_field"].isin(["MMSE", "CDRSB", "MOCA", "ABETA", "TAU", "PTAU", "ADAS-Cog", "APOE"])]

    return f"""# Locked-Model Clinical Error Phenotype Audit Result

Updated: `{updated_utc}`

Source audit: `{AUDIT_DIR.relative_to(PROJECT_ROOT)}/`

## Scope and Safety

This was a read-only descriptive audit. It did not train a model, score data, fit thresholds, select a model, or derive any subject exclusion rule.

## Continuous Phenotype Comparison

{markdown_table(continuous)}

Key descriptive findings:

- Stable FP CN were older than true-negative CN (`75.80` vs `72.04` years).
- Stable FP CN had fewer original timepoints (`180.44` vs `193.81`).
- Stable FN AD had similar age to true-positive AD (`74.48` vs `74.26` years).
- Stable FN AD had higher original timepoints (`190.14` vs `174.38`).

## Categorical Phenotype Comparison

{markdown_table(categorical)}

Key subgroup findings:

- Philips CN false positives remain elevated.
- Stable FN AD were enriched for male and GE cases.

## Clinical Suspicion and Milder/Atypical Flags

{markdown_table(flag_summary)}

Interpretation: `32/86` stable FP CN had at least one heuristic clinical/cognitive/atrophy/biomarker suspicion flag. `16/21` stable FN AD had at least one milder/atypical flag. These flags are descriptive only and are limited by sparse cognitive/biomarker coverage.

## Longitudinal Evidence

{markdown_table(longitudinal)}

Local longitudinal fMRI-linked clinical sources showed `0` later MCI/AD conversions for stable FP CN. This does not rule out future conversion in external ADNI records, but it means no local evidence supports a deterministic relabeling or exclusion rule.

## Metadata Coverage

{markdown_table(requested_coverage)}

Available locally: MMSE, CDRSB, MOCA, education, structural proxies, ABETA, TAU, and PTAU where present. ADAS-Cog and APOE were not found in local clinical metadata headers.

## Manuscript-Safe Interpretation

Residual errors likely reflect disease heterogeneity, scanner/site structure, threshold-transfer limits, and incomplete clinical phenotype capture. The audit does not justify excluding subjects or changing the locked model. It should be used as reviewer-facing robustness and error-phenotyping evidence.
"""


def update_readme() -> None:
    body = """## Locked-Model Clinical Error Phenotype Audit

Audit path: `results/revision_bspc_2026/locked_model_clinical_error_phenotype_audit/`

This read-only descriptive audit compared stable false-positive CN subjects with true-negative CN subjects and stable false-negative AD subjects with true-positive AD subjects. It did not train a model, score data, fit thresholds, select a model, or derive a subject exclusion rule.

Stable FP CN were older than true-negative CN (`75.80` vs `72.04`) and had fewer original timepoints (`180.44` vs `193.81`). Philips CN false positives remained elevated. `32/86` stable FP CN had at least one heuristic clinical/cognitive/atrophy/biomarker suspicion flag, but local longitudinal fMRI-linked sources showed `0` later MCI/AD conversions for stable FP CN.

Stable FN AD had similar age to true-positive AD (`74.48` vs `74.26`) and higher original timepoints (`190.14` vs `174.38`). Stable FN AD were enriched for male/GE cases, and `16/21` had at least one milder/atypical flag.

Metadata coverage was incomplete: MMSE, CDRSB, MOCA, education, structural proxies, ABETA, TAU, and PTAU were available where present, while ADAS-Cog and APOE were not found locally. Decision: do not exclude subjects and do not change the model; use this as reviewer-facing robustness/error-phenotyping analysis.
"""
    upsert_marker(README, "LOCKED_MODEL_CLINICAL_ERROR_PHENOTYPE_AUDIT", body)


def update_final_recommendation() -> None:
    body = """## Locked-Model Clinical Error Phenotype Audit

The clinical error phenotype audit does not justify changing the locked model or excluding subjects. It was read-only and descriptive: no training, scoring, threshold fitting, model selection, or subject exclusion rule was performed.

Stable FP CN were older (`75.80` vs `72.04`) and had fewer original timepoints (`180.44` vs `193.81`) than true-negative CN. Philips CN false positives remain elevated. Although `32/86` stable FP CN had at least one heuristic clinical/cognitive/atrophy/biomarker suspicion flag, local longitudinal fMRI-linked sources showed `0` later MCI/AD conversions.

Stable FN AD had similar age to true-positive AD (`74.48` vs `74.26`) but higher original timepoints (`190.14` vs `174.38`), were enriched for male/GE cases, and `16/21` had at least one milder/atypical flag.

Interpretation: residual errors likely reflect disease heterogeneity, scanner/site structure, threshold-transfer limits, and incomplete clinical phenotype capture. Decision: retain the locked model and proceed with OASIS external calibration/test validation.
"""
    upsert_marker(FINAL_RECOMMENDATION, "LOCKED_MODEL_CLINICAL_ERROR_PHENOTYPE_AUDIT", body)


def update_reviewer_text() -> None:
    body = """## Reviewer Response: Clinical Error Phenotyping

We added a read-only clinical error phenotype audit of stable false-positive and false-negative subjects. This analysis was descriptive only: it did not train a model, refit thresholds, select models, or derive a subject exclusion rule.

Stable false-positive CN subjects were older than true-negative CN subjects (`75.80` vs `72.04` years) and had fewer original timepoints (`180.44` vs `193.81`). Philips CN false positives remained elevated. `32/86` stable FP controls had at least one heuristic clinical/cognitive/atrophy/biomarker suspicion flag, but local longitudinal fMRI-linked sources showed `0` later MCI/AD conversions for these stable FP controls.

Stable false-negative AD subjects had similar age to true-positive AD subjects (`74.48` vs `74.26`) and higher original timepoint counts (`190.14` vs `174.38`). They were enriched for male/GE cases, and `16/21` had at least one milder/atypical flag. Available local metadata included MMSE, CDRSB, MOCA, education, structural proxies, ABETA, TAU, and PTAU where present; ADAS-Cog and APOE were not found in local headers.

We therefore interpret residual errors as reflecting a mixture of disease heterogeneity, scanner/site structure, threshold-transfer limits, and incomplete clinical phenotype capture. These findings support transparent error reporting, not post-hoc subject exclusion or model replacement.
"""
    upsert_marker(REVIEWER_TEXT, "LOCKED_MODEL_CLINICAL_ERROR_PHENOTYPE_AUDIT", body)


def update_command_log(updated_utc: str) -> None:
    log = json.loads(COMMAND_LOG.read_text(encoding="utf-8")) if COMMAND_LOG.exists() else {}
    log["last_updated_utc"] = updated_utc
    log["latest_update_script"] = str(Path(__file__).resolve())
    updates = log.setdefault("updates", [])
    updates[:] = [u for u in updates if u.get("action") != "add_locked_model_clinical_error_phenotype_audit"]
    updates.append(
        {
            "action": "add_locked_model_clinical_error_phenotype_audit",
            "source_dir": str(AUDIT_DIR),
            "recorded_findings": {
                "read_only_descriptive": True,
                "stable_fp_cn_vs_true_negative_cn": {
                    "age_mean": [75.80, 72.04],
                    "original_n_tr_mean": [180.44, 193.81],
                    "philips_cn_false_positives_elevated": True,
                    "clinical_suspicion_flags": "32/86",
                    "later_mci_ad_conversions_in_local_fmri_linked_sources": 0,
                },
                "stable_fn_ad_vs_true_positive_ad": {
                    "age_mean": [74.48, 74.26],
                    "original_n_tr_mean": [190.14, 174.38],
                    "enriched_for_male_ge_cases": True,
                    "milder_atypical_flags": "16/21",
                },
                "metadata_coverage": {
                    "available": ["MMSE", "CDRSB", "MOCA", "education", "structural proxies", "ABETA", "TAU", "PTAU"],
                    "not_found_locally": ["ADAS-Cog", "APOE"],
                },
                "interpretation": "residual_errors_reflect_disease_heterogeneity_scanner_site_structure_threshold_transfer_limits_incomplete_phenotype_capture",
                "decision": "do_not_exclude_subjects_do_not_change_model_use_as_reviewer_facing_error_phenotyping",
            },
            "training_launched": False,
            "scoring_launched": False,
            "threshold_fitting": False,
            "model_selection": False,
            "subject_exclusion_rule_derived": False,
            "tensor_modified": False,
            "metadata_modified": False,
            "ledger_modified": False,
            "model_outputs_modified": False,
            "updated_utc": updated_utc,
        }
    )
    COMMAND_LOG.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    updated_utc = now_utc()
    required = [
        "stable_fp_cn_vs_true_negative_cn_continuous.csv",
        "stable_fn_ad_vs_true_positive_ad_continuous.csv",
        "stable_fp_cn_vs_true_negative_cn_categorical.csv",
        "stable_fn_ad_vs_true_positive_ad_categorical.csv",
        "clinical_flag_summary.csv",
        "longitudinal_evidence_summary.csv",
        "clinical_variable_coverage.csv",
    ]
    for name in required:
        if not (AUDIT_DIR / name).exists():
            raise FileNotFoundError(AUDIT_DIR / name)

    RESULT_MD.write_text(build_result_doc(updated_utc), encoding="utf-8")
    update_readme()
    update_final_recommendation()
    update_reviewer_text()
    update_command_log(updated_utc)
    print(f"Updated manuscript defense package with clinical error phenotype audit: {DEFENSE_DIR}")


if __name__ == "__main__":
    main()
