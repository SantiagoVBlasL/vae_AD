#!/usr/bin/env python3
"""Prepare a read-only/dry-run preflight for longer ADNI ROI time series.

The script consumes the ADNI timepoint availability/confounding audit and writes
a controlled rebuild plan for channels [1,0,2]. It does not build tensors and
does not modify existing data, configs, ledgers, or model outputs.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = ROOT / "results/revision_bspc_2026/adni_timepoint_availability_confounding_audit"
OUT_DIR = ROOT / "results/revision_bspc_2026/adni_longer_timeseries_connectome_preflight"

CHANNELS = [
    {
        "channel_index": 1,
        "channel_name": "Pearson_Full_FisherZ_Signed",
        "build_status": "planned_if_branch_unblocked",
        "notes": "Pearson full connectivity, Fisher-z signed.",
    },
    {
        "channel_index": 0,
        "channel_name": "Pearson_OMST_GCE_Signed_Weighted",
        "build_status": "planned_if_branch_unblocked",
        "notes": "OMST/Pearson graph-filtered signed weighted channel.",
    },
    {
        "channel_index": 2,
        "channel_name": "MI_KNN_Symmetric",
        "build_status": "planned_if_branch_unblocked",
        "notes": "kNN mutual information symmetric channel.",
    },
]


def to_markdown(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.write_text(df.to_markdown(index=index) + "\n", encoding="utf-8")


def write_table(df: pd.DataFrame, name: str, index: bool = False) -> None:
    df.to_csv(OUT_DIR / f"{name}.csv", index=index)
    to_markdown(df, OUT_DIR / f"{name}.md", index=index)


def load_audit_recommendation() -> str:
    log_path = AUDIT_DIR / "command_log.json"
    if log_path.exists():
        data = json.loads(log_path.read_text(encoding="utf-8"))
        rec = data.get("recommendation")
        if rec:
            return str(rec)
    text = (AUDIT_DIR / "final_recommendation.md").read_text(encoding="utf-8")
    match = re.search(r"Final recommendation:\s+\*\*([^*]+)\*\*", text)
    return match.group(1).strip() if match else "unknown"


def branch_statuses(recommendation: str) -> dict[str, dict[str, Any]]:
    all_available_allowed = recommendation == "all_timepoints_safe_to_test"
    fixed_preferred = recommendation == "fixed_longer_length_preferred"
    return {
        "locked_140TR_reference": {
            "branch_type": "reference",
            "required_n_TR": 140,
            "variable_length": False,
            "guardrail_status": "reference_only",
            "allowed_by_confounding_audit": True,
            "reason": "Current locked 140-TR time-homogenized reference.",
        },
        "all_available_timepoints": {
            "branch_type": "candidate",
            "required_n_TR": "subject_specific_all_available",
            "variable_length": True,
            "guardrail_status": "blocked" if not all_available_allowed else "allowed",
            "allowed_by_confounding_audit": all_available_allowed,
            "reason": (
                "Allowed only when the timepoint audit finds no diagnosis/site/manufacturer confounding. "
                f"Current audit recommendation is {recommendation}."
            ),
        },
        "fixed_160TR": {
            "branch_type": "candidate",
            "required_n_TR": 160,
            "variable_length": False,
            "guardrail_status": "not_preferred" if not fixed_preferred else "allowed",
            "allowed_by_confounding_audit": fixed_preferred,
            "reason": (
                "Fixed longer length avoids variable-length confounding but drops subjects with <160 TR. "
                f"Current audit recommendation is {recommendation}."
            ),
        },
        "fixed_180TR": {
            "branch_type": "candidate",
            "required_n_TR": 180,
            "variable_length": False,
            "guardrail_status": "not_preferred" if not fixed_preferred else "allowed",
            "allowed_by_confounding_audit": fixed_preferred,
            "reason": (
                "Equivalent retention to 160 TR in this cohort because observed lengths are 140/147/197/200. "
                f"Current audit recommendation is {recommendation}."
            ),
        },
    }


def build_subject_retention(subjects: pd.DataFrame, recommendation: str) -> pd.DataFrame:
    statuses = branch_statuses(recommendation)
    out = subjects.copy()
    out["branch_all_available_allowed"] = statuses["all_available_timepoints"][
        "allowed_by_confounding_audit"
    ]
    out["branch_fixed_160_allowed"] = statuses["fixed_160TR"]["allowed_by_confounding_audit"]
    out["branch_fixed_180_allowed"] = statuses["fixed_180TR"]["allowed_by_confounding_audit"]
    out["retain_locked_140TR_reference"] = out["original_n_timepoints"] >= 140
    out["retain_all_available_timepoints_if_overridden"] = out["original_n_timepoints"].notna()
    out["retain_fixed_160TR_if_overridden"] = out["original_n_timepoints"] >= 160
    out["retain_fixed_180TR_if_overridden"] = out["original_n_timepoints"] >= 180
    out["pool_impact_fixed_160TR"] = np.where(
        out["retain_fixed_160TR_if_overridden"],
        "retained",
        "dropped_insufficient_TR",
    )
    out["pool_impact_fixed_180TR"] = np.where(
        out["retain_fixed_180TR_if_overridden"],
        "retained",
        "dropped_insufficient_TR",
    )
    cols = [
        "SubjectID",
        "tensor_index",
        "ResearchGroup_Mapped",
        "Diagnosis",
        "classifier_pool_role",
        "classifier_outer_fold",
        "in_vae_pool_any_fold",
        "SiteCode",
        "Manufacturer",
        "Age",
        "Sex",
        "source_batch",
        "source_label",
        "roi_signal_path",
        "raw_shape",
        "original_n_timepoints",
        "locked_n_timepoints_used",
        "retain_locked_140TR_reference",
        "retain_all_available_timepoints_if_overridden",
        "retain_fixed_160TR_if_overridden",
        "retain_fixed_180TR_if_overridden",
        "pool_impact_fixed_160TR",
        "pool_impact_fixed_180TR",
        "branch_all_available_allowed",
        "branch_fixed_160_allowed",
        "branch_fixed_180_allowed",
        "motion_qc_fields_available",
        "motion_qc_note",
    ]
    return out[[c for c in cols if c in out.columns]].copy()


def summarize_branch(subjects: pd.DataFrame, branch_name: str, retain_col: str) -> dict[str, Any]:
    retained = subjects[subjects[retain_col]]
    classifier = subjects[subjects["ResearchGroup_Mapped"].isin(["CN", "AD"])]
    classifier_retained = retained[retained["ResearchGroup_Mapped"].isin(["CN", "AD"])]
    row: dict[str, Any] = {
        "branch_name": branch_name,
        "n_vae_pool_total": int(len(subjects)),
        "n_vae_pool_retained": int(len(retained)),
        "n_classifier_total": int(len(classifier)),
        "n_classifier_retained": int(len(classifier_retained)),
        "planned_tensor_shape_if_built": f"({len(retained)}, 3, 131, 131)",
        "exact_locked_subject_pool_preserved": bool(len(retained) == len(subjects)),
    }
    for dx in ["CN", "AD", "MCI"]:
        total = int((subjects["ResearchGroup_Mapped"] == dx).sum())
        kept = int((retained["ResearchGroup_Mapped"] == dx).sum())
        row[f"{dx}_retained"] = kept
        row[f"{dx}_total"] = total
        row[f"{dx}_pct_retained"] = float(kept / total) if total else np.nan
    return row


def build_channel_summary(subjects: pd.DataFrame, recommendation: str) -> pd.DataFrame:
    statuses = branch_statuses(recommendation)
    branch_rows = [
        summarize_branch(subjects, "locked_140TR_reference", "retain_locked_140TR_reference"),
        summarize_branch(
            subjects,
            "all_available_timepoints",
            "retain_all_available_timepoints_if_overridden",
        ),
        summarize_branch(subjects, "fixed_160TR", "retain_fixed_160TR_if_overridden"),
        summarize_branch(subjects, "fixed_180TR", "retain_fixed_180TR_if_overridden"),
    ]
    rows: list[dict[str, Any]] = []
    for branch in branch_rows:
        status = statuses[branch["branch_name"]]
        for channel in CHANNELS:
            row = dict(branch)
            row.update(channel)
            row.update(
                {
                    "branch_type": status["branch_type"],
                    "required_n_TR": status["required_n_TR"],
                    "variable_length": status["variable_length"],
                    "guardrail_status": status["guardrail_status"],
                    "allowed_by_confounding_audit": status["allowed_by_confounding_audit"],
                    "guardrail_reason": status["reason"],
                    "roi_order": "ADNI locked 131 ROI order",
                    "metadata_policy": "preserve v5.1b metadata; no existing metadata files modified",
                    "split_policy": "preserve ResearchGroup_Mapped + Manufacturer split logic; rerun splits only in new branch if built",
                }
            )
            if branch["branch_name"] == "locked_140TR_reference":
                row["build_recommendation"] = "reference_no_rebuild"
            elif row["allowed_by_confounding_audit"]:
                row["build_recommendation"] = "eligible_for_explicit_confirm_build"
            else:
                row["build_recommendation"] = "blocked_by_guardrail"
            rows.append(row)
    return pd.DataFrame(rows)


def write_tensor_shape_plan(summary: pd.DataFrame, recommendation: str) -> None:
    branch_summary = (
        summary.drop_duplicates("branch_name")[
            [
                "branch_name",
                "guardrail_status",
                "required_n_TR",
                "n_vae_pool_retained",
                "n_classifier_retained",
                "planned_tensor_shape_if_built",
                "exact_locked_subject_pool_preserved",
                "guardrail_reason",
            ]
        ]
        .sort_values("branch_name")
        .reset_index(drop=True)
    )
    text = [
        "# Planned Tensor Shapes",
        "",
        f"Confounding-audit recommendation: **{recommendation}**.",
        "",
        "Only channels `[1,0,2]` are in scope, so every candidate tensor would have shape "
        "`N x 3 x 131 x 131` after connectome construction.",
        "",
        branch_summary.to_markdown(index=False),
        "",
        "No tensor was built in this preflight.",
    ]
    (OUT_DIR / "planned_tensor_shape.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def write_guardrail_summary(
    recommendation: str, retention: pd.DataFrame, confounding_tests: pd.DataFrame
) -> None:
    important_tests = confounding_tests[
        confounding_tests["test_name"].isin(
            [
                "welch_t_CN_vs_AD",
                "mann_whitney_CN_vs_AD",
                "logit_AD_vs_CN_adjusted",
                "kruskal_by_Manufacturer",
                "kruskal_by_SiteCode",
            ]
        )
    ].copy()
    text = [
        "# Confounding Guardrail Summary",
        "",
        f"Prior audit recommendation: **{recommendation}**.",
        "",
        "The guardrail blocks variable all-available-timepoint connectomes unless timepoint count is "
        "not associated with diagnosis, Manufacturer, or SiteCode. That condition is not met.",
        "",
        "Fixed longer-length branches avoid variable-length connectomes but do not preserve the locked "
        "subject pool. In the current audit, 160/180 TR retain only 321/396 CN/AD classifier subjects "
        "and disproportionately drop AD subjects.",
        "",
        "## Key Tests",
        "",
        important_tests.to_markdown(index=False),
        "",
        "## Retention Summary",
        "",
        retention.to_markdown(index=False),
    ]
    (OUT_DIR / "confounding_guardrail_summary.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def write_dry_run_report(summary: pd.DataFrame, recommendation: str) -> None:
    eligible = summary[
        (summary["branch_type"] == "candidate") & (summary["allowed_by_confounding_audit"])
    ]["branch_name"].unique()
    blocked = summary[
        (summary["branch_type"] == "candidate") & (~summary["allowed_by_confounding_audit"])
    ]["branch_name"].unique()
    text = [
        "# Dry-Run Report",
        "",
        "Mode: **preflight only**.",
        "",
        "- No ROI matrices were rewritten.",
        "- No connectomes were computed.",
        "- No tensors were built.",
        "- No training or scoring was launched.",
        "- Existing locked tensors, metadata, ledgers, configs, and model outputs were not modified.",
        "",
        f"Input guardrail recommendation: **{recommendation}**.",
        "",
        f"Eligible candidate branches under guardrail: {list(eligible) if len(eligible) else 'none'}.",
        f"Blocked/not-preferred candidate branches: {list(blocked)}.",
        "",
        "Dry-run decision: **do not launch a longer-time-series rebuild from this preflight**. "
        "The only branch preserving the exact 646-subject VAE pool is all-available timepoints, "
        "which is blocked by diagnosis/manufacturer/site confounding. Fixed 160/180 TR branches "
        "are less confounded in principle but shrink the cohort and change class balance.",
        "",
        "A future rebuild would need an explicit scientific override and a new analysis plan that treats "
        "the resulting cohort as a sensitivity analysis, not as a direct replacement for the locked 140-TR model.",
    ]
    (OUT_DIR / "dry_run_report.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = datetime.now().isoformat(timespec="seconds")
    subjects = pd.read_csv(AUDIT_DIR / "subject_timepoint_table.csv")
    retention = pd.read_csv(AUDIT_DIR / "candidate_length_retention.csv")
    confounding_tests = pd.read_csv(AUDIT_DIR / "confounding_tests.csv")
    recommendation = load_audit_recommendation()

    planned_subjects = build_subject_retention(subjects, recommendation)
    write_table(planned_subjects, "planned_subject_retention")

    channel_summary = build_channel_summary(planned_subjects, recommendation)
    write_table(channel_summary, "planned_channel_build_summary")

    write_tensor_shape_plan(channel_summary, recommendation)
    write_guardrail_summary(recommendation, retention, confounding_tests)
    write_dry_run_report(channel_summary, recommendation)

    command_log = {
        "script": str(Path(__file__).resolve()),
        "started_at": started,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "read_only_dry_run_preflight",
        "inputs": {
            "audit_dir": str(AUDIT_DIR),
            "subject_timepoint_table": str(AUDIT_DIR / "subject_timepoint_table.csv"),
            "candidate_length_retention": str(AUDIT_DIR / "candidate_length_retention.csv"),
            "confounding_tests": str(AUDIT_DIR / "confounding_tests.csv"),
        },
        "outputs": [
            "planned_subject_retention.csv/.md",
            "planned_channel_build_summary.csv/.md",
            "planned_tensor_shape.md",
            "confounding_guardrail_summary.md",
            "dry_run_report.md",
            "command_log.json",
        ],
        "channels_to_build_if_unblocked": [1, 0, 2],
        "recommendation_from_timepoint_audit": recommendation,
        "eligible_candidate_branches": sorted(
            set(
                channel_summary.loc[
                    (channel_summary["branch_type"] == "candidate")
                    & (channel_summary["allowed_by_confounding_audit"]),
                    "branch_name",
                ]
            )
        ),
        "no_build_launched": True,
        "no_training_launched": True,
        "no_existing_data_modified": True,
    }
    (OUT_DIR / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
    print(f"Wrote preflight package to {OUT_DIR}")
    print(f"Guardrail recommendation: {recommendation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
