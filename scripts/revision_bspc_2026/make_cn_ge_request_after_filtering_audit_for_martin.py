#!/usr/bin/env python3
"""Create the CN-GE request for Martin after filtering provenance audit.

Read-only with respect to tensors and training artifacts. The outputs are
communication tables only.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "final_cn_ge_inventory_before_martin_request"
)
FILTERING_AUDIT_DIR = (
    PROJECT_ROOT / "results" / "revision_bspc_2026" / "ge_cn_filtering_provenance_audit"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Martin CN-GE request after filtering provenance audit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--final-dir", type=Path, default=FINAL_DIR)
    parser.add_argument("--filtering-audit-dir", type=Path, default=FILTERING_AUDIT_DIR)
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def subject_metadata(inventory: pd.DataFrame) -> pd.DataFrame:
    cols = ["SubjectID", "ImageID", "Visit", "StudyDate", "Age", "Sex", "Manufacturer"]
    available = [c for c in cols if c in inventory.columns]
    meta = inventory[available].copy()
    meta = meta.drop_duplicates("SubjectID", keep="first").sort_values("SubjectID").reset_index(drop=True)
    return meta


def priority_map(final_dir: Path) -> Dict[str, str]:
    compact_path = final_dir / "cn_ge_request_compact_for_martin.csv"
    if not compact_path.exists():
        return {}
    compact = read_csv(compact_path)
    out: Dict[str, str] = {}
    for _, row in compact.iterrows():
        out[clean(row.get("SubjectID"))] = (
            "YES" if clean(row.get("priority_group")) == "priority_first_67" else "NO"
        )
    return out


def missing_reason(row: pd.Series) -> str:
    final_reason = clean(row.get("final_reason"))
    if final_reason:
        return final_reason
    return "No local ROISignals .mat/.txt were found in the audited local roots."


def action_for_audit_row(row: pd.Series) -> str:
    decision = clean(row.get("final_use_decision"))
    was_included = clean(row.get("was_included_in_v5_1_gecn9")).lower() == "true"
    if decision == "quarantine_needs_martin_confirmation":
        prefix = (
            "Current v5.1_gecn9 entry is provisional/quarantined; "
            if was_included
            else ""
        )
        return (
            prefix
            + "confirm with DPARSF/MATLAB logs that this first-visit AAL3 ROI signal was already "
            "bandpass filtered at 0.01-0.08 Hz with TR=3s before export; if that cannot be documented, "
            "re-export/reprocess the first-visit AAL3 ROISignals."
        )
    prefix = "Current v5.1_gecn9 entry is provisional/quarantined; " if was_included else ""
    return (
        prefix
        + "re-export/reprocess first-visit AAL3 ROISignals from DPARSF/MATLAB with 170 ROIs, "
        "TR=3s, passband 0.01-0.08 Hz already applied in MATLAB/DPARSF, finite signal, and "
        "10000-compatible scale."
    )


def action_for_missing() -> str:
    return (
        "process/export first-visit CN-GE ROISignals AAL3 with 170 ROIs, TR=3s, finite signal, "
        "10000-compatible scale, and MATLAB/DPARSF passband 0.01-0.08 Hz already applied."
    )


def request_category(row: pd.Series, audit_by_sid: Dict[str, pd.Series]) -> str:
    sid = clean(row.get("SubjectID"))
    audit = audit_by_sid.get(sid)
    if audit is None:
        return "no_roisignals_found_needs_preprocessing"
    was_included = clean(audit.get("was_included_in_v5_1_gecn9")).lower() == "true"
    if was_included:
        return "previously_added_to_v5_1_gecn9_but_now_quarantined"
    if clean(audit.get("final_use_decision")) == "quarantine_needs_martin_confirmation":
        return "local_but_martin_stage_confirmation_required"
    return "local_but_reprocess_required"


def action_category(audit: pd.Series | None) -> str:
    if audit is None:
        return "no_roisignals_found_needs_preprocessing"
    if clean(audit.get("final_use_decision")) == "quarantine_needs_martin_confirmation":
        return "local_but_martin_stage_confirmation_required"
    return "local_but_reprocess_required"


def build_outputs(final_dir: Path, filtering_audit_dir: Path, date: str) -> Dict[str, Path]:
    inventory = read_csv(final_dir / "cn_ge_final_inventory.csv")
    decisions = read_csv(filtering_audit_dir / "ge_cn_final_use_decision.csv")
    meta = subject_metadata(inventory)
    if len(meta) != 110:
        raise RuntimeError(f"Expected 110 CN-GE first-visit candidates, found {len(meta)}")

    audit_by_sid = {clean(row["SubjectID"]): row for _, row in decisions.iterrows()}
    priorities = priority_map(final_dir)
    representative_inventory = (
        inventory.drop_duplicates("SubjectID", keep="first").set_index("SubjectID", drop=False)
    )

    rows: List[Dict[str, str]] = []
    for _, base in meta.iterrows():
        sid = clean(base["SubjectID"])
        audit = audit_by_sid.get(sid)
        inv_row = representative_inventory.loc[sid] if sid in representative_inventory.index else pd.Series(dtype=str)
        category = request_category(base, audit_by_sid)
        priority_first_67 = priorities.get(sid, "NO")
        if audit is None:
            local_path = ""
            stage_guess = ""
            spectral_class = ""
            delta_class = ""
            reason = missing_reason(inv_row)
            action_needed = action_for_missing()
            final_decision = ""
            was_previously_added = "NO"
        else:
            local_path = clean(audit.get("best_local_path"))
            stage_guess = clean(audit.get("stage_guess"))
            spectral_class = clean(audit.get("spectrum_class"))
            delta_class = clean(audit.get("python_bandpass_delta_class"))
            reason = clean(audit.get("decision_reason"))
            action_needed = action_for_audit_row(audit)
            final_decision = clean(audit.get("final_use_decision"))
            was_previously_added = (
                "YES" if clean(audit.get("was_included_in_v5_1_gecn9")).lower() == "true" else "NO"
            )

        rows.append(
            {
                "SubjectID": sid,
                "request_category": category,
                "martin_action_category": action_category(audit),
                "was_previously_added_to_v5_1_gecn9": was_previously_added,
                "filtering_audit_decision": final_decision,
                "local_path": local_path,
                "stage_guess": stage_guess,
                "spectral_class": spectral_class,
                "python_bandpass_delta_class": delta_class,
                "reason": reason,
                "action_needed": action_needed,
                "priority_first_67": priority_first_67,
                "python_bandpass_requested": "NO",
            }
        )

    request = pd.DataFrame(rows).sort_values(
        ["priority_first_67", "request_category", "SubjectID"],
        ascending=[False, True, True],
    )
    if len(request) != 110:
        raise RuntimeError(f"Request table must have 110 rows, found {len(request)}")
    if set(request["python_bandpass_requested"]) != {"NO"}:
        raise RuntimeError("python_bandpass_requested must be NO for every row")

    quarantined = request[request["was_previously_added_to_v5_1_gecn9"].eq("YES")].copy()
    if len(quarantined) != 9:
        raise RuntimeError(f"Expected 9 previously added/quarantined local subjects, found {len(quarantined)}")

    request_path = final_dir / f"cn_ge_request_after_filtering_audit_{date}.csv"
    readme_path = final_dir / f"README_for_martin_after_filtering_audit_{date}.md"
    quarantine_path = final_dir / f"cn_ge_quarantined_local_subjects_{date}.csv"

    request.to_csv(request_path, index=False)
    quarantined.to_csv(quarantine_path, index=False)
    write_readme(readme_path, request, quarantined, date)
    return {
        "request": request_path,
        "readme": readme_path,
        "quarantined": quarantine_path,
    }


def write_readme(path: Path, request: pd.DataFrame, quarantined: pd.DataFrame, date: str) -> None:
    category_counts = request["request_category"].value_counts().to_dict()
    action_counts = request["martin_action_category"].value_counts().to_dict()
    q_reprocess = quarantined["martin_action_category"].eq("local_but_reprocess_required").sum()
    q_confirm = quarantined["martin_action_category"].eq("local_but_martin_stage_confirmation_required").sum()
    no_local = int(action_counts.get("no_roisignals_found_needs_preprocessing", 0))
    local_reprocess = int(action_counts.get("local_but_reprocess_required", 0))
    local_confirm = int(action_counts.get("local_but_martin_stage_confirmation_required", 0))
    priority_yes = int(request["priority_first_67"].eq("YES").sum())

    lines = [
        f"# CN-GE request after filtering/provenance audit ({date})",
        "",
        "This request supersedes the previous CN-GE request tables for the filtering/provenance question. It does not train models, build tensors, or modify existing tensors.",
        "",
        "## Summary",
        "",
        f"- CN-GE first-visit candidates in the request: `{len(request)}`.",
        f"- Rows marked `priority_first_67=YES`: `{priority_yes}`.",
        f"- No local ROISignals found: `{no_local}`.",
        f"- Local but reprocess/re-export required: `{local_reprocess}`.",
        f"- Local but Martin stage confirmation required: `{local_confirm}`.",
        f"- Previously added to `v5.1_gecn9` and now quarantined: `{len(quarantined)}`.",
        "",
        "## Why the 9 v5.1_gecn9 CN-GE are quarantined",
        "",
        f"The filtering/provenance audit found `0/9` previously added CN-GE safe for final v5.1 use. All 9 came from local ARWSDCF-looking paths, but the available local provenance did not document a MATLAB/DPARSF 0.01-0.08 Hz bandpass. Seven showed an unfiltered-like spectrum or material diagnostic Python-bandpass delta and should be re-exported/reprocessed. Two (`{', '.join(quarantined[quarantined['martin_action_category'].eq('local_but_martin_stage_confirmation_required')]['SubjectID'].tolist())}`) remain confirmation cases: they need DPARSF/MATLAB logs proving the exported ROISignals were already filtered, otherwise they also need re-export/reprocess.",
        "",
        "`v5.1_gecn9` should therefore be treated as provisional/quarantine, not as a final dataset.",
        "",
        "## Why the 10 other local CN-GE are not incorporated",
        "",
        "The 10 local CN-GE that were already rejected remain unusable for final v5.1 because their local files are CovRegressed/non-F-stage, ARWSDC/non-F-stage, wrong ROI count, or spectrally inconsistent with the known DPARSF passband reference. This is a preprocessing/stage compatibility problem, not a diagnosis problem.",
        "",
        "## What Martin should provide",
        "",
        "For every row in the CSV, Martin should provide or re-export the first-visit CN-GE ROISignals from DPARSF/MATLAB with:",
        "",
        "- AAL3, 170 ROIs.",
        "- TR=3s.",
        "- Bandpass 0.01-0.08 Hz already applied in MATLAB/DPARSF before ROISignal export.",
        "- Finite signal and scale compatible with the 10000-level DPARSF outputs.",
        "- Clear stage/provenance naming or logs showing the filtering stage.",
        "",
        "For the stage-confirmation rows, logs that prove the current local ARWSDCF ROISignals were already MATLAB/DPARSF filtered are acceptable; without that proof, those subjects should be re-exported/reprocessed too.",
        "",
        "## Python bandpass",
        "",
        "Python bandpass is not requested and is not part of the final recommended path. We will compute connectivity without applying Python bandpass. Python bandpass was used only as a diagnostic sensitivity test in the audit.",
        "",
        "## Files",
        "",
        f"- `cn_ge_request_after_filtering_audit_{date}.csv`",
        f"- `cn_ge_quarantined_local_subjects_{date}.csv`",
        f"- `README_for_martin_after_filtering_audit_{date}.md`",
        "",
        "## Request category counts",
        "",
    ]
    for key in sorted(category_counts):
        lines.append(f"- `{key}`: `{int(category_counts[key])}`")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    outputs = build_outputs(args.final_dir, args.filtering_audit_dir, args.date)
    print("Created Martin request after filtering audit:")
    for label, output_path in outputs.items():
        print(f"{label}: {output_path}")
    print("No training. No tensor construction. No tensor modification. Python bandpass requested: NO.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
