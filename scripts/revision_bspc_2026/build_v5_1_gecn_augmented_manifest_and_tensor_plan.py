#!/usr/bin/env python3
"""Build a candidate ADNI v5.1 manifest augmented with compatible local CN-GE signals.

This script does not compute a full tensor and does not train models. It creates
a separate candidate manifest under results/ and leaves the current v5 outputs
untouched.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIRST_VISIT = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_master_manifest"
    / "adni_v5_1_master_subject_manifest_first_visit.csv"
)
DEFAULT_AUDIT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_gecn_local_roisignals_audit"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_gecn_augmented_manifest"
)
DEFAULT_V5_TENSOR = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_dparsf10000_no_pybandpass/subject_tensors/"
    "GLOBAL_TENSOR_ADNI_expanded_v5_dparsf10000_no_pybandpass.npz"
)
DEFAULT_V5_TRAINING_METADATA = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_dparsf10000_no_pybandpass/"
    "training_ready_metadata_v5_dparsf10000_no_pybandpass.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build candidate v5.1 manifest augmented with compatible local CN-GE signals.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--first-visit-manifest", type=Path, default=DEFAULT_FIRST_VISIT)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--v5-tensor", type=Path, default=DEFAULT_V5_TENSOR)
    parser.add_argument("--v5-training-metadata", type=Path, default=DEFAULT_V5_TRAINING_METADATA)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(f"Output directory exists and is not empty; pass --overwrite: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def clean_string(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def yes_mask(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.lower().isin({"yes", "true", "1"})


def is_cn_ge(df: pd.DataFrame) -> pd.Series:
    return df["ResearchGroup_Mapped"].fillna("").eq("CN") & df["Manufacturer"].fillna("").eq("GE")


def load_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def augment_manifest(first: pd.DataFrame, recommended: pd.DataFrame) -> pd.DataFrame:
    out = first.copy()
    out["gecn_augmented_candidate"] = "no"
    out["gecn_augmented_status"] = ""
    out["gecn_augmented_warning"] = ""
    out["gecn_source_from_audit"] = ""
    out["gecn_previous_preprocessing_status"] = out.get("preprocessing_status", "")
    out["gecn_previous_action_needed"] = out.get("action_needed", "")
    rec_by_sid = recommended.drop_duplicates("SubjectID", keep="first").set_index("SubjectID", drop=False)
    for idx, row in out.iterrows():
        sid = row["SubjectID"]
        if sid not in rec_by_sid.index:
            continue
        rec = rec_by_sid.loc[sid]
        compat = clean_string(rec.get("recommended_compatibility", ""))
        if compat == "needs_preprocessing":
            continue
        warning = compat == "direct_compatible_with_warning"
        has_path = bool(clean_string(rec.get("recommended_path", "")))
        out.at[idx, "has_roisignals"] = has_path
        out.at[idx, "roisignals_path"] = rec.get("recommended_path", "")
        out.at[idx, "source_root"] = rec.get("source_root", "")
        out.at[idx, "stage_guess"] = rec.get("recommended_stage_guess", "")
        for col in [
            "shape",
            "n_timepoints",
            "n_rois",
            "finite_fraction",
            "nan_count",
            "columns_all_nan",
            "mean",
            "std",
            "median",
            "min",
            "max",
            "scale_label",
            "bandpassed_like",
            "energy_below_0p01",
            "energy_0p01_0p08",
            "energy_above_0p08",
        ]:
            if col in rec.index:
                out.at[idx, col] = rec.get(col, "")
        out.at[idx, "gecn_augmented_candidate"] = "yes"
        out.at[idx, "gecn_augmented_status"] = compat
        out.at[idx, "gecn_augmented_warning"] = "yes" if warning else "no"
        out.at[idx, "gecn_source_from_audit"] = rec.get("recommended_path", "")
        if compat not in {"direct_compatible", "direct_compatible_with_warning"}:
            out.at[idx, "compatible_for_v5_1_direct"] = "no"
            out.at[idx, "preprocessing_status"] = "exclude" if compat == "reject" else compat
            out.at[idx, "action_needed"] = rec.get("compatibility_reason", compat)
            continue
        out.at[idx, "compatible_for_v5_1_direct"] = "yes"
        out.at[idx, "preprocessing_status"] = (
            "ready_for_v5_1_direct_gecn_warning" if warning else "ready_for_v5_1_direct_gecn_clean"
        )
        out.at[idx, "action_needed"] = (
            "stage_warning_existing_roisignals_selected" if warning else "ready_for_v5_1_direct"
        )
    return out


def balance_summary(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    working = df.copy()
    for col in ["ResearchGroup_Mapped", "Manufacturer", "Sex", "compatible_for_v5_1_direct", "preprocessing_status"]:
        if col not in working.columns:
            working[col] = "UNKNOWN"
        working[col] = working[col].fillna("").replace("", "UNKNOWN")

    def add_counts(summary_type: str, group_cols: List[str], sub: pd.DataFrame) -> None:
        for keys, part in sub.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {"dataset": label, "summary_type": summary_type, "n": len(part)}
            for idx, key in enumerate(keys, start=1):
                row[f"group{idx}"] = key
            rows.append(row)

    add_counts("Diagnosis", ["ResearchGroup_Mapped"], working)
    add_counts("Manufacturer", ["Manufacturer"], working)
    add_counts("Diagnosis_x_Manufacturer", ["ResearchGroup_Mapped", "Manufacturer"], working)
    add_counts("Sex", ["Sex"], working)
    add_counts("preprocessing_status", ["preprocessing_status"], working)
    add_counts("direct_compatible_yes_no", ["compatible_for_v5_1_direct"], working)
    direct = working[working["compatible_for_v5_1_direct"].eq("yes")].copy()
    add_counts("Direct_Diagnosis", ["ResearchGroup_Mapped"], direct)
    add_counts("Direct_Diagnosis_x_Manufacturer", ["ResearchGroup_Mapped", "Manufacturer"], direct)
    age_df = working.copy()
    age_df["Age_num"] = pd.to_numeric(age_df.get("Age", ""), errors="coerce")
    for dx, sub in age_df.groupby(age_df["ResearchGroup_Mapped"].fillna("UNKNOWN")):
        rows.append(
            {
                "dataset": label,
                "summary_type": "Age_by_Diagnosis",
                "group1": dx,
                "n": int(sub["Age_num"].notna().sum()),
                "age_mean": float(sub["Age_num"].mean()) if sub["Age_num"].notna().any() else np.nan,
                "age_median": float(sub["Age_num"].median()) if sub["Age_num"].notna().any() else np.nan,
                "age_min": float(sub["Age_num"].min()) if sub["Age_num"].notna().any() else np.nan,
                "age_max": float(sub["Age_num"].max()) if sub["Age_num"].notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def tensor_plan(
    output_dir: Path,
    first: pd.DataFrame,
    augmented: pd.DataFrame,
    sources_to_add: pd.DataFrame,
    remaining: pd.DataFrame,
    v5_tensor: Path,
    v5_training_metadata: Path,
) -> None:
    v5_subjects = set()
    if v5_training_metadata.exists():
        v5_meta = pd.read_csv(v5_training_metadata, dtype=str, keep_default_na=False)
        v5_subjects = set(v5_meta.get("SubjectID", pd.Series(dtype=str)))
    add_subjects = sorted(set(sources_to_add["SubjectID"]) - v5_subjects) if not sources_to_add.empty else []
    already_in_v5 = sorted(set(sources_to_add["SubjectID"]) & v5_subjects) if not sources_to_add.empty else []
    lines = [
        "# ADNI v5.1 GECN Tensor Plan",
        "",
        "No full tensor was computed by this script. Current v5 outputs are not overwritten.",
        "",
        "## Method",
        "",
        "- Python bandpass: OFF.",
        "- Input: DPARSF/MATLAB ROISignals centered around global mean 10000 when compatible.",
        "- Primary set: first visit only.",
        "- ROI reduction/reorder: same v5 no-Python-bandpass extraction code, 170 -> 131, Yeo-17 order.",
        "- Channels for final extraction: all 7 existing channels.",
        "",
        "## Current Inputs",
        "",
        f"- Existing v5 tensor: `{v5_tensor}`.",
        f"- Existing v5 training metadata: `{v5_training_metadata}`.",
        f"- Augmented manifest: `{output_dir / 'adni_v5_1_gecn_augmented_first_visit_manifest.csv'}`.",
        "",
        "## New Subjects To Add Relative To Current v5 Training Metadata",
        "",
        f"- CN-GE source rows selected for add: `{len(sources_to_add)}`.",
        f"- Selected subjects already in current v5 metadata: `{len(already_in_v5)}`.",
        f"- Selected subjects absent from current v5 metadata: `{len(add_subjects)}`.",
        f"- New subject IDs: `{', '.join(add_subjects) if add_subjects else 'none'}`.",
        "",
        "## Remaining Blockers",
        "",
        f"- Remaining non-direct first-visit subjects after augmentation: `{len(remaining)}`.",
        "",
        "## Recommended Next Command",
        "",
        "`/home/diego/anaconda3/envs/vae_ad/bin/python scripts/revision_bspc_2026/compare_v5_vs_v5_1_gecn_balance.py --overwrite`",
        "",
        "After scientific review of warning rows, run only a smoke extraction from the augmented manifest before any full tensor build.",
    ]
    (output_dir / "adni_v5_1_gecn_tensor_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(
    output_dir: Path,
    first: pd.DataFrame,
    augmented: pd.DataFrame,
    recommended: pd.DataFrame,
    sources_to_add: pd.DataFrame,
    remaining: pd.DataFrame,
    before_after_summary: pd.DataFrame,
) -> None:
    universe = first[is_cn_ge(first)].copy()
    found = recommended[recommended["recommended_path"].fillna("").astype(bool)] if "recommended_path" in recommended else pd.DataFrame()
    clean = int(sources_to_add["recommended_compatibility"].eq("direct_compatible").sum()) if not sources_to_add.empty else 0
    warn = int(sources_to_add["recommended_compatibility"].eq("direct_compatible_with_warning").sum()) if not sources_to_add.empty else 0
    remaining_cn_ge = augmented[is_cn_ge(augmented) & ~yes_mask(augmented["compatible_for_v5_1_direct"])]
    before_direct = int(yes_mask(first["compatible_for_v5_1_direct"]).sum())
    after_direct = int(yes_mask(augmented["compatible_for_v5_1_direct"]).sum())
    status = "YES" if len(remaining) == 0 and warn == 0 else ("PARTIAL" if clean + warn > 0 else "NO")
    dxman = before_after_summary[
        before_after_summary["summary_type"].eq("Direct_Diagnosis_x_Manufacturer")
    ][["dataset", "n", "group1", "group2"]]
    dxman_text = (
        "```text\n" + dxman.to_string(index=False) + "\n```"
        if not dxman.empty
        else "No diagnosis x manufacturer summary available."
    )
    lines = [
        "# ADNI v5.1 GECN Augmented Manifest",
        "",
        "Separate candidate manifest. No current v5 outputs were overwritten. No full tensor and no training were run.",
        "",
        "## Explicit Answers",
        "",
        f"1. CN-GE first-visit subjects in v5.1 universe: `{len(universe)}`.",
        f"2. CN-GE subjects physically found as ROISignals: `{found['SubjectID'].nunique() if not found.empty else 0}`.",
        f"3. CN-GE direct-compatible clean: `{clean}`.",
        f"4. CN-GE direct-compatible with warning: `{warn}`.",
        f"5. CN-GE still needing confirmation/preprocessing/rejection resolution: `{len(remaining_cn_ge)}`.",
        f"6. Direct-compatible first-visit subjects after incorporating compatible CN-GE: `{after_direct}` (`{before_direct}` before).",
        "7. Diagnosis x Manufacturer before/after is summarized below and fully in `adni_v5_1_gecn_balance_summary.csv`.",
        "",
        dxman_text,
        "",
        f"8. Can we build tensor v5.1 now? `{status}`.",
        f"9. Exact remaining request list for Martin: `{output_dir / 'adni_v5_1_gecn_remaining_request_for_martin.csv'}` (`{len(remaining)}` rows).",
        "10. Recommended next command: `/home/diego/anaconda3/envs/vae_ad/bin/python scripts/revision_bspc_2026/compare_v5_vs_v5_1_gecn_balance.py --overwrite`.",
        "",
        "## Output Files",
        "",
        "- `adni_v5_1_gecn_augmented_first_visit_manifest.csv`",
        "- `adni_v5_1_gecn_sources_to_add.csv`",
        "- `adni_v5_1_gecn_remaining_request_for_martin.csv`",
        "- `adni_v5_1_gecn_balance_summary.csv`",
        "- `adni_v5_1_gecn_tensor_plan.md`",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    prepare_output_dir(args.output_dir, args.overwrite)
    first = load_required_csv(args.first_visit_manifest)
    recommended = load_required_csv(args.audit_dir / "ge_cn_recommended_roisignals.csv")
    augmented = augment_manifest(first, recommended)

    sources_to_add = recommended[recommended["recommended_to_add"].fillna("").eq("yes")].copy()
    remaining = augmented[~yes_mask(augmented["compatible_for_v5_1_direct"])].copy()

    before_summary = balance_summary(first, "v5_1_master_before_gecn")
    after_summary = balance_summary(augmented, "v5_1_gecn_augmented_after")
    summary = pd.concat([before_summary, after_summary], ignore_index=True, sort=False)

    augmented.to_csv(args.output_dir / "adni_v5_1_gecn_augmented_first_visit_manifest.csv", index=False)
    sources_to_add.to_csv(args.output_dir / "adni_v5_1_gecn_sources_to_add.csv", index=False)
    remaining.to_csv(args.output_dir / "adni_v5_1_gecn_remaining_request_for_martin.csv", index=False)
    summary.to_csv(args.output_dir / "adni_v5_1_gecn_balance_summary.csv", index=False)
    tensor_plan(args.output_dir, first, augmented, sources_to_add, remaining, args.v5_tensor, args.v5_training_metadata)
    write_readme(args.output_dir, first, augmented, recommended, sources_to_add, remaining, summary)

    command = {
        "script": str(Path(__file__).resolve()),
        "first_visit_manifest": str(args.first_visit_manifest),
        "audit_dir": str(args.audit_dir),
        "output_dir": str(args.output_dir),
        "v5_tensor": str(args.v5_tensor),
        "v5_training_metadata": str(args.v5_training_metadata),
        "overwrite": bool(args.overwrite),
        "python_bandpass_applied": False,
        "full_tensor_computed": False,
        "training_run": False,
    }
    (args.output_dir / "command_log.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote augmented v5.1 GECN manifest to {args.output_dir}")
    print(
        "direct_before="
        f"{yes_mask(first['compatible_for_v5_1_direct']).sum()} "
        "direct_after="
        f"{yes_mask(augmented['compatible_for_v5_1_direct']).sum()} "
        f"gecn_sources_to_add={len(sources_to_add)} remaining_request={len(remaining)}"
    )
    print("No full tensor computed. No training run.")


if __name__ == "__main__":
    main()
