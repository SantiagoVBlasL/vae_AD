#!/usr/bin/env python3
"""Compare current v5 balance against the candidate v5.1 GECN-augmented manifest.

Read-only analysis. No tensor construction and no model training.
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
DEFAULT_V5_TRAINING_METADATA = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_dparsf10000_no_pybandpass/"
    "training_ready_metadata_v5_dparsf10000_no_pybandpass.csv"
)
DEFAULT_MASTER_FIRST_VISIT = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_master_manifest"
    / "adni_v5_1_master_subject_manifest_first_visit.csv"
)
DEFAULT_AUGMENTED = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_gecn_augmented_manifest"
    / "adni_v5_1_gecn_augmented_first_visit_manifest.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_gecn_balance_comparison"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare v5 and candidate v5.1 GECN-augmented balance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--v5-training-metadata", type=Path, default=DEFAULT_V5_TRAINING_METADATA)
    parser.add_argument("--master-first-visit", type=Path, default=DEFAULT_MASTER_FIRST_VISIT)
    parser.add_argument("--augmented-manifest", type=Path, default=DEFAULT_AUGMENTED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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


def canonical_manufacturer(value: Any) -> str:
    text = clean_string(value).upper()
    if "GE" in text:
        return "GE"
    if "PHILIPS" in text:
        return "Philips"
    if "SIEMENS" in text:
        return "SIEMENS"
    return clean_string(value) or "UNKNOWN"


def yes_mask(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.lower().isin({"yes", "true", "1"})


def canonical_dataset(df: pd.DataFrame, label: str, direct_only: bool = False) -> pd.DataFrame:
    out = df.copy()
    if direct_only and "compatible_for_v5_1_direct" in out.columns:
        out = out[yes_mask(out["compatible_for_v5_1_direct"])].copy()
    if "exclude_from_supervised" in out.columns:
        out = out[~yes_mask(out["exclude_from_supervised"])].copy()
    for col in ["SubjectID", "ResearchGroup_Mapped", "Manufacturer", "Sex", "Age"]:
        if col not in out.columns:
            out[col] = ""
    out["ResearchGroup_Mapped"] = out["ResearchGroup_Mapped"].fillna("").replace("", "UNKNOWN")
    out["Manufacturer"] = out["Manufacturer"].map(canonical_manufacturer).fillna("").replace("", "UNKNOWN")
    out["Sex"] = out["Sex"].fillna("").replace("", "UNKNOWN")
    out["dataset"] = label
    out = out.drop_duplicates("SubjectID", keep="first")
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    def add_counts(summary_type: str, group_cols: List[str], sub: pd.DataFrame) -> None:
        for keys, part in sub.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {"dataset": sub["dataset"].iloc[0] if not sub.empty else "", "summary_type": summary_type, "n": len(part)}
            for idx, key in enumerate(keys, start=1):
                row[f"group{idx}"] = key
            rows.append(row)

    if df.empty:
        return pd.DataFrame()
    add_counts("Diagnosis", ["ResearchGroup_Mapped"], df)
    add_counts("Manufacturer", ["Manufacturer"], df)
    add_counts("Diagnosis_x_Manufacturer", ["ResearchGroup_Mapped", "Manufacturer"], df)
    add_counts("Sex", ["Sex"], df)
    ad_cn = df[df["ResearchGroup_Mapped"].isin(["AD", "CN"])].copy()
    if not ad_cn.empty:
        add_counts("AD_CN_Diagnosis", ["ResearchGroup_Mapped"], ad_cn)
        add_counts("AD_CN_Diagnosis_x_Manufacturer", ["ResearchGroup_Mapped", "Manufacturer"], ad_cn)
    age_df = df.copy()
    age_df["Age_num"] = pd.to_numeric(age_df["Age"], errors="coerce")
    for dx, sub in age_df.groupby("ResearchGroup_Mapped", dropna=False):
        rows.append(
            {
                "dataset": df["dataset"].iloc[0],
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


def diagnosis_manufacturer_delta(summary: pd.DataFrame) -> pd.DataFrame:
    sub = summary[summary["summary_type"].eq("Diagnosis_x_Manufacturer")].copy()
    if sub.empty:
        return pd.DataFrame()
    pivot = sub.pivot_table(index=["group1", "group2"], columns="dataset", values="n", aggfunc="sum", fill_value=0)
    pivot = pivot.reset_index()
    if "v5_current_training_ready" in pivot.columns and "v5_1_gecn_augmented_direct" in pivot.columns:
        pivot["delta_augmented_minus_v5_current"] = (
            pivot["v5_1_gecn_augmented_direct"] - pivot["v5_current_training_ready"]
        )
    if "v5_1_master_direct_before_gecn" in pivot.columns and "v5_1_gecn_augmented_direct" in pivot.columns:
        pivot["delta_augmented_minus_master_before"] = (
            pivot["v5_1_gecn_augmented_direct"] - pivot["v5_1_master_direct_before_gecn"]
        )
    return pivot


def write_readme(output_dir: Path, datasets: Dict[str, pd.DataFrame], summary: pd.DataFrame, delta: pd.DataFrame) -> None:
    def count(label: str, dx: str = "") -> int:
        df = datasets.get(label, pd.DataFrame())
        if dx:
            return int(df["ResearchGroup_Mapped"].eq(dx).sum()) if not df.empty else 0
        return len(df)

    cn_ge_before = 0
    cn_ge_after = 0
    before = datasets.get("v5_1_master_direct_before_gecn", pd.DataFrame())
    after = datasets.get("v5_1_gecn_augmented_direct", pd.DataFrame())
    if not before.empty:
        cn_ge_before = int((before["ResearchGroup_Mapped"].eq("CN") & before["Manufacturer"].eq("GE")).sum())
    if not after.empty:
        cn_ge_after = int((after["ResearchGroup_Mapped"].eq("CN") & after["Manufacturer"].eq("GE")).sum())
    delta_text = "No delta table available."
    if not delta.empty:
        cols = [c for c in ["group1", "group2", "v5_current_training_ready", "v5_1_master_direct_before_gecn", "v5_1_gecn_augmented_direct", "delta_augmented_minus_master_before"] if c in delta.columns]
        delta_text = "```text\n" + delta[cols].to_string(index=False) + "\n```"
    lines = [
        "# v5 vs v5.1 GECN Balance Comparison",
        "",
        "Read-only comparison. No connectivity and no training were run.",
        "",
        "## Headline",
        "",
        f"- Current v5 training-ready subjects: `{count('v5_current_training_ready')}`.",
        f"- v5.1 master direct-compatible before GECN augmentation: `{count('v5_1_master_direct_before_gecn')}`.",
        f"- v5.1 GECN augmented direct-compatible subjects: `{count('v5_1_gecn_augmented_direct')}`.",
        f"- CN-GE direct-compatible before augmentation: `{cn_ge_before}`.",
        f"- CN-GE direct-compatible after augmentation: `{cn_ge_after}`.",
        f"- AD/CN after augmentation: `AD={count('v5_1_gecn_augmented_direct', 'AD')}`, `CN={count('v5_1_gecn_augmented_direct', 'CN')}`.",
        "",
        "## Diagnosis x Manufacturer Delta",
        "",
        delta_text,
        "",
        "## Outputs",
        "",
        "- `v5_vs_v5_1_gecn_balance_summary.csv`",
        "- `v5_vs_v5_1_gecn_diagnosis_manufacturer_delta.csv`",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    prepare_output_dir(args.output_dir, args.overwrite)
    v5 = pd.read_csv(args.v5_training_metadata, dtype=str, keep_default_na=False)
    master = pd.read_csv(args.master_first_visit, dtype=str, keep_default_na=False)
    augmented = pd.read_csv(args.augmented_manifest, dtype=str, keep_default_na=False)

    datasets = {
        "v5_current_training_ready": canonical_dataset(v5, "v5_current_training_ready"),
        "v5_1_master_direct_before_gecn": canonical_dataset(master, "v5_1_master_direct_before_gecn", direct_only=True),
        "v5_1_gecn_augmented_direct": canonical_dataset(augmented, "v5_1_gecn_augmented_direct", direct_only=True),
    }
    summary = pd.concat([summarize(df) for df in datasets.values()], ignore_index=True, sort=False)
    delta = diagnosis_manufacturer_delta(summary)
    summary.to_csv(args.output_dir / "v5_vs_v5_1_gecn_balance_summary.csv", index=False)
    delta.to_csv(args.output_dir / "v5_vs_v5_1_gecn_diagnosis_manufacturer_delta.csv", index=False)
    command = {
        "script": str(Path(__file__).resolve()),
        "v5_training_metadata": str(args.v5_training_metadata),
        "master_first_visit": str(args.master_first_visit),
        "augmented_manifest": str(args.augmented_manifest),
        "output_dir": str(args.output_dir),
        "overwrite": bool(args.overwrite),
        "python_bandpass_applied": False,
        "connectivity_computed": False,
        "training_run": False,
    }
    (args.output_dir / "command_log.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
    write_readme(args.output_dir, datasets, summary, delta)
    print(f"Wrote v5/v5.1 GECN balance comparison to {args.output_dir}")
    print(
        "subjects: "
        f"v5={len(datasets['v5_current_training_ready'])} "
        f"master_direct={len(datasets['v5_1_master_direct_before_gecn'])} "
        f"augmented_direct={len(datasets['v5_1_gecn_augmented_direct'])}"
    )
    print("No connectivity computed. No training run.")


if __name__ == "__main__":
    main()
