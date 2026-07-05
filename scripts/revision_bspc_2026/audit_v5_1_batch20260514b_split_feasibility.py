#!/usr/bin/env python3
"""Audit manufacturer-aware split feasibility for ADNI v5.1 batch20260514b.

Metadata-only audit. It does not train, load tensors, or modify ledgers.
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass/"
    "training_ready_metadata_v5_1_batch20260514b_no_pybandpass.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_batch20260514b_split_feasibility"
)

SCHEMES = [
    {
        "scheme": "A_current_equivalent_diagnosis_sex",
        "description": "Current/equivalent: Diagnosis + Sex",
        "strat_cols": ["Diagnosis", "Sex"],
    },
    {
        "scheme": "B_manufacturer_aware_diagnosis_manufacturer",
        "description": "Manufacturer-aware: Diagnosis + Manufacturer",
        "strat_cols": ["Diagnosis", "Manufacturer"],
    },
    {
        "scheme": "C_full_diagnosis_manufacturer_sex",
        "description": "Full: Diagnosis + Manufacturer + Sex",
        "strat_cols": ["Diagnosis", "Manufacturer", "Sex"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit v5.1 batch20260514b split feasibility.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def normalize_manufacturer(value: Any) -> str:
    text = clean(value).upper()
    if "GE" in text:
        return "GE"
    if "SIEMENS" in text:
        return "SIEMENS"
    if "PHILIPS" in text:
        return "Philips"
    return clean(value) or "UNKNOWN"


def normalize_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = ["SubjectID", "ResearchGroup_Mapped", "Diagnosis", "Manufacturer", "Sex", "Age"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"Metadata missing required columns: {missing}")
    out = df.copy()
    out["Diagnosis"] = out["ResearchGroup_Mapped"].map(clean)
    out["Manufacturer"] = out["Manufacturer"].map(normalize_manufacturer)
    out["Sex"] = out["Sex"].map(lambda x: clean(x) or "UNKNOWN")
    out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
    return out


def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    generated = [
        "cell_counts_diagnosis.csv",
        "cell_counts_diagnosis_x_manufacturer.csv",
        "cell_counts_diagnosis_x_manufacturer_x_sex.csv",
        "scheme_stratum_counts.csv",
        "fold_balance_summary.csv",
        "fold_cell_counts.csv",
        "scheme_feasibility_summary.csv",
        "split_assignment_preview.csv",
        "README.md",
        "command_log.json",
    ]
    existing = [path / name for name in generated if (path / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{path} already contains outputs; pass --overwrite")
    if overwrite:
        for file_path in existing:
            file_path.unlink()
    return path


def stratum_key(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    temp = df[list(cols)].copy()
    for col in cols:
        temp[col] = temp[col].map(lambda x: clean(x) or "UNKNOWN")
    return temp.apply(lambda row: " | ".join(row.astype(str).tolist()), axis=1)


def count_table(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    return (
        df.groupby(list(cols), dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(list(cols))
        .reset_index(drop=True)
    )


def fold_counts_for_variable(fold_df: pd.DataFrame, scheme: str, fold: int, variable: str) -> pd.DataFrame:
    if variable == "Diagnosis":
        table = count_table(fold_df, ["Diagnosis"])
    elif variable == "Manufacturer":
        table = count_table(fold_df, ["Manufacturer"])
    elif variable == "Sex":
        table = count_table(fold_df, ["Sex"])
    elif variable == "Diagnosis_x_Manufacturer":
        table = count_table(fold_df, ["Diagnosis", "Manufacturer"])
    elif variable == "Diagnosis_x_Manufacturer_x_Sex":
        table = count_table(fold_df, ["Diagnosis", "Manufacturer", "Sex"])
    else:
        raise ValueError(variable)
    table.insert(0, "fold", fold)
    table.insert(0, "scheme", scheme)
    table.insert(2, "variable", variable)
    return table


def fold_summary_row(fold_df: pd.DataFrame, scheme: str, fold: int, min_cell_count: int) -> Dict[str, Any]:
    classifier = fold_df[fold_df["Diagnosis"].isin(["AD", "CN"])]
    row: Dict[str, Any] = {
        "scheme": scheme,
        "fold": fold,
        "vae_pool_n": int(len(fold_df)),
        "vae_AD": int(fold_df["Diagnosis"].eq("AD").sum()),
        "vae_CN": int(fold_df["Diagnosis"].eq("CN").sum()),
        "vae_MCI": int(fold_df["Diagnosis"].eq("MCI").sum()),
        "classifier_n": int(len(classifier)),
        "classifier_AD": int(classifier["Diagnosis"].eq("AD").sum()),
        "classifier_CN": int(classifier["Diagnosis"].eq("CN").sum()),
        "sex_F": int(fold_df["Sex"].eq("F").sum()),
        "sex_M": int(fold_df["Sex"].eq("M").sum()),
        "minimum_stratification_cell_count": int(min_cell_count),
        "age_mean": float(fold_df["Age"].mean()),
        "age_std": float(fold_df["Age"].std(ddof=1)),
    }
    for manufacturer in ["GE", "SIEMENS", "Philips"]:
        key = "Siemens" if manufacturer == "SIEMENS" else manufacturer
        row[f"manufacturer_{key}"] = int(fold_df["Manufacturer"].eq(manufacturer).sum())
        row[f"AD_{key}"] = int((fold_df["Diagnosis"].eq("AD") & fold_df["Manufacturer"].eq(manufacturer)).sum())
        row[f"CN_{key}"] = int((fold_df["Diagnosis"].eq("CN") & fold_df["Manufacturer"].eq(manufacturer)).sum())
        row[f"MCI_{key}"] = int((fold_df["Diagnosis"].eq("MCI") & fold_df["Manufacturer"].eq(manufacturer)).sum())
    return row


def max_range(summary: pd.DataFrame, cols: Sequence[str]) -> Dict[str, int]:
    return {f"{col}_range": int(summary[col].max() - summary[col].min()) for col in cols if col in summary}


def simulate_scheme(df: pd.DataFrame, scheme: Dict[str, Any], n_splits: int, seed: int) -> Dict[str, pd.DataFrame | Dict[str, Any]]:
    key = stratum_key(df, scheme["strat_cols"])
    stratum_counts = key.value_counts().sort_index()
    min_cell = int(stratum_counts.min())
    feasible = bool(min_cell >= n_splits)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_rows: List[Dict[str, Any]] = []
    cell_count_tables: List[pd.DataFrame] = []
    assignments: List[pd.DataFrame] = []
    warnings_seen: List[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        splits = list(splitter.split(np.zeros(len(df)), key))
        warnings_seen = [str(w.message) for w in caught]
    for fold, (_train_idx, test_idx) in enumerate(splits, start=1):
        fold_df = df.iloc[test_idx].copy()
        fold_df["fold"] = fold
        fold_df["scheme"] = scheme["scheme"]
        fold_df["stratum"] = key.iloc[test_idx].values
        assignments.append(
            fold_df[["scheme", "fold", "SubjectID", "Diagnosis", "Manufacturer", "Sex", "Age", "stratum"]]
        )
        fold_rows.append(fold_summary_row(fold_df, scheme["scheme"], fold, min_cell))
        for variable in [
            "Diagnosis",
            "Manufacturer",
            "Sex",
            "Diagnosis_x_Manufacturer",
            "Diagnosis_x_Manufacturer_x_Sex",
        ]:
            cell_count_tables.append(fold_counts_for_variable(fold_df, scheme["scheme"], fold, variable))
    fold_summary = pd.DataFrame(fold_rows)
    fold_cells = pd.concat(cell_count_tables, ignore_index=True, sort=False)
    ranges = max_range(
        fold_summary,
        [
            "vae_AD",
            "vae_CN",
            "vae_MCI",
            "classifier_AD",
            "classifier_CN",
            "manufacturer_GE",
            "manufacturer_Siemens",
            "manufacturer_Philips",
            "sex_F",
            "sex_M",
        ],
    )
    summary = {
        "scheme": scheme["scheme"],
        "description": scheme["description"],
        "stratification_cols": "+".join(scheme["strat_cols"]),
        "n_strata": int(len(stratum_counts)),
        "minimum_cell_count": min_cell,
        "cells_below_5": int((stratum_counts < n_splits).sum()),
        "feasible_5fold_without_sparse_warning": feasible,
        "warnings": " | ".join(warnings_seen),
        **ranges,
    }
    summary["combined_balance_risk"] = (
        summary.get("vae_AD_range", 0)
        + summary.get("vae_CN_range", 0)
        + summary.get("vae_MCI_range", 0)
        + summary.get("manufacturer_GE_range", 0)
        + summary.get("manufacturer_Siemens_range", 0)
        + summary.get("manufacturer_Philips_range", 0)
        + 0.5 * (summary.get("sex_F_range", 0) + summary.get("sex_M_range", 0))
    )
    strata = stratum_counts.rename_axis("stratum").reset_index(name="n")
    strata.insert(0, "scheme", scheme["scheme"])
    strata.insert(1, "description", scheme["description"])
    return {
        "stratum_counts": strata,
        "fold_summary": fold_summary,
        "fold_cells": fold_cells,
        "assignments": pd.concat(assignments, ignore_index=True, sort=False),
        "summary": summary,
    }


def choose_recommendation(summaries: pd.DataFrame) -> str:
    row_b = summaries[summaries["scheme"].eq("B_manufacturer_aware_diagnosis_manufacturer")]
    row_c = summaries[summaries["scheme"].eq("C_full_diagnosis_manufacturer_sex")]
    if row_b.empty:
        return "no_B_scheme_available"
    b_safe = bool(row_b["minimum_cell_count"].iloc[0] >= 5)
    c_safe = bool(not row_c.empty and row_c["minimum_cell_count"].iloc[0] >= 5)
    if b_safe and not c_safe:
        return "prefer_B_manufacturer_aware; C_has_sparse_cells"
    if b_safe and c_safe:
        b_risk = float(row_b["combined_balance_risk"].iloc[0])
        c_risk = float(row_c["combined_balance_risk"].iloc[0])
        if c_risk <= b_risk:
            return "C_is_feasible_but_use_B_unless_sex_balance_is_required; C_adds_complexity"
        return "prefer_B_manufacturer_aware; C_feasible_but_not_better"
    return "A_current_equivalent_only; manufacturer_cells_too_sparse"


def write_readme(output_dir: Path, summaries: pd.DataFrame, cell_c: pd.DataFrame, recommendation: str) -> None:
    c_min = int(summaries.loc[summaries["scheme"].eq("C_full_diagnosis_manufacturer_sex"), "minimum_cell_count"].iloc[0])
    b_min = int(summaries.loc[summaries["scheme"].eq("B_manufacturer_aware_diagnosis_manufacturer"), "minimum_cell_count"].iloc[0])
    lines = [
        "# ADNI v5.1 batch20260514b Split Feasibility",
        "",
        "Metadata-only audit. No training, tensor access, or ledger modification.",
        "",
        "## Recommendation",
        "",
        f"- Decision: `{recommendation}`.",
        f"- B minimum Diagnosis x Manufacturer cell count: `{b_min}`.",
        f"- C minimum Diagnosis x Manufacturer x Sex cell count: `{c_min}`.",
        "- Preferred split for next experiment: B (`Diagnosis + Manufacturer`) unless the training code explicitly needs Sex in the outer split. Keep Sex as metadata/covariate and report fold-wise Sex counts.",
        "",
        "## Scheme Summary",
        "",
        "```text",
        summaries.to_string(index=False),
        "```",
        "",
        "## Sparsest Full-Split Cells",
        "",
        "```text",
        cell_c.sort_values("n").head(12).to_string(index=False),
        "```",
        "",
        "## Files",
        "",
        "- `cell_counts_diagnosis.csv`",
        "- `cell_counts_diagnosis_x_manufacturer.csv`",
        "- `cell_counts_diagnosis_x_manufacturer_x_sex.csv`",
        "- `scheme_stratum_counts.csv`",
        "- `fold_balance_summary.csv`",
        "- `fold_cell_counts.csv`",
        "- `scheme_feasibility_summary.csv`",
        "- `split_assignment_preview.csv`",
        "- `command_log.json`",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = prepare_output_dir(args.output_dir, args.overwrite)
    metadata = normalize_metadata(args.metadata)

    cell_diag = count_table(metadata, ["Diagnosis"])
    cell_dx_man = count_table(metadata, ["Diagnosis", "Manufacturer"])
    cell_dx_man_sex = count_table(metadata, ["Diagnosis", "Manufacturer", "Sex"])

    results = [simulate_scheme(metadata, scheme, args.n_splits, args.seed) for scheme in SCHEMES]
    stratum_counts = pd.concat([r["stratum_counts"] for r in results], ignore_index=True, sort=False)
    fold_summary = pd.concat([r["fold_summary"] for r in results], ignore_index=True, sort=False)
    fold_cells = pd.concat([r["fold_cells"] for r in results], ignore_index=True, sort=False)
    assignments = pd.concat([r["assignments"] for r in results], ignore_index=True, sort=False)
    summaries = pd.DataFrame([r["summary"] for r in results])
    recommendation = choose_recommendation(summaries)
    summaries["recommended_decision"] = recommendation
    summaries["recommended"] = summaries["scheme"].eq("B_manufacturer_aware_diagnosis_manufacturer") & ("B" in recommendation)

    cell_diag.to_csv(output_dir / "cell_counts_diagnosis.csv", index=False)
    cell_dx_man.to_csv(output_dir / "cell_counts_diagnosis_x_manufacturer.csv", index=False)
    cell_dx_man_sex.to_csv(output_dir / "cell_counts_diagnosis_x_manufacturer_x_sex.csv", index=False)
    stratum_counts.to_csv(output_dir / "scheme_stratum_counts.csv", index=False)
    fold_summary.to_csv(output_dir / "fold_balance_summary.csv", index=False)
    fold_cells.to_csv(output_dir / "fold_cell_counts.csv", index=False)
    summaries.to_csv(output_dir / "scheme_feasibility_summary.csv", index=False)
    assignments.to_csv(output_dir / "split_assignment_preview.csv", index=False)

    command_log = {
        "script": str(Path(__file__).resolve()),
        "generated": datetime.now().isoformat(timespec="seconds"),
        "metadata": str(args.metadata),
        "output_dir": str(output_dir),
        "n_subjects": int(len(metadata)),
        "n_splits": int(args.n_splits),
        "seed": int(args.seed),
        "recommendation": recommendation,
        "training_run": False,
        "tensor_loaded": False,
        "ledger_modified": False,
    }
    (output_dir / "command_log.json").write_text(
        json.dumps(command_log, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_readme(output_dir, summaries, cell_dx_man_sex, recommendation)

    print(f"output_dir={output_dir}")
    print(f"n_subjects={len(metadata)}")
    print(f"recommendation={recommendation}")
    print(summaries[["scheme", "minimum_cell_count", "feasible_5fold_without_sparse_warning", "combined_balance_risk"]].to_string(index=False))
    print("training_run=False")
    print("tensor_loaded=False")
    print("ledger_modified=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
