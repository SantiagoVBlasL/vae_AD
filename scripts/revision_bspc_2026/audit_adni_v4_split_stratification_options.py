#!/usr/bin/env python3
"""Evaluate ADNI v4 CN/AD outer-fold stratification options.

This is a metadata-only audit. It does not train models or load tensors.
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA = PROJECT_ROOT / "data/revision_bspc_2026/adni_expanded_v4_all_available/subject_metadata_adni_expanded_v4_all_available.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/auc_sprint_adni_only/split_stratification_audit"

SCHEMES = [
    {
        "scheme": "A_current_rg_sex",
        "description": "current: ResearchGroup_Mapped + Sex",
        "strat_cols": ["ResearchGroup_Mapped", "Sex"],
        "sex_stratified": True,
    },
    {
        "scheme": "B_proposed_rg_manufacturer",
        "description": "proposed: ResearchGroup_Mapped + Manufacturer",
        "strat_cols": ["ResearchGroup_Mapped", "Manufacturer"],
        "sex_stratified": False,
    },
    {
        "scheme": "C_extended_rg_manufacturer_sex",
        "description": "proposed extended: ResearchGroup_Mapped + Manufacturer + Sex",
        "strat_cols": ["ResearchGroup_Mapped", "Manufacturer", "Sex"],
        "sex_stratified": True,
    },
    {
        "scheme": "D_relaxed_rg_manufacturer_report_sex",
        "description": "proposed relaxed: ResearchGroup_Mapped + Manufacturer; Sex reported only",
        "strat_cols": ["ResearchGroup_Mapped", "Manufacturer"],
        "sex_stratified": False,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    path = resolve(path)
    path.mkdir(parents=True, exist_ok=True)
    generated = [
        "scheme_stratum_counts.csv",
        "fold_balance_summary.csv",
        "fold_category_counts.csv",
        "imbalance_scores.csv",
        "scheme_recommendation.csv",
        "README.md",
        "audit_manifest.json",
    ]
    existing = [path / name for name in generated if (path / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{path} already contains audit outputs; pass --overwrite")
    if overwrite:
        for file_path in existing:
            file_path.unlink()
    return path


def normalize_metadata(df: pd.DataFrame) -> pd.DataFrame:
    required = ["SubjectID", "ResearchGroup_Mapped", "Manufacturer", "Sex", "Age"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required metadata columns: {missing}")
    out = df.copy()
    for col in ["SubjectID", "ResearchGroup_Mapped", "Manufacturer", "Sex"]:
        out[col] = out[col].fillna(f"{col}_Unknown").astype(str).str.strip()
    out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
    if "Site3" in out.columns:
        out["Site3"] = out["Site3"].fillna("Site3_Unknown").astype(str)
    else:
        out["Site3"] = out["SubjectID"].str.extract(r"^(\d{3})", expand=False).fillna("Site3_Unknown")
    return out


def stratum_key(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    temp = df[cols].copy()
    for col in cols:
        temp[col] = temp[col].fillna(f"{col}_Unknown").astype(str)
    return temp.apply(lambda row: "_".join(row.values.astype(str)), axis=1)


def category_counts(fold_df: pd.DataFrame, scheme: str, fold: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for variable in ["ResearchGroup_Mapped", "Manufacturer", "Sex", "Site3"]:
        if variable == "ResearchGroup_Mapped":
            counts = fold_df.groupby([variable], dropna=False).size().reset_index(name="n")
            counts["diagnosis"] = counts[variable]
        else:
            counts = fold_df.groupby([variable, "ResearchGroup_Mapped"], dropna=False).size().reset_index(name="n")
            counts = counts.rename(columns={"ResearchGroup_Mapped": "diagnosis"})
        for _, row in counts.iterrows():
            rows.append(
                {
                    "scheme": scheme,
                    "fold": fold,
                    "variable": variable,
                    "value": row[variable],
                    "diagnosis": row["diagnosis"],
                    "n": int(row["n"]),
                }
            )
    return rows


def fold_summary(fold_df: pd.DataFrame, scheme: str, fold: int) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "scheme": scheme,
        "fold": fold,
        "n": int(len(fold_df)),
        "n_CN": int((fold_df["ResearchGroup_Mapped"] == "CN").sum()),
        "n_AD": int((fold_df["ResearchGroup_Mapped"] == "AD").sum()),
        "age_mean": float(fold_df["Age"].mean()),
        "age_std": float(fold_df["Age"].std(ddof=1)),
        "n_manufacturers": int(fold_df["Manufacturer"].nunique()),
        "n_sites": int(fold_df["Site3"].nunique()),
        "n_F": int((fold_df["Sex"] == "F").sum()),
        "n_M": int((fold_df["Sex"] == "M").sum()),
    }
    for manufacturer, count in fold_df["Manufacturer"].value_counts().items():
        key = str(manufacturer).replace(" ", "_").replace("/", "_")
        row[f"manufacturer_{key}"] = int(count)
    return row


def max_count_range_by_value(fold_counts: pd.DataFrame, variable: str) -> int:
    subset = fold_counts[fold_counts["variable"] == variable]
    if subset.empty:
        return 0
    ranges: List[int] = []
    for (_value, _dx), grp in subset.groupby(["value", "diagnosis"]):
        per_fold = grp.set_index("fold")["n"].reindex(range(1, 6), fill_value=0)
        ranges.append(int(per_fold.max() - per_fold.min()))
    return max(ranges) if ranges else 0


def simulate_scheme(df: pd.DataFrame, scheme: Dict[str, Any], n_splits: int, seed: int) -> Dict[str, Any]:
    key = stratum_key(df, scheme["strat_cols"])
    stratum_counts = key.value_counts().sort_index()
    feasible = bool((stratum_counts >= n_splits).all())
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_rows: List[Dict[str, Any]] = []
    fold_count_rows: List[Dict[str, Any]] = []
    warnings_seen: List[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        splits = list(splitter.split(np.zeros(len(df)), key))
        warnings_seen = [str(w.message) for w in caught]
    for fold, (_train_idx, test_idx) in enumerate(splits, start=1):
        fold_df = df.iloc[test_idx].copy()
        fold_rows.append(fold_summary(fold_df, scheme["scheme"], fold))
        fold_count_rows.extend(category_counts(fold_df, scheme["scheme"], fold))
    fold_summary_df = pd.DataFrame(fold_rows)
    fold_counts_df = pd.DataFrame(fold_count_rows)
    imbalance = {
        "scheme": scheme["scheme"],
        "description": scheme["description"],
        "stratification_cols": "+".join(scheme["strat_cols"]),
        "min_stratum_count": int(stratum_counts.min()),
        "n_strata": int(stratum_counts.shape[0]),
        "feasible_5fold": feasible,
        "warnings": " | ".join(warnings_seen),
        "cn_fold_range": int(fold_summary_df["n_CN"].max() - fold_summary_df["n_CN"].min()),
        "ad_fold_range": int(fold_summary_df["n_AD"].max() - fold_summary_df["n_AD"].min()),
        "age_mean_range": float(fold_summary_df["age_mean"].max() - fold_summary_df["age_mean"].min()),
        "sex_max_dx_count_range": max_count_range_by_value(fold_counts_df, "Sex"),
        "manufacturer_max_dx_count_range": max_count_range_by_value(fold_counts_df, "Manufacturer"),
        "site3_max_dx_count_range": max_count_range_by_value(fold_counts_df, "Site3"),
    }
    imbalance["combined_imbalance_score"] = (
        imbalance["cn_fold_range"]
        + imbalance["ad_fold_range"]
        + imbalance["manufacturer_max_dx_count_range"]
        + 0.5 * imbalance["sex_max_dx_count_range"]
        + 0.1 * imbalance["age_mean_range"]
    )
    stratum_df = stratum_counts.rename_axis("stratum").reset_index(name="n")
    stratum_df.insert(0, "scheme", scheme["scheme"])
    stratum_df.insert(1, "description", scheme["description"])
    return {
        "stratum_counts": stratum_df,
        "fold_summary": fold_summary_df,
        "fold_counts": fold_counts_df,
        "imbalance": imbalance,
    }


def recommendation(imbalance_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    feasible = imbalance_df[imbalance_df["feasible_5fold"]].copy()
    if feasible.empty:
        chosen = imbalance_df.sort_values("combined_imbalance_score").iloc[0]
        note = "No candidate scheme has all strata >= 5; use label-only or collect more data."
    else:
        preferred_names = ["B_proposed_rg_manufacturer", "D_relaxed_rg_manufacturer_report_sex"]
        preferred = feasible[feasible["scheme"].isin(preferred_names)].copy()
        pool = preferred if not preferred.empty else feasible
        chosen = pool.sort_values(["manufacturer_max_dx_count_range", "combined_imbalance_score"]).iloc[0]
        note = (
            "Use manufacturer in fold stratification and keep Sex as a reported covariate unless the extended "
            "Manufacturer+Sex scheme is clearly feasible and materially better."
        )
    for _, row in imbalance_df.iterrows():
        rows.append(
            {
                "scheme": row["scheme"],
                "recommended": bool(row["scheme"] == chosen["scheme"]),
                "reason": note if row["scheme"] == chosen["scheme"] else "",
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    outdir = prepare_output_dir(args.output_dir, args.overwrite)
    metadata_path = resolve(args.metadata_path)
    df = normalize_metadata(pd.read_csv(metadata_path))
    cn_ad = df[df["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy()
    if cn_ad["SubjectID"].duplicated().any():
        raise ValueError("Duplicate SubjectID rows found among CN/AD metadata.")

    all_strata: List[pd.DataFrame] = []
    all_folds: List[pd.DataFrame] = []
    all_counts: List[pd.DataFrame] = []
    all_imbalance: List[Dict[str, Any]] = []
    for scheme in SCHEMES:
        result = simulate_scheme(cn_ad, scheme, args.n_splits, args.seed)
        all_strata.append(result["stratum_counts"])
        all_folds.append(result["fold_summary"])
        all_counts.append(result["fold_counts"])
        all_imbalance.append(result["imbalance"])

    stratum_df = pd.concat(all_strata, ignore_index=True)
    fold_df = pd.concat(all_folds, ignore_index=True)
    count_df = pd.concat(all_counts, ignore_index=True)
    imbalance_df = pd.DataFrame(all_imbalance).sort_values("combined_imbalance_score")
    rec_df = recommendation(imbalance_df)

    stratum_df.to_csv(outdir / "scheme_stratum_counts.csv", index=False)
    fold_df.to_csv(outdir / "fold_balance_summary.csv", index=False)
    count_df.to_csv(outdir / "fold_category_counts.csv", index=False)
    imbalance_df.to_csv(outdir / "imbalance_scores.csv", index=False)
    rec_df.to_csv(outdir / "scheme_recommendation.csv", index=False)

    chosen = rec_df[rec_df["recommended"]].iloc[0]["scheme"]
    chosen_row = imbalance_df[imbalance_df["scheme"] == chosen].iloc[0]
    readme = [
        "# ADNI V4 Split Stratification Audit",
        "",
        "Metadata-only audit for CN/AD outer-fold split options.",
        "",
        f"- CN/AD subjects: {len(cn_ad)}",
        f"- CN: {(cn_ad['ResearchGroup_Mapped'] == 'CN').sum()}",
        f"- AD: {(cn_ad['ResearchGroup_Mapped'] == 'AD').sum()}",
        f"- Recommended scheme: `{chosen}`",
        f"- Recommended stratification columns: `{chosen_row['stratification_cols']}`",
        f"- Minimum stratum count: {int(chosen_row['min_stratum_count'])}",
        f"- 5-fold feasible: {bool(chosen_row['feasible_5fold'])}",
        "",
        "## Methodological Note",
        "",
        "Manufacturer should be used for fold stratification because it is a strong acquisition-domain variable. "
        "It should not be used as a predictive feature because that would encourage scanner/domain shortcut learning rather than diagnostic signal.",
        "",
        "This is a minimal methodological change: architecture, beta, latent dimensionality, dropout, final activation, selected channels, and metadata features remain fixed.",
        "",
        "LOSO/site-held-out validation should be run after model selection, because using LOSO during model selection would mix stress-test design with tuning and reduce interpretability.",
    ]
    (outdir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_path": str(metadata_path),
        "output_dir": str(outdir),
        "n_cn_ad": int(len(cn_ad)),
        "n_splits": args.n_splits,
        "seed": args.seed,
        "no_training": True,
    }
    (outdir / "audit_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("\n".join(readme))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
