#!/usr/bin/env python3
"""Audit diagnostic/manufacturer/site balance for ADNI_expanded_v3_all_available."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA = (
    PROJECT_ROOT
    / "data"
    / "revision_bspc_2026"
    / "adni_expanded_v3_all_available"
    / "subject_metadata_adni_expanded_v3_all_available.csv"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "revision_bspc_2026" / "adni_expanded_v3_balance_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit balance tables and missing cells for ADNI_expanded_v3_all_available.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-cn-ad-per-manufacturer", type=int, default=5)
    return parser.parse_args()


def count_table(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    return df.groupby(list(cols), dropna=False).size().reset_index(name="n").sort_values("n", ascending=False)


def pivot(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    table = (
        df.groupby(list(cols) + ["ResearchGroup_Mapped"], dropna=False)
        .size()
        .unstack(fill_value=0)
    )
    for col in ["CN", "AD", "MCI"]:
        if col not in table.columns:
            table[col] = 0
    return table[["CN", "AD", "MCI"]].reset_index()


def build_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    tables = {
        "research_group_counts": count_table(df, ["ResearchGroup_Mapped"]),
        "manufacturer_by_research_group": pivot(df, ["Manufacturer"]),
        "source_by_research_group": pivot(df, ["SourceCohort"]),
        "site_by_research_group": pivot(df, ["Site3"]),
        "manufacturer_site_by_research_group": pivot(df, ["Manufacturer", "Site3"]),
    }
    ms = tables["manufacturer_site_by_research_group"].copy()
    tables["sites_only_cn_or_only_ad"] = ms[
        ((ms["CN"] > 0) & (ms["AD"] == 0)) | ((ms["AD"] > 0) & (ms["CN"] == 0))
    ].sort_values(["Manufacturer", "Site3"])
    man = tables["manufacturer_by_research_group"].copy()
    man["CN_AD_abs_diff"] = (man["CN"] - man["AD"]).abs()
    man["CN_AD_ratio_CN_over_AD"] = man["CN"] / man["AD"].replace(0, pd.NA)
    tables["manufacturer_ad_cn_imbalance"] = man.sort_values("CN_AD_abs_diff", ascending=False)
    return tables


def recommendations(tables: Dict[str, pd.DataFrame], min_target: int) -> pd.DataFrame:
    rows = []
    ms = tables["manufacturer_site_by_research_group"]
    for row in ms.itertuples(index=False):
        if row.CN > 0 and row.AD == 0:
            rows.append(
                {
                    "level": "Manufacturer_Site3",
                    "Manufacturer": row.Manufacturer,
                    "Site3": row.Site3,
                    "needed_diagnosis": "AD",
                    "current_CN": int(row.CN),
                    "current_AD": int(row.AD),
                    "reason": "site_has_CN_without_AD",
                }
            )
        elif row.AD > 0 and row.CN == 0:
            rows.append(
                {
                    "level": "Manufacturer_Site3",
                    "Manufacturer": row.Manufacturer,
                    "Site3": row.Site3,
                    "needed_diagnosis": "CN",
                    "current_CN": int(row.CN),
                    "current_AD": int(row.AD),
                    "reason": "site_has_AD_without_CN",
                }
            )
    man = tables["manufacturer_by_research_group"]
    for row in man.itertuples(index=False):
        if row.CN < min_target:
            rows.append(
                {
                    "level": "Manufacturer",
                    "Manufacturer": row.Manufacturer,
                    "Site3": "",
                    "needed_diagnosis": "CN",
                    "current_CN": int(row.CN),
                    "current_AD": int(row.AD),
                    "reason": f"manufacturer_CN_below_{min_target}",
                }
            )
        if row.AD < min_target:
            rows.append(
                {
                    "level": "Manufacturer",
                    "Manufacturer": row.Manufacturer,
                    "Site3": "",
                    "needed_diagnosis": "AD",
                    "current_CN": int(row.CN),
                    "current_AD": int(row.AD),
                    "reason": f"manufacturer_AD_below_{min_target}",
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    if not args.metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {args.metadata_path}")
    df = pd.read_csv(args.metadata_path)
    required = {"SubjectID", "ResearchGroup_Mapped", "Manufacturer", "Site3", "SourceCohort"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Metadata missing columns: {missing}")
    tables = build_tables(df)
    recs = recommendations(tables, args.min_cn_ad_per_manufacturer)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(args.output_dir / f"{name}.csv", index=False)
    recs.to_csv(args.output_dir / "missing_cells_recommendation.csv", index=False)
    print("ResearchGroup_Mapped counts:")
    print(tables["research_group_counts"].to_string(index=False))
    print(f"\nOutputs written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
