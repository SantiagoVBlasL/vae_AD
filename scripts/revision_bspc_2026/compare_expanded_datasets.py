#!/usr/bin/env python3
"""
Compare ADNI_expanded_v1 vs ADNI_expanded_v2 datasets.

Reads the per-subject manifests from both versions and produces:
  v1_vs_v2_dataset_comparison.csv   — per-metric table
  v1_vs_v2_dataset_comparison.md    — human-readable report
  v1_vs_v2_dataset_comparison.json  — machine-readable summary
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_V1_MANIFEST = (
    PROJECT_ROOT / "data" / "revision_bspc_2026" / "adni_expanded_v1"
    / "expanded_subject_manifest.csv"
)
DEFAULT_V2_MANIFEST = (
    PROJECT_ROOT / "data" / "revision_bspc_2026" / "adni_expanded_v2"
    / "expanded_subject_manifest.csv"
)
DEFAULT_V1_METADATA = (
    PROJECT_ROOT / "data" / "revision_bspc_2026" / "adni_expanded_v1"
    / "subject_metadata_adni_expanded_v1.csv"
)
DEFAULT_V2_METADATA = (
    PROJECT_ROOT / "data" / "revision_bspc_2026" / "adni_expanded_v2"
    / "subject_metadata_adni_expanded_v2.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "revision_bspc_2026" / "adni_expanded_v2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare ADNI_expanded_v1 vs ADNI_expanded_v2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--v1-manifest", type=Path, default=DEFAULT_V1_MANIFEST)
    p.add_argument("--v2-manifest", type=Path, default=DEFAULT_V2_MANIFEST)
    p.add_argument("--v1-metadata", type=Path, default=DEFAULT_V1_METADATA)
    p.add_argument("--v2-metadata", type=Path, default=DEFAULT_V2_METADATA)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def load_df(path: Path, label: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"WARNING: {label} not found: {path}")
        return None
    return pd.read_csv(path)


def count_by(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=int)
    return df[col].value_counts().sort_index()


def compare_counts(s1: pd.Series, s2: pd.Series, label: str) -> List[Dict]:
    rows = []
    all_keys = sorted(set(s1.index) | set(s2.index))
    for k in all_keys:
        v1 = int(s1.get(k, 0))
        v2 = int(s2.get(k, 0))
        rows.append({
            "metric_group": label,
            "value": str(k),
            "v1": v1,
            "v2": v2,
            "delta_v2_minus_v1": v2 - v1,
        })
    return rows


def scalar_row(label: str, metric: str, v1, v2) -> Dict:
    v1i = int(v1) if v1 is not None else None
    v2i = int(v2) if v2 is not None else None
    delta = (v2i - v1i) if (v1i is not None and v2i is not None) else None
    return {"metric_group": label, "value": metric, "v1": v1i, "v2": v2i, "delta_v2_minus_v1": delta}


def main() -> int:
    args = parse_args()

    m1 = load_df(args.v1_manifest, "v1 manifest")
    m2 = load_df(args.v2_manifest, "v2 manifest")
    d1 = load_df(args.v1_metadata, "v1 metadata")
    d2 = load_df(args.v2_metadata, "v2 metadata")

    if m1 is None and d1 is None:
        print("ERROR: Neither v1 manifest nor metadata found.")
        return 1
    if m2 is None and d2 is None:
        print("ERROR: Neither v2 manifest nor metadata found.")
        return 1

    # Use metadata if manifest not available (manifest has more columns but meta is sufficient)
    df1 = m1 if m1 is not None else d1
    df2 = m2 if m2 is not None else d2

    rows: List[Dict] = []

    # ── N total ────────────────────────────────────────────────────────────────
    rows.append(scalar_row("N_total", "N", len(df1), len(df2)))

    # ── N by ResearchGroup ─────────────────────────────────────────────────────
    rg1 = count_by(df1, "ResearchGroup_Mapped")
    rg2 = count_by(df2, "ResearchGroup_Mapped")
    rows += compare_counts(rg1, rg2, "ResearchGroup_Mapped")

    # ── N by SourceCohort ──────────────────────────────────────────────────────
    sc1 = count_by(df1, "SourceCohort")
    sc2 = count_by(df2, "SourceCohort")
    rows += compare_counts(sc1, sc2, "SourceCohort")

    # ── N by Manufacturer ──────────────────────────────────────────────────────
    mf1 = count_by(df1, "Manufacturer")
    mf2 = count_by(df2, "Manufacturer")
    rows += compare_counts(mf1, mf2, "Manufacturer")

    # ── N by Site3 ─────────────────────────────────────────────────────────────
    s1 = count_by(df1, "Site3")
    s2 = count_by(df2, "Site3")
    rows += compare_counts(s1, s2, "Site3")

    # ── Expansion flags ────────────────────────────────────────────────────────
    def flag_count(df: pd.DataFrame, col: str):
        return int(df[col].sum()) if col in df.columns else None

    for flag in ["IsExpansionSubject", "IsMartin59", "IsSantiProcessed", "IsStressCandidate"]:
        rows.append(scalar_row("Flags", flag, flag_count(df1, flag), flag_count(df2, flag)))

    # ── Subjects added in v2 vs v1 ─────────────────────────────────────────────
    ids1 = set(df1["SubjectID"].astype(str).tolist()) if "SubjectID" in df1.columns else set()
    ids2 = set(df2["SubjectID"].astype(str).tolist()) if "SubjectID" in df2.columns else set()
    added = sorted(ids2 - ids1)
    removed = sorted(ids1 - ids2)
    rows.append(scalar_row("SubjectID_changes", "added_in_v2", None, len(added)))
    rows.append(scalar_row("SubjectID_changes", "removed_in_v2", len(removed), None))
    rows.append(scalar_row("SubjectID_changes", "common", len(ids1 & ids2), len(ids1 & ids2)))

    comparison_df = pd.DataFrame(rows)

    # ── CN/AD distribution by SourceCohort × Manufacturer ────────────────────
    cross_rows: List[Dict] = []
    for version, df in [("v1", df1), ("v2", df2)]:
        if "ResearchGroup_Mapped" not in df.columns:
            continue
        for (sc, mfr), sub in df.groupby(
            ["SourceCohort", "Manufacturer"], dropna=False
        ):
            rg_counts = sub["ResearchGroup_Mapped"].value_counts().to_dict()
            cross_rows.append({
                "version": version,
                "SourceCohort": str(sc),
                "Manufacturer": str(mfr),
                "N": len(sub),
                **{f"N_{k}": int(v) for k, v in rg_counts.items()},
            })
    cross_df = pd.DataFrame(cross_rows).fillna(0)

    # ── Added subjects detail ─────────────────────────────────────────────────
    added_detail = pd.DataFrame()
    if added:
        added_mask = df2["SubjectID"].astype(str).isin(added)
        cols = [c for c in ["SubjectID", "SourceCohort", "ResearchGroup_Mapped",
                             "Manufacturer", "Site3", "Age", "Sex"] if c in df2.columns]
        added_detail = df2.loc[added_mask, cols].copy().reset_index(drop=True)

    # ── Save ─────────────────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)

    comparison_df.to_csv(args.output_dir / "v1_vs_v2_dataset_comparison.csv", index=False)
    cross_df.to_csv(args.output_dir / "v1_vs_v2_cross_table.csv", index=False)
    if not added_detail.empty:
        added_detail.to_csv(args.output_dir / "v2_added_subjects.csv", index=False)

    # ── Markdown report ───────────────────────────────────────────────────────
    n1, n2 = len(df1), len(df2)
    md_lines = [
        "# ADNI Expanded: v1 vs v2 Comparison",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"| Metric | v1 | v2 | Δ (v2-v1) |",
        f"|--------|----|----|-----------|",
        f"| N total | {n1} | {n2} | {n2 - n1:+d} |",
    ]
    for flag in ["IsExpansionSubject", "IsMartin59", "IsSantiProcessed", "IsStressCandidate"]:
        f1 = flag_count(df1, flag)
        f2 = flag_count(df2, flag)
        d = (f2 - f1) if (f1 is not None and f2 is not None) else "?"
        md_lines.append(
            f"| {flag} | {f1 if f1 is not None else '—'} "
            f"| {f2 if f2 is not None else '—'} "
            f"| {f'+{d}' if isinstance(d, int) and d >= 0 else d} |"
        )
    md_lines += [
        "",
        "## By ResearchGroup_Mapped",
        "",
    ]
    for k in sorted(set(rg1.index) | set(rg2.index)):
        v1v = int(rg1.get(k, 0))
        v2v = int(rg2.get(k, 0))
        md_lines.append(f"- {k}: v1={v1v}, v2={v2v}, Δ={v2v - v1v:+d}")

    md_lines += [
        "",
        "## By SourceCohort",
        "",
    ]
    for k in sorted(set(sc1.index) | set(sc2.index)):
        v1v = int(sc1.get(k, 0))
        v2v = int(sc2.get(k, 0))
        md_lines.append(f"- {k}: v1={v1v}, v2={v2v}, Δ={v2v - v1v:+d}")

    md_lines += [
        "",
        "## By Manufacturer",
        "",
    ]
    for k in sorted(set(mf1.index) | set(mf2.index)):
        v1v = int(mf1.get(k, 0))
        v2v = int(mf2.get(k, 0))
        md_lines.append(f"- {k}: v1={v1v}, v2={v2v}, Δ={v2v - v1v:+d}")

    md_lines += [
        "",
        f"## Subjects added in v2: {len(added)}",
        "",
    ]
    for sid in added[:20]:
        row = df2.loc[df2["SubjectID"].astype(str) == sid]
        if not row.empty:
            r = row.iloc[0]
            md_lines.append(
                f"- {sid}: {r.get('SourceCohort','?')} | "
                f"{r.get('ResearchGroup_Mapped','?')} | "
                f"{r.get('Manufacturer','?')} | Site3={r.get('Site3','?')}"
            )
    if len(added) > 20:
        md_lines.append(f"  ... and {len(added) - 20} more (see v2_added_subjects.csv)")

    if removed:
        md_lines += ["", f"## Subjects removed in v2 vs v1: {len(removed)}", ""]
        for sid in removed[:10]:
            md_lines.append(f"- {sid}")

    (args.output_dir / "v1_vs_v2_dataset_comparison.md").write_text(
        "\n".join(md_lines), encoding="utf-8"
    )

    # ── JSON summary ──────────────────────────────────────────────────────────
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "v1_n": n1,
        "v2_n": n2,
        "delta_n": n2 - n1,
        "v1_source_cohorts": sc1.to_dict(),
        "v2_source_cohorts": sc2.to_dict(),
        "v1_research_groups": rg1.to_dict(),
        "v2_research_groups": rg2.to_dict(),
        "v1_manufacturers": mf1.to_dict(),
        "v2_manufacturers": mf2.to_dict(),
        "subjects_added_in_v2": len(added),
        "subjects_removed_in_v2": len(removed),
        "subjects_common": len(ids1 & ids2),
        "added_subject_ids": added,
        "removed_subject_ids": removed,
    }
    with (args.output_dir / "v1_vs_v2_dataset_comparison.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"v1 N={n1}, v2 N={n2}, Δ={n2 - n1:+d}")
    print(f"Subjects added in v2: {len(added)}")
    print(f"Comparison files written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
