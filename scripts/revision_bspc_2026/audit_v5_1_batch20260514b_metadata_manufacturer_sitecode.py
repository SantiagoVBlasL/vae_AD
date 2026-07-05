"""
Read-only metadata audit: scanner manufacturer vs. ADNI acquisition site.

Input:
    training_ready_metadata_v5_1_batch20260514b_no_pybandpass.csv

Outputs (all written to OUTPUT_DIR):
    README.md
    manufacturer_counts.csv / .md
    sitecode_counts.csv / .md
    manufacturer_sitecode_table.csv / .md
    diagnosis_by_manufacturer.csv / .md
    diagnosis_by_sitecode.csv / .md
    within_site_auc_feasibility.csv / .md
    terminology_recommendation.md
    command_log.json

Constraints:
    - Read-only: no tensor, metadata, ledger, config, or model-output modification.
    - No training, no inference.
"""

import json
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_METADATA_PATH = (
    Path("/media/diego/Datos/vae_AD_data")
    / "revision_bspc_2026"
    / "adni_expanded_v5_1_batch20260514b_no_pybandpass"
    / "training_ready_metadata_v5_1_batch20260514b_no_pybandpass.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_batch20260514b_metadata_manufacturer_sitecode_audit"
)

# ---------------------------------------------------------------------------
# Feasibility thresholds
# ---------------------------------------------------------------------------
MIN_TOTAL = 10
MIN_AD = 3
MIN_CN = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def df_to_markdown(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a GitHub-flavoured Markdown table."""
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join([header, sep] + rows)


def save_csv_md(df: pd.DataFrame, stem: str, out_dir: Path, title: str = "") -> None:
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    md_lines = []
    if title:
        md_lines.append(f"# {title}\n")
    md_lines.append(df_to_markdown(df))
    (out_dir / f"{stem}.md").write_text("\n".join(md_lines) + "\n")


def extract_sitecode_from_ptid(ptid: str) -> str:
    """Extract 3-digit zero-padded site code from ADNI PTID (format: XXX_S_NNNN)."""
    parts = str(ptid).split("_S_")
    return parts[0].zfill(3) if parts else "UNKNOWN"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_audit(
    metadata_path: Path = DEFAULT_METADATA_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    t0 = datetime.now(timezone.utc)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load metadata                                                     #
    # ------------------------------------------------------------------ #
    df = pd.read_csv(metadata_path)
    n_total = len(df)

    # ------------------------------------------------------------------ #
    # 2. Confirm required columns                                          #
    # ------------------------------------------------------------------ #
    required_cols = ["SubjectID", "Manufacturer", "Site3", "ResearchGroup_Mapped"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in metadata: {missing_cols}")

    # ------------------------------------------------------------------ #
    # 3. Derive SiteCode columns                                           #
    # ------------------------------------------------------------------ #
    df["SiteCode_from_SubjectID"] = df["SubjectID"].apply(extract_sitecode_from_ptid)

    # Site3 is stored as float (e.g. 2.0, 130.0) → zero-padded string
    def site3_to_str(val) -> str:
        try:
            if pd.isna(val):
                return None
            return str(int(val)).zfill(3)
        except (ValueError, TypeError):
            return None

    df["Site3_str"] = df["Site3"].apply(site3_to_str)

    # Best: use Site3_str when available, else PTID-derived
    df["SiteCode_best"] = df["Site3_str"].combine_first(df["SiteCode_from_SubjectID"])

    # Simplify diagnosis: keep ResearchGroup_Mapped (CN / MCI / AD)
    df["Diagnosis_group"] = df["ResearchGroup_Mapped"]

    # ------------------------------------------------------------------ #
    # 4. Manufacturer counts                                               #
    # ------------------------------------------------------------------ #
    mfr_counts = (
        df["Manufacturer"].value_counts(dropna=False)
        .reset_index()
        .rename(columns={"index": "Manufacturer", "count": "n"})
    )
    # pandas ≥2.0 value_counts columns differ
    if "Manufacturer" not in mfr_counts.columns:
        mfr_counts.columns = ["Manufacturer", "n"]
    mfr_counts["pct"] = (mfr_counts["n"] / n_total * 100).round(1)
    save_csv_md(mfr_counts, "manufacturer_counts", output_dir, "Manufacturer Counts")

    # ------------------------------------------------------------------ #
    # 5. Site3 missing count                                               #
    # ------------------------------------------------------------------ #
    site3_missing = int(df["Site3"].isna().sum())
    site3_present = n_total - site3_missing

    # ------------------------------------------------------------------ #
    # 6. SiteCode_best counts                                              #
    # ------------------------------------------------------------------ #
    sc_counts = (
        df["SiteCode_best"].value_counts()
        .reset_index()
        .rename(columns={"index": "SiteCode_best", "count": "n"})
    )
    if "SiteCode_best" not in sc_counts.columns:
        sc_counts.columns = ["SiteCode_best", "n"]
    sc_counts = sc_counts.sort_values("SiteCode_best").reset_index(drop=True)
    save_csv_md(sc_counts, "sitecode_counts", output_dir, "SiteCode_best Counts")

    n_sites = sc_counts["SiteCode_best"].nunique()

    # ------------------------------------------------------------------ #
    # 7. Manufacturer × SiteCode_best table                               #
    # ------------------------------------------------------------------ #
    mfr_sc = (
        df.groupby(["Manufacturer", "SiteCode_best"])
        .size()
        .reset_index(name="n")
        .sort_values(["Manufacturer", "SiteCode_best"])
        .reset_index(drop=True)
    )
    save_csv_md(mfr_sc, "manufacturer_sitecode_table", output_dir,
                "Manufacturer × SiteCode_best")

    # Sites using >1 manufacturer
    sites_multi_mfr = (
        df.groupby("SiteCode_best")["Manufacturer"].nunique()
    )
    multi_mfr_sites = sites_multi_mfr[sites_multi_mfr > 1].index.tolist()

    # ------------------------------------------------------------------ #
    # 8. Diagnosis × Manufacturer                                          #
    # ------------------------------------------------------------------ #
    diag_mfr_raw = (
        df.groupby(["Diagnosis_group", "Manufacturer"])
        .size()
        .unstack(fill_value=0)
    )
    diag_mfr_raw["Total"] = diag_mfr_raw.sum(axis=1)
    diag_mfr_pct = diag_mfr_raw.copy()
    # add pct AD within each Manufacturer
    diag_mfr_flat = (
        df.groupby(["Manufacturer", "Diagnosis_group"])
        .size()
        .reset_index(name="n")
    )
    diag_mfr_total = diag_mfr_flat.groupby("Manufacturer")["n"].sum().rename("total")
    diag_mfr_flat = diag_mfr_flat.merge(diag_mfr_total, on="Manufacturer")
    diag_mfr_flat["pct"] = (diag_mfr_flat["n"] / diag_mfr_flat["total"] * 100).round(1)
    save_csv_md(diag_mfr_flat, "diagnosis_by_manufacturer", output_dir,
                "Diagnosis × Manufacturer")

    # Compute AD% per Manufacturer for QC note
    ad_pct_by_mfr = {}
    for mfr, grp in df.groupby("Manufacturer"):
        n_ad = (grp["Diagnosis_group"] == "AD").sum()
        n_cn = (grp["Diagnosis_group"] == "CN").sum()
        ad_pct_by_mfr[mfr] = {
            "n_CN": int(n_cn),
            "n_AD": int(n_ad),
            "AD_pct_of_CN_AD": round(n_ad / (n_ad + n_cn) * 100, 1) if (n_ad + n_cn) > 0 else None,
        }

    # ------------------------------------------------------------------ #
    # 9. Diagnosis × SiteCode_best                                         #
    # ------------------------------------------------------------------ #
    diag_sc_flat = (
        df.groupby(["SiteCode_best", "Diagnosis_group"])
        .size()
        .reset_index(name="n")
    )
    diag_sc_total = diag_sc_flat.groupby("SiteCode_best")["n"].sum().rename("total")
    diag_sc_flat = diag_sc_flat.merge(diag_sc_total, on="SiteCode_best")
    diag_sc_flat["pct"] = (diag_sc_flat["n"] / diag_sc_flat["total"] * 100).round(1)
    save_csv_md(diag_sc_flat, "diagnosis_by_sitecode", output_dir,
                "Diagnosis × SiteCode_best")

    # ------------------------------------------------------------------ #
    # 10. Sites with both CN and AD                                        #
    # ------------------------------------------------------------------ #
    site_groups = df.groupby("SiteCode_best")["Diagnosis_group"].apply(set)
    sites_both = sorted(
        site_groups[site_groups.apply(lambda s: "CN" in s and "AD" in s)].index.tolist()
    )

    # ------------------------------------------------------------------ #
    # 11. Within-site AUC feasibility (CN+AD only)                        #
    # ------------------------------------------------------------------ #
    df_cn_ad = df[df["Diagnosis_group"].isin(["CN", "AD"])].copy()
    feasibility_rows = []
    for site, grp in df_cn_ad.groupby("SiteCode_best"):
        n_ad = int((grp["Diagnosis_group"] == "AD").sum())
        n_cn = int((grp["Diagnosis_group"] == "CN").sum())
        n_tot = n_ad + n_cn
        manufacturers = "/".join(sorted(grp["Manufacturer"].unique().tolist()))
        feasible = (n_tot >= MIN_TOTAL) and (n_ad >= MIN_AD) and (n_cn >= MIN_CN)
        feasibility_rows.append({
            "SiteCode_best": site,
            "Manufacturer_s": manufacturers,
            "n_CN": n_cn,
            "n_AD": n_ad,
            "n_total": n_tot,
            "feasible": feasible,
        })

    feasibility_df = (
        pd.DataFrame(feasibility_rows)
        .sort_values("n_total", ascending=False)
        .reset_index(drop=True)
    )
    save_csv_md(feasibility_df, "within_site_auc_feasibility", output_dir,
                "Within-Site AUC Feasibility")

    feasible_sites = feasibility_df[feasibility_df["feasible"]].reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # 12. Terminology recommendation                                       #
    # ------------------------------------------------------------------ #
    term_md = textwrap.dedent(f"""\
    # Manuscript Terminology Recommendations

    ## Definitions used in this audit

    | Term | Definition | Levels |
    | --- | --- | --- |
    | **scanner manufacturer** | Mapped `Manufacturer` column: GE / Philips / SIEMENS | 3 |
    | **acquisition site** | `SiteCode_best`: 3-digit ADNI site code (from `Site3` or PTID prefix) | {n_sites} |
    | **ADNI site** | Synonym for acquisition site (preferred in Methods) | {n_sites} |

    ## Recommended manuscript phrasing

    - **Stratification**: "Outer and inner cross-validation folds were stratified jointly by
      diagnostic group (CN/MCI/AD) and **scanner manufacturer** (GE / Philips / SIEMENS)."
    - **QC leakage test**: "We tested for **scanner manufacturer** leakage using the
      `qc_check_scanner_leakage` module, which predicts scanner manufacturer from latent
      representations using a stratified 5-fold cross-validated SVM; this tests the
      3-level manufacturer variable, **not** the {n_sites}-level acquisition-site variable."
    - **Confound statement**: "GE Medical Systems scanners are over-represented in CN
      ({ad_pct_by_mfr.get('GE', {}).get('AD_pct_of_CN_AD', 'N/A')}% AD) relative to
      Philips ({ad_pct_by_mfr.get('Philips', {}).get('AD_pct_of_CN_AD', 'N/A')}% AD) and
      SIEMENS ({ad_pct_by_mfr.get('SIEMENS', {}).get('AD_pct_of_CN_AD', 'N/A')}% AD) in
      the CN+AD subset."

    ## Terms to avoid

    - **"scanner site"**: ambiguous compound; use "acquisition site" or "scanner manufacturer"
      explicitly.
    - **"site"** alone when referring to manufacturer: always qualify as "ADNI site" (for
      site codes) or "scanner manufacturer" (for GE/Philips/SIEMENS).
    - **"vendor"**: acceptable synonym for "scanner manufacturer" but "scanner manufacturer"
      is preferred for clarity in a neuroimaging methods section.

    ## Notes on QC coverage

    - `qc_check_scanner_leakage=True` in the locked config tests **Manufacturer leakage**
      (3 levels), not SiteCode leakage ({n_sites} levels).
    - To test acquisition-site leakage specifically, a separate audit using SiteCode_best
      as the target would be required; this is outside the current locked pipeline.
    - {len(multi_mfr_sites)} sites used >1 manufacturer
      ({', '.join(str(s) for s in sorted(multi_mfr_sites)) if multi_mfr_sites else 'none'}),
      so Manufacturer stratification does **not** perfectly separate acquisition sites.
    """)
    (output_dir / "terminology_recommendation.md").write_text(term_md)

    # ------------------------------------------------------------------ #
    # 13. README                                                           #
    # ------------------------------------------------------------------ #
    readme_md = textwrap.dedent(f"""\
    # Metadata Audit: Scanner Manufacturer vs. ADNI Acquisition Site
    ## adni_v5_1_batch20260514b_no_pybandpass

    **Generated:** {t0.strftime('%Y-%m-%d %H:%M:%S UTC')}
    **Metadata:** `{metadata_path}`
    **Script:** `scripts/revision_bspc_2026/audit_v5_1_batch20260514b_metadata_manufacturer_sitecode.py`

    ## Summary

    | Item | Value |
    | --- | --- |
    | Total subjects | {n_total} |
    | Subjects with Site3 present | {site3_present} |
    | Subjects with Site3 missing (filled from PTID) | {site3_missing} |
    | Unique SiteCode_best values | {n_sites} |
    | Manufacturers | 3 (GE / Philips / SIEMENS) |
    | Sites using >1 manufacturer | {len(multi_mfr_sites)} ({', '.join(str(s) for s in sorted(multi_mfr_sites)) if multi_mfr_sites else 'none'}) |
    | Sites with both CN and AD | {len(sites_both)} |
    | Sites feasible for within-site AUC (n≥{MIN_TOTAL}, AD≥{MIN_AD}, CN≥{MIN_CN}) | {len(feasible_sites)} |

    ## Feasible sites for within-site AUC

    {df_to_markdown(feasible_sites)}

    ## Files

    | File | Contents |
    | --- | --- |
    | manufacturer_counts.csv/.md | Counts and % for each scanner manufacturer |
    | sitecode_counts.csv/.md | Counts for each SiteCode_best |
    | manufacturer_sitecode_table.csv/.md | Cross-tabulation: Manufacturer × SiteCode_best |
    | diagnosis_by_manufacturer.csv/.md | Diagnosis breakdown by scanner manufacturer |
    | diagnosis_by_sitecode.csv/.md | Diagnosis breakdown by SiteCode_best |
    | within_site_auc_feasibility.csv/.md | Per-site n_CN, n_AD, feasibility flag |
    | terminology_recommendation.md | Manuscript-safe terminology guide |
    | command_log.json | Execution metadata |

    ## Important note on QC

    `qc_check_scanner_leakage=True` in the locked pipeline config tests **scanner manufacturer**
    leakage (3-level: GE/Philips/SIEMENS), **not** acquisition-site leakage ({n_sites}-level).
    """)
    (output_dir / "README.md").write_text(readme_md)

    # ------------------------------------------------------------------ #
    # 14. command_log.json                                                 #
    # ------------------------------------------------------------------ #
    t1 = datetime.now(timezone.utc)
    log = {
        "script": str(Path(__file__).name),
        "started_utc": t0.isoformat(),
        "finished_utc": t1.isoformat(),
        "elapsed_s": round((t1 - t0).total_seconds(), 2),
        "metadata_path": str(metadata_path),
        "output_dir": str(output_dir),
        "n_subjects": n_total,
        "site3_missing": site3_missing,
        "n_unique_sitecodes": n_sites,
        "n_manufacturers": df["Manufacturer"].nunique(),
        "sites_multi_manufacturer": sorted(multi_mfr_sites),
        "sites_with_cn_and_ad": sites_both,
        "n_feasible_sites": int(len(feasible_sites)),
        "feasibility_thresholds": {
            "min_total": MIN_TOTAL,
            "min_AD": MIN_AD,
            "min_CN": MIN_CN,
        },
        "python": sys.version,
        "read_only": True,
    }
    (output_dir / "command_log.json").write_text(
        json.dumps(log, indent=2, default=str)
    )

    # ------------------------------------------------------------------ #
    # Console summary                                                      #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print("METADATA AUDIT COMPLETE")
    print(f"{'='*60}")
    print(f"  Subjects         : {n_total}")
    print(f"  Site3 missing    : {site3_missing} (filled from PTID)")
    print(f"  Unique SiteCodes : {n_sites}")
    print(f"  Manufacturers    : 3 (GE={ad_pct_by_mfr.get('GE',{}).get('n_CN','?')}CN "
          f"/ Philips={ad_pct_by_mfr.get('Philips',{}).get('n_CN','?')}CN "
          f"/ SIEMENS={ad_pct_by_mfr.get('SIEMENS',{}).get('n_CN','?')}CN)")
    print(f"  AD% by Mfr (CN+AD subset):")
    for mfr, vals in sorted(ad_pct_by_mfr.items()):
        print(f"    {mfr:8s}: {vals['AD_pct_of_CN_AD']}% AD "
              f"({vals['n_AD']} AD / {vals['n_CN']} CN)")
    print(f"  Sites multi-mfr  : {len(multi_mfr_sites)} "
          f"({', '.join(str(s) for s in sorted(multi_mfr_sites)) if multi_mfr_sites else 'none'})")
    print(f"  Sites both CN+AD : {len(sites_both)}")
    print(f"  Feasible sites   : {len(feasible_sites)}")
    if len(feasible_sites) > 0:
        for _, row in feasible_sites.iterrows():
            print(f"    Site {row['SiteCode_best']} ({row['Manufacturer_s']}): "
                  f"n={row['n_total']}  CN={row['n_CN']} AD={row['n_AD']}")
    print(f"\n  QC NOTE: qc_check_scanner_leakage tests MANUFACTURER (3-level),")
    print(f"           NOT SiteCode ({n_sites}-level).")
    print(f"\n  Output: {output_dir}")
    print(f"  Elapsed: {log['elapsed_s']}s")
    print("="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Read-only metadata audit: manufacturer vs acquisition site"
    )
    parser.add_argument("--metadata_path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    run_audit(metadata_path=args.metadata_path, output_dir=args.output_dir)
