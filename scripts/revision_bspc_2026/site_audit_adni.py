"""
ADNI Cohort / Site Audit for BSPC 2026 Revision
=================================================
Audits the ADNI cohort used in the AD-likeness paper and determines LOSO
(Leave-One-Site-Out) feasibility for CN vs AD classification.

Outputs written to: results/revision_bspc_2026/site_audit/

Usage:
    conda run -n vae_ad python scripts/revision_bspc_2026/site_audit_adni.py

Author: auto-generated for revision_bspc_2026
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data"
RESULTS_DIR  = PROJECT_ROOT / "results" / "revision_bspc_2026" / "site_audit"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

METADATA_PATH = DATA_DIR / "SubjectsData_AAL3_procesado2.csv"
TENSOR_NPZ    = (
    DATA_DIR
    / "AAL3_dynamicROIs_fmri_tensor_NeuroEnhanced_v6.5.17_"
      "AAL3_131ROIs_OMST_GCE_Signed_GrangerLag1_ChNorm_ROIreorderedYeo17_"
      "ParallelTuned"
    / "GLOBAL_TENSOR_from_AAL3_dynamicROIs_fmri_tensor_NeuroEnhanced_v6.5.17_"
      "AAL3_131ROIs_OMST_GCE_Signed_GrangerLag1_ChNorm_ROIreorderedYeo17_"
      "ParallelTuned.npz"
)
FOLD_PRED_PATTERN = (
    PROJECT_ROOT
    / "results"
    / "vae_3channels_beta65_pro"
    / "fold_{fold}"
    / "test_predictions_logreg.csv"
)

# LOSO feasibility thresholds
LOSO_MAIN_MIN_PER_CLASS       = 5   # strict: both CN and AD >= 5
LOSO_SENSITIVITY_MIN_PER_CLASS = 3  # relaxed: both CN and AD >= 3

# Paper-reported numbers (used in cohort comparison)
PAPER_CN = 89
PAPER_AD = 95   # 1 AD subject unmapped from tensor → final analysis uses 94

# ─── Column requirements ───────────────────────────────────────────────────────
REQUIRED_META_COLS = {
    "SubjectID", "ResearchGroup_Mapped", "Site3",
    "Manufacturer", "Phase",
}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_metadata() -> pd.DataFrame:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata not found: {METADATA_PATH}")
    df = pd.read_csv(METADATA_PATH)
    missing = REQUIRED_META_COLS - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in metadata: {missing}")
    return df


def _load_tensor_subject_ids() -> np.ndarray:
    if not TENSOR_NPZ.exists():
        raise FileNotFoundError(f"Global tensor not found: {TENSOR_NPZ}")
    data = np.load(TENSOR_NPZ, allow_pickle=True)
    return data["subject_ids"]


def _load_analysis_cohort(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Load subjects that were actually used in the CN/AD classification pipeline
    (test-set predictions across all 5 folds = full analysis cohort).
    Returns merged DataFrame with metadata columns.
    """
    dfs = []
    for fold in range(1, 6):
        p = Path(str(FOLD_PRED_PATTERN).format(fold=fold))
        if not p.exists():
            raise FileNotFoundError(f"Fold prediction file missing: {p}")
        df = pd.read_csv(p)
        df["fold"] = fold
        dfs.append(df)
    preds = pd.concat(dfs, ignore_index=True)

    # Each subject appears in exactly one fold's test set
    assert preds["SubjectID"].nunique() == len(preds), (
        "Duplicate SubjectID across folds — unexpected."
    )

    analysis = preds.drop_duplicates("SubjectID").merge(
        meta[list(REQUIRED_META_COLS)],
        on="SubjectID",
        how="left",
    )
    return analysis


def _save_csv(df: pd.DataFrame, name: str) -> Path:
    p = RESULTS_DIR / name
    df.to_csv(p, index=False)
    return p


def _save_fig(fig: plt.Figure, name: str) -> Path:
    p = RESULTS_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


# ─── Analysis functions ───────────────────────────────────────────────────────

def build_cohort_overview(
    meta: pd.DataFrame,
    tensor_ids: np.ndarray,
    analysis: pd.DataFrame,
) -> pd.DataFrame:
    """Comparison table: metadata vs tensor vs analysis cohort."""
    tensor_set   = set(tensor_ids)
    analysis_set = set(analysis["SubjectID"])

    rows = []
    for grp in ["CN", "AD", "MCI"]:
        meta_n    = (meta["ResearchGroup_Mapped"] == grp).sum()
        tensor_n  = meta.loc[meta["ResearchGroup_Mapped"] == grp, "SubjectID"].isin(tensor_set).sum()
        analysis_n = (
            int((analysis["y_true"] == 0).sum()) if grp == "CN"
            else int((analysis["y_true"] == 1).sum()) if grp == "AD"
            else 0
        )
        rows.append({
            "group":            grp,
            "n_metadata":       int(meta_n),
            "n_tensor":         int(tensor_n),
            "n_analysis_cohort": analysis_n,
            "dropped_meta2tensor":   int(meta_n - tensor_n),
            "note": (
                "binary classifier uses CN vs AD only"
                if grp == "MCI"
                else ""
            ),
        })
    df = pd.DataFrame(rows)
    return df


def build_site_by_class_all(meta: pd.DataFrame) -> pd.DataFrame:
    """Cross-tab: Site3 × ResearchGroup_Mapped (all three classes, full metadata)."""
    ct = (
        meta.groupby(["Site3", "ResearchGroup_Mapped"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    ct.columns.name = None
    ct["total"] = ct.drop(columns="Site3").sum(axis=1)
    return ct.sort_values("total", ascending=False).reset_index(drop=True)


def build_site_by_class_cn_ad(analysis: pd.DataFrame) -> pd.DataFrame:
    """Cross-tab: Site3 × class (CN / AD) for the analysis cohort only."""
    analysis = analysis.copy()
    analysis["class"] = analysis["y_true"].map({0: "CN", 1: "AD"})
    ct = (
        analysis.groupby(["Site3", "class"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    ct.columns.name = None
    # Ensure both columns exist
    for col in ["CN", "AD"]:
        if col not in ct.columns:
            ct[col] = 0
    ct["total"] = ct["CN"] + ct["AD"]
    return ct.sort_values("total", ascending=False).reset_index(drop=True)


def build_site_by_manufacturer(meta: pd.DataFrame) -> pd.DataFrame:
    ct = (
        meta.groupby(["Site3", "Manufacturer"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    ct.columns.name = None
    ct["total"] = ct.drop(columns="Site3").sum(axis=1)
    return ct.sort_values("total", ascending=False).reset_index(drop=True)


def build_class_by_manufacturer(analysis: pd.DataFrame) -> pd.DataFrame:
    analysis = analysis.copy()
    analysis["class"] = analysis["y_true"].map({0: "CN", 1: "AD"})
    ct = (
        analysis.groupby(["Manufacturer", "class"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    ct.columns.name = None
    for col in ["CN", "AD"]:
        if col not in ct.columns:
            ct[col] = 0
    ct["total"] = ct["CN"] + ct["AD"]
    pct_cn = ct["CN"] / ct["total"]
    pct_ad = ct["AD"] / ct["total"]
    ct["pct_CN"] = (pct_cn * 100).round(1)
    ct["pct_AD"] = (pct_ad * 100).round(1)
    return ct


def build_phase_by_class(meta: pd.DataFrame) -> pd.DataFrame:
    ct = (
        meta.groupby(["Phase", "ResearchGroup_Mapped"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    ct.columns.name = None
    ct["total"] = ct.drop(columns="Phase").sum(axis=1)
    return ct


def build_loso_feasibility(site_cn_ad: pd.DataFrame) -> pd.DataFrame:
    """
    Per-site LOSO feasibility assessment.

    Columns:
      - site, n_CN, n_AD, n_total
      - has_both_classes   : True if both CN > 0 and AD > 0
      - main_loso          : True if both CN >= LOSO_MAIN_MIN_PER_CLASS
                                      and AD >= LOSO_MAIN_MIN_PER_CLASS
      - sensitivity_loso   : True if both >= LOSO_SENSITIVITY_MIN_PER_CLASS
      - loso_set           : 'main' | 'sensitivity' | 'excluded'
      - exclusion_reason   : why the site was excluded (if applicable)
    """
    df = site_cn_ad.copy().rename(columns={"Site3": "site"})
    df["has_both_classes"] = (df["CN"] > 0) & (df["AD"] > 0)
    df["main_loso"] = (
        (df["CN"] >= LOSO_MAIN_MIN_PER_CLASS)
        & (df["AD"] >= LOSO_MAIN_MIN_PER_CLASS)
    )
    df["sensitivity_loso"] = (
        (df["CN"] >= LOSO_SENSITIVITY_MIN_PER_CLASS)
        & (df["AD"] >= LOSO_SENSITIVITY_MIN_PER_CLASS)
    )

    def categorize(row):
        if row["main_loso"]:
            return "main"
        if row["sensitivity_loso"]:
            return "sensitivity"
        return "excluded"

    def exclusion_reason(row):
        if row["main_loso"]:
            return ""
        if not row["has_both_classes"]:
            return "single class only"
        min_n = min(row["CN"], row["AD"])
        return (
            f"too small (min class n={min_n}, "
            f"need {LOSO_SENSITIVITY_MIN_PER_CLASS} for sensitivity, "
            f"{LOSO_MAIN_MIN_PER_CLASS} for main)"
        )

    df["loso_set"] = df.apply(categorize, axis=1)
    df["exclusion_reason"] = df.apply(exclusion_reason, axis=1)
    return df.sort_values(["loso_set", "total"], ascending=[True, False]).reset_index(drop=True)


# ─── Confound detection ───────────────────────────────────────────────────────

def _compute_chi2_cramersV(ct_values: np.ndarray) -> tuple[float, float]:
    """Chi-squared test + Cramér's V for a contingency table (2D array)."""
    from scipy.stats import chi2_contingency
    chi2, p, dof, _ = chi2_contingency(ct_values, correction=False)
    n = ct_values.sum()
    k = min(ct_values.shape) - 1
    V = np.sqrt(chi2 / (n * max(k, 1)))
    return float(p), float(V)


def detect_confounds(
    analysis: pd.DataFrame,
    site_cn_ad: pd.DataFrame,
) -> dict:
    """
    Detect major confounds:
      - class × manufacturer imbalance
      - class × site imbalance (restricted to LOSO-eligible sites)
    Returns a dict of findings for use in the summary report.
    """
    findings = {}

    # Class × Manufacturer
    analysis = analysis.copy()
    analysis["class"] = analysis["y_true"].map({0: "CN", 1: "AD"})
    cm = pd.crosstab(analysis["class"], analysis["Manufacturer"]).values
    p_mfg, V_mfg = _compute_chi2_cramersV(cm)
    findings["class_manufacturer_p"]        = p_mfg
    findings["class_manufacturer_cramersV"] = V_mfg

    # Class × Site (all sites with both classes)
    dual_sites = site_cn_ad.loc[
        (site_cn_ad["CN"] > 0) & (site_cn_ad["AD"] > 0), "Site3"
    ]
    analysis_dual = analysis[analysis["Site3"].isin(dual_sites)]
    if len(analysis_dual) > 0:
        cs = pd.crosstab(analysis_dual["class"], analysis_dual["Site3"]).values
        p_site, V_site = _compute_chi2_cramersV(cs)
        findings["class_site_p"]        = p_site
        findings["class_site_cramersV"] = V_site
    else:
        findings["class_site_p"]        = None
        findings["class_site_cramersV"] = None

    return findings


# ─── Figures ──────────────────────────────────────────────────────────────────

def plot_heatmap_site_class_cn_ad(site_cn_ad: pd.DataFrame) -> plt.Figure:
    """
    Heatmap: sites (rows) × {CN, AD} (columns), cell values = subject counts.
    Only sites with at least 1 subject shown; sorted by total descending.
    """
    df = site_cn_ad.set_index("Site3")[["CN", "AD"]].sort_values(
        "CN", ascending=False
    )
    fig, ax = plt.subplots(figsize=(5, max(6, len(df) * 0.35)))
    im = ax.imshow(df.values, aspect="auto", cmap="YlOrRd", vmin=0)
    plt.colorbar(im, ax=ax, label="n subjects")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["CN", "AD"], fontsize=11)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f"Site {s}" for s in df.index], fontsize=7)
    for i, row in enumerate(df.values):
        for j, val in enumerate(row):
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=7, color="black" if val < df.values.max() * 0.7 else "white")
    ax.set_title("CN / AD subjects per site\n(analysis cohort)", fontsize=11, pad=8)
    fig.tight_layout()
    return fig


def plot_barplot_subjects_per_site_cn_ad(site_cn_ad: pd.DataFrame, loso: pd.DataFrame) -> plt.Figure:
    """
    Stacked bar chart per site (CN/AD counts), with LOSO category colour-coded.
    """
    # Merge LOSO category
    loso_map = loso.set_index("site")["loso_set"].to_dict()
    df = site_cn_ad.copy()
    df["loso_set"] = df["Site3"].map(loso_map)
    df = df.sort_values("total", ascending=False)

    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.55), 5))
    x = np.arange(len(df))
    colors = {"main": "#2196F3", "sensitivity": "#FF9800", "excluded": "#BDBDBD"}
    width = 0.6

    bar_cn = ax.bar(x, df["CN"].values, width, label="CN",
                    color=[colors[c] for c in df["loso_set"]], alpha=0.85)
    bar_ad = ax.bar(x, df["AD"].values, width, bottom=df["CN"].values,
                    label="AD",
                    color=[colors[c] for c in df["loso_set"]], alpha=0.45)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Site {s}" for s in df["Site3"]], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("N subjects")
    ax.set_title(
        "Subjects per site — CN (solid) / AD (faded)\n"
        "Colour: blue=main LOSO, orange=sensitivity, grey=excluded",
        fontsize=10
    )

    # Threshold lines
    ax.axhline(LOSO_MAIN_MIN_PER_CLASS * 2, color="blue", lw=0.8, ls="--",
               label=f"main threshold ({LOSO_MAIN_MIN_PER_CLASS}/class)")
    ax.axhline(LOSO_SENSITIVITY_MIN_PER_CLASS * 2, color="orange", lw=0.8, ls="--",
               label=f"sensitivity threshold ({LOSO_SENSITIVITY_MIN_PER_CLASS}/class)")

    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_barplot_class_by_manufacturer(class_mfg: pd.DataFrame) -> plt.Figure:
    """Grouped bar chart: class counts per manufacturer."""
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(class_mfg))
    w = 0.35
    ax.bar(x - w / 2, class_mfg["CN"], w, label="CN", color="#4CAF50")
    ax.bar(x + w / 2, class_mfg["AD"], w, label="AD", color="#F44336")
    ax.set_xticks(x)
    ax.set_xticklabels(class_mfg["Manufacturer"], fontsize=10)
    ax.set_ylabel("N subjects")
    ax.set_title("CN / AD distribution by manufacturer\n(analysis cohort)", fontsize=11)
    for i, (cn, ad, tot) in enumerate(
        zip(class_mfg["CN"], class_mfg["AD"], class_mfg["total"])
    ):
        ax.text(i - w / 2, cn + 0.5, str(cn), ha="center", va="bottom", fontsize=9)
        ax.text(i + w / 2, ad + 0.5, str(ad), ha="center", va="bottom", fontsize=9)
    ax.legend(fontsize=10)
    fig.tight_layout()
    return fig


# ─── Markdown summary ─────────────────────────────────────────────────────────

def build_summary_md(
    overview: pd.DataFrame,
    site_cn_ad: pd.DataFrame,
    loso: pd.DataFrame,
    class_mfg: pd.DataFrame,
    confounds: dict,
    analysis: pd.DataFrame,
) -> str:
    total_analysis = len(analysis)
    n_cn = int((analysis["y_true"] == 0).sum())
    n_ad = int((analysis["y_true"] == 1).sum())
    n_sites_total   = site_cn_ad["Site3"].nunique()
    n_sites_dual    = int(((site_cn_ad["CN"] > 0) & (site_cn_ad["AD"] > 0)).sum())
    main_sites      = loso[loso["loso_set"] == "main"]["site"].tolist()
    sens_sites      = loso[loso["loso_set"] == "sensitivity"]["site"].tolist()
    excl_sites      = loso[loso["loso_set"] == "excluded"]["site"].tolist()

    p_mfg = confounds["class_manufacturer_p"]
    V_mfg = confounds["class_manufacturer_cramersV"]
    p_site = confounds.get("class_site_p")
    V_site = confounds.get("class_site_cramersV")

    mfg_warning = (
        f"**WARNING — class×manufacturer confound: p={p_mfg:.3e}, V={V_mfg:.3f}**"
        if p_mfg < 0.05
        else f"Class×manufacturer: p={p_mfg:.3f}, V={V_mfg:.3f} (no significant imbalance)"
    )
    site_warning = (
        f"**WARNING — class×site confound: p={p_site:.3e}, V={V_site:.3f}**"
        if p_site is not None and p_site < 0.05
        else (
            f"Class×site: p={p_site:.3f}, V={V_site:.3f} (no significant imbalance)"
            if p_site is not None
            else "Class×site: insufficient data"
        )
    )

    main_list  = ", ".join(f"Site {s}" for s in sorted(main_sites))
    sens_list  = ", ".join(f"Site {s}" for s in sorted(sens_sites))

    # Manufacturer percentages per class
    mfg_lines = []
    for _, row in class_mfg.iterrows():
        mfg_lines.append(
            f"  - {row['Manufacturer']}: {int(row['CN'])} CN ({row['pct_CN']}%), "
            f"{int(row['AD'])} AD ({row['pct_AD']}%)"
        )
    mfg_text = "\n".join(mfg_lines)

    md = textwrap.dedent(f"""
    # ADNI Site Audit — BSPC 2026 Revision
    Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

    ## 1. Cohort Overview

    | Group | Metadata | Tensor | Analysis cohort |
    |-------|---------|--------|----------------|
    {chr(10).join(f"| {r.group} | {r.n_metadata} | {r.n_tensor} | {r.n_analysis_cohort} |" for r in overview.itertuples())}

    - **Total analysis cohort (CN + AD)**: {total_analysis} subjects
      ({n_cn} CN, {n_ad} AD)
    - MCI subjects are present in the tensor but excluded from the binary
      CN/AD classifier.
    - 3 metadata subjects are not present in the tensor (1 AD, 2 MCI);
      see cohort_overview.csv for details.
    - Paper reported: CN={PAPER_CN}, AD={PAPER_AD} (pre-pipeline drop) →
      final analysis: CN={n_cn}, AD={n_ad}.

    ## 2. Site Distribution

    - **Unique sites in analysis cohort**: {n_sites_total}
    - **Sites with both CN and AD**: {n_sites_dual}

    ## 3. Manufacturer Distribution (analysis cohort)

{mfg_text}

    ## 4. Confound Assessment

    - {mfg_warning}
    - {site_warning}

    > Philips dominates both CN and AD but at different proportions.
    > SIEMENS is over-represented in AD relative to CN.
    > Manufacturer effects should be controlled in LOSO analysis.

    ## 5. LOSO Feasibility

    Criteria:
    - **Main LOSO set**: both CN ≥ {LOSO_MAIN_MIN_PER_CLASS} and AD ≥ {LOSO_MAIN_MIN_PER_CLASS}
      → {len(main_sites)} sites
    - **Sensitivity set**: both CN ≥ {LOSO_SENSITIVITY_MIN_PER_CLASS} and AD ≥ {LOSO_SENSITIVITY_MIN_PER_CLASS}
      (includes main + smaller sites)
      → {len(main_sites) + len(sens_sites)} sites
    - **Excluded**: single class or too small
      → {len(excl_sites)} sites

    **Main LOSO candidates** ({len(main_sites)} sites):
    {main_list}

    **Sensitivity-only candidates** (add {len(sens_sites)} sites):
    {sens_list}

    ## 6. Recommended LOSO Design

    1. **Primary LOSO evaluation**: leave out one site at a time from the
       main LOSO set ({len(main_sites)} sites). Train on remaining sites,
       test on left-out site. Report per-site AUC and aggregate.
    2. **Sensitivity analysis**: repeat with the extended set
       ({len(main_sites) + len(sens_sites)} sites).
    3. **Manufacturer control**: include manufacturer as covariate or stratify;
       investigate whether SIEMENS–Philips imbalance drives LOSO variance.
    4. Sites in the excluded set (n={len(excl_sites)}) should be pooled into the
       training set only — never used as a LOSO test site.

    ## 7. Open Questions

    - Does site == scanner hardware? Verify that each site uses exactly one
      manufacturer (see site_by_manufacturer.csv).
    - Is acquisition protocol controlled within ADNI phases? Phase imbalance
      may confound LOSO if train/test splits happen to split phases.
    - 1 AD subject (013_S_6768) dropped at tensor stage — verify exclusion
      reason in the pipeline log.
    """).strip()
    return md


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading data...")
    meta       = _load_metadata()
    tensor_ids = _load_tensor_subject_ids()
    analysis   = _load_analysis_cohort(meta)

    print(f"  Metadata:          {len(meta)} subjects")
    print(f"  Tensor subjects:   {len(tensor_ids)}")
    print(f"  Analysis cohort:   {len(analysis)} subjects "
          f"({int((analysis['y_true']==0).sum())} CN, "
          f"{int((analysis['y_true']==1).sum())} AD)")

    # ── Build tables ──────────────────────────────────────────────────────────
    print("\nBuilding tables...")
    overview     = build_cohort_overview(meta, tensor_ids, analysis)
    site_all     = build_site_by_class_all(meta)
    site_cn_ad   = build_site_by_class_cn_ad(analysis)
    site_mfg     = build_site_by_manufacturer(meta)
    class_mfg    = build_class_by_manufacturer(analysis)
    phase_class  = build_phase_by_class(meta)
    loso         = build_loso_feasibility(site_cn_ad)
    confounds    = detect_confounds(analysis, site_cn_ad)

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    print("Saving CSVs...")
    _save_csv(overview,    "cohort_overview.csv")
    _save_csv(site_all,    "site_by_class_all.csv")
    _save_csv(site_cn_ad,  "site_by_class_cn_ad.csv")
    _save_csv(site_mfg,    "site_by_manufacturer.csv")
    _save_csv(class_mfg,   "class_by_manufacturer.csv")
    _save_csv(phase_class, "phase_by_class.csv")
    _save_csv(loso,        "site_loso_feasibility.csv")

    # ── Save figures ──────────────────────────────────────────────────────────
    print("Saving figures...")
    _save_fig(
        plot_heatmap_site_class_cn_ad(site_cn_ad),
        "heatmap_site_class_cn_ad.png",
    )
    _save_fig(
        plot_barplot_subjects_per_site_cn_ad(site_cn_ad, loso),
        "barplot_subjects_per_site_cn_ad.png",
    )
    _save_fig(
        plot_barplot_class_by_manufacturer(class_mfg),
        "barplot_class_by_manufacturer.png",
    )

    # ── Markdown summary ──────────────────────────────────────────────────────
    print("Writing summary...")
    summary_md = build_summary_md(
        overview, site_cn_ad, loso, class_mfg, confounds, analysis
    )
    p_md = RESULTS_DIR / "site_audit_summary.md"
    p_md.write_text(summary_md, encoding="utf-8")

    # ── Terminal summary ───────────────────────────────────────────────────────
    n_cn     = int((analysis["y_true"] == 0).sum())
    n_ad     = int((analysis["y_true"] == 1).sum())
    n_mci    = int((meta["ResearchGroup_Mapped"] == "MCI").sum())
    main_sites = loso[loso["loso_set"] == "main"]["site"].tolist()
    sens_sites = loso[loso["loso_set"] == "sensitivity"]["site"].tolist()

    p_mfg = confounds["class_manufacturer_p"]
    V_mfg = confounds["class_manufacturer_cramersV"]
    p_site = confounds.get("class_site_p")
    V_site = confounds.get("class_site_cramersV")

    print("\n" + "=" * 62)
    print("  ADNI SITE AUDIT — TERMINAL SUMMARY")
    print("=" * 62)
    print(f"  Total analysis cohort (CN+AD): {n_cn + n_ad}")
    print(f"    CN  : {n_cn}")
    print(f"    AD  : {n_ad}")
    print(f"    MCI : {n_mci}  (in metadata; excluded from classifier)")
    print(f"  Unique sites in analysis cohort: {site_cn_ad['Site3'].nunique()}")
    print(f"  Sites with both CN and AD: "
          f"{int(((site_cn_ad['CN']>0)&(site_cn_ad['AD']>0)).sum())}")
    print()
    print(f"  LOSO feasibility (main, ≥{LOSO_MAIN_MIN_PER_CLASS}/class):")
    print(f"    {len(main_sites)} sites: {sorted(main_sites)}")
    print(f"  LOSO feasibility (sensitivity, ≥{LOSO_SENSITIVITY_MIN_PER_CLASS}/class):")
    print(f"    +{len(sens_sites)} additional sites: {sorted(sens_sites)}")
    print()
    print("  Confound warnings:")
    mfg_ok = "OK" if p_mfg >= 0.05 else "WARNING"
    site_ok = ("OK" if (p_site is not None and p_site >= 0.05) else "WARNING")
    print(f"    Class×Manufacturer: p={p_mfg:.3e}, V={V_mfg:.3f}  [{mfg_ok}]")
    print(f"    Class×Site:         p={p_site:.3e}, V={V_site:.3f}  [{site_ok}]"
          if p_site is not None else "    Class×Site:         insufficient data")
    print()
    print(f"  Outputs → {RESULTS_DIR}")
    print("=" * 62)


if __name__ == "__main__":
    main()
