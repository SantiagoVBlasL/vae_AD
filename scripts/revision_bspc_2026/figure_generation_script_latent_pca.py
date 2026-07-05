#!/usr/bin/env python3
"""
Latent-space PCA visualization for Philips CN false-positive audit.

Promoted model: recover035_latent384_beta3p75_T80_h10000_p560_full5x5

Coordinate-system audit outcome:
  CASE 2 (strict OOF): each subject's OOF test latent comes from a
  fold-specific encoder -> pooling 5 OOF test sets into one PCA is invalid.

  Modified CASE 1 (used here for primary figures): fold 1's latent_cache
  covers ALL 397 CN+AD subjects (test_n=80 + trainDev_n=317) from the
  same fold-1 VAE encoder.  This gives a valid common coordinate system
  for structural visualization.  trainDev subjects were in fold 1's
  training set (encoder has seen them); test subjects were held out.
  Both groups are from the same encoder -> PCA axes are consistent.

  Fig 6 (exploratory): Procrustes-aligned OOF test latents across 5 folds.

Outputs (to results/revision_bspc_2026/philips_director_figures_20260612/):
  latent_coordinate_system_audit.md
  fig4_latent_pca3d_manufacturer.{png,svg}
  fig5_latent_pca3d_philips_site.{png,svg}
  fig6_procrustes_aligned_pca_exploratory.{png,svg}
  plotted_values_fig4.csv
  plotted_values_fig5.csv
  plotted_values_fig6.csv
  fig4_caption.md, fig5_caption.md, fig6_caption.md
  00_LATENT_FIGURES_README.md
  command_log.json

Hard constraints: read-only data, no retraining, no tensor/metadata edits.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results" / "revision_bspc_2026"
OUT = RESULTS / "philips_director_figures_20260612"
OUT.mkdir(parents=True, exist_ok=True)

PROMOTED_RUN = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
LATENT_CACHE = PROMOTED_RUN / "classifier_only_readout" / "latent_cache"
MANIFEST_PATH = PROMOTED_RUN / "classifier_only_readout" / "latent_feature_manifest.json"
MASTER_DB_PATH = RESULTS / "promoted_model_master_database_20260610" / "promoted_model_master_database.csv"

# ---------------------------------------------------------------------------
# Style constants (Wong / Tol colorblind-safe palette)
# ---------------------------------------------------------------------------
MFR_COLORS = {
    "Philips":  "#D55E00",   # vermilion
    "SIEMENS":  "#56B4E9",   # sky blue
    "GE":       "#009E73",   # teal green
}
DIAG_MARKERS = {"CN": "o", "AD": "^", "MCI": "s"}
FP_EDGE   = "#000000"  # black outline for Philips CN FP
FP_LW     = 1.6
TN_EDGE   = "#FFFFFF"
TN_LW     = 0.4

# Site palette for Philips (up to 15 distinct sites, Set1 + extras)
_SITE_PALETTE = [
    "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00",
    "#A65628", "#F781BF", "#999999", "#66C2A5", "#FC8D62",
    "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F", "#B3B3B3",
]

plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "font.size":     11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "figure.dpi":    150,
})

_LOG: list[dict[str, Any]] = []
_T0 = datetime.now(timezone.utc)


def _log(event: str, detail: Any = None) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    entry = {"ts": ts, "event": event}
    if detail is not None:
        entry["detail"] = str(detail) if not isinstance(detail, (dict, list, int, float, bool)) else detail
    _LOG.append(entry)
    print(f"[{ts}] {event}" + (f": {detail}" if detail is not None else ""))


def _savefig(fig: plt.Figure, stem: str) -> None:
    for ext, kw in [("png", {"dpi": 300}), ("svg", {})]:
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", facecolor="white", **kw)
    plt.close(fig)
    _log(f"Saved figure", stem)


def _write_md(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")
    _log("Saved MD", path.name)


# ---------------------------------------------------------------------------
# STEP 1 — Coordinate-system audit
# ---------------------------------------------------------------------------
def _run_coordinate_audit() -> dict:
    """Return a dict of audit facts and write the audit markdown."""
    manifest = json.loads(MANIFEST_PATH.read_text())

    fold_checkpoints = {e["fold"]: Path(e["checkpoint"]) for e in manifest["folds"]}
    n_distinct_encoders = len(set(str(p) for p in fold_checkpoints.values()))

    # Check OOF non-overlap
    test_sids: dict[int, list] = {}
    for f in range(1, 6):
        df = pd.read_csv(LATENT_CACHE / f"fold_{f}_test_latent_mu.csv", usecols=["SubjectID"])
        test_sids[f] = df.SubjectID.tolist()
    all_test = [s for sids in test_sids.values() for s in sids]
    oof_unique = len(set(all_test))
    oof_total  = len(all_test)
    oof_non_overlapping = oof_unique == oof_total

    # Fold 1 full-cache coverage
    t1  = pd.read_csv(LATENT_CACHE / "fold_1_test_latent_mu.csv",     usecols=["SubjectID"])
    td1 = pd.read_csv(LATENT_CACHE / "fold_1_trainDev_latent_mu.csv", usecols=["SubjectID"])
    f1_combined_n   = len(t1) + len(td1)
    f1_unique_n     = len(set(t1.SubjectID) | set(td1.SubjectID))
    f1_has_duplicates = f1_combined_n != f1_unique_n

    mdb = pd.read_csv(MASTER_DB_PATH, usecols=["SubjectID", "ResearchGroup_Mapped"])
    clf_pool = mdb[mdb.ResearchGroup_Mapped.isin(["CN", "AD"])]
    f1_covers_all = set(clf_pool.SubjectID).issubset(set(t1.SubjectID) | set(td1.SubjectID))

    facts = {
        "n_folds": len(manifest["folds"]),
        "n_distinct_encoders": n_distinct_encoders,
        "oof_non_overlapping": oof_non_overlapping,
        "oof_total_test": oof_total,
        "oof_unique_subjects": oof_unique,
        "fold1_combined_n": f1_combined_n,
        "fold1_unique_n": f1_unique_n,
        "fold1_has_duplicates": f1_has_duplicates,
        "fold1_covers_all_clf_pool": f1_covers_all,
        "clf_pool_n": len(clf_pool),
        "checkpoints": {k: str(v) for k, v in fold_checkpoints.items()},
        "common_encoder_available": f1_covers_all and not f1_has_duplicates,
    }

    verdict = "MODIFIED CASE 1" if facts["common_encoder_available"] else "CASE 2"
    case_text = (
        "A single-fold full-cache view is available (fold 1 test + trainDev covers all 397 "
        "CN+AD subjects from the same fold-1 VAE encoder).  Primary figures use this common "
        "coordinate system.  trainDev subjects (n=317) were in fold 1's training set; test "
        "subjects (n=80) were held out.  Both groups are encoded by the same VAE checkpoint."
        if facts["common_encoder_available"] else
        "No valid common coordinate system found.  Fold-wise small multiples only."
    )

    audit_md = f"""# Latent Coordinate-System Audit

**Script**: figure_generation_script_latent_pca.py
**Date**: {datetime.now(timezone.utc).isoformat()}
**Promoted model**: recover035_latent384_beta3p75_T80_h10000_p560_full5x5

---

## Question 1 — Were all OOF latent vectors from the same encoder?

**NO.**

The promoted model uses a 5-fold outer cross-validation.  Each outer fold trains
an independent VAE.  The OOF test latents for each subject come from the encoder
that did NOT see that subject during training.

- Number of outer folds: {facts["n_folds"]}
- Number of distinct VAE checkpoints: {facts["n_distinct_encoders"]}
- OOF test subjects per fold: {[len(v) for v in test_sids.values()]}
- OOF non-overlapping (each subject in exactly one test set): {facts["oof_non_overlapping"]}
- Total unique OOF subjects: {facts["oof_unique_subjects"]}

**Fold checkpoints (different for each fold):**
{chr(10).join(f"  - Fold {k}: {v}" for k, v in facts["checkpoints"].items())}

---

## Question 2 — Is a pooled OOF PCA valid?

**NO** (strict OOF).

Pooling OOF test latents from 5 different fold-specific encoders creates a
composite space with up to 5 arbitrary rotational degrees of freedom.  Structure
visible in such a plot could reflect encoder-to-encoder coordinate variation
rather than genuine data geometry.  A pooled OOF PCA is not scientifically
defensible without prior alignment.

---

## Question 3 — Is any common coordinate system available?

**YES — Modified CASE 1 (single-fold full cache).**

For each fold k, the latent cache contains:
  - fold_k_test_latent_mu.csv     (held-out test subjects)
  - fold_k_trainDev_latent_mu.csv (training subjects)

Together, fold 1's test + trainDev covers **all {facts["fold1_unique_n"]} CN+AD subjects**
without duplicates.  All {facts["fold1_unique_n"]} latent vectors were generated by the
**same fold-1 VAE checkpoint**.  This is a valid common coordinate system.

- Fold 1 test n:     {len(t1)}
- Fold 1 trainDev n: {len(td1)}
- Fold 1 combined n: {facts["fold1_combined_n"]}
- Fold 1 unique n:   {facts["fold1_unique_n"]}
- Covers full classifier pool ({facts["clf_pool_n"]} subjects): {facts["fold1_covers_all_clf_pool"]}
- Duplicates: {facts["fold1_has_duplicates"]}

**Caveat**: trainDev subjects were seen by fold 1's encoder during training.  Their
latent representations may reflect in-distribution memorization.  For structural
visualization this is acceptable; for generalization claims it is not.

---

## Audit Verdict: {verdict}

{case_text}

---

## Visualization Choices

| Figure | Source | Description |
|---|---|---|
| fig4 | Fold 1 full cache (n=397) | 3D PCA, color by Manufacturer |
| fig5 | Fold 1 full cache, Philips (n=145) | 3D PCA, color by Site3 |
| fig6 | Procrustes-aligned OOF test (n=397) | Exploratory only |

---

## Safest Alternative if Common Encoder Were Unavailable

If the fold 1 full-cache approach were rejected, the hierarchy of alternatives is:

A) Fold-wise PCA small multiples — each panel uses only one fold's OOF test
   subjects (~79–80 subjects); coordinate system is consistent within each panel
   but not comparable across panels.

B) Procrustes-aligned pooled latents — align fold 2–5 test latents to fold 1
   as reference using shared anchor subjects.  Exploratory; removes arbitrary
   rotations but cannot correct for learned representation differences.

C) No PCA — use only centroid/distance geometry (already implemented in
   fig2_philips_latent_centroid_shift.png from the prior director figures session).

---

## Files

- **Primary latent source**: {LATENT_CACHE}
- **Manifest**: {MANIFEST_PATH}
- **Master DB**: {MASTER_DB_PATH}
"""

    _write_md(OUT / "latent_coordinate_system_audit.md", audit_md)
    return facts


# ---------------------------------------------------------------------------
# STEP 2 — Load fold 1 full cache + master DB merge
# ---------------------------------------------------------------------------
def _load_data(facts: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (full_df, mu_cols_list) where full_df has metadata + fold1 mu coords."""
    _log("Loading fold 1 full cache")

    t1  = pd.read_csv(LATENT_CACHE / "fold_1_test_latent_mu.csv")
    td1 = pd.read_csv(LATENT_CACHE / "fold_1_trainDev_latent_mu.csv")
    t1["split_in_fold1"]  = "test"
    td1["split_in_fold1"] = "trainDev"
    lat = pd.concat([t1, td1], ignore_index=True)

    mu_cols = [c for c in lat.columns if c.startswith("mu_")]
    assert len(mu_cols) == 384, f"Expected 384 mu cols, got {len(mu_cols)}"

    # Load master DB for extra metadata
    mdb_cols = [
        "SubjectID", "Manufacturer", "ResearchGroup_Mapped", "Site3",
        "is_philips_cn", "philips_cn_error_type", "y_score_final",
        "raw_tp_group", "inferred_ADNI_phase", "Age",
    ]
    mdb_avail = pd.read_csv(MASTER_DB_PATH, low_memory=False)
    mdb = mdb_avail[[c for c in mdb_cols if c in mdb_avail.columns]].copy()

    # Merge
    df = lat.merge(mdb, on="SubjectID", how="left", suffixes=("", "_mdb"))

    # Derive confusion label for Philips CN
    def _philips_confusion(row):
        if row.get("Manufacturer") != "Philips":
            return None
        if row.get("ResearchGroup_Mapped") not in ("CN",):
            return None
        err = str(row.get("philips_cn_error_type", "")).upper()
        return "FP" if "FP" in err else ("TN" if "TN" in err else None)

    df["philips_cn_confusion"] = df.apply(_philips_confusion, axis=1)

    _log("Fold 1 full cache loaded", {"n": len(df), "mu_cols": len(mu_cols)})
    return df, mu_cols


# ---------------------------------------------------------------------------
# STEP 3 — PCA helpers
# ---------------------------------------------------------------------------
def _fit_pca(df: pd.DataFrame, mu_cols: list[str], n: int = 3) -> tuple[np.ndarray, np.ndarray]:
    X = df[mu_cols].to_numpy(dtype=np.float32)
    pca = PCA(n_components=n, random_state=0)
    coords = pca.fit_transform(X)
    return coords, pca.explained_variance_ratio_


def _add_pca_cols(df: pd.DataFrame, coords: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    for i in range(coords.shape[1]):
        df[f"PC{i+1}"] = coords[:, i]
    return df


# ---------------------------------------------------------------------------
# STEP 4 — Figure 4: Manufacturer
# ---------------------------------------------------------------------------
def _make_fig4(df: pd.DataFrame, evr: np.ndarray) -> None:
    _log("Generating fig4 (manufacturer PCA)")

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("white")

    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax12 = fig.add_subplot(2, 2, 2)
    ax13 = fig.add_subplot(2, 2, 3)
    ax23 = fig.add_subplot(2, 2, 4)

    axes_2d = {
        "PC1-PC2": (ax12, "PC1", "PC2"),
        "PC1-PC3": (ax13, "PC1", "PC3"),
        "PC2-PC3": (ax23, "PC2", "PC3"),
    }

    # Build plot groups: Manufacturer × Diagnosis (CN, AD)
    # Order: non-FP first, FP last (so FP markers draw on top)
    def _get_marker(diag):
        return DIAG_MARKERS.get(diag, "o")

    def _plot_scatter3d(ax, x, y, z, color, marker, ec, lw, alpha, size, label):
        ax.scatter(x, y, z, c=color, marker=marker, edgecolors=ec, linewidths=lw,
                   alpha=alpha, s=size, label=label, depthshade=True)

    def _plot_scatter2d(ax, x, y, color, marker, ec, lw, alpha, size, label):
        ax.scatter(x, y, c=color, marker=marker, edgecolors=ec, linewidths=lw,
                   alpha=alpha, s=size, label=label)

    legend_handles = []
    legend_labels  = []
    seen_labels    = set()

    for mfr in ["GE", "SIEMENS", "Philips"]:
        for diag in ["CN", "AD"]:
            sub = df[(df["Manufacturer"] == mfr) & (df["ResearchGroup_Mapped"] == diag)]
            if len(sub) == 0:
                continue
            color  = MFR_COLORS[mfr]
            marker = _get_marker(diag)

            # Split Philips CN into TN and FP
            if mfr == "Philips" and diag == "CN":
                for confusion, ec, lw, alpha, sz, suffix in [
                    ("TN", TN_EDGE, TN_LW, 0.70, 55, " CN-TN"),
                    ("FP", FP_EDGE, FP_LW, 0.92, 90, " CN-FP"),
                    (None, TN_EDGE, TN_LW, 0.55, 45, " CN-?"),  # unknown confusion
                ]:
                    if confusion is None:
                        sub2 = sub[sub["philips_cn_confusion"].isna()]
                    else:
                        sub2 = sub[sub["philips_cn_confusion"] == confusion]
                    if len(sub2) == 0:
                        continue
                    lbl = f"{mfr}{suffix}"
                    _plot_scatter3d(ax3d, sub2.PC1, sub2.PC2, sub2.PC3,
                                    color, marker, ec, lw, alpha, sz, lbl)
                    for ax, xcol, ycol in axes_2d.values():
                        _plot_scatter2d(ax, sub2[xcol], sub2[ycol],
                                        color, marker, ec, lw, alpha, sz, lbl)
                    if lbl not in seen_labels:
                        import matplotlib.lines as mlines
                        h = mlines.Line2D([], [], color=color, marker=marker, linestyle="None",
                                          markeredgecolor=ec, markeredgewidth=lw,
                                          markersize=8 if confusion != "FP" else 10, label=lbl)
                        legend_handles.append(h)
                        legend_labels.append(lbl)
                        seen_labels.add(lbl)
            else:
                alpha = 0.65
                sz    = 45 if diag == "CN" else 60
                lbl   = f"{mfr} {diag}"
                _plot_scatter3d(ax3d, sub.PC1, sub.PC2, sub.PC3,
                                color, marker, TN_EDGE, TN_LW, alpha, sz, lbl)
                for ax, xcol, ycol in axes_2d.values():
                    _plot_scatter2d(ax, sub[xcol], sub[ycol],
                                    color, marker, TN_EDGE, TN_LW, alpha, sz, lbl)
                if lbl not in seen_labels:
                    import matplotlib.lines as mlines
                    h = mlines.Line2D([], [], color=color, marker=marker, linestyle="None",
                                      markeredgecolor=TN_EDGE, markeredgewidth=TN_LW,
                                      markersize=8, label=lbl)
                    legend_handles.append(h)
                    legend_labels.append(lbl)
                    seen_labels.add(lbl)

    # 3D labels
    ax3d.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)", labelpad=8)
    ax3d.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)", labelpad=8)
    ax3d.set_zlabel(f"PC3 ({evr[2]*100:.1f}%)", labelpad=8)
    ax3d.set_title("3D PCA — Manufacturer")
    ax3d.view_init(elev=22, azim=-55)
    ax3d.grid(True, linewidth=0.5, alpha=0.4)

    # 2D labels
    for name, (ax, xcol, ycol) in axes_2d.items():
        pc_x = int(xcol[-1])
        pc_y = int(ycol[-1])
        ax.set_xlabel(f"{xcol} ({evr[pc_x-1]*100:.1f}%)")
        ax.set_ylabel(f"{ycol} ({evr[pc_y-1]*100:.1f}%)")
        ax.set_title(name)
        ax.grid(True, color="#EEEEEE", linewidth=0.8)
        ax.set_axisbelow(True)

    pct_total = sum(evr) * 100
    fig.suptitle(
        f"Latent PCA — Manufacturer  |  {len(df)} subjects (fold-1 encoder)\n"
        f"PC1–PC3 cumulative variance: {pct_total:.1f}%\n"
        f"(descriptive/internal — fold-1 full cache: test n=80 + trainDev n=317)",
        fontsize=11, y=1.01,
    )

    fig.legend(legend_handles, legend_labels,
               loc="upper right", bbox_to_anchor=(1.18, 0.95),
               frameon=True, title="Manufacturer / Group", ncol=1)
    fig.tight_layout()
    _savefig(fig, "fig4_latent_pca3d_manufacturer")


# ---------------------------------------------------------------------------
# STEP 5 — Figure 5: Philips site
# ---------------------------------------------------------------------------
def _make_fig5(df: pd.DataFrame, evr: np.ndarray) -> None:
    _log("Generating fig5 (Philips site PCA)")

    ph = df[df["Manufacturer"] == "Philips"].copy()
    _log("Philips subjects", len(ph))

    # Site3 as string
    ph["Site3_str"] = ph["Site3"].apply(
        lambda x: str(int(x)) if pd.notna(x) and x != "" else "MISS"
    )
    sites_ordered = (
        ph.groupby("Site3_str").size()
        .sort_values(ascending=False)
        .index.tolist()
    )
    label_sites = {"2", "13", "18", "31", "130", "177", "301", "6", "100"}

    site_color = {}
    for i, s in enumerate(sites_ordered):
        site_color[s] = _SITE_PALETTE[i % len(_SITE_PALETTE)]

    # Marker by TN/FP/AD
    def _row_marker(row):
        diag = row.get("ResearchGroup_Mapped", "?")
        if diag == "AD":
            return "^"
        conf = row.get("philips_cn_confusion", None)
        if conf == "FP":
            return "*"   # star for FP
        return "o"       # circle for TN or unknown CN

    def _row_size(row):
        diag = row.get("ResearchGroup_Mapped", "?")
        if diag == "AD":
            return 65
        conf = row.get("philips_cn_confusion", None)
        return 110 if conf == "FP" else 55

    def _row_alpha(row):
        return 0.92 if row.get("philips_cn_confusion") == "FP" else 0.68

    # TP group edge style: 140TP = thick edge, 197TP = thin edge
    def _row_lw(row):
        tp = str(row.get("raw_tp_group", "")).strip()
        return 1.8 if "140" in tp else 0.4

    def _row_edge(row):
        conf = row.get("philips_cn_confusion", None)
        return FP_EDGE if conf == "FP" else "#555555"

    ph["_marker"] = ph.apply(_row_marker, axis=1)
    ph["_size"]   = ph.apply(_row_size, axis=1)
    ph["_alpha"]  = ph.apply(_row_alpha, axis=1)
    ph["_lw"]     = ph.apply(_row_lw, axis=1)
    ph["_edge"]   = ph.apply(_row_edge, axis=1)

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("white")
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax12 = fig.add_subplot(2, 2, 2)
    ax13 = fig.add_subplot(2, 2, 3)
    ax23 = fig.add_subplot(2, 2, 4)
    axes_2d = {
        "PC1-PC2": (ax12, "PC1", "PC2"),
        "PC1-PC3": (ax13, "PC1", "PC3"),
        "PC2-PC3": (ax23, "PC2", "PC3"),
    }

    for site in sites_ordered:
        sub = ph[ph["Site3_str"] == site]
        c   = site_color[site]
        for _, row in sub.iterrows():
            kw = dict(c=c, marker=row["_marker"], edgecolors=row["_edge"],
                      linewidths=row["_lw"], alpha=row["_alpha"], s=row["_size"])
            ax3d.scatter(row.PC1, row.PC2, row.PC3, depthshade=True, **kw)
            for ax, xcol, ycol in axes_2d.values():
                ax.scatter(row[xcol], row[ycol], **kw)

    # Site labels on PC1-PC2 for labellable sites (centroid)
    for site in label_sites:
        sub = ph[ph["Site3_str"] == site]
        if len(sub) == 0:
            continue
        cx, cy = sub.PC1.mean(), sub.PC2.mean()
        ax12.annotate(
            f"s{site}", (cx, cy),
            fontsize=8, ha="center", va="bottom",
            color=site_color.get(site, "#333333"),
            fontweight="bold",
        )

    # 3D labels
    ax3d.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)", labelpad=8)
    ax3d.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)", labelpad=8)
    ax3d.set_zlabel(f"PC3 ({evr[2]*100:.1f}%)", labelpad=8)
    ax3d.set_title("3D PCA — Philips subjects, by Site")
    ax3d.view_init(elev=22, azim=-55)
    ax3d.grid(True, linewidth=0.5, alpha=0.4)

    for name, (ax, xcol, ycol) in axes_2d.items():
        pc_x, pc_y = int(xcol[-1]), int(ycol[-1])
        ax.set_xlabel(f"{xcol} ({evr[pc_x-1]*100:.1f}%)")
        ax.set_ylabel(f"{ycol} ({evr[pc_y-1]*100:.1f}%)")
        ax.set_title(name)
        ax.grid(True, color="#EEEEEE", linewidth=0.8)
        ax.set_axisbelow(True)

    # Legend: sites (marker=circle), shape legend, tp edge legend
    import matplotlib.patches as mpatches
    import matplotlib.lines  as mlines

    site_patches = [
        mpatches.Patch(color=site_color[s], label=f"Site {s}" + (" ★" if s in label_sites else ""))
        for s in sites_ordered[:12]  # cap at 12 site swatches
    ]
    shape_handles = [
        mlines.Line2D([], [], marker="o", linestyle="None", color="#888888", label="CN-TN", markersize=8),
        mlines.Line2D([], [], marker="*", linestyle="None", color="#888888", label="CN-FP", markersize=12,
                      markeredgecolor=FP_EDGE, markeredgewidth=1.4),
        mlines.Line2D([], [], marker="^", linestyle="None", color="#888888", label="AD",    markersize=8),
    ]
    tp_handles = [
        mlines.Line2D([], [], marker="o", linestyle="None", color="#888888", label="140 TP (thick edge)",
                      markeredgecolor="#555555", markeredgewidth=1.8, markersize=8),
        mlines.Line2D([], [], marker="o", linestyle="None", color="#888888", label="197 TP (thin edge)",
                      markeredgecolor="#555555", markeredgewidth=0.4, markersize=8),
    ]
    all_handles = site_patches + shape_handles + tp_handles

    pct_total = sum(evr) * 100
    fig.suptitle(
        f"Latent PCA — Philips subjects (n={len(ph)})  |  fold-1 encoder\n"
        f"Color = Site3, Shape = group (○ CN-TN, ★ CN-FP, △ AD), Edge width = 140/197 TP\n"
        f"PC1–PC3 cumulative variance: {pct_total:.1f}%",
        fontsize=11, y=1.01,
    )
    fig.legend(all_handles, [h.get_label() for h in all_handles],
               loc="upper right", bbox_to_anchor=(1.22, 0.95),
               frameon=True, ncol=1, fontsize=8)
    fig.tight_layout()
    _savefig(fig, "fig5_latent_pca3d_philips_site")


# ---------------------------------------------------------------------------
# STEP 6 — Figure 6: Procrustes-aligned OOF test latents (exploratory)
# ---------------------------------------------------------------------------
def _make_fig6(mu_cols: list[str]) -> None:
    _log("Generating fig6 (Procrustes-aligned OOF test latents)")
    from scipy.linalg import orthogonal_procrustes

    # Load all 5 fold full caches, keyed by SubjectID
    # Each fold's cache: test + trainDev = 397 subjects from that fold's encoder
    fold_dfs: dict[int, pd.DataFrame] = {}
    for f in range(1, 6):
        t  = pd.read_csv(LATENT_CACHE / f"fold_{f}_test_latent_mu.csv")
        td = pd.read_csv(LATENT_CACHE / f"fold_{f}_trainDev_latent_mu.csv")
        combined = pd.concat([t, td], ignore_index=True)
        combined["fold_source"] = f
        fold_dfs[f] = combined.sort_values("SubjectID").reset_index(drop=True)

    # Reference: fold 1 sorted by SubjectID
    ref = fold_dfs[1]
    X_ref = ref[mu_cols].to_numpy(dtype=np.float64)
    X_ref_centered = X_ref - X_ref.mean(axis=0)

    # Align each fold to fold 1
    aligned_oof: list[pd.DataFrame] = []
    # Fold 1 OOF test subjects (encoded by fold 1 — held out)
    f1_test = pd.read_csv(LATENT_CACHE / "fold_1_test_latent_mu.csv")

    for f in range(1, 6):
        other = fold_dfs[f]
        # Verify same subjects and order
        assert list(other.SubjectID) == list(ref.SubjectID), f"Fold {f} subject order mismatch"
        X_other = other[mu_cols].to_numpy(dtype=np.float64)
        X_other_centered = X_other - X_other.mean(axis=0)
        if f == 1:
            X_aligned = X_other_centered
        else:
            R, _ = orthogonal_procrustes(X_other_centered, X_ref_centered)
            X_aligned = X_other_centered @ R

        # Extract OOF test subjects for this fold
        oof_test_f = pd.read_csv(LATENT_CACHE / f"fold_{f}_test_latent_mu.csv",
                                  usecols=["SubjectID"])
        oof_sids = set(oof_test_f.SubjectID)
        mask = other.SubjectID.isin(oof_sids)
        oof_aligned = other.loc[mask, ["SubjectID"]].copy()
        coords_oof  = X_aligned[mask.to_numpy()]
        for i, col in enumerate(mu_cols):
            oof_aligned[col] = coords_oof[:, i]
        oof_aligned["oof_fold"] = f
        aligned_oof.append(oof_aligned)
        _log(f"  Fold {f} OOF test after Procrustes alignment", len(oof_aligned))

    pooled = pd.concat(aligned_oof, ignore_index=True)
    assert pooled.SubjectID.nunique() == 397, "Procrustes OOF pool should have 397 unique subjects"

    # PCA on aligned OOF
    pca = PCA(n_components=3, random_state=0)
    coords = pca.fit_transform(pooled[mu_cols].to_numpy(dtype=np.float32))
    evr = pca.explained_variance_ratio_
    pooled["PC1"] = coords[:, 0]
    pooled["PC2"] = coords[:, 1]
    pooled["PC3"] = coords[:, 2]

    # Merge metadata
    mdb = pd.read_csv(MASTER_DB_PATH, usecols=[
        "SubjectID", "Manufacturer", "ResearchGroup_Mapped",
        "philips_cn_error_type",
    ])
    pooled = pooled.merge(mdb, on="SubjectID", how="left")

    # Save CSV
    pooled_out = pooled[["SubjectID", "oof_fold", "Manufacturer", "ResearchGroup_Mapped",
                          "philips_cn_confusion"] if "philips_cn_confusion" in pooled.columns
                         else ["SubjectID", "oof_fold", "Manufacturer", "ResearchGroup_Mapped",
                               "PC1", "PC2", "PC3"]].copy()
    # add PC cols
    pooled_out = pooled[["SubjectID", "oof_fold", "Manufacturer", "ResearchGroup_Mapped",
                          "PC1", "PC2", "PC3"]].copy()
    pooled_out.to_csv(OUT / "plotted_values_fig6.csv", index=False)
    _log("Saved plotted_values_fig6.csv")

    # --- Figure ---
    fig = plt.figure(figsize=(20, 7))
    fig.patch.set_facecolor("white")
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax12 = fig.add_subplot(1, 2, 2)

    import matplotlib.lines as mlines
    for mfr in ["GE", "SIEMENS", "Philips"]:
        for diag in ["CN", "AD"]:
            sub = pooled[(pooled["Manufacturer"] == mfr) & (pooled["ResearchGroup_Mapped"] == diag)]
            if len(sub) == 0:
                continue
            color  = MFR_COLORS[mfr]
            marker = DIAG_MARKERS.get(diag, "o")
            alpha  = 0.65
            sz     = 40
            # Highlight Philips CN FP
            if mfr == "Philips" and diag == "CN":
                ec_col = pooled.loc[sub.index, "philips_cn_error_type"].apply(
                    lambda x: FP_EDGE if "FP" in str(x).upper() else TN_EDGE
                )
                lw_col = pooled.loc[sub.index, "philips_cn_error_type"].apply(
                    lambda x: FP_LW if "FP" in str(x).upper() else TN_LW
                )
                sz_col = pooled.loc[sub.index, "philips_cn_error_type"].apply(
                    lambda x: 80 if "FP" in str(x).upper() else 45
                )
                for _, row in sub.iterrows():
                    ec  = FP_EDGE if "FP" in str(row.get("philips_cn_error_type", "")).upper() else TN_EDGE
                    lw2 = FP_LW  if "FP" in str(row.get("philips_cn_error_type", "")).upper() else TN_LW
                    sz2 = 80     if "FP" in str(row.get("philips_cn_error_type", "")).upper() else 45
                    ax3d.scatter(row.PC1, row.PC2, row.PC3, c=color, marker=marker,
                                 edgecolors=ec, linewidths=lw2, alpha=0.75, s=sz2, depthshade=True)
                    ax12.scatter(row.PC1, row.PC2, c=color, marker=marker,
                                 edgecolors=ec, linewidths=lw2, alpha=0.75, s=sz2)
            else:
                lbl = f"{mfr} {diag}"
                ax3d.scatter(sub.PC1, sub.PC2, sub.PC3, c=color, marker=marker,
                             edgecolors=TN_EDGE, linewidths=TN_LW, alpha=alpha, s=sz,
                             label=lbl, depthshade=True)
                ax12.scatter(sub.PC1, sub.PC2, c=color, marker=marker,
                             edgecolors=TN_EDGE, linewidths=TN_LW, alpha=alpha, s=sz, label=lbl)

    ax3d.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)", labelpad=6)
    ax3d.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)", labelpad=6)
    ax3d.set_zlabel(f"PC3 ({evr[2]*100:.1f}%)", labelpad=6)
    ax3d.set_title("Procrustes-aligned OOF — 3D")
    ax3d.view_init(elev=22, azim=-55)
    ax3d.grid(True, linewidth=0.5, alpha=0.4)

    ax12.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax12.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax12.set_title("PC1 vs PC2")
    ax12.grid(True, color="#EEEEEE", linewidth=0.8)
    ax12.set_axisbelow(True)

    pct_total = sum(evr) * 100
    fig.suptitle(
        "⚠ EXPLORATORY ONLY — Procrustes-aligned OOF test latents (5 fold-specific encoders)\n"
        f"n=397 (each subject from their OOF fold, aligned to fold-1 coordinate system)\n"
        f"PC1–PC3 cumulative variance: {pct_total:.1f}%  |  Alignment may not fully correct encoder divergence",
        fontsize=10, y=1.02, color="#AA3300",
    )

    handles, labels = ax12.get_legend_handles_labels()
    philips_fp_handle = mlines.Line2D(
        [], [], marker="o", linestyle="None", color=MFR_COLORS["Philips"],
        markeredgecolor=FP_EDGE, markeredgewidth=FP_LW, markersize=10, label="Philips CN-FP (black edge)"
    )
    fig.legend(handles + [philips_fp_handle], labels + ["Philips CN-FP (black edge)"],
               loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=5, frameon=True)
    fig.tight_layout()
    _savefig(fig, "fig6_procrustes_aligned_pca_exploratory")

    return evr


# ---------------------------------------------------------------------------
# STEP 7 — Save CSV outputs
# ---------------------------------------------------------------------------
def _save_csvs(df: pd.DataFrame) -> None:
    # fig4 CSV: all subjects
    cols4 = ["SubjectID", "Manufacturer", "ResearchGroup_Mapped", "Age",
             "split_in_fold1", "philips_cn_confusion", "y_score_final", "PC1", "PC2", "PC3"]
    out4 = df[[c for c in cols4 if c in df.columns]].copy()
    out4.to_csv(OUT / "plotted_values_fig4.csv", index=False)
    _log("Saved plotted_values_fig4.csv")

    # fig5 CSV: Philips only
    ph = df[df["Manufacturer"] == "Philips"].copy()
    cols5 = ["SubjectID", "Manufacturer", "ResearchGroup_Mapped", "Site3",
             "split_in_fold1", "philips_cn_confusion", "raw_tp_group",
             "y_score_final", "PC1", "PC2", "PC3"]
    out5 = ph[[c for c in cols5 if c in ph.columns]].copy()
    out5.to_csv(OUT / "plotted_values_fig5.csv", index=False)
    _log("Saved plotted_values_fig5.csv")


# ---------------------------------------------------------------------------
# STEP 8 — Captions and README
# ---------------------------------------------------------------------------
def _write_captions(evr4: np.ndarray, evr6: np.ndarray, n_subjects: int) -> None:
    pct4 = evr4.sum() * 100
    pct6 = evr6.sum() * 100 if evr6 is not None else float("nan")

    _write_md(OUT / "fig4_caption.md", f"""
## Figure 4 — Latent PCA: Manufacturer

Principal component analysis (PCA) of 384-dimensional latent mean (μ) vectors
from the promoted β-VAE model (recover035_latent384_beta3p75_T80_h10000_p560_full5x5).
All {n_subjects} CN+AD subjects from the classifier pool are shown, encoded by the
fold-1 VAE encoder (consistent coordinate system; trainDev subjects were in fold-1's
training set and are so labelled).  PC1–PC3 explain {pct4:.1f}% of the variance.

Color indicates scanner manufacturer (Philips, SIEMENS, GE).
Marker shape distinguishes CN (circle) and AD (triangle).
Philips CN false positives (score ≥ 0.5 at OOF-ECDF threshold) are highlighted
with a thick black outline.

Interpretation is descriptive and exploratory.  Any apparent clustering by
manufacturer is consistent with the known scanner-level confound in the training
cohort (all CN training subjects are Philips; see main text), but this figure does
not establish causality.  trainDev latent positions may differ from held-out subjects
of the same encoder due to in-distribution fit.
""")

    _write_md(OUT / "fig5_caption.md", f"""
## Figure 5 — Latent PCA: Philips Subjects by Site

Principal component analysis of latent μ vectors for Philips subjects only (n=145),
using the same fold-1 encoder coordinate system as Figure 4.
Color indicates ADNI acquisition site (Site3 code); marker shape distinguishes
CN true-negative (circle), CN false-positive (star, black edge), and AD (triangle).
Edge linewidth encodes scan duration: thick edge = 140 time-point protocol (ADNI 1/2);
thin edge = 197 time-point protocol (ADNI 3).  Selected sites of interest are
annotated on the PC1–PC2 panel (sites 2, 13, 18, 31, 130, 177, 301, 6, 100).

Any apparent clustering by site or protocol group is descriptive.  The figure
suggests that site/protocol variation is partially organised in latent space,
consistent with the protocol-risk findings (140 TP FPR ≈ 63%; 197 TP FPR ≈ 33%),
but should be interpreted with caution given the limited sample size per site.
""")

    _write_md(OUT / "fig6_caption.md", f"""
## Figure 6 — Procrustes-Aligned OOF Latents (EXPLORATORY ONLY)

⚠ **This figure is exploratory and should not be used as primary evidence.**

OOF test latent vectors for all 397 CN+AD subjects are pooled after Procrustes
orthogonal alignment (folds 2–5 aligned to fold 1 as reference, using all 397
subjects as anchor points; scipy.linalg.orthogonal_procrustes).  PC1–PC3 explain
{pct6:.1f}% of the post-alignment variance.

Alignment removes arbitrary global rotations between fold-specific latent spaces
but cannot correct for encoder-specific learned representations or dimension ordering
differences.  Any structure visible in this figure may reflect genuine data geometry,
residual encoder-to-encoder variation, or both.  Color = manufacturer;
Philips CN false positives are highlighted with a black marker edge.
""")

    _write_md(OUT / "00_LATENT_FIGURES_README.md", f"""
# Latent PCA Figures — README

**Date**: {datetime.now(timezone.utc).isoformat()}
**Script**: scripts/revision_bspc_2026/figure_generation_script_latent_pca.py
**Output directory**: results/revision_bspc_2026/philips_director_figures_20260612/

---

## Audit Conclusion

OOF test latents for the promoted model (recover035_latent384_beta3p75_T80_h10000_p560_full5x5)
come from 5 fold-specific VAE encoders.  A pooled OOF PCA is NOT scientifically
defensible without alignment (CASE 2).

However, fold-1's full cache (test n=80 + trainDev n=317) covers ALL 397 CN+AD
subjects from the same encoder.  Primary figures use this consistent coordinate system.

Full audit: `latent_coordinate_system_audit.md`

---

## Files

| File | Description |
|---|---|
| latent_coordinate_system_audit.md | Encoder provenance audit and coordinate-system decision |
| fig4_latent_pca3d_manufacturer.{{png,svg}} | 3D PCA, all subjects, color = manufacturer |
| fig5_latent_pca3d_philips_site.{{png,svg}} | 3D PCA, Philips only, color = Site3 |
| fig6_procrustes_aligned_pca_exploratory.{{png,svg}} | Procrustes-aligned OOF — exploratory |
| plotted_values_fig4.csv | Coordinates and metadata for fig4 |
| plotted_values_fig5.csv | Coordinates and metadata for fig5 |
| plotted_values_fig6.csv | Coordinates and metadata for fig6 |
| fig4_caption.md, fig5_caption.md, fig6_caption.md | Draft figure captions |

---

## Hard Constraints (honoured)

- Read-only: no model, tensor, or metadata modification
- No retraining, threshold fitting, or subject exclusion
- All figures are descriptive / internal research discussion
""")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    _log("START figure_generation_script_latent_pca")
    _log("Output directory", str(OUT))

    # Coordinate system audit
    facts = _run_coordinate_audit()

    # Load data
    df, mu_cols = _load_data(facts)

    # PCA on fold 1 full cache
    _log("Computing 3D PCA on fold-1 full cache (n=397, d=384)")
    coords, evr = _fit_pca(df, mu_cols, n=3)
    _log("Explained variance ratio", {f"PC{i+1}": f"{v*100:.2f}%" for i, v in enumerate(evr)})
    df = _add_pca_cols(df, coords)

    # Save CSVs
    _save_csvs(df)

    # Fig4: Manufacturer
    _make_fig4(df, evr)

    # Fig5: Philips site
    _make_fig5(df, evr)

    # Fig6: Procrustes (returns its own evr)
    evr6 = _make_fig6(mu_cols)

    # Captions + README
    _write_captions(evr, evr6, n_subjects=len(df))

    # Command log
    elapsed = (datetime.now(timezone.utc) - _T0).total_seconds()
    _LOG.append({"event": "DONE", "elapsed_seconds": elapsed,
                 "n_output_files": len(list(OUT.glob("*")))})
    (OUT / "command_log_latent_pca.json").write_text(
        json.dumps({"script": "figure_generation_script_latent_pca.py",
                    "run_date": _T0.isoformat(),
                    "elapsed_seconds": elapsed,
                    "log": _LOG}, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n{'='*60}")
    print(f"Done.  Output: {OUT}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
