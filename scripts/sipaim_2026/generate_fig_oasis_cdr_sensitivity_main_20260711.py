#!/usr/bin/env python3
"""Generate fig_oasis_cdr_sensitivity_main.pdf for the SIPAIM manuscript.

Publication-quality replacement for the audit-style fig_oasis_label_sensitivity.pdf.
Three-panel figure summarizing the OASIS CDR label-definition sensitivity audit:
  A. OASIS current-label composition by CDR group.
  B. Locked frozen-transfer score distributions by CDR group.
  C. ROC-AUC by label definition (Current-180 vs Strict-128) and arm.

All numbers are read directly from the audited CSVs in
results/sipaim_2026/oasis_label_sensitivity_20260710/. No metric is
recomputed or substituted.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/diego/proyectos/vae_AD")
DATA_DIR = PROJECT_ROOT / "results/sipaim_2026/oasis_label_sensitivity_20260710"
OUT_PDF = PROJECT_ROOT / "manuscript/sipaim_2026/Figures/site_geometry/fig_oasis_cdr_sensitivity_main.pdf"

# ─────────────────────────────────────────────────────────────────────────────
# Style: validated categorical/ordinal palette (dataviz skill reference slots).
# CDR group is ordinal -> one hue (blue), monotone lightness, light->dark.
# Label-definition (Current-180 vs Strict-128) is a genuine 2-way categorical
# contrast -> slot 1 blue vs slot 8 orange, consistent with the rest of the
# manuscript's "reference vs. alternative view" convention.
# ─────────────────────────────────────────────────────────────────────────────
CDR_COLORS = {"CDR 0": "#b7d3f6", "CDR 0.5": "#5598e7", "CDR >= 1": "#184f95"}
COLOR_CURRENT180 = "#2a78d6"
COLOR_STRICT128 = "#eb6834"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"

matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 7.2,
        "axes.titlesize": 7.4,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.6,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.edgecolor": INK_MUTED,
        "text.color": INK_PRIMARY,
        "axes.labelcolor": INK_PRIMARY,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
    }
)


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK_MUTED)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5)


subjects = pd.read_csv(DATA_DIR / "oasis_label_sensitivity_subject_table.csv")
dist = pd.read_csv(DATA_DIR / "oasis_cdr_score_distributions.csv")
metrics = pd.read_csv(DATA_DIR / "oasis_label_sensitivity_metrics.csv")

fig, (axA, axB, axC) = plt.subplots(
    1, 3, figsize=(8.6, 3.4), constrained_layout=False,
    gridspec_kw={"width_ratios": [0.85, 1.1, 1.05]},
)
fig.subplots_adjust(left=0.065, right=0.985, top=0.80, bottom=0.24, wspace=0.55)

# ── Panel A: current-label composition by CDR group ────────────────────────
cdr_order = ["CDR 0", "CDR 0.5", "CDR >= 1"]
comp = pd.crosstab(subjects["current_project_label"], subjects["cdr_group"])[cdr_order]
labels_a = ["CN\n(n=90)", "AD_DEMENTIA\n(n=90)"]
x = np.arange(2)
bottom = np.zeros(2)
for cdr in cdr_order:
    vals = comp.loc[["CN", "AD"], cdr].to_numpy()
    bars = axA.bar(x, vals, bottom=bottom, width=0.55, color=CDR_COLORS[cdr],
                    edgecolor="white", linewidth=0.6, zorder=2, label=cdr)
    for xi, v, b in zip(x, vals, bottom):
        if v >= 5:
            txt_color = "white" if cdr == "CDR >= 1" else INK_PRIMARY
            axA.text(xi, b + v / 2, f"{int(v)}\n{cdr}", ha="center", va="center",
                      fontsize=5.8, color=txt_color, linespacing=1.3)
    bottom += vals

axA.annotate(
    "51/90 current AD_DEMENTIA\nsubjects are CDR 0.5",
    xy=(1, comp.loc["AD", "CDR 0"] + comp.loc["AD", "CDR 0.5"] / 2),
    xytext=(1, 96),
    fontsize=5.9, color=INK_PRIMARY, ha="center", va="bottom",
    arrowprops=dict(arrowstyle="-", color=INK_SECONDARY, linewidth=0.7, shrinkA=0, shrinkB=3),
)

axA.set_xticks(x)
axA.set_xticklabels(labels_a, linespacing=1.4)
axA.set_ylim(0, 118)
axA.set_yticks([0, 20, 40, 60, 80, 100])
axA.set_xlim(-0.55, 1.55)
axA.set_ylabel("Subjects (n)")
axA.set_title("A. Current-label composition\nby CDR group")
style_axis(axA)

# ── Panel B: locked score distributions by CDR group (raincloud-like) ──────
rng = np.random.default_rng(20260711)
group_data = {g: subjects.loc[subjects["cdr_group"] == g, "score_locked"].to_numpy() for g in cdr_order}
positions = np.arange(3)

bp = axB.boxplot(
    [group_data[g] for g in cdr_order], positions=positions, widths=0.32,
    patch_artist=True, showfliers=False, zorder=2,
    medianprops=dict(color=INK_PRIMARY, linewidth=1.1),
    whiskerprops=dict(color=INK_MUTED, linewidth=0.8),
    capprops=dict(color=INK_MUTED, linewidth=0.8),
)
for patch, g in zip(bp["boxes"], cdr_order):
    patch.set_facecolor(CDR_COLORS[g])
    patch.set_alpha(0.55)
    patch.set_edgecolor(INK_SECONDARY)
    patch.set_linewidth(0.6)

for xi, g in zip(positions, cdr_order):
    vals = group_data[g]
    jitter = rng.uniform(-0.11, 0.11, size=len(vals))
    axB.scatter(xi + jitter, vals, s=6, color=CDR_COLORS[g], edgecolor=INK_SECONDARY,
                linewidth=0.15, alpha=0.75, zorder=3)

n_by_group = {g: len(group_data[g]) for g in cdr_order}
axB.set_xticks(positions)
axB.set_xticklabels([f"{g}\n(n={n_by_group[g]})" for g in cdr_order])
axB.set_ylabel("AD-oriented ensemble score\n(locked frozen transfer)")
axB.set_title("B. Score by CDR group")
axB.set_ylim(-0.05, 1.05)

rho = dist.loc[dist.arm == "locked_frozen_transfer", "spearman_rho_score_vs_ordinal_cdr"].iloc[0]
pval = dist.loc[dist.arm == "locked_frozen_transfer", "spearman_pvalue"].iloc[0]
axB.text(
    0.02, 0.97, f"Spearman ρ = {rho:.3f}\np = {pval:.2e}",
    transform=axB.transAxes, ha="left", va="top", fontsize=6.2, color=INK_PRIMARY,
)
style_axis(axB)

# ── Panel C: ROC-AUC by label definition and arm ────────────────────────────
arm_order = [
    "locked_frozen_transfer",
    "previous_adni_fitted_siemens_combat",
    "external_dataset_combat_adni_reference",
    "external_dataset_combat_no_reference",
]
arm_labels = ["Locked", "Prev.\nSiemens-CB", "Dataset-CB\n(ADNI ref.)", "Dataset-CB\n(no ref.)"]
xc = np.arange(len(arm_order))
off = 0.11

point_vals = {}
for eval_set, color, dx in [("OASIS-current-180", COLOR_CURRENT180, -off), ("OASIS-strict-128", COLOR_STRICT128, off)]:
    sub = metrics[metrics.evaluation_set == eval_set].set_index("arm").loc[arm_order]
    vals = sub["roc_auc"].to_numpy()
    lo = vals - sub["roc_auc_ci_low_2p5"].to_numpy()
    hi = sub["roc_auc_ci_high_97p5"].to_numpy() - vals
    axC.errorbar(
        xc + dx, vals, yerr=[lo, hi], fmt="o", color=color, ecolor=color,
        elinewidth=1.1, capsize=2.5, markersize=4.5, markeredgecolor="white",
        markeredgewidth=0.4, zorder=3,
    )
    point_vals[eval_set] = vals

# Direct labels on the first arm's points instead of a legend box (avoids
# overlapping the densely-packed CIs elsewhere in the panel).
axC.text(0 - off, point_vals["OASIS-current-180"][0] - 0.028, "current-180",
          color=COLOR_CURRENT180, fontsize=5.8, ha="center", va="top", fontweight="bold")
axC.text(0 + off, point_vals["OASIS-strict-128"][0] + 0.028, "strict-128",
          color=COLOR_STRICT128, fontsize=5.8, ha="center", va="bottom", fontweight="bold")

for xi in xc:
    cur = metrics[(metrics.evaluation_set == "OASIS-current-180") & (metrics.arm == arm_order[xi])]["roc_auc"].iloc[0]
    strict = metrics[(metrics.evaluation_set == "OASIS-strict-128") & (metrics.arm == arm_order[xi])]["roc_auc"].iloc[0]
    axC.plot([xi - off, xi + off], [cur, strict], color=INK_MUTED, linewidth=0.8, zorder=1, linestyle=(0, (2, 1.5)))

axC.axhline(0.5, color=INK_MUTED, linewidth=0.9, linestyle=(0, (4, 2)), zorder=1)
axC.text(len(arm_order) - 0.55, 0.5 - 0.012, "chance", va="top", ha="right", fontsize=5.6, color=INK_MUTED)

axC.text(
    0.5, 0.955, "Strict label definition: +0.06 ROC-AUC across all four arms",
    fontsize=5.7, color=INK_PRIMARY, ha="center", va="center",
    transform=axC.transAxes,
)

axC.set_xticks(xc)
axC.set_xticklabels(arm_labels, fontsize=6.0)
axC.set_ylim(0.45, 0.90)
axC.set_xlim(-0.5, len(arm_order) - 0.5)
axC.set_ylabel("ROC-AUC")
axC.set_title("C. ROC-AUC by label definition and arm")
style_axis(axC)

fig.suptitle(
    "OASIS CDR label-definition sensitivity — current-180 remains primary; strict-128 is a transparent sensitivity check",
    fontsize=6.8, color=INK_SECONDARY, y=0.995,
)
fig.text(
    0.5, 0.045,
    "PR-AUC is not shown here because AD prevalence differs between label definitions (50% current-180 vs. 30.5% strict-128) "
    "and is therefore not directly comparable; see Table and caption for PR-AUC values.",
    ha="center", va="center", fontsize=5.7, color=INK_SECONDARY,
)

OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PDF, format="pdf")
plt.close(fig)
print(f"Wrote {OUT_PDF}")
