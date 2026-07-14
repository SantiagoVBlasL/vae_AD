#!/usr/bin/env python3
"""Generate fig_oasis_external_batch_combat.pdf for the SIPAIM manuscript.

Three-panel horizontal figure summarizing the audited external OASIS
Dataset-batch ComBat analysis (unsupervised, batch=Dataset in {ADNI,OASIS},
covariates=Age+Sex, Diagnosis excluded, OASIS labels never used for
harmonization/calibration/threshold/model selection) against the locked
frozen-transfer and previous ADNI-fitted Siemens-ComBat baselines:
  A. Feature-space Wasserstein (W1) distance to OASIS, before vs. after
     Dataset-ComBat, both reference-batch variants, per fold.
  B. OASIS ROC-AUC / PR-AUC by arm (4 arms).
  C. Sensitivity/specificity trade-off by arm (2-D scatter).

All numbers are read directly from the audited CSVs in
results/revision_bspc_2026/post_revision_exploratory_20260630/oasis_external_batch_combat_20260710/ .
No metric is recomputed or substituted; only trivial fold-level mean/SD
aggregation of already-audited per-fold values is performed.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/post_revision_exploratory_20260630"
    / "oasis_external_batch_combat_20260710"
)
OUT_PDF = (
    PROJECT_ROOT
    / "manuscript/sipaim_2026/Figures/site_geometry"
    / "fig_oasis_external_batch_combat.pdf"
)

# ─────────────────────────────────────────────────────────────────────────────
# Style: validated 4-slot categorical order (blue/aqua/yellow/green, dataviz
# skill reference palette slots 1-4), fixed order, never cycled. Blue is kept
# for "locked" to stay visually consistent with fig_transportability_summary.pdf.
# ─────────────────────────────────────────────────────────────────────────────
COLOR_LOCKED = "#2a78d6"          # slot 1 blue  -- locked_frozen_transfer
COLOR_PREV_COMBAT = "#1baf7a"     # slot 2 aqua  -- previous_adni_fitted_siemens_combat
COLOR_DATASET_ADNI_REF = "#eda100"  # slot 3 yellow -- external_dataset_combat_adni_reference
COLOR_DATASET_NO_REF = "#008300"  # slot 4 green -- external_dataset_combat_no_reference
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
FOLD_LINE = "#c3c2b7"

ARM_COLORS = {
    "locked_frozen_transfer": COLOR_LOCKED,
    "previous_adni_fitted_siemens_combat": COLOR_PREV_COMBAT,
    "external_dataset_combat_adni_reference": COLOR_DATASET_ADNI_REF,
    "external_dataset_combat_no_reference": COLOR_DATASET_NO_REF,
}
ARM_LABELS = {
    "locked_frozen_transfer": "Locked frozen transfer",
    "previous_adni_fitted_siemens_combat": "Prev. ADNI-fitted Siemens-ComBat",
    "external_dataset_combat_adni_reference": "Dataset-ComBat (ADNI reference)",
    "external_dataset_combat_no_reference": "Dataset-ComBat (no reference)",
}
ARM_ORDER = [
    "locked_frozen_transfer",
    "previous_adni_fitted_siemens_combat",
    "external_dataset_combat_adni_reference",
    "external_dataset_combat_no_reference",
]

matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,  # embed as TrueType, not Type-3 bitmap
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 7.2,
        "axes.titlesize": 7.6,
        "axes.labelsize": 7.2,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "legend.fontsize": 6.4,
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


def bar_value_label(ax, x, y, text, color=INK_PRIMARY, dy=0.012) -> None:
    ax.text(x, y + dy, text, ha="center", va="bottom", fontsize=6.0, color=color)


# ─────────────────────────────────────────────────────────────────────────────
# Load audited data
# ─────────────────────────────────────────────────────────────────────────────
metrics = pd.read_csv(AUDIT_DIR / "oasis_external_batch_combat_metrics.csv")
wasserstein = pd.read_csv(AUDIT_DIR / "oasis_external_batch_combat_wasserstein.csv")

metrics = metrics.set_index("arm")

w1_noref = wasserstein[wasserstein["arm"] == "external_dataset_combat_no_reference"].sort_values("fold")
w1_adniref = wasserstein[wasserstein["arm"] == "external_dataset_combat_adni_reference"].sort_values("fold")
before_noref = w1_noref[w1_noref["state"] == "before_raw"]["w1_mean_all_features"].to_numpy()
after_noref = w1_noref[w1_noref["state"] == "after_external_batch_combat"]["w1_mean_all_features"].to_numpy()
before_adniref = w1_adniref[w1_adniref["state"] == "before_raw"]["w1_mean_all_features"].to_numpy()
after_adniref = w1_adniref[w1_adniref["state"] == "after_external_batch_combat"]["w1_mean_all_features"].to_numpy()
assert np.allclose(before_noref, before_adniref), "before_raw W1 must match across variants"
n_folds = len(before_noref)

# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────
fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(7.2, 3.15), constrained_layout=False)
fig.subplots_adjust(left=0.075, right=0.985, top=0.76, bottom=0.185, wspace=0.50)

# ── Panel A: feature W1 to OASIS, before vs. after Dataset-ComBat ──────────
x_before, x_noref, x_adniref = 0.0, 1.0, 2.0
for i in range(n_folds):
    axA.plot([x_before, x_noref], [before_noref[i], after_noref[i]],
             color=COLOR_DATASET_NO_REF, alpha=0.55, linewidth=0.8, zorder=1)
    axA.plot([x_before, x_adniref], [before_adniref[i], after_adniref[i]],
             color=COLOR_DATASET_ADNI_REF, alpha=0.55, linewidth=0.8, zorder=1)

axA.scatter(np.full(n_folds, x_before), before_noref, s=14, color=INK_MUTED,
            zorder=2, linewidths=0)
axA.scatter(np.full(n_folds, x_noref), after_noref, s=14, color=COLOR_DATASET_NO_REF,
            zorder=2, linewidths=0)
axA.scatter(np.full(n_folds, x_adniref), after_adniref, s=14, color=COLOR_DATASET_ADNI_REF,
            zorder=2, linewidths=0)

for x, vals, color in [
    (x_before, before_noref, INK_MUTED),
    (x_noref, after_noref, COLOR_DATASET_NO_REF),
    (x_adniref, after_adniref, COLOR_DATASET_ADNI_REF),
]:
    m, sd = vals.mean(), vals.std(ddof=1)
    axA.errorbar([x], [m], yerr=[sd], fmt="D", color=color, ecolor=color,
                 elinewidth=1.1, capsize=2.5, markersize=5,
                 markeredgecolor="white", markeredgewidth=0.5, zorder=3)

axA.set_xlim(-0.5, 2.5)
axA.set_xticks([x_before, x_noref, x_adniref])
axA.set_xticklabels(["Before\n(raw)", "After\n(no ref.)", "After\n(ADNI ref.)"])
axA.set_ylim(0.09, 0.26)
axA.set_ylabel("Mean feature W1 (ADNI vs. OASIS)")
axA.set_title("A. Feature-space alignment\n(lower = closer to OASIS)")
style_axis(axA)

# ── Panel B: OASIS ROC-AUC / PR-AUC by arm ─────────────────────────────────
metric_groups = ["ROC-AUC", "PR-AUC"]
metric_cols = ["roc_auc", "pr_auc"]
group_x = np.arange(len(metric_groups))
n_arms = len(ARM_ORDER)
bar_w = 0.19
offsets = (np.arange(n_arms) - (n_arms - 1) / 2) * (bar_w + 0.02)

for arm, off in zip(ARM_ORDER, offsets):
    vals = [metrics.loc[arm, c] for c in metric_cols]
    axB.bar(group_x + off, vals, width=bar_w, color=ARM_COLORS[arm], zorder=2,
            edgecolor="white", linewidth=0.4)

axB.axhline(0.5, color=INK_MUTED, linewidth=0.9, linestyle=(0, (4, 2)), zorder=1)
axB.text(0.5, 0.5 + 0.02, "chance", va="bottom", ha="center", fontsize=5.6, color=INK_MUTED)
axB.text(0.5, 0.755, "all pairwise Δ n.s. (paired bootstrap)", ha="center", va="top",
         fontsize=5.6, color=INK_SECONDARY, style="italic")

axB.set_xlim(-0.55, len(metric_groups) - 1 + 0.55)
axB.set_xticks(group_x)
axB.set_xticklabels(metric_groups)
axB.set_ylim(0.0, 0.80)
axB.set_ylabel("Score")
axB.set_title("B. OASIS diagnostic ranking\n(higher = better ranking)")
style_axis(axB)

# ── Panel C: sensitivity/specificity trade-off by arm ──────────────────────
for arm in ARM_ORDER:
    sens = metrics.loc[arm, "sensitivity"]
    spec = metrics.loc[arm, "specificity"]
    axC.scatter([spec], [sens], s=34, color=ARM_COLORS[arm], zorder=3,
                edgecolor="white", linewidth=0.5)

axC.set_xlim(0.70, 0.92)
axC.set_ylim(0.20, 0.53)
axC.invert_xaxis()  # higher specificity (better) to the left, mirrors ROC convention
axC.set_xlabel("Specificity")
axC.set_ylabel("Sensitivity")
axC.set_title("C. Operating-point trade-off\n(not a transport-quality metric)")
style_axis(axC)

# ── Shared legend (single, not repeated per panel) ─────────────────────────
legend_handles = [
    mlines.Line2D([], [], marker="D", markersize=5, color=ARM_COLORS[arm],
                  markeredgecolor="white", markeredgewidth=0.5, linestyle="none",
                  label=ARM_LABELS[arm])
    for arm in ARM_ORDER
]
fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, 1.0), handletextpad=0.5, columnspacing=1.2)

OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PDF, format="pdf")
plt.close(fig)
print(f"Wrote {OUT_PDF}")
