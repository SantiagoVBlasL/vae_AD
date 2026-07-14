#!/usr/bin/env python3
"""Generate fig_transportability_summary.pdf for the BSPC/SIPAIM manuscript.

Three-panel horizontal figure summarizing the audited fold-wise input-ComBat
result versus the locked (non-harmonized) reference:
  A. Manufacturer information in latent space (fold-level, paired).
  B. Diagnostic ranking (ROC-AUC) on ADNI (internal) and OASIS (external).
  C. Manufacturer-specific CN false-positive rate (GE / Philips / SIEMENS).

All numbers are read directly from the audited CSVs in
results/revision_bspc_2026/post_revision_exploratory_20260630/foldcombat_cleanrepro_final_audit_20260706/ .
No metric is recomputed or substituted; only trivial fold-level mean/SD
aggregation of already-audited per-fold values is performed (verified to
match the audit's own reported aggregates).
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
    / "foldcombat_cleanrepro_final_audit_20260706"
)
OUT_PDF = (
    PROJECT_ROOT
    / "docs/revision_bspc_2026/site_geometry_analysis/Figures/site_geometry"
    / "fig_transportability_summary.pdf"
)

# ─────────────────────────────────────────────────────────────────────────────
# Style: validated two-slot categorical pair (blue/orange, dataviz skill
# reference palette slots 1 and 8) — CVD-safe, no "bad=red" valence since
# ComBat does not uniformly under/over-perform locked across panels.
# ─────────────────────────────────────────────────────────────────────────────
COLOR_LOCKED = "#2a78d6"
COLOR_COMBAT = "#eb6834"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
FOLD_LINE = "#c3c2b7"

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
        "legend.fontsize": 6.8,
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
    ax.text(
        x, y + dy, text, ha="center", va="bottom", fontsize=6.2, color=color,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Load audited data
# ─────────────────────────────────────────────────────────────────────────────
mfr_decoding = pd.read_csv(AUDIT_DIR / "cleanrepro_manufacturer_decoding.csv")
mfr_transport = pd.read_csv(AUDIT_DIR / "cleanrepro_manufacturer_transport.csv")
adni_bootstrap = pd.read_csv(AUDIT_DIR / "cleanrepro_paired_bootstrap_vs_locked.csv")
oasis_bootstrap = pd.read_csv(AUDIT_DIR / "cleanrepro_oasis_paired_bootstrap.csv")

locked_fold = mfr_decoding[mfr_decoding["arm"] == "locked"].sort_values("fold")
combat_fold = mfr_decoding[
    mfr_decoding["arm"] == "clean_foldwise_input_combat"
].sort_values("fold")
assert list(locked_fold["fold"]) == list(combat_fold["fold"]), "fold mismatch"
locked_bacc = locked_fold["manufacturer_balanced_accuracy"].to_numpy()
combat_bacc = combat_fold["manufacturer_balanced_accuracy"].to_numpy()
n_folds = len(locked_bacc)

adni_roc = adni_bootstrap[adni_bootstrap["metric"] == "roc_auc"].iloc[0]
oasis_roc = oasis_bootstrap[oasis_bootstrap["metric"] == "roc_auc"].iloc[0]

mfr_order = ["GE", "Philips", "SIEMENS"]
transport_locked = mfr_transport[mfr_transport["arm"] == "locked"].set_index(
    "manufacturer"
).loc[mfr_order]
transport_combat = mfr_transport[
    mfr_transport["arm"] == "clean_foldwise_input_combat"
].set_index("manufacturer").loc[mfr_order]

# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────
fig, (axA, axB, axC) = plt.subplots(
    1, 3, figsize=(7.2, 3.05), constrained_layout=False
)
fig.subplots_adjust(left=0.065, right=0.985, top=0.80, bottom=0.19, wspace=0.48)

# ── Panel A: manufacturer decodability, paired fold-level slope plot ───────
x_locked, x_combat = 0.0, 1.0
for i in range(n_folds):
    axA.plot(
        [x_locked, x_combat],
        [locked_bacc[i], combat_bacc[i]],
        color=FOLD_LINE,
        linewidth=0.8,
        zorder=1,
    )
axA.scatter(
    np.full(n_folds, x_locked), locked_bacc, s=16, color=COLOR_LOCKED,
    alpha=0.55, zorder=2, linewidths=0,
)
axA.scatter(
    np.full(n_folds, x_combat), combat_bacc, s=16, color=COLOR_COMBAT,
    alpha=0.55, zorder=2, linewidths=0,
)

locked_mean, locked_sd = locked_bacc.mean(), locked_bacc.std(ddof=1)
combat_mean, combat_sd = combat_bacc.mean(), combat_bacc.std(ddof=1)
axA.errorbar(
    [x_locked], [locked_mean], yerr=[locked_sd], fmt="D", color=COLOR_LOCKED,
    ecolor=COLOR_LOCKED, elinewidth=1.1, capsize=2.5, markersize=5,
    markeredgecolor="white", markeredgewidth=0.5, zorder=3,
)
axA.errorbar(
    [x_combat], [combat_mean], yerr=[combat_sd], fmt="D", color=COLOR_COMBAT,
    ecolor=COLOR_COMBAT, elinewidth=1.1, capsize=2.5, markersize=5,
    markeredgecolor="white", markeredgewidth=0.5, zorder=3,
)

axA.axhline(1 / 3, color=INK_MUTED, linewidth=0.9, linestyle=(0, (4, 2)), zorder=1)
axA.text(
    x_combat + 0.30, 1 / 3 - 0.012, "chance (1/3)", va="top", ha="left",
    fontsize=5.8, color=INK_MUTED,
)

axA.set_xlim(-0.45, 1.55)
axA.set_xticks([x_locked, x_combat])
axA.set_xticklabels(["Locked", "ComBat"])
axA.set_ylim(0.25, 0.80)
axA.set_ylabel("Manufacturer balanced accuracy")
axA.set_title("A. Manufacturer info in latent space\n(lower = less manufacturer info)")
style_axis(axA)

# ── Panel B: diagnostic ranking (ROC-AUC), ADNI vs OASIS ───────────────────
groups = ["ADNI\n(internal)", "OASIS\n(external)"]
group_x = np.arange(len(groups))
bar_w = 0.32
locked_vals = [adni_roc["locked"], oasis_roc["locked"]]
combat_vals = [adni_roc["clean_foldwise_input_combat"], oasis_roc["clean_foldwise_input_combat"]]
deltas = [adni_roc["observed_delta"], oasis_roc["observed_delta"]]
ci_lows = [adni_roc["ci_low_2p5"], oasis_roc["ci_low_2p5"]]
ci_highs = [adni_roc["ci_high_97p5"], oasis_roc["ci_high_97p5"]]

axB.bar(
    group_x - bar_w / 2 - 0.01, locked_vals, width=bar_w, color=COLOR_LOCKED,
    zorder=2, edgecolor="white", linewidth=0.4,
)
axB.bar(
    group_x + bar_w / 2 + 0.01, combat_vals, width=bar_w, color=COLOR_COMBAT,
    zorder=2, edgecolor="white", linewidth=0.4,
)
for gx, lv, cv in zip(group_x, locked_vals, combat_vals):
    bar_value_label(axB, gx - bar_w / 2 - 0.01, lv, f"{lv:.3f}")
    bar_value_label(axB, gx + bar_w / 2 + 0.01, cv, f"{cv:.3f}")

axB.axhline(0.5, color=INK_MUTED, linewidth=0.9, linestyle=(0, (4, 2)), zorder=1)
axB.text(
    -0.50, 0.5 + 0.012, "chance", va="bottom", ha="left",
    fontsize=5.8, color=INK_MUTED,
)

for gx, d, lo, hi in zip(group_x, deltas, ci_lows, ci_highs):
    sig = "sig." if (lo > 0 or hi < 0) else "n.s."
    axB.text(
        gx, 0.885,
        f"Δ={d:+.3f}\n[{lo:+.3f},{hi:+.3f}] {sig}",
        ha="center", va="top", fontsize=5.5, color=INK_SECONDARY, linespacing=1.35,
    )

axB.set_xlim(-0.55, len(groups) - 1 + 0.55)
axB.set_xticks(group_x)
axB.set_xticklabels(groups)
axB.set_ylim(0.45, 0.95)
axB.set_ylabel("ROC-AUC")
axB.set_title("B. Diagnostic ranking\n(higher = better ranking)")
style_axis(axB)

# ── Panel C: manufacturer-specific CN false-positive rate ─────────────────
mfr_x = np.arange(len(mfr_order))
locked_fpr = transport_locked["cn_false_positive_rate"].to_numpy()
combat_fpr = transport_combat["cn_false_positive_rate"].to_numpy()

axC.bar(
    mfr_x - bar_w / 2 - 0.01, locked_fpr, width=bar_w, color=COLOR_LOCKED,
    zorder=2, edgecolor="white", linewidth=0.4,
)
axC.bar(
    mfr_x + bar_w / 2 + 0.01, combat_fpr, width=bar_w, color=COLOR_COMBAT,
    zorder=2, edgecolor="white", linewidth=0.4,
)
for gx, lv, cv in zip(mfr_x, locked_fpr, combat_fpr):
    bar_value_label(axC, gx - bar_w / 2 - 0.01, lv, f"{lv:.0%}")
    bar_value_label(axC, gx + bar_w / 2 + 0.01, cv, f"{cv:.0%}")

philips_idx = mfr_order.index("Philips")
axC.annotate(
    "",
    xy=(philips_idx - bar_w / 2 - 0.01, locked_fpr[philips_idx] + 0.015),
    xytext=(philips_idx - 0.05, 0.535),
    arrowprops=dict(arrowstyle="-", color=INK_SECONDARY, linewidth=0.7, shrinkA=0, shrinkB=2),
)
axC.annotate(
    "",
    xy=(philips_idx + bar_w / 2 + 0.01, combat_fpr[philips_idx] + 0.015),
    xytext=(philips_idx + 0.05, 0.535),
    arrowprops=dict(arrowstyle="-", color=INK_SECONDARY, linewidth=0.7, shrinkA=0, shrinkB=2),
)
axC.text(
    philips_idx, 0.55,
    f"{locked_fpr[philips_idx]:.1%} → {combat_fpr[philips_idx]:.1%}",
    fontsize=6.2, color=INK_PRIMARY, ha="center", va="bottom",
)

axC.set_xlim(-0.55, len(mfr_order) - 1 + 0.55)
axC.set_xticks(mfr_x)
axC.set_xticklabels(mfr_order)
axC.set_ylim(0, 0.62)
axC.set_ylabel("CN false-positive rate")
axC.set_title("C. CN false positives by manufacturer\n(lower = fewer false positives)")
style_axis(axC)

# ── Shared legend (single, not repeated per panel) ─────────────────────────
legend_handles = [
    mlines.Line2D(
        [], [], marker="D", markersize=5, color=COLOR_LOCKED,
        markeredgecolor="white", markeredgewidth=0.5, linestyle="none",
        label="Locked (non-harmonized)",
    ),
    mlines.Line2D(
        [], [], marker="D", markersize=5, color=COLOR_COMBAT,
        markeredgecolor="white", markeredgewidth=0.5, linestyle="none",
        label="Fold-wise input ComBat",
    ),
]
fig.legend(
    handles=legend_handles, loc="upper center", ncol=2, frameon=False,
    bbox_to_anchor=(0.5, 1.0), handletextpad=0.5, columnspacing=1.4,
)

OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PDF, format="pdf")
plt.close(fig)
print(f"Wrote {OUT_PDF}")
