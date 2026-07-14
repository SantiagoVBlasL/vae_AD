#!/usr/bin/env python3
"""Generate fig_sipaim_transportability_main.{pdf,png} for the SIPAIM manuscript.

Read-only. No training, no VAE inference, no model selection, no threshold
optimization. No manuscript .tex file is read or edited -- the Overleaf
manuscript remains sole authoritative; this script only reads already-computed,
audited CSV artifacts and writes a standalone figure + data table.

Panel A: paired point/CI comparisons, locked vs input-ComBat, on two
  *separate* y-axes (manufacturer balanced accuracy; ADNI OOF ROC-AUC).
  Source: results/sipaim_2026/manufacturer_decoding/cleanrepro_manufacturer_decoding.csv
          results/sipaim_2026/clean_foldwise_combat_training_and_audit/
              cleanrepro_downstream_foldwise_metrics.csv, cleanrepro_paired_bootstrap_vs_locked.csv
          results/revision_bspc_2026/recover035_latent384_beta3p75_stageB_oof_score_calibration/
              calib_foldwise_metrics.csv (locked ADNI OOF foldwise reference)

Panel B: three valid shared-coordinate external arms (Siemens-ComBat VAE
  excluded), absolute standardized marginal shift (x) vs fold-local projected
  AUC (y), with arrows from locked to each adapted arm.
  Source: results/sipaim_2026/standardized_locked_space_geometry_20260712/
              standardized_shared_geometry.csv

Panel C: frozen OASIS locked scores under the primary nearest-CDR-within-180-days
  mapping, by CDR group (CDR>=1 labeled "high severity", never "AD").
  Source: results/sipaim_2026/final_blocker_resolution_20260712/
              oasis_cdr_scan_alignment.csv, cdr_temporal_sensitivity_metrics.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/diego/proyectos/vae_AD")
OUT_DIR = PROJECT_ROOT / "results/sipaim_2026/fig_sipaim_transportability_main_20260712"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Source paths (all pre-computed, audited artifacts; nothing recomputed here) ──
MFR_DECODING = PROJECT_ROOT / "results/sipaim_2026/manufacturer_decoding/cleanrepro_manufacturer_decoding.csv"
COMBAT_FOLDWISE = PROJECT_ROOT / "results/sipaim_2026/clean_foldwise_combat_training_and_audit/cleanrepro_downstream_foldwise_metrics.csv"
COMBAT_PAIRED_BOOT = PROJECT_ROOT / "results/sipaim_2026/clean_foldwise_combat_training_and_audit/cleanrepro_paired_bootstrap_vs_locked.csv"
LOCKED_FOLDWISE = PROJECT_ROOT / "results/revision_bspc_2026/recover035_latent384_beta3p75_stageB_oof_score_calibration/calib_foldwise_metrics.csv"

GEOMETRY = PROJECT_ROOT / "results/sipaim_2026/standardized_locked_space_geometry_20260712/standardized_shared_geometry.csv"

CDR_ALIGNMENT = PROJECT_ROOT / "results/sipaim_2026/final_blocker_resolution_20260712/oasis_cdr_scan_alignment.csv"
CDR_METRICS = PROJECT_ROOT / "results/sipaim_2026/final_blocker_resolution_20260712/cdr_temporal_sensitivity_metrics.csv"

# ── Colorblind-safe palette (Okabe-Ito derived; consistent with this project's
#    established conventions -- ordinal CDR ramp reused verbatim from the
#    oasis_cdr_sensitivity figure work, blue/orange reused for 2-way contrasts) ──
COLOR_BASELINE = "#2a78d6"       # blue -- locked/baseline
COLOR_COMBAT = "#eb6834"         # orange -- input-ComBat / adapted
COLOR_LOCKED_B = "#000000"       # black -- Panel B reference (locked)
COLOR_ADNI_REF = "#009E73"       # bluish green -- Dataset-ComBat, ADNI reference (primary adapted)
COLOR_NO_REF = "#56B4E9"         # sky blue -- Dataset-ComBat, no reference (secondary)
CDR_COLORS = {"CDR 0": "#b7d3f6", "CDR 0.5": "#5598e7", "CDR >= 1 (high severity)": "#184f95"}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8.0, "axes.titlesize": 8.5, "axes.labelsize": 8.0,
    "xtick.labelsize": 7.3, "ytick.labelsize": 7.3, "legend.fontsize": 7.0,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.edgecolor": INK_MUTED, "text.color": INK_PRIMARY, "axes.labelcolor": INK_PRIMARY,
    "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
})


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK_MUTED)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.2)


figure_data_rows = []  # collected across panels -> fig_sipaim_transportability_main_data.csv

# ═══════════════════════════════════════════════════════════════════════════
# Load & validate Panel A data
# ═══════════════════════════════════════════════════════════════════════════
mfr = pd.read_csv(MFR_DECODING)
mfr_fold = mfr[mfr["arm"].isin(["locked", "clean_foldwise_input_combat"])]
mfr_locked_vals = mfr_fold[mfr_fold.arm == "locked"]["manufacturer_balanced_accuracy"].to_numpy()
mfr_combat_vals = mfr_fold[mfr_fold.arm == "clean_foldwise_input_combat"]["manufacturer_balanced_accuracy"].to_numpy()
mfr_locked_mean, mfr_locked_sd = mfr_locked_vals.mean(), mfr_locked_vals.std(ddof=1)
mfr_combat_mean, mfr_combat_sd = mfr_combat_vals.mean(), mfr_combat_vals.std(ddof=1)
assert abs(mfr_locked_mean - 0.666) < 0.001, f"mfr locked mean mismatch: {mfr_locked_mean}"
assert abs(mfr_combat_mean - 0.440) < 0.001, f"mfr combat mean mismatch: {mfr_combat_mean}"

combat_fold = pd.read_csv(COMBAT_FOLDWISE)
combat_auc_vals = combat_fold["roc_auc"].to_numpy()
combat_auc_mean, combat_auc_sd = combat_auc_vals.mean(), combat_auc_vals.std(ddof=1)

locked_fold = pd.read_csv(LOCKED_FOLDWISE)
locked_fold_sub = locked_fold[(locked_fold.model_name == "logreg_l2_original") & (locked_fold.feature_set == "z_plus_age_sex")
                               & (locked_fold.calib_method == "oof_ecdf") & (locked_fold.threshold_strategy == "fixed_0p5")]
locked_auc_vals = locked_fold_sub["auc"].to_numpy()
locked_auc_fold_mean, locked_auc_fold_sd = locked_auc_vals.mean(), locked_auc_vals.std(ddof=1)

paired_boot = pd.read_csv(COMBAT_PAIRED_BOOT)
auc_delta_row = paired_boot[paired_boot.metric == "roc_auc"].iloc[0]
locked_pooled_auc = float(auc_delta_row["locked"])
combat_pooled_auc = float(auc_delta_row["clean_foldwise_input_combat"])
assert abs(locked_pooled_auc - 0.795) < 0.001, f"locked pooled AUC mismatch: {locked_pooled_auc}"
assert abs(combat_pooled_auc - 0.749) < 0.001, f"combat pooled AUC mismatch: {combat_pooled_auc}"
auc_delta_ci = (float(auc_delta_row["ci_low_2p5"]), float(auc_delta_row["ci_high_97p5"]))
auc_delta_p = float(auc_delta_row["p_delta_gt_0"])

for label, mean, sd, vals in [
    ("manufacturer_bacc_locked", mfr_locked_mean, mfr_locked_sd, mfr_locked_vals),
    ("manufacturer_bacc_input_combat", mfr_combat_mean, mfr_combat_sd, mfr_combat_vals),
]:
    figure_data_rows.append(dict(panel="A1", quantity=label, point_value=mean, error_low=mean - sd,
                                  error_high=mean + sd, n_folds=len(vals), source=str(MFR_DECODING)))
figure_data_rows.append(dict(panel="A2", quantity="adni_oof_rocauc_locked_pooled", point_value=locked_pooled_auc,
                              error_low=locked_pooled_auc - locked_auc_fold_sd, error_high=locked_pooled_auc + locked_auc_fold_sd,
                              n_folds=5, source=str(COMBAT_PAIRED_BOOT) + " (point); " + str(LOCKED_FOLDWISE) + " (fold SD)"))
figure_data_rows.append(dict(panel="A2", quantity="adni_oof_rocauc_input_combat_pooled", point_value=combat_pooled_auc,
                              error_low=combat_pooled_auc - combat_auc_sd, error_high=combat_pooled_auc + combat_auc_sd,
                              n_folds=5, source=str(COMBAT_PAIRED_BOOT) + " (point); " + str(COMBAT_FOLDWISE) + " (fold SD)"))
figure_data_rows.append(dict(panel="A2", quantity="adni_oof_rocauc_paired_delta", point_value=auc_delta_row["observed_delta"],
                              error_low=auc_delta_ci[0], error_high=auc_delta_ci[1], n_folds=5, source=str(COMBAT_PAIRED_BOOT)))

# ═══════════════════════════════════════════════════════════════════════════
# Load & validate Panel B data
# ═══════════════════════════════════════════════════════════════════════════
geom = pd.read_csv(GEOMETRY)
assert set(geom["arm"]) == {"locked_frozen_transfer", "external_dataset_combat_adni_reference",
                             "external_dataset_combat_no_reference"}, "unexpected arm set in geometry source"
geom = geom.set_index("arm")
for arm in geom.index:
    x = abs(geom.loc[arm, "marginal_standardized_shift"])
    ci_lo_raw = geom.loc[arm, "marginal_standardized_shift_ci_low_2p5"]
    ci_hi_raw = geom.loc[arm, "marginal_standardized_shift_ci_high_97p5"]
    assert ci_lo_raw < 0 and ci_hi_raw < 0, f"{arm}: expected entirely-negative marginal-shift CI, got [{ci_lo_raw},{ci_hi_raw}]"
    x_ci = (abs(ci_hi_raw), abs(ci_lo_raw))
    y = geom.loc[arm, "projected_auc"]
    y_ci = (geom.loc[arm, "projected_auc_ci_low_2p5"], geom.loc[arm, "projected_auc_ci_high_97p5"])
    figure_data_rows.append(dict(panel="B", quantity=f"{arm}__abs_marginal_standardized_shift", point_value=x,
                                  error_low=x_ci[0], error_high=x_ci[1], n_folds=None, source=str(GEOMETRY)))
    figure_data_rows.append(dict(panel="B", quantity=f"{arm}__projected_auc", point_value=y,
                                  error_low=y_ci[0], error_high=y_ci[1], n_folds=None, source=str(GEOMETRY)))

# ═══════════════════════════════════════════════════════════════════════════
# Load & validate Panel C data
# ═══════════════════════════════════════════════════════════════════════════
align = pd.read_csv(CDR_ALIGNMENT)
metrics = pd.read_csv(CDR_METRICS)
mapping_a = metrics[metrics.mapping == "A_nearest_180d"].iloc[0]
mapped_a = align[align["A_nearest_180d_mapped"]].copy()
assert len(mapped_a) == int(mapping_a["n_mapped"]) == 169
for grp, n_expect in [("CDR 0", 81), ("CDR 0.5", 51), ("CDR >= 1 (high severity)", 37)]:
    n_actual = int((mapped_a["A_nearest_180d_cdr_group"] == grp).sum())
    assert n_actual == n_expect, f"{grp}: expected {n_expect}, got {n_actual}"
    grp_scores = mapped_a.loc[mapped_a["A_nearest_180d_cdr_group"] == grp, "score_locked"]
    figure_data_rows.append(dict(panel="C", quantity=f"score_locked__{grp}", point_value=grp_scores.mean(),
                                  error_low=grp_scores.min(), error_high=grp_scores.max(), n_folds=n_actual, source=str(CDR_ALIGNMENT)))
figure_data_rows.append(dict(panel="C", quantity="pairwise_auc_cdr0_vs_ge1", point_value=mapping_a["pairwise_auc"],
                              error_low=mapping_a["pairwise_auc_ci_low"], error_high=mapping_a["pairwise_auc_ci_high"],
                              n_folds=None, source=str(CDR_METRICS)))
figure_data_rows.append(dict(panel="C", quantity="partial_spearman_rho_age_sex_adj", point_value=mapping_a["partial_spearman_rho_age_sex_adj"],
                              error_low=None, error_high=None, n_folds=None, source=str(CDR_METRICS)))

print("All panel data loaded and validated against source artifacts.")

# ═══════════════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════════════
FIG_WIDTH_IN = 7.16
FIG_HEIGHT_IN = 3.35
fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
gs = fig.add_gridspec(1, 4, width_ratios=[0.78, 0.78, 1.05, 1.35], wspace=0.68,
                       left=0.058, right=0.99, top=0.88, bottom=0.235)
axA1 = fig.add_subplot(gs[0, 0])
axA2 = fig.add_subplot(gs[0, 1])
axB = fig.add_subplot(gs[0, 2])
axC = fig.add_subplot(gs[0, 3])

# ── Panel A1: manufacturer BACC ──
xA = [0, 1]
axA1.errorbar(xA, [mfr_locked_mean, mfr_combat_mean], yerr=[mfr_locked_sd, mfr_combat_sd],
              fmt="o", color=INK_PRIMARY, ecolor=INK_PRIMARY, elinewidth=1.0, capsize=2.5, markersize=0, zorder=3)
axA1.scatter(xA, [mfr_locked_mean, mfr_combat_mean], s=26, color=[COLOR_BASELINE, COLOR_COMBAT],
             edgecolor="white", linewidth=0.5, zorder=4)
for xi, vals, c in zip(xA, [mfr_locked_vals, mfr_combat_vals], [COLOR_BASELINE, COLOR_COMBAT]):
    jit = np.random.default_rng(20260712).uniform(-0.06, 0.06, size=len(vals))
    axA1.scatter(np.full(len(vals), xi) + jit, vals, s=6, color=c, alpha=0.45, zorder=2, linewidth=0)
axA1.axhline(1/3, color=INK_MUTED, linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)
axA1.text(0.99, 1/3 + 0.025, "chance (3-class)", fontsize=5.0, color=INK_MUTED, va="bottom", ha="right",
          transform=axA1.get_yaxis_transform())
axA1.set_xticks(xA)
axA1.set_xticklabels(["Locked", "Input-\nComBat"], fontsize=7.0)
axA1.set_xlim(-0.5, 1.5)
axA1.set_ylim(0.25, 0.80)
axA1.set_ylabel("Manufacturer BACC")
axA1.set_title("A1. Manufacturer\ndecodability", fontsize=7.4, pad=4)
style_axis(axA1)

# ── Panel A2: ADNI OOF ROC-AUC (separate y-axis, deliberately not shared with A1) ──
axA2.errorbar(xA, [locked_pooled_auc, combat_pooled_auc], yerr=[locked_auc_fold_sd, combat_auc_sd],
              fmt="o", color=INK_PRIMARY, ecolor=INK_PRIMARY, elinewidth=1.0, capsize=2.5, markersize=0, zorder=3)
axA2.scatter(xA, [locked_pooled_auc, combat_pooled_auc], s=26, color=[COLOR_BASELINE, COLOR_COMBAT],
             edgecolor="white", linewidth=0.5, zorder=4)
for xi, vals, c in zip(xA, [locked_auc_vals, combat_auc_vals], [COLOR_BASELINE, COLOR_COMBAT]):
    jit = np.random.default_rng(20260712).uniform(-0.06, 0.06, size=len(vals))
    axA2.scatter(np.full(len(vals), xi) + jit, vals, s=6, color=c, alpha=0.45, zorder=2, linewidth=0)
axA2.axhline(0.5, color=INK_MUTED, linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)
axA2.text(0.99, 0.5 + 0.02, "chance", fontsize=5.0, color=INK_MUTED, va="bottom", ha="right",
          transform=axA2.get_yaxis_transform())
axA2.set_xticks(xA)
axA2.set_xticklabels(["Locked", "Input-\nComBat"], fontsize=7.0)
axA2.set_xlim(-0.5, 1.5)
axA2.set_ylim(0.45, 0.97)
axA2.set_ylabel("ADNI OOF ROC-AUC")
axA2.set_title("A2. ADNI diagnostic\nperformance", fontsize=7.4, pad=4)
style_axis(axA2)
axA2.text(0.5, 0.93, f"$\\Delta$={auc_delta_row['observed_delta']:.3f}\n95% CI [{auc_delta_ci[0]:.3f}, {auc_delta_ci[1]:.3f}]",
          transform=axA2.transAxes, ha="center", va="top", fontsize=5.2, color=INK_SECONDARY, linespacing=1.3)

# ── Panel B: standardized marginal shift vs projected AUC, 3 shared-coordinate arms ──
arm_style = {
    "locked_frozen_transfer": dict(color=COLOR_LOCKED_B, marker="s", label="Locked", label_dy=18, label_dx=0),
    "external_dataset_combat_adni_reference": dict(color=COLOR_ADNI_REF, marker="o", label="ADNI ref.", label_dy=18, label_dx=0),
    "external_dataset_combat_no_reference": dict(color=COLOR_NO_REF, marker="D", label="No ref.\n(secondary)", label_dy=-30, label_dx=0),
}
locked_x = abs(geom.loc["locked_frozen_transfer", "marginal_standardized_shift"])
locked_y = geom.loc["locked_frozen_transfer", "projected_auc"]
for arm, style in arm_style.items():
    x = abs(geom.loc[arm, "marginal_standardized_shift"])
    y = geom.loc[arm, "projected_auc"]
    x_ci = sorted([abs(geom.loc[arm, "marginal_standardized_shift_ci_low_2p5"]),
                   abs(geom.loc[arm, "marginal_standardized_shift_ci_high_97p5"])])
    y_ci = [geom.loc[arm, "projected_auc_ci_low_2p5"], geom.loc[arm, "projected_auc_ci_high_97p5"]]
    if arm != "locked_frozen_transfer":
        y_off = -0.012 if "no_reference" not in arm else -0.030  # stagger the two arrows so they don't fully overlap
        arrow_y = y + y_off
        arrow = mpatches.FancyArrowPatch((locked_x, locked_y + y_off), (x, arrow_y),
                                          arrowstyle="-|>", mutation_scale=8, linewidth=1.3,
                                          color=style["color"], alpha=0.85, zorder=4.5,
                                          linestyle="dashed" if "no_reference" in arm else "solid",
                                          shrinkA=6, shrinkB=6, transform=axB.transData)
        axB.add_patch(arrow)
    axB.errorbar([x], [y], xerr=[[x - x_ci[0]], [x_ci[1] - x]], yerr=[[y - y_ci[0]], [y_ci[1] - y]],
                 fmt="none", ecolor=style["color"], elinewidth=0.9, capsize=2.2, zorder=3, alpha=0.55)
    axB.scatter([x], [y], s=30, color=style["color"], marker=style["marker"],
                edgecolor="white", linewidth=0.5, zorder=5,
                facecolor="white" if "no_reference" in arm else style["color"])
    axB.annotate(style["label"], xy=(x, y), xytext=(style["label_dx"], style["label_dy"]),
                 textcoords="offset points", ha="center", va="center", fontsize=5.4,
                 color=style["color"], fontweight="bold", linespacing=1.2, zorder=6)
axB.axhline(0.5, color=INK_MUTED, linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)
axB.set_xlabel("Absolute standardized marginal shift\n(OASIS vs. ADNI, unit-norm axis)", fontsize=7.0)
axB.set_ylabel("Fold-local projected AUC")
axB.set_xlim(0.15, 1.3)
axB.set_ylim(0.50, 0.80)
axB.set_title("B. Shared-coordinate geometry\n(locked VAE latents only)", fontsize=7.4, pad=4)
style_axis(axB)

# ── Panel C: CDR score distributions, mapping A (nearest within 180 days) ──
cdr_order = ["CDR 0", "CDR 0.5", "CDR >= 1 (high severity)"]
positions = np.arange(3)
group_data = {g: mapped_a.loc[mapped_a["A_nearest_180d_cdr_group"] == g, "score_locked"].to_numpy() for g in cdr_order}
bp = axC.boxplot([group_data[g] for g in cdr_order], positions=positions, widths=0.34, patch_artist=True,
                  showfliers=False, zorder=2, medianprops=dict(color=INK_PRIMARY, linewidth=1.0),
                  whiskerprops=dict(color=INK_MUTED, linewidth=0.7), capprops=dict(color=INK_MUTED, linewidth=0.7))
for patch, g in zip(bp["boxes"], cdr_order):
    patch.set_facecolor(CDR_COLORS[g]); patch.set_alpha(0.55); patch.set_edgecolor(INK_SECONDARY); patch.set_linewidth(0.6)
rng = np.random.default_rng(20260712)
for xi, g in zip(positions, cdr_order):
    vals = group_data[g]
    jitter = rng.uniform(-0.12, 0.12, size=len(vals))
    axC.scatter(xi + jitter, vals, s=5, color=CDR_COLORS[g], edgecolor=INK_SECONDARY, linewidth=0.12, alpha=0.7, zorder=3)
n_by_group = {g: len(group_data[g]) for g in cdr_order}
axC.set_xticks(positions)
axC.set_xticklabels([f"CDR 0\n(n={n_by_group['CDR 0']})", f"CDR 0.5\n(n={n_by_group['CDR 0.5']})",
                      f"CDR$\\geq$1\n(n={n_by_group['CDR >= 1 (high severity)']})"], fontsize=6.6)
axC.set_ylabel("AD-oriented ensemble score\n(locked frozen transfer)")
axC.set_ylim(-0.05, 1.18)
axC.set_title("C. Score by CDR severity\n(+/-180d mapping, n=169/180)", fontsize=7.4, pad=4)
style_axis(axC)
axC.text(0.03, 0.99, f"AUC(CDR0 vs CDR$\\geq$1)={mapping_a['pairwise_auc']:.3f}\n"
                     f"95% CI [{mapping_a['pairwise_auc_ci_low']:.3f}, {mapping_a['pairwise_auc_ci_high']:.3f}]\n"
                     f"partial $\\rho$(Age,Sex-adj)={mapping_a['partial_spearman_rho_age_sex_adj']:.3f},\n"
                     f"p={mapping_a['partial_spearman_pvalue_age_sex_adj']:.1e}",
         transform=axC.transAxes, ha="left", va="top", fontsize=5.1, color=INK_PRIMARY, linespacing=1.3)

fig.text(0.5, 0.055,
         "CDR is a severity indicator, not an AD-etiology confirmation; \"CDR $\\geq$1\" is not labeled Alzheimer's disease.",
         ha="center", va="center", fontsize=5.3, color=INK_SECONDARY)
fig.text(0.5, 0.018,
         "Siemens-ComBat VAE (separately retrained) is excluded from Panel B: it does not share the locked VAE's latent coordinate system.",
         ha="center", va="center", fontsize=5.3, color=INK_SECONDARY)

OUT_PDF = OUT_DIR / "fig_sipaim_transportability_main.pdf"
OUT_PNG = OUT_DIR / "fig_sipaim_transportability_main.png"
fig.savefig(OUT_PDF, format="pdf")
fig.savefig(OUT_PNG, format="png", dpi=600)
plt.close(fig)
print(f"Wrote {OUT_PDF}")
print(f"Wrote {OUT_PNG}")

# ═══════════════════════════════════════════════════════════════════════════
# Data table
# ═══════════════════════════════════════════════════════════════════════════
data_df = pd.DataFrame(figure_data_rows)
data_df.to_csv(OUT_DIR / "fig_sipaim_transportability_main_data.csv", index=False)
print(f"Wrote {OUT_DIR / 'fig_sipaim_transportability_main_data.csv'} ({len(data_df)} rows)")
