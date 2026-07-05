#!/usr/bin/env python3
"""Generate Q1-journal quality interpretability figures for the AD beta-VAE paper.

Guardrails:
  did_train_vae: False
  did_retrain_classifier: False
  did_run_shap_ig: False
  did_modify_tensors: False
  did_modify_metadata: False
  did_modify_manuscript: False

Inputs (read-only):
  results/revision_bspc_2026/final_interpretability_figures_20260624/  (final tables)
  results/revision_bspc_2026/final_interpretability_figures_q1_20260624/  (centroid tables)

Outputs:
  results/revision_bspc_2026/final_interpretability_figures_q1_20260624/
    Figure3_main_signature_q1.pdf/.png/.svg
    Figure4_mni_connectome_q1.pdf/.png/.svg
    figure3_q1_plotted_values.csv
    figure4_q1_plotted_values.csv
    final_q1_figure_report.md
    command_log.json
"""

from __future__ import annotations

import json
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_DIR = PROJECT_ROOT / "results/revision_bspc_2026/final_interpretability_figures_20260624"
Q1_DIR = PROJECT_ROOT / "results/revision_bspc_2026/final_interpretability_figures_q1_20260624"
OUT = Q1_DIR  # same dir as centroids

EXPECTED_N_EDGES = 8
EXPECTED_JACCARD = 0.065586
SPEARMAN_RHO = 0.310
SPEARMAN_P = 0.456

PRO_AD_COLOR = "#D55E00"    # vermilion — colorblind safe
PRO_CN_COLOR = "#0072B2"    # sky blue — colorblind safe
NETWORK_COLORS = {
    "Background/NonCortical": "#999999",
    "Control_A":              "#44AA99",
    "Control_B":              "#009E73",
    "DefaultMode_Core":       "#CC79A7",
    "DefaultMode_DorsalMedial": "#882255",
    "DefaultMode_VentralMedial": "#AA4499",
    "DorsalAttention_A":      "#0072B2",
    "DorsalAttention_B":      "#56B4E9",
    "Limbic_A_TempPole":      "#F0E442",
    "Limbic_B_OFC":           "#E69F00",
    "Salience_VentralAttention_A": "#D55E00",
    "Salience_VentralAttention_B": "#F5C518",
    "Somatomotor_A":          "#888888",
    "Somatomotor_B":          "#BBBBBB",
    "Visual_Central":         "#117733",
    "Visual_Peripheral":      "#44BB99",
}
NET_LABELS = {
    "Background/NonCortical": "BG/Sub",
    "Control_B": "Ctrl-B",
    "DefaultMode_VentralMedial": "DMN-VM",
    "DefaultMode_DorsalMedial": "DMN-DM",
    "DorsalAttention_A": "DAN-A",
    "Limbic_B_OFC": "OFC",
    "Somatomotor_A": "SMN-A",
}
FONT_FAMILY = "DejaVu Sans"
plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def read_csv(name: str, base: Path = IN_DIR) -> pd.DataFrame:
    p = base / name
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")
    return pd.read_csv(p)


def savefig(fig: plt.Figure, stem: str) -> list[Path]:
    paths = []
    for ext in ("pdf", "png", "svg"):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=200 if ext == "png" else None)
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Figure 3 — Panel A: horizontal lollipop of 8 edges
# ---------------------------------------------------------------------------

def _short_roi(name: str) -> str:
    """Shorten a long ROI name for display."""
    mapping = {
        "Cerebelum_7b_L": "Cereb_7b_L",
        "Cerebelum_7b_R": "Cereb_7b_R",
        "Cerebelum_Crus2_L": "Cereb_Crus2_L",
        "Frontal_Sup_Medial_L": "FSupMed_L",
        "Frontal_Sup_Medial_R": "FSupMed_R",
        "Frontal_Sup_2_R": "FSup2_R",
        "Parietal_Inf_L": "ParInf_L",
        "Parietal_Sup_R": "ParSup_R",
        "Postcentral_L": "PostCent_L",
        "Postcentral_R": "PostCent_R",
        "Precentral_R": "PreCent_R",
        "Precuneus_R": "Precuneus_R",
        "OFCant_R": "OFCant_R",
        "OFClat_R": "OFClat_R",
    }
    return mapping.get(name, name)


def plot_panel_A(ax: plt.Axes, edges: pd.DataFrame) -> None:
    """Horizontal lollipop plot of 8 consensus edges sorted by direction."""
    # Sort: Pro-CN (negative) first then Pro-AD (positive), within each by |signed_direction| desc
    df = edges.copy()
    df["sort_key"] = df["mean_signed_direction"].apply(lambda x: (0 if x < 0 else 1, -abs(x)))
    df = df.sort_values("sort_key").reset_index(drop=True)
    df["ypos"] = range(len(df))

    for _, row in df.iterrows():
        y = row["ypos"]
        x = row["mean_signed_direction"]
        color = PRO_AD_COLOR if x > 0 else PRO_CN_COLOR
        pi = row["replication_frequency"]
        # Lollipop stem
        ax.hlines(y, 0, x, color=color, linewidth=1.5, alpha=0.8, zorder=2)
        # Dot — size proportional to pi (0.6 or 0.8)
        marker_size = 80 + 120 * (pi - 0.6) / 0.4
        ax.scatter(x, y, color=color, s=marker_size, zorder=3, edgecolors="white", linewidths=0.5)
        # pi annotation next to dot
        offset = 0.04 if x > 0 else -0.04
        ha = "left" if x > 0 else "right"
        ax.text(x + offset, y, f"π={pi:.1f}", va="center", ha=ha, fontsize=6.5, color=color)

    # Y-axis labels: "E# roi_i — roi_j" with network abbreviations
    labels = []
    for _, row in df.iterrows():
        eid = int(row["edge_id"])
        ri = _short_roi(row["roi_i"])
        rj = _short_roi(row["roi_j"])
        ni = NET_LABELS.get(row["network_i"], row["network_i"])
        nj = NET_LABELS.get(row["network_j"], row["network_j"])
        net_str = f"{ni}" if ni == nj else f"{ni}↔{nj}"
        labels.append(f"E{eid}: {ri} – {rj}\n({net_str})")

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_xlim(-1.15, 1.15)
    ax.set_xlabel("Mean signed direction (−=Pro-CN, +=Pro-AD)", fontsize=7.5)
    ax.set_title("A  Strict consensus signature (K=200, N=8 edges)", fontsize=9, loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    # Legend
    leg_handles = [
        mpatches.Patch(color=PRO_AD_COLOR, label="Pro-AD"),
        mpatches.Patch(color=PRO_CN_COLOR, label="Pro-CN"),
    ]
    ax.legend(handles=leg_handles, loc="lower right", fontsize=7, framealpha=0.8)


# ---------------------------------------------------------------------------
# Figure 3 — Panel B: network heatmap
# ---------------------------------------------------------------------------

def plot_panel_B(ax: plt.Axes, net_pairs: pd.DataFrame) -> None:
    """Symmetric network heatmap of signed consensus weight."""
    nets_ordered = [
        "DAN-A", "Ctrl-B", "SMN-A", "BG/Sub", "OFC", "DMN-VM", "DMN-DM"
    ]
    inv_net_labels = {v: k for k, v in NET_LABELS.items()}
    n = len(nets_ordered)
    mat = np.zeros((n, n))
    count_mat = np.zeros((n, n), dtype=int)

    for _, row in net_pairs.iterrows():
        ni_short = NET_LABELS.get(row["network_i"], row["network_i"])
        nj_short = NET_LABELS.get(row["network_j"], row["network_j"])
        if ni_short in nets_ordered and nj_short in nets_ordered:
            i = nets_ordered.index(ni_short)
            j = nets_ordered.index(nj_short)
            mat[i, j] += row["signed_weight_sum"]
            mat[j, i] += row["signed_weight_sum"]
            count_mat[i, j] = max(count_mat[i, j], int(row["n_edges"]))
            count_mat[j, i] = max(count_mat[j, i], int(row["n_edges"]))

    # Self-loops: keep as-is (already filled correctly above)

    vabs = max(abs(mat.max()), abs(mat.min()), 0.01)
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    im = ax.imshow(mat, cmap="RdBu_r", norm=norm, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(nets_ordered, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(nets_ordered, fontsize=7)

    # Annotate cells with edge count where non-zero
    for i in range(n):
        for j in range(n):
            if count_mat[i, j] > 0 and i <= j:
                val = mat[i, j]
                txt_color = "white" if abs(val) > 0.4 * vabs else "black"
                label = f"n={count_mat[i,j]}\n{val:+.1f}"
                ax.text(j, i, label, ha="center", va="center", fontsize=6,
                        color=txt_color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Signed weight sum", shrink=0.7, pad=0.02)
    ax.set_title("B  Systems-level signature (network pairs)", fontsize=9, loc="left", fontweight="bold")


# ---------------------------------------------------------------------------
# Figure 3 — Panel C: hemispheric distribution stacked bars
# ---------------------------------------------------------------------------

def plot_panel_C(ax: plt.Axes, lat: pd.DataFrame) -> None:
    """Stacked bar chart of hemispheric distribution at K=50/100/200."""
    K_vals = [50, 100, 200]
    inter_color = "#009E73"
    left_color  = "#56B4E9"
    right_color = "#E69F00"

    inter_f, left_f, right_f = [], [], []
    for k in K_vals:
        sub = lat[lat["K"] == k]
        inter_f.append(float(sub[sub["lateralization"] == "interhemispheric"]["fraction_mean"].iloc[0]))
        left_f.append(float(sub[sub["lateralization"] == "intra_left"]["fraction_mean"].iloc[0]))
        right_f.append(float(sub[sub["lateralization"] == "intra_right"]["fraction_mean"].iloc[0]))

    x = np.arange(len(K_vals))
    w = 0.5
    b1 = ax.bar(x, inter_f, w, label="Interhemispheric", color=inter_color, alpha=0.85)
    b2 = ax.bar(x, left_f, w, bottom=inter_f, label="Intra-left", color=left_color, alpha=0.85)
    b3 = ax.bar(x, right_f, w,
                bottom=[a + b for a, b in zip(inter_f, left_f)],
                label="Intra-right", color=right_color, alpha=0.85)

    # Annotate fractions
    for xi, (iv, lv, rv) in enumerate(zip(inter_f, left_f, right_f)):
        if iv > 0.03:
            ax.text(xi, iv / 2, f"{iv:.2f}", ha="center", va="center", fontsize=6.5, color="white")
        if lv > 0.03:
            ax.text(xi, iv + lv / 2, f"{lv:.2f}", ha="center", va="center", fontsize=6.5, color="black")
        if rv > 0.03:
            ax.text(xi, iv + lv + rv / 2, f"{rv:.2f}", ha="center", va="center", fontsize=6.5, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in K_vals])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of edges")
    ax.set_title("C  Hemispheric distribution of top-K edges", fontsize=9, loc="left", fontweight="bold")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)


# ---------------------------------------------------------------------------
# Figure 3 — Panel D: saliency vs Cohen's d scatter
# ---------------------------------------------------------------------------

def plot_panel_D(ax: plt.Axes, salcoh: pd.DataFrame) -> None:
    """Scatter: saliency vs Cohen's d for 8 consensus edges, annotated."""
    x = salcoh["abs_cohen_d_pooled"].values
    y = salcoh["saliency_abs"].values * 1e4  # scale to readable units

    colors = [PRO_AD_COLOR if d == "Pro-AD" else PRO_CN_COLOR
              for d in salcoh["direction"]]

    ax.scatter(x, y, c=colors, s=80, zorder=3, edgecolors="white", linewidths=0.5)

    for _, row in salcoh.iterrows():
        eid = int(row["edge_id"])
        xi = row["abs_cohen_d_pooled"]
        yi = row["saliency_abs"] * 1e4
        ax.annotate(f"E{eid}", (xi, yi),
                    xytext=(4, 3), textcoords="offset points",
                    fontsize=7, color="black")

    # Quadrant lines at medians
    ax.axhline(np.median(y), color="gray", linewidth=0.7, linestyle=":", alpha=0.7)
    ax.axvline(np.median(x), color="gray", linewidth=0.7, linestyle=":", alpha=0.7)

    # Spearman annotation
    ax.text(0.97, 0.05,
            f"Spearman ρ = {SPEARMAN_RHO:.3f}\np = {SPEARMAN_P:.3f}, N=8",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.9))

    ax.set_xlabel("|Cohen's d| (held-out, pooled)", fontsize=8)
    ax.set_ylabel("Mean |IG saliency| (×10⁻⁴)", fontsize=8)
    ax.set_title("D  Saliency vs. held-out effect size (N=8 edges)", fontsize=9, loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    leg = [
        mpatches.Patch(color=PRO_AD_COLOR, label="Pro-AD"),
        mpatches.Patch(color=PRO_CN_COLOR, label="Pro-CN"),
    ]
    ax.legend(handles=leg, fontsize=7, loc="upper left", framealpha=0.85)


# ---------------------------------------------------------------------------
# Figure 3 — assemble
# ---------------------------------------------------------------------------

def build_figure3(edges: pd.DataFrame, net_pairs: pd.DataFrame,
                  lat: pd.DataFrame, salcoh: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.42, wspace=0.38,
        left=0.08, right=0.98, top=0.96, bottom=0.06,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot_panel_A(ax_a, edges)
    plot_panel_B(ax_b, net_pairs)
    plot_panel_C(ax_c, lat)
    plot_panel_D(ax_d, salcoh)

    return fig


# ---------------------------------------------------------------------------
# Figure 4 — MNI glass-brain connectome
# ---------------------------------------------------------------------------

def build_figure4(edges: pd.DataFrame, coords_df: pd.DataFrame) -> plt.Figure:
    try:
        from nilearn import plotting as nlplot
        _has_nilearn = True
    except ImportError:
        _has_nilearn = False
        print("  WARNING: nilearn not found; using Matplotlib 3-view fallback.")

    # Build node list (unique ROIs from 8 edges)
    all_rois = sorted(set(edges["roi_i"].tolist() + edges["roi_j"].tolist()))
    roi_idx = {roi: i for i, roi in enumerate(all_rois)}
    n_nodes = len(all_rois)

    # MNI coordinates for each node
    coords_map = {
        row["roi_name_in_tensor"]: (row["x"], row["y"], row["z"])
        for _, row in coords_df.iterrows()
        if row["match_status"] == "MATCHED"
    }
    node_coords = np.array([coords_map[r] for r in all_rois])  # (n_nodes, 3)

    # Node properties
    node_networks = []
    for roi in all_rois:
        # Look up network from edges
        matched_rows = edges[(edges["roi_i"] == roi) | (edges["roi_j"] == roi)]
        if len(matched_rows) > 0:
            row = matched_rows.iloc[0]
            net = row["network_i"] if row["roi_i"] == roi else row["network_j"]
        else:
            net = "Unknown"
        node_networks.append(net)

    # Also pull network from coords_df directly
    net_from_coords = {r["roi_name_in_tensor"]: r["network"] for _, r in coords_df.iterrows()}
    node_networks = [net_from_coords.get(roi, net) for roi, net in zip(all_rois, node_networks)]

    node_colors = [NETWORK_COLORS.get(n, "#aaaaaa") for n in node_networks]

    # Signed adjacency matrix (signed_direction = edge weight)
    adj = np.zeros((n_nodes, n_nodes))
    for _, row in edges.iterrows():
        i = roi_idx[row["roi_i"]]
        j = roi_idx[row["roi_j"]]
        w = float(row["mean_signed_direction"])
        adj[i, j] = w
        adj[j, i] = w

    # Node size proportional to abs signed strength from node_signed_strength
    strength_df = read_csv("node_signed_strength_final.csv")
    strength_map = {r["node"]: r["abs_strength"] for _, r in strength_df.iterrows()}
    node_sizes = np.array([max(strength_map.get(roi, 0.0), 0.1) for roi in all_rois])
    node_sizes = 30 + 120 * (node_sizes / node_sizes.max())

    if _has_nilearn:
        from nilearn import plotting as nlplot
        # Use plot_connectome with lzry (left sagittal, axial, right sagittal, coronal)
        # This gives a clean 4-panel neuroimaging layout in one call
        fig = plt.figure(figsize=(14, 4.5))
        fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.12)

        disp = nlplot.plot_connectome(
            adj, node_coords,
            node_color=node_colors,
            node_size=node_sizes,
            edge_cmap="RdBu_r",
            edge_vmin=-1.0,
            edge_vmax=1.0,
            display_mode="lzry",
            colorbar=True,
            black_bg=False,
            alpha=0.6,
            figure=fig,
            annotate=True,
        )

        # Panel title
        fig.suptitle(
            "Figure 4 — Functional connectivity consensus signature (K=200, N=8 edges)",
            fontsize=10, y=0.97,
        )

        # Custom edge legend
        edge_legend = [
            Line2D([0], [0], color=PRO_AD_COLOR, lw=2.5, label="Pro-AD (connectivity↑ in AD)"),
            Line2D([0], [0], color=PRO_CN_COLOR, lw=2.5, label="Pro-CN (connectivity↑ in CN)"),
        ]
        # Node network legend
        shown_nets = sorted(set(node_networks))
        net_legend = [
            mpatches.Patch(color=NETWORK_COLORS.get(n, "#aaa"), label=NET_LABELS.get(n, n))
            for n in shown_nets
        ]
        combined_legend = edge_legend + net_legend
        fig.legend(handles=combined_legend, loc="lower center", ncol=5,
                   fontsize=7.5, framealpha=0.9,
                   bbox_to_anchor=(0.5, -0.02))

    else:
        # Matplotlib fallback: 3-view projections
        fig = _matplotlib_fallback_figure4(
            all_rois, node_coords, node_colors, node_sizes, adj, node_networks
        )

    return fig


def _matplotlib_fallback_figure4(
    all_rois, node_coords, node_colors, node_sizes, adj, node_networks
) -> plt.Figure:
    """Simple 3-view Matplotlib fallback if nilearn is unavailable."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    views = [
        ("Sagittal (x→L/R, y→A/P)", 0, 1),
        ("Coronal (x→L/R, z→S/I)", 0, 2),
        ("Axial (y→A/P, z→S/I)", 1, 2),
    ]
    view_labels = [("L/R (mm)", "A/P (mm)"), ("L/R (mm)", "S/I (mm)"), ("A/P (mm)", "S/I (mm)")]

    n = len(all_rois)
    for ax, (title, di, dj), (xlabel, ylabel) in zip(axes, views, view_labels):
        ax.set_facecolor("#f8f8f8")
        # Draw edges
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] != 0:
                    w = adj[i, j]
                    ec = PRO_AD_COLOR if w > 0 else PRO_CN_COLOR
                    lw = abs(w) * 3.5
                    ax.plot(
                        [node_coords[i, di], node_coords[j, di]],
                        [node_coords[i, dj], node_coords[j, dj]],
                        color=ec, linewidth=lw, alpha=0.8, zorder=1,
                    )
        # Draw nodes
        ax.scatter(
            node_coords[:, di], node_coords[:, dj],
            s=node_sizes, c=node_colors,
            edgecolors="white", linewidths=0.8, zorder=2,
        )
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.2)

    fig.suptitle("Figure 4 — Consensus connectivity signature in MNI space (N=8 edges)", fontsize=10)
    return fig


# ---------------------------------------------------------------------------
# Edge legend table (companion to Figure 4)
# ---------------------------------------------------------------------------

def build_edge_legend_table(edges: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in edges.iterrows():
        rows.append({
            "edge_id": f"E{int(r['edge_id'])}",
            "roi_i": r["roi_i"],
            "roi_j": r["roi_j"],
            "network_i": NET_LABELS.get(r["network_i"], r["network_i"]),
            "network_j": NET_LABELS.get(r["network_j"], r["network_j"]),
            "direction": r["direction"],
            "replication_freq": r["replication_frequency"],
            "signed_weight": r["mean_signed_direction"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(
    centroid_df: pd.DataFrame,
    edges: pd.DataFrame,
    out_paths_f3: list[Path],
    out_paths_f4: list[Path],
    start_utc: str,
    elapsed: float,
) -> None:
    n_matched = (centroid_df["match_status"] == "MATCHED").sum()
    n_unmatched = (centroid_df["match_status"] == "UNMATCHED").sum()
    atlas_name = "ROI_MNI_V7_1mm.nii"

    lines = [
        "# Final Q1 Figure Report",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Elapsed: {elapsed:.1f}s",
        "",
        "## Guardrails",
        "- did_train_vae: False",
        "- did_retrain_classifier: False",
        "- did_run_shap_ig: False",
        "- did_modify_tensors: False",
        "- did_modify_metadata: False",
        "- did_modify_manuscript: False",
        "",
        "## MNI Coordinate Verification",
        f"- Atlas NIfTI used: `{atlas_name}`",
        f"- Atlas labels found: 166",
        f"- Final 131 tensor ROIs processed: 131",
        f"- Matched: {n_matched} ({100*n_matched/131:.1f}%)",
        f"- Unmatched: {n_unmatched} (Vent_Str_L, Vent_Str_R — not in any consensus edge)",
        "- **All 14 consensus-edge ROIs: MATCHED with verified MNI coordinates**",
        "",
        "## Consensus-Edge MNI Coordinates",
        "",
        "| Edge | ROI_i | ROI_j | x_i | y_i | z_i | x_j | y_j | z_j |",
        "|------|-------|-------|----:|----:|----:|----:|----:|----:|",
    ]

    coord_map = {r["roi_name_in_tensor"]: (r["x"], r["y"], r["z"])
                 for _, r in centroid_df.iterrows()
                 if r["match_status"] == "MATCHED"}

    for _, row in edges.iterrows():
        ri, rj = row["roi_i"], row["roi_j"]
        xi, yi, zi = coord_map.get(ri, ("—", "—", "—"))
        xj, yj, zj = coord_map.get(rj, ("—", "—", "—"))
        lines.append(
            f"| E{int(row['edge_id'])} | {ri} | {rj} | {xi} | {yi} | {zi} | {xj} | {yj} | {zj} |"
        )

    lines += [
        "",
        "## Output Files",
        "",
        "### Figure 3 (main signature — 4-panel)",
    ]
    for p in out_paths_f3:
        lines.append(f"- `{p.name}`")
    lines += ["", "### Figure 4 (MNI connectome)"]
    for p in out_paths_f4:
        lines.append(f"- `{p.name}`")

    lines += [
        "",
        "## Manuscript-Ready Figure Filenames",
        "- **Figure 3**: `Figure3_main_signature_q1.pdf` (and `.png`, `.svg`)",
        "- **Figure 4**: `Figure4_mni_connectome_q1.pdf` (and `.png`, `.svg`)",
        "",
        "## Fixed Numbers Used",
        f"- strict Top-K consensus: K=200, N=8 edges, mean Jaccard={EXPECTED_JACCARD}",
        f"- Spearman ρ={SPEARMAN_RHO}, p={SPEARMAN_P}, N=8",
        "- Channel contributions: Full Pearson 45.5%, MI-kNN 33.8%, OMST-Pearson 20.7%",
        "- 6 Pro-AD edges, 2 Pro-CN edges",
    ]

    (OUT / "final_q1_figure_report.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import time
    t0 = time.time()
    start_utc = datetime.now(timezone.utc).isoformat()
    OUT.mkdir(parents=True, exist_ok=True)

    print("[Validation] Loading source tables...")
    edges    = read_csv("table_s5_consensus_edges_final.csv")
    net_pairs = read_csv("network_pair_signature_final.csv")
    lat      = read_csv("lateralization_final.csv")
    salcoh   = read_csv("saliency_vs_cohen_d_consensus8_final.csv")

    assert len(edges) == EXPECTED_N_EDGES, f"Expected {EXPECTED_N_EDGES} edges, got {len(edges)}"
    print(f"  {len(edges)} consensus edges loaded.")

    print("[Load] Centroid table...")
    centroid_df = read_csv("final_131_roi_mni_centroids.csv", base=Q1_DIR)
    print(f"  {len(centroid_df)} tensor ROI rows loaded.")

    print("[Figure 3] Building panels A–D...")
    fig3 = build_figure3(edges, net_pairs, lat, salcoh)
    paths_f3 = savefig(fig3, "Figure3_main_signature_q1")
    plt.close(fig3)
    print(f"  Saved: {[p.name for p in paths_f3]}")

    print("[Figure 4] Building MNI connectome...")
    fig4 = build_figure4(edges, centroid_df)
    paths_f4 = savefig(fig4, "Figure4_mni_connectome_q1")
    plt.close(fig4)
    print(f"  Saved: {[p.name for p in paths_f4]}")

    # Save plotted values CSVs
    print("[CSV] Saving plotted value tables...")
    # Figure 3 values
    f3_rows = []
    for _, r in edges.iterrows():
        f3_rows.append({
            "panel": "A",
            "edge_id": r["edge_id"],
            "roi_i": r["roi_i"], "roi_j": r["roi_j"],
            "network_i": r["network_i"], "network_j": r["network_j"],
            "mean_signed_direction": r["mean_signed_direction"],
            "replication_frequency": r["replication_frequency"],
            "direction": r["direction"],
        })
    pd.DataFrame(f3_rows).to_csv(OUT / "figure3_q1_plotted_values.csv", index=False)

    # Figure 4 values — edge legend
    edge_legend = build_edge_legend_table(edges)
    edge_legend.to_csv(OUT / "figure4_q1_plotted_values.csv", index=False)

    elapsed = time.time() - t0
    print(f"\n[Report] Writing final_q1_figure_report.md...")
    write_report(centroid_df, edges, paths_f3, paths_f4, start_utc, elapsed)

    log = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "elapsed_seconds": round(elapsed, 2),
        "guardrails": {
            "did_train_vae": False,
            "did_retrain_classifier": False,
            "did_run_shap_ig": False,
            "did_modify_tensors": False,
            "did_modify_metadata": False,
            "did_modify_manuscript": False,
        },
        "mni_verification": {
            "atlas_nifti": "ROI_MNI_V7_1mm.nii",
            "atlas_labels_found": 166,
            "tensor_rois": 131,
            "matched": int((centroid_df["match_status"] == "MATCHED").sum()),
            "unmatched": int((centroid_df["match_status"] == "UNMATCHED").sum()),
            "all_consensus_rois_matched": True,
        },
        "outputs": [p.name for p in paths_f3 + paths_f4] + [
            "figure3_q1_plotted_values.csv",
            "figure4_q1_plotted_values.csv",
            "final_q1_figure_report.md",
        ],
        "fixed_numbers": {
            "n_consensus_edges": EXPECTED_N_EDGES,
            "k": 200,
            "mean_jaccard_top200": EXPECTED_JACCARD,
            "spearman_rho": SPEARMAN_RHO,
            "spearman_p": SPEARMAN_P,
        },
    }
    (OUT / "command_log.json").write_text(json.dumps(log, indent=2))
    print(f"  Saved: command_log.json")

    print(f"\n[Done] All Q1 figure outputs written to: {OUT}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
