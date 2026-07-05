"""Utilities for final BSPC interpretability tables and figures.

These helpers are intentionally data-only: they do not train models, run SHAP,
run Integrated Gradients, or modify model artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRO_AD_COLOR = "#D55E00"
PRO_CN_COLOR = "#0072B2"
NEUTRAL_COLOR = "#4D4D4D"


def write_csv_md(df: pd.DataFrame, csv_path: Path, md_path: Path | None = None) -> None:
    """Write a CSV plus a compact Markdown rendering."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    if md_path is not None:
        try:
            md_text = df.to_markdown(index=False)
        except Exception:
            md_text = df.to_string(index=False)
        md_path.write_text(md_text + "\n", encoding="utf-8")


def hemisphere_from_roi(roi_name: str) -> str:
    """Return L/R/Mixed for AAL-style ROI names."""
    name = str(roi_name)
    if name.endswith("_L"):
        return "L"
    if name.endswith("_R"):
        return "R"
    return "M"


def edge_lateralization(roi_i: str, roi_j: str) -> str:
    """Classify an edge by endpoint hemispheres."""
    hi = hemisphere_from_roi(roi_i)
    hj = hemisphere_from_roi(roi_j)
    if hi == "L" and hj == "L":
        return "intra_left"
    if hi == "R" and hj == "R":
        return "intra_right"
    if {hi, hj} == {"L", "R"}:
        return "interhemispheric"
    return "midline_or_unknown"


def cohen_d(x_ad: Iterable[float], x_cn: Iterable[float]) -> float:
    """AD minus CN Cohen's d with pooled sample SD."""
    a = np.asarray(list(x_ad), dtype=float)
    c = np.asarray(list(x_cn), dtype=float)
    a = a[np.isfinite(a)]
    c = c[np.isfinite(c)]
    if len(a) < 2 or len(c) < 2:
        return float("nan")
    va = np.var(a, ddof=1)
    vc = np.var(c, ddof=1)
    denom = ((len(a) - 1) * va + (len(c) - 1) * vc) / (len(a) + len(c) - 2)
    if denom <= 0:
        return float("nan")
    return float((np.mean(a) - np.mean(c)) / np.sqrt(denom))


def infer_node_positions(edges: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Build deterministic 2D positions for consensus nodes.

    This is a schematic connectome projection because the final artifact set
    does not include verified MNI coordinates. Left/right naming controls the
    horizontal axis; network order controls the vertical axis.
    """
    nodes: dict[str, str] = {}
    for _, row in edges.iterrows():
        nodes[str(row["roi_i"])] = str(row["network_i"])
        nodes[str(row["roi_j"])] = str(row["network_j"])
    networks = sorted(set(nodes.values()))
    y_lookup = {net: 1.0 - i / max(1, len(networks) - 1) for i, net in enumerate(networks)}
    hemi_counts: dict[str, int] = {"L": 0, "R": 0, "M": 0}
    positions: dict[str, tuple[float, float]] = {}
    for node, net in sorted(nodes.items()):
        hemi = hemisphere_from_roi(node)
        base_x = {"L": 0.18, "R": 0.82, "M": 0.50}.get(hemi, 0.50)
        offset = (hemi_counts[hemi] % 5 - 2) * 0.018
        hemi_counts[hemi] += 1
        positions[node] = (base_x + offset, y_lookup[net])
    return positions


def plot_consensus_schematic(ax: plt.Axes, edges: pd.DataFrame) -> None:
    """Draw an 8-edge schematic connectome map."""
    pos = infer_node_positions(edges)
    for _, row in edges.iterrows():
        a = str(row["roi_i"])
        b = str(row["roi_j"])
        x1, y1 = pos[a]
        x2, y2 = pos[b]
        direction = str(row.get("direction", ""))
        color = PRO_AD_COLOR if direction == "Pro-AD" else PRO_CN_COLOR if direction == "Pro-CN" else NEUTRAL_COLOR
        width = 1.2 + 2.5 * float(row.get("replication_frequency", 0.6))
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, alpha=0.75, zorder=1)
    for node, (x, y) in pos.items():
        ax.scatter([x], [y], s=52, color="white", edgecolor="black", linewidth=0.8, zorder=2)
        ax.text(x, y + 0.025, node.replace("_", " "), ha="center", va="bottom", fontsize=6.5)
    ax.text(0.18, -0.06, "Left", ha="center", va="top", fontsize=8)
    ax.text(0.82, -0.06, "Right", ha="center", va="top", fontsize=8)
    ax.set_xlim(0.02, 0.98)
    ax.set_ylim(-0.10, 1.08)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("A. Strict 8-edge consensus", loc="left", fontweight="bold")
    ax.set_frame_on(False)


def plot_network_heatmap(ax: plt.Axes, network_sig: pd.DataFrame) -> None:
    """Plot signed network-pair signature."""
    nets = sorted(set(network_sig["network_i"]).union(set(network_sig["network_j"])))
    mat = pd.DataFrame(0.0, index=nets, columns=nets)
    for _, row in network_sig.iterrows():
        mat.loc[row["network_i"], row["network_j"]] += float(row["signed_weight_sum"])
        mat.loc[row["network_j"], row["network_i"]] += float(row["signed_weight_sum"])
    vmax = max(1.0, float(np.nanmax(np.abs(mat.values)))) if mat.size else 1.0
    im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(nets)))
    ax.set_yticks(range(len(nets)))
    ax.set_xticklabels(nets, rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels(nets, fontsize=6)
    ax.set_title("B. Systems-level signature", loc="left", fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="signed edge count")


def plot_lateralization(ax: plt.Axes, lateralization: pd.DataFrame) -> None:
    """Stacked bar lateralization fractions by K."""
    categories = ["intra_left", "intra_right", "interhemispheric", "midline_or_unknown"]
    colors = ["#56B4E9", "#E69F00", "#009E73", "#999999"]
    pivot = lateralization.pivot_table(index="K", columns="lateralization", values="fraction_mean", fill_value=0)
    bottom = np.zeros(len(pivot))
    x = np.arange(len(pivot))
    for cat, color in zip(categories, colors):
        vals = pivot[cat].values if cat in pivot.columns else np.zeros(len(pivot))
        ax.bar(x, vals, bottom=bottom, color=color, label=cat.replace("_", " "), width=0.72)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in pivot.index])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Top-K")
    ax.set_ylabel("Mean fraction across folds")
    ax.set_title("C. Fold-level lateralization", loc="left", fontweight="bold")
    ax.legend(fontsize=6, frameon=False, ncol=1, loc="upper right")


def plot_saliency_vs_effect(ax: plt.Axes, saliency: pd.DataFrame) -> None:
    """Scatter abs saliency magnitude vs abs held-out Cohen's d."""
    x = saliency["saliency_abs"].astype(float).values
    y = saliency["abs_cohen_d_pooled"].astype(float).values
    colors = [PRO_AD_COLOR if d == "Pro-AD" else PRO_CN_COLOR for d in saliency["direction"]]
    ax.scatter(x, y, s=65, c=colors, edgecolor="black", linewidth=0.6)
    for _, row in saliency.iterrows():
        ax.text(float(row["saliency_abs"]), float(row["abs_cohen_d_pooled"]), str(row["edge_id"]), fontsize=7, ha="center", va="center")
    rho = saliency["spearman_rho"].dropna().iloc[0] if "spearman_rho" in saliency and saliency["spearman_rho"].notna().any() else np.nan
    pval = saliency["spearman_p"].dropna().iloc[0] if "spearman_p" in saliency and saliency["spearman_p"].notna().any() else np.nan
    ax.set_xlabel("|model saliency|")
    ax.set_ylabel("|held-out Cohen's d|")
    ax.set_title("D. Saliency vs empirical effect", loc="left", fontweight="bold")
    ax.text(0.02, 0.98, f"Spearman rho={rho:.3f}, p={pval:.3f}", transform=ax.transAxes, va="top", fontsize=8)


def plot_node_strength(ax: plt.Axes, node_strength: pd.DataFrame) -> None:
    """Horizontal bar plot of node-wise signed strength."""
    df = node_strength.sort_values("signed_strength")
    colors = [PRO_AD_COLOR if v > 0 else PRO_CN_COLOR for v in df["signed_strength"]]
    ax.barh(df["node"], df["signed_strength"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Signed consensus strength")
    ax.set_title("B. Node-wise signed strength", loc="left", fontweight="bold")
    ax.tick_params(axis="y", labelsize=7)

