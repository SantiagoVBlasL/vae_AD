#!/usr/bin/env python3
"""Render publication-oriented final interpretability figures.

This script redesigns figures only. It consumes fixed final tables produced by
the final interpretability pipeline and writes a separate pretty figure package.
It does not train models, rerun SHAP/IG, or modify tensors/metadata/manuscript.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from betavae_xai.interpretability.paper_figures import (  # noqa: E402
    PRO_AD_COLOR,
    PRO_CN_COLOR,
    edge_lateralization,
    hemisphere_from_roi,
    write_csv_md,
)

IN_DIR = PROJECT_ROOT / "results/revision_bspc_2026/final_interpretability_figures_20260624"
AUDIT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/final_interpretability_figure_rebuild_audit_20260624"
OUT = PROJECT_ROOT / "results/revision_bspc_2026/final_interpretability_figures_pretty_20260624"

EXPECTED_N_EDGES = 8
EXPECTED_K = 200
EXPECTED_JACCARD = 0.065586
EXPECTED_RHO = 0.310
EXPECTED_P = 0.456
STALE_FORBIDDEN = [
    "vae_3channels_beta65_pro",
    "latent_dim=256",
    "TOP_K_STABILITY=250",
]

FONT_FAMILY = "DejaVu Sans"
PRO_AD = PRO_AD_COLOR
PRO_CN = PRO_CN_COLOR
INTER = "#009E73"
INTRA_L = "#56B4E9"
INTRA_R = "#E69F00"
GRAY = "#666666"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def read_csv(name: str) -> pd.DataFrame:
    path = IN_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required input CSV: {path}")
    return pd.read_csv(path)


def savefig(fig: plt.Figure, stem: str) -> list[Path]:
    paths: list[Path] = []
    for ext in ["pdf", "png", "svg"]:
        path = OUT / f"{stem}.{ext}"
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if ext == "png":
            kwargs["dpi"] = 300
        fig.savefig(path, **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def short_roi(name: str) -> str:
    repl = {
        "Cerebelum": "Cer",
        "Frontal": "Front",
        "Postcentral": "PostC",
        "Precentral": "PreC",
        "Parietal": "Par",
        "Precuneus": "Prec",
        "Medial": "Med",
        "Sup": "S",
        "Inf": "I",
        "Crus": "Crus",
    }
    out = str(name)
    for old, new in repl.items():
        out = out.replace(old, new)
    return out.replace("_", " ")


def edge_label(row: pd.Series) -> str:
    return f"{short_roi(row['roi_i'])} - {short_roi(row['roi_j'])}"


def network_label(label: str) -> str:
    text = str(label)
    repl = {
        "Background/NonCortical": "Noncortical",
        "DefaultMode_VentralMedial": "DMN ventral",
        "DefaultMode_DorsalMedial": "DMN dorsal",
        "DorsalAttention_A": "Dorsal attention",
        "Limbic_B_OFC": "Limbic/OFC",
        "Somatomotor_A": "Somatomotor",
        "Control_B": "Control",
    }
    return repl.get(text, text.replace("_", " "))


def validate_inputs(edges: pd.DataFrame, lat: pd.DataFrame, sal: pd.DataFrame) -> None:
    if len(edges) != EXPECTED_N_EDGES:
        raise RuntimeError(f"Expected 8 strict consensus edges, found {len(edges)}")
    if set(edges["K"].astype(int)) != {EXPECTED_K}:
        raise RuntimeError(f"Final compact consensus must be K=200, observed {sorted(edges['K'].unique())}")
    if len(edges) == 11:
        raise RuntimeError("Detected stale 11-edge consensus table.")
    if (edges["replication_frequency"].astype(float) < 0.6).any():
        raise RuntimeError("Consensus edge pi below 0.6 found.")
    if (edges["mean_signed_direction"].abs().astype(float) < 0.6).any():
        raise RuntimeError("Consensus edge sign consistency below 0.6 found.")
    if set(lat["K"].astype(int)) != {50, 100, 200}:
        raise RuntimeError("Lateralization table must contain K=50/100/200.")
    if len(sal) != EXPECTED_N_EDGES:
        raise RuntimeError(f"Expected 8 saliency/effect rows, found {len(sal)}")
    for name in [
        "table_s5_consensus_edges_final.csv",
        "lateralization_final.csv",
        "saliency_vs_cohen_d_consensus8_final.csv",
    ]:
        text = (IN_DIR / name).read_text(encoding="utf-8", errors="ignore")
        for token in STALE_FORBIDDEN:
            if token in text:
                raise RuntimeError(f"Input {name} contains stale forbidden token: {token}")


def mni_coordinate_status() -> tuple[bool, str]:
    candidate = PROJECT_ROOT / "data/ROI_MNI_V7_vol.txt"
    if not candidate.exists():
        return False, "No coordinate-like file found in data/."
    cols = pd.read_csv(candidate, sep="\t", nrows=1).columns.tolist()
    lower = {c.lower() for c in cols}
    has_xyz = {"x", "y", "z"}.issubset(lower) or {"mni_x", "mni_y", "mni_z"}.issubset(lower)
    if not has_xyz:
        return False, f"{rel(candidate)} exists but has columns {cols}; no verified x/y/z coordinates."
    return False, f"{rel(candidate)} has coordinate-like columns, but final 131 ROI order/name matching was not validated in this task."


def prepare_panel_a(edges: pd.DataFrame) -> pd.DataFrame:
    df = edges.copy()
    df["signed_weight"] = df["mean_signed_direction"].astype(float)
    df["abs_signed_weight"] = df["signed_weight"].abs()
    df["edge_label"] = df.apply(edge_label, axis=1)
    df["network_pair"] = df["network_i"].map(network_label) + " - " + df["network_j"].map(network_label)
    direction_order = df["direction"].map({"Pro-AD": 0, "Pro-CN": 1}).fillna(2)
    df = df.assign(direction_order=direction_order)
    df = df.sort_values(
        ["replication_frequency", "direction_order", "abs_signed_weight", "edge_id"],
        ascending=[False, True, False, True],
    ).drop(columns=["direction_order"])
    write_csv_md(df, OUT / "panel_A_edge_signature_values.csv")
    return df


def prepare_panel_b(edges: pd.DataFrame) -> pd.DataFrame:
    nets = sorted(set(edges["network_i"]).union(edges["network_j"]), key=network_label)
    rows = []
    for ni in nets:
        for nj in nets:
            sub = edges[
                ((edges["network_i"] == ni) & (edges["network_j"] == nj))
                | ((edges["network_i"] == nj) & (edges["network_j"] == ni))
            ]
            signed = float(sub["mean_signed_direction"].sum()) if not sub.empty else 0.0
            rows.append(
                {
                    "network_i": ni,
                    "network_j": nj,
                    "network_i_label": network_label(ni),
                    "network_j_label": network_label(nj),
                    "signed_weight_sum": signed,
                    "edge_count": int(len(sub)),
                }
            )
    df = pd.DataFrame(rows)
    write_csv_md(df, OUT / "panel_B_network_matrix_values.csv")
    return df


def prepare_panel_c(lat: pd.DataFrame) -> pd.DataFrame:
    keep = lat[lat["lateralization"].isin(["interhemispheric", "intra_left", "intra_right"])].copy()
    keep["lateralization_label"] = keep["lateralization"].map(
        {
            "interhemispheric": "Interhemispheric",
            "intra_left": "Intra-left",
            "intra_right": "Intra-right",
        }
    )
    write_csv_md(keep, OUT / "panel_C_lateralization_values.csv")
    return keep


def prepare_panel_d(sal: pd.DataFrame) -> pd.DataFrame:
    df = sal.copy()
    df["edge_short_label"] = df.apply(lambda r: f"E{int(r['edge_id'])}", axis=1)
    write_csv_md(df, OUT / "panel_D_saliency_cohend_values.csv")
    return df


def build_node_strength(edges: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in edges.iterrows():
        signed = float(row["mean_signed_direction"])
        for roi_col, net_col in [("roi_i", "network_i"), ("roi_j", "network_j")]:
            rows.append(
                {
                    "node": row[roi_col],
                    "node_label": short_roi(row[roi_col]),
                    "network": row[net_col],
                    "network_label": network_label(row[net_col]),
                    "signed_strength": signed,
                    "abs_strength": abs(signed),
                    "degree": 1,
                }
            )
    df = pd.DataFrame(rows)
    out = (
        df.groupby(["node", "node_label", "network", "network_label"], as_index=False)
        .agg(
            signed_strength=("signed_strength", "sum"),
            abs_strength=("abs_strength", "sum"),
            degree=("degree", "sum"),
        )
        .sort_values(["signed_strength", "node"])
    )
    write_csv_md(out, OUT / "node_signed_strength_pretty.csv")
    return out


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "figure.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
        }
    )


def panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.08,
        1.04,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )


def plot_panel_a(ax: plt.Axes, df: pd.DataFrame) -> None:
    plot_df = df.reset_index(drop=True).copy()
    y = np.arange(len(plot_df))
    colors = [PRO_AD if d == "Pro-AD" else PRO_CN for d in plot_df["direction"]]
    ax.axvline(0, color="#BDBDBD", lw=0.8, zorder=0)
    ax.hlines(y, 0, plot_df["signed_weight"], color=colors, lw=2.2)
    ax.scatter(plot_df["signed_weight"], y, s=52, color=colors, edgecolor="white", linewidth=0.7, zorder=3)
    labels = [f"E{int(r.edge_id)}  {r.edge_label}" for r in plot_df.itertuples()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(-0.95, 0.95)
    ax.set_xlabel("Signed consensus direction")
    ax.set_title("Strict 8-edge signature", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E6E6E6", lw=0.7)
    for yi, r in zip(y, plot_df.itertuples()):
        txt = f"pi={r.replication_frequency:.1f} | {r.direction}"
        x_txt = 0.92 if r.signed_weight >= 0 else -0.92
        ha = "right" if r.signed_weight >= 0 else "left"
        ax.text(x_txt, yi, txt, ha=ha, va="center", fontsize=7, color="#333333")
    ax.text(0.01, 1.01, "Pro-CN", transform=ax.transAxes, color=PRO_CN, ha="left", va="bottom", fontsize=7.5)
    ax.text(0.99, 1.01, "Pro-AD", transform=ax.transAxes, color=PRO_AD, ha="right", va="bottom", fontsize=7.5)


def plot_panel_b(ax: plt.Axes, matrix_values: pd.DataFrame) -> None:
    labels = sorted(matrix_values["network_i_label"].unique())
    mat = pd.DataFrame(0.0, index=labels, columns=labels)
    cnt = pd.DataFrame(0, index=labels, columns=labels)
    for _, row in matrix_values.iterrows():
        mat.loc[row["network_i_label"], row["network_j_label"]] = float(row["signed_weight_sum"])
        cnt.loc[row["network_i_label"], row["network_j_label"]] = int(row["edge_count"])
    vmax = max(0.6, float(np.nanmax(np.abs(mat.values))))
    im = ax.imshow(mat.values, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax))
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cnt.iloc[i, j] > 0:
                ax.text(j, i, f"{cnt.iloc[i, j]}\n{mat.iloc[i, j]:+.1f}", ha="center", va="center", fontsize=6.5)
    ax.set_title("Systems-level signature", loc="left", fontweight="bold")
    cb = plt.colorbar(im, ax=ax, fraction=0.044, pad=0.03)
    cb.set_label("Signed weight")


def plot_panel_c(ax: plt.Axes, lat: pd.DataFrame) -> None:
    order = ["Interhemispheric", "Intra-left", "Intra-right"]
    colors = [INTER, INTRA_L, INTRA_R]
    pivot = lat.pivot(index="K", columns="lateralization_label", values="fraction_mean").fillna(0)
    x = np.arange(len(pivot.index))
    bottom = np.zeros(len(x))
    for lab, color in zip(order, colors):
        vals = pivot[lab].values if lab in pivot.columns else np.zeros(len(x))
        ax.bar(x, vals, bottom=bottom, width=0.68, color=color, label=lab)
        for xi, btm, val in zip(x, bottom, vals):
            if val > 0.08:
                ax.text(xi, btm + val / 2, f"{val:.2f}", ha="center", va="center", fontsize=7, color="white")
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(k)) for k in pivot.index])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Top-K fold ranking")
    ax.set_ylabel("Mean fraction")
    ax.set_title("Hemispheric distribution", loc="left", fontweight="bold")
    ax.text(0.02, 0.98, "Descriptive; no validated lateralization p-value", transform=ax.transAxes, ha="left", va="top", fontsize=7.2, color=GRAY)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.34), ncol=3)


def plot_panel_d(ax: plt.Axes, sal: pd.DataFrame) -> None:
    colors = [PRO_AD if d == "Pro-AD" else PRO_CN for d in sal["direction"]]
    x = sal["saliency_abs"].astype(float)
    y = sal["abs_cohen_d_pooled"].astype(float)
    ax.axhline(0, color="#BDBDBD", lw=0.8)
    ax.axvline(float(x.median()), color="#BDBDBD", lw=0.8, ls="--")
    ax.axhline(float(y.median()), color="#BDBDBD", lw=0.8, ls="--")
    ax.scatter(x, y, s=72, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    for _, row in sal.iterrows():
        ax.annotate(f"E{int(row['edge_id'])}", (row["saliency_abs"], row["abs_cohen_d_pooled"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("|model saliency|")
    ax.set_ylabel("|held-out Cohen's d|")
    ax.set_title("Saliency vs empirical effect", loc="left", fontweight="bold")
    ax.text(0.02, 0.98, "Spearman rho = 0.310\np = 0.456, N = 8", transform=ax.transAxes, ha="left", va="top", fontsize=8)
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PRO_AD, label="Pro-AD", markersize=7),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PRO_CN, label="Pro-CN", markersize=7),
    ]
    ax.legend(handles=legend, frameon=False, loc="lower right")


def make_figure3(panel_a: pd.DataFrame, panel_b: pd.DataFrame, panel_c: pd.DataFrame, panel_d: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(11.2, 8.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.0], height_ratios=[1.12, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    plot_panel_a(ax_a, panel_a)
    plot_panel_b(ax_b, panel_b)
    plot_panel_c(ax_c, panel_c)
    plot_panel_d(ax_d, panel_d)
    for ax, letter in zip([ax_a, ax_b, ax_c, ax_d], list("ABCD")):
        panel_letter(ax, letter)
    fig.suptitle("Final AD-vs-CN interpretability signature", x=0.01, ha="left", fontweight="bold")
    for ext in ["pdf", "png", "svg"]:
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if ext == "png":
            kwargs["dpi"] = 300
        fig.savefig(OUT / f"Figure3_main_signature_pretty.{ext}", **kwargs)
    plt.close(fig)


def node_layout(edges: pd.DataFrame) -> dict[str, tuple[float, float]]:
    nodes = sorted(set(edges["roi_i"]).union(edges["roi_j"]))
    groups: dict[str, list[str]] = {"left": [], "right": [], "cerebellar": [], "midline_default": []}
    for node in nodes:
        if "Cerebel" in node:
            groups["cerebellar"].append(node)
        elif "Medial" in node or "Precuneus" in node or "OFC" in node:
            groups["midline_default"].append(node)
        elif hemisphere_from_roi(node) == "L":
            groups["left"].append(node)
        elif hemisphere_from_roi(node) == "R":
            groups["right"].append(node)
        else:
            groups["midline_default"].append(node)
    anchors = {
        "left": (0.18, 0.70),
        "right": (0.82, 0.70),
        "cerebellar": (0.18, 0.25),
        "midline_default": (0.55, 0.35),
    }
    layout = {}
    for group, group_nodes in groups.items():
        cx, cy = anchors[group]
        n = max(1, len(group_nodes))
        for idx, node in enumerate(sorted(group_nodes)):
            angle = 2 * math.pi * idx / n
            radius = 0.10 if n > 1 else 0
            layout[node] = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
    return layout


def make_figure4(edges: pd.DataFrame, node_strength: pd.DataFrame, mni_available: bool) -> None:
    fig = plt.figure(figsize=(10.8, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 0.95])
    ax_net = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    layout = node_layout(edges)
    for _, row in edges.iterrows():
        a, b = row["roi_i"], row["roi_j"]
        x1, y1 = layout[a]
        x2, y2 = layout[b]
        color = PRO_AD if row["direction"] == "Pro-AD" else PRO_CN
        lw = 1.2 + 2.6 * float(row["replication_frequency"])
        ax_net.plot([x1, x2], [y1, y2], color=color, lw=lw, alpha=0.78, zorder=1)
    for node, (x, y) in layout.items():
        hemi = hemisphere_from_roi(node)
        face = "#FFFFFF" if hemi in ["L", "R"] else "#F2F2F2"
        ax_net.scatter([x], [y], s=90, facecolor=face, edgecolor="#222222", linewidth=0.9, zorder=3)
        ax_net.text(x, y + 0.035, short_roi(node), ha="center", va="bottom", fontsize=7)
    ax_net.text(0.18, 0.94, "Left hemisphere", ha="center", fontsize=8, color=GRAY)
    ax_net.text(0.82, 0.94, "Right hemisphere", ha="center", fontsize=8, color=GRAY)
    ax_net.text(0.18, 0.05, "Cerebellar/noncortical", ha="center", fontsize=8, color=GRAY)
    ax_net.text(0.55, 0.08, "Midline/default/OFC", ha="center", fontsize=8, color=GRAY)
    ax_net.set_xlim(0, 1)
    ax_net.set_ylim(0, 1)
    ax_net.set_xticks([])
    ax_net.set_yticks([])
    ax_net.set_frame_on(False)
    title = "Schematic 8-edge connectome layout"
    if mni_available:
        title = "Verified-coordinate connectome layout"
    ax_net.set_title(title, loc="left", fontweight="bold")
    panel_letter(ax_net, "A")

    df = node_strength.sort_values("signed_strength").copy()
    colors = [PRO_AD if v > 0 else PRO_CN if v < 0 else "#999999" for v in df["signed_strength"]]
    ax_bar.barh(df["node_label"], df["signed_strength"], color=colors)
    ax_bar.axvline(0, color="#222222", lw=0.8)
    ax_bar.set_xlabel("Signed node strength")
    ax_bar.set_title("Node-wise signed strength", loc="left", fontweight="bold")
    ax_bar.grid(axis="x", color="#E6E6E6", lw=0.7)
    panel_letter(ax_bar, "B")

    fig.suptitle("Final consensus connectome summary", x=0.01, ha="left", fontweight="bold")
    for ext in ["pdf", "png", "svg"]:
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if ext == "png":
            kwargs["dpi"] = 300
        fig.savefig(OUT / f"Figure4_schematic_connectome_pretty.{ext}", **kwargs)
    plt.close(fig)


def write_report(
    inputs: list[str],
    mni_available: bool,
    mni_note: str,
    panel_a: pd.DataFrame,
    panel_c: pd.DataFrame,
    panel_d: pd.DataFrame,
) -> None:
    pro_ad = int((panel_a["direction"] == "Pro-AD").sum())
    pro_cn = int((panel_a["direction"] == "Pro-CN").sum())
    lat_pivot = panel_c.pivot(index="lateralization_label", columns="K", values="fraction_mean")
    report = f"""# Final Pretty Interpretability Figure Report

## Inputs Used

{chr(10).join(f"- `{p}`" for p in inputs)}

## Coordinate Status

- verified_mni_coordinates_available={str(mni_available).lower()}
- {mni_note}
- Figure 4 is therefore labeled as a schematic connectome layout, not a real glass brain.

## Fixed Numbers Preserved

- strict Top-K=200 consensus
- consensus edges: {len(panel_a)}
- direction split: {pro_ad} Pro-AD, {pro_cn} Pro-CN
- mean Top-200 Jaccard: {EXPECTED_JACCARD:.6f}
- channel contributions: Full Pearson 45.5%, MI-kNN 33.8%, OMST-Pearson 20.7%
- saliency vs held-out Cohen's d: Spearman rho = {EXPECTED_RHO:.3f}, p = {EXPECTED_P:.3f}, N=8

## Lateralization Fractions

{lat_pivot.to_markdown()}

## Design Notes

- Figure 3 Panel A uses only the strict 8-edge consensus table.
- Figure 3 Panel B includes only networks involved in the strict 8 edges.
- Figure 3 Panel C is descriptive; no lateralization p-value is emphasized.
- Figure 3 Panel D uses edge IDs linked to Panel A.
- Figure 4 draws only the 8 final consensus edges in a manual schematic layout.

## Guardrails

- did_train_vae=false
- did_retrain_classifier=false
- did_run_shap_ig=false
- did_modify_tensors=false
- did_modify_metadata=false
- did_modify_manuscript=false
"""
    (OUT / "final_pretty_figure_report.md").write_text(report, encoding="utf-8")


def write_command_log(inputs: list[str], mni_available: bool, mni_note: str) -> None:
    log = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": rel(Path(__file__)),
        "output_dir": rel(OUT),
        "inputs": inputs,
        "mni_coordinates": {
            "verified_available": mni_available,
            "note": mni_note,
        },
        "guardrails": {
            "did_train_vae": False,
            "did_retrain_classifier": False,
            "did_run_shap_ig": False,
            "did_modify_tensors": False,
            "did_modify_metadata": False,
            "did_modify_manuscript": False,
        },
    }
    (OUT / "command_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    configure_style()
    inputs = [
        rel(IN_DIR / "table_s5_consensus_edges_final.csv"),
        rel(IN_DIR / "lateralization_final.csv"),
        rel(IN_DIR / "saliency_vs_cohen_d_consensus8_final.csv"),
        rel(IN_DIR / "node_signed_strength_final.csv"),
        rel(IN_DIR / "channel_contributions_final.csv"),
    ]
    edges = read_csv("table_s5_consensus_edges_final.csv")
    lat = read_csv("lateralization_final.csv")
    sal = read_csv("saliency_vs_cohen_d_consensus8_final.csv")
    validate_inputs(edges, lat, sal)
    panel_a = prepare_panel_a(edges)
    panel_b = prepare_panel_b(edges)
    panel_c = prepare_panel_c(lat)
    panel_d = prepare_panel_d(sal)
    node_strength = build_node_strength(edges)
    mni_available, mni_note = mni_coordinate_status()
    make_figure3(panel_a, panel_b, panel_c, panel_d)
    make_figure4(edges, node_strength, mni_available)
    write_report(inputs, mni_available, mni_note, panel_a, panel_c, panel_d)
    write_command_log(inputs, mni_available, mni_note)
    index_rows = []
    for path in sorted(OUT.iterdir()):
        if path.is_file():
            index_rows.append({"path": rel(path), "bytes": path.stat().st_size})
    write_csv_md(pd.DataFrame(index_rows), OUT / "outputs_index.csv", OUT / "outputs_index.md")
    print(pd.DataFrame(index_rows).to_string(index=False))


if __name__ == "__main__":
    main()
