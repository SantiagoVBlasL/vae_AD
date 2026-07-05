#!/usr/bin/env python3
"""Render final BSPC interpretability figures from derived final tables."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from betavae_xai.interpretability.paper_figures import (  # noqa: E402
    PRO_AD_COLOR,
    PRO_CN_COLOR,
    infer_node_positions,
    plot_consensus_schematic,
    plot_lateralization,
    plot_network_heatmap,
    plot_node_strength,
    plot_saliency_vs_effect,
    write_csv_md,
)

OUT = PROJECT_ROOT / "results/revision_bspc_2026/final_interpretability_figures_20260624"
EXPECTED_N_EDGES = 8


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def read_required_csv(name: str) -> pd.DataFrame:
    path = OUT / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required table: {path}. Run build_final_interpretability_tables.py first.")
    return pd.read_csv(path)


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    paths = []
    for ext in ["pdf", "png", "svg"]:
        path = OUT / f"{stem}.{ext}"
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 300
        fig.savefig(path, **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def build_node_strength(edges: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in edges.iterrows():
        signed = float(row["mean_signed_direction"])
        for col in ["roi_i", "roi_j"]:
            rows.append(
                {
                    "node": row[col],
                    "network": row["network_i"] if col == "roi_i" else row["network_j"],
                    "signed_strength": signed,
                    "abs_strength": abs(signed),
                }
            )
    df = pd.DataFrame(rows)
    out = (
        df.groupby(["node", "network"], as_index=False)
        .agg(
            signed_strength=("signed_strength", "sum"),
            abs_strength=("abs_strength", "sum"),
            degree=("node", "count"),
        )
        .sort_values("abs_strength", ascending=False)
    )
    write_csv_md(out, OUT / "node_signed_strength_final.csv", OUT / "node_signed_strength_final.md")
    return out


def render_figure3() -> None:
    edges = read_required_csv("table_s5_consensus_edges_final.csv")
    lateralization = read_required_csv("lateralization_final.csv")
    saliency_vs = read_required_csv("saliency_vs_cohen_d_consensus8_final.csv")
    network_sig = read_required_csv("network_pair_signature_final.csv")
    if len(edges) != EXPECTED_N_EDGES:
        raise RuntimeError(f"Figure 3 requires 8 consensus edges, observed {len(edges)}")

    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    plot_consensus_schematic(ax_a, edges)
    plot_network_heatmap(ax_b, network_sig)
    plot_lateralization(ax_c, lateralization)
    plot_saliency_vs_effect(ax_d, saliency_vs)
    fig.suptitle("Final interpretability signature: strict Top-K=200 consensus", fontsize=13, fontweight="bold")
    save_figure(fig, "Figure3_main_signature_final")


def render_figure4() -> None:
    edges = read_required_csv("table_s5_consensus_edges_final.csv")
    node_strength = build_node_strength(edges)
    if len(edges) != EXPECTED_N_EDGES:
        raise RuntimeError(f"Figure 4 requires 8 consensus edges, observed {len(edges)}")

    fig = plt.figure(figsize=(11, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    pos = infer_node_positions(edges)
    for _, row in edges.iterrows():
        a = str(row["roi_i"])
        b = str(row["roi_j"])
        x1, y1 = pos[a]
        x2, y2 = pos[b]
        color = PRO_AD_COLOR if row["direction"] == "Pro-AD" else PRO_CN_COLOR
        width = 1.0 + 2.6 * float(row["replication_frequency"])
        ax_a.plot([x1, x2], [y1, y2], color=color, linewidth=width, alpha=0.75)
    for node, (x, y) in pos.items():
        ax_a.scatter([x], [y], s=72, color="white", edgecolor="black", linewidth=0.9, zorder=3)
        ax_a.text(x, y + 0.03, node.replace("_", " "), ha="center", va="bottom", fontsize=7)
    ax_a.set_xlim(0.02, 0.98)
    ax_a.set_ylim(-0.08, 1.08)
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    ax_a.set_frame_on(False)
    ax_a.set_title("A. Consensus connectome projection", loc="left", fontweight="bold")
    ax_a.text(
        0.02,
        -0.07,
        "Schematic projection; no verified MNI coordinates were present in the final artifact set.",
        fontsize=7,
        ha="left",
        va="top",
    )

    plot_node_strength(ax_b, node_strength)
    fig.suptitle("Final consensus nodes and signed strengths", fontsize=13, fontweight="bold")
    save_figure(fig, "Figure4_glass_brain_final")


def refresh_outputs_index() -> pd.DataFrame:
    rows = []
    for path in sorted(OUT.iterdir()):
        if path.is_file():
            rows.append({"path": rel(path), "bytes": path.stat().st_size})
    df = pd.DataFrame(rows)
    write_csv_md(df, OUT / "outputs_index.csv", OUT / "outputs_index.md")
    return df


def write_command_log() -> None:
    log = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": rel(Path(__file__)),
        "table_script": "scripts/revision_bspc_2026/build_final_interpretability_tables.py",
        "helper_module": "src/betavae_xai/interpretability/paper_figures.py",
        "output_dir": rel(OUT),
        "guardrails": {
            "did_train_vae": False,
            "did_retrain_classifier": False,
            "did_run_shap_ig": False,
            "did_modify_tensors": False,
            "did_modify_metadata": False,
            "did_modify_manuscript": False,
        },
    }
    (OUT / "command_log_figures.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    (OUT / "command_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    render_figure3()
    render_figure4()
    write_command_log()
    idx = refresh_outputs_index()
    print(f"Figures written to: {OUT}")
    print(idx.to_string(index=False))


if __name__ == "__main__":
    main()
