#!/usr/bin/env python3
"""
Figure generation for interpretability stability audit.
Reads pre-computed CSV/JSON artifacts from oof_logitz_interpretability_stability_audit/
and writes publication-quality figures to the same directory.

No model loading. No retraining. No threshold fitting. Read-only.

Figures produced:
  fig_01_shap_barplot.png         SHAP latent importance bar chart
  fig_02_shap_beeswarm.png        Per-fold SHAP signed coefficient scatter
  fig_03_ig_roi_heatmap.png       IG ROI×ROI saliency heatmap (sparse top-20/fold)
  fig_04_yeo_network_heatmap.png  Yeo network-pair IG intensity heatmap
  fig_05_consensus_edges_table.png  Consensus + near-miss edge table
  fig_06_channel_contributions.png  Channel L1 contribution barplot
"""
from __future__ import annotations
import sys
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns

# ─── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = (ROOT / "results" / "revision_bspc_2026" /
             "oof_logitz_interpretability_stability_audit")
ROI_INFO_CSV = (ROOT / "results" / "revision_bspc_2026" /
                "recover035_latent384_beta3p75_T80_h10000_p560_full5x5" /
                "roi_info_from_tensor.csv")
OUT_DIR = AUDIT_DIR
DPI = 300

# ─── global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 100,
})

PRO_AD_COL = "#d62728"
PRO_CN_COL = "#1f77b4"
LOCKED_COL = "#4575b4"
PROMOTED_COL = "#d73027"

YEO_ORDER = [
    "Visual_Peripheral", "Visual_Central",
    "Somatomotor_A", "Somatomotor_B",
    "DorsalAttention_A", "DorsalAttention_B",
    "Salience_VentralAttention_A", "Salience_VentralAttention_B",
    "Limbic_A_TempPole", "Limbic_B_OFC",
    "Control_A", "Control_B",
    "DefaultMode_Core", "DefaultMode_DorsalMedial", "DefaultMode_VentralMedial",
    "Background/NonCortical",
]
YEO_COLORS = {
    "Visual_Peripheral":           "#7b2d8b",
    "Visual_Central":              "#9b59d0",
    "Somatomotor_A":               "#4393c3",
    "Somatomotor_B":               "#74add1",
    "DorsalAttention_A":           "#4dac26",
    "DorsalAttention_B":           "#7fbf7b",
    "Salience_VentralAttention_A": "#f0027f",
    "Salience_VentralAttention_B": "#f7a7c9",
    "Limbic_A_TempPole":           "#e6ab02",
    "Limbic_B_OFC":                "#fdd99a",
    "Control_A":                   "#f4a502",
    "Control_B":                   "#f7c96e",
    "DefaultMode_Core":            "#ca0020",
    "DefaultMode_DorsalMedial":    "#e84848",
    "DefaultMode_VentralMedial":   "#f4a582",
    "Background/NonCortical":      "#aaaaaa",
}


# ─── helpers ──────────────────────────────────────────────────────────────────
def load_roi_info():
    df = pd.read_csv(ROI_INFO_CSV)
    df["net_order"] = df["network_label_in_tensor"].map(
        lambda x: YEO_ORDER.index(x) if x in YEO_ORDER else len(YEO_ORDER)
    )
    df_sorted = df.sort_values(["net_order", "roi_name_in_tensor"]).reset_index(drop=True)
    roi_to_sorted_idx = {name: i for i, name in
                         enumerate(df_sorted["roi_name_in_tensor"].tolist())}
    return df_sorted, roi_to_sorted_idx


def compute_near_miss_edges(fold_df: pd.DataFrame, min_folds: int = 2) -> pd.DataFrame:
    edge_folds: dict = defaultdict(list)
    edge_signs: dict = defaultdict(list)
    edge_meta: dict = {}
    edge_vals: dict = defaultdict(list)
    for _, row in fold_df.iterrows():
        k = (row["src"], row["dst"])
        edge_folds[k].append(int(row["fold"]))
        edge_signs[k].append(int(row["sign"]))
        edge_vals[k].append(float(row["ig_diff_signed"]))
        if k not in edge_meta:
            edge_meta[k] = {
                "src_network": row["src_network"],
                "dst_network": row["dst_network"],
            }
    records = []
    for k, folds in edge_folds.items():
        freq = len(folds)
        if freq < min_folds:
            continue
        signs = edge_signs[k]
        sign_sum = sum(signs)
        direction = "pro-AD" if sign_sum > 0 else ("pro-CN" if sign_sum < 0 else "mixed")
        meta = edge_meta[k]
        records.append({
            "src": k[0], "dst": k[1],
            "src_network": meta["src_network"],
            "dst_network": meta["dst_network"],
            "freq": freq, "sign_sum": sign_sum,
            "direction": direction,
            "mean_ig": float(np.mean(edge_vals[k])),
            "folds": sorted(folds),
        })
    if not records:
        return pd.DataFrame(columns=["src", "dst", "src_network", "dst_network",
                                      "freq", "sign_sum", "direction", "mean_ig", "folds"])
    return pd.DataFrame(records).sort_values(["freq", "src"], ascending=[False, True])


def save_fig(fig: plt.Figure, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ─── Figure 1: SHAP barplot ───────────────────────────────────────────────────
def fig_shap_barplot() -> None:
    top_locked = pd.read_csv(AUDIT_DIR / "shap_top20_locked.csv")
    top_promoted = pd.read_csv(AUDIT_DIR / "shap_top20_promoted.csv")
    shap_locked = pd.read_csv(AUDIT_DIR / "shap_latent_importance_locked.csv")
    shap_promoted = pd.read_csv(AUDIT_DIR / "shap_latent_importance_promoted.csv")

    def _add_std(top_df, shap_df):
        lat = shap_df[shap_df["feature_type"] == "latent"]
        std_map = lat.groupby("feature")["abs_coef"].std(ddof=1).to_dict()
        meta = shap_df[shap_df["feature_type"] == "metadata"]
        for feat in ["Age", "Sex"]:
            sub = meta[meta["feature"] == feat]
            if len(sub):
                std_map[feat] = float(sub["abs_coef"].std(ddof=1))
        top_df = top_df.copy()
        top_df["std_coef"] = top_df["feature"].map(std_map).fillna(0.0)
        return top_df

    top_locked = _add_std(top_locked, shap_locked)
    top_promoted = _add_std(top_promoted, shap_promoted)

    # Also add metadata rows to top_df (Age, Sex from shap_locked/promoted)
    def _append_meta(top_df, shap_df):
        meta = shap_df[shap_df["feature_type"] == "metadata"]
        rows = []
        for feat in ["Age", "Sex"]:
            sub = meta[meta["feature"] == feat]
            if len(sub):
                rows.append({
                    "latent_idx": -1,
                    "feature": feat,
                    "abs_coef": float(sub["abs_coef"].mean()),
                    "mean_coef_signed": float(sub["coef_raw"].mean()),
                    "direction": "pro-AD" if sub["coef_raw"].mean() > 0 else "pro-CN",
                    "std_coef": float(sub["abs_coef"].std(ddof=1)),
                })
        if rows:
            top_df = pd.concat([top_df, pd.DataFrame(rows)], ignore_index=True)
            top_df = top_df.sort_values("abs_coef", ascending=False)
        return top_df

    top_locked = _append_meta(top_locked, shap_locked)
    top_promoted = _append_meta(top_promoted, shap_promoted)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=False)

    for ax, top_df, title, latent_dim in zip(
        axes,
        [top_locked, top_promoted],
        ["locked_v5p1b  (ld256, β=2.5)", "promoted_beta3p75  (ld384, β=3.75)"],
        [256, 384],
    ):
        top_df = top_df.copy().reset_index(drop=True)
        labels = []
        for _, row in top_df.iterrows():
            if row["feature"] in ("Age", "Sex"):
                labels.append(row["feature"])
            else:
                labels.append(f"L.{int(row['latent_idx'])}")

        colors = [PRO_AD_COL if d == "pro-AD" else PRO_CN_COL
                  for d in top_df["direction"]]
        y = np.arange(len(top_df))[::-1]

        ax.barh(y, top_df["abs_coef"], xerr=top_df["std_coef"],
                color=colors, edgecolor="none", height=0.7,
                error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": "#555"})
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.set_xlabel("Mean |coefficient| (across folds)")
        ax.set_title(title)
        ax.tick_params(axis="y", length=0)

    # Shared legend
    handles = [
        mpatches.Patch(color=PRO_AD_COL, label="pro-AD"),
        mpatches.Patch(color=PRO_CN_COL, label="pro-CN"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle("SHAP Latent Feature Importance (top-20 + metadata)", y=1.01)
    fig.tight_layout()
    save_fig(fig, "fig_01_shap_barplot.png")


# ─── Figure 2: SHAP beeswarm (fold-level scatter) ────────────────────────────
def fig_shap_beeswarm() -> None:
    shap_locked = pd.read_csv(AUDIT_DIR / "shap_latent_importance_locked.csv")
    shap_promoted = pd.read_csv(AUDIT_DIR / "shap_latent_importance_promoted.csv")
    top_locked = pd.read_csv(AUDIT_DIR / "shap_top20_locked.csv")
    top_promoted = pd.read_csv(AUDIT_DIR / "shap_top20_promoted.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=False)

    for ax, shap_df, top_df, title in zip(
        axes,
        [shap_locked, shap_promoted],
        [top_locked, top_promoted],
        ["locked_v5p1b  (ld256, β=2.5)", "promoted_beta3p75  (ld384, β=3.75)"],
    ):
        top_feats = top_df["feature"].tolist()
        direction_map = dict(zip(top_df["feature"], top_df["direction"]))

        lat_sub = shap_df[
            (shap_df["feature_type"] == "latent") &
            (shap_df["feature"].isin(top_feats))
        ]
        meta_sub = shap_df[shap_df["feature_type"] == "metadata"]

        # Rank positions: top feat = top of y-axis
        rank_map = {feat: i for i, feat in enumerate(top_feats)}

        rng = np.random.default_rng(42)
        for feat in top_feats:
            sub = lat_sub[lat_sub["feature"] == feat]
            y_pos = len(top_feats) - 1 - rank_map[feat]
            # signed coef, with jitter
            x_vals = sub["coef_raw"].values
            jitter = rng.uniform(-0.15, 0.15, len(x_vals))
            col = PRO_AD_COL if direction_map.get(feat) == "pro-AD" else PRO_CN_COL
            ax.scatter(x_vals, y_pos + jitter, color=col, s=18, alpha=0.85,
                       linewidths=0, zorder=3)
            ax.scatter([sub["coef_raw"].mean()], [y_pos], color=col, s=45,
                       marker="D", edgecolors="k", linewidths=0.5, zorder=4)

        # Metadata points
        for feat in ["Age", "Sex"]:
            sub_m = meta_sub[meta_sub["feature"] == feat]
            if len(sub_m):
                y_pos = -1 if feat == "Age" else -2
                x_vals = sub_m["coef_raw"].values
                col = PRO_AD_COL if sub_m["coef_raw"].mean() > 0 else PRO_CN_COL
                jitter = rng.uniform(-0.15, 0.15, len(x_vals))
                ax.scatter(x_vals, y_pos + jitter, color=col, s=18, alpha=0.85,
                           linewidths=0, zorder=3)
                ax.scatter([sub_m["coef_raw"].mean()], [y_pos], color=col, s=45,
                           marker="D", edgecolors="k", linewidths=0.5, zorder=4)

        total_labels = len(top_feats) + 2
        yticks = list(range(-2, len(top_feats)))[::-1]
        meta_labels = ["Sex", "Age"]
        feat_labels = [f"L.{top_df.loc[top_df['feature']==f, 'latent_idx'].values[0]:.0f}"
                       for f in top_feats]
        all_labels = feat_labels + meta_labels
        ax.set_yticks(list(range(-2, len(top_feats))))
        ax.set_yticklabels(all_labels[::-1], fontsize=7)
        ax.axvline(0, color="#333", lw=0.8, ls="--", alpha=0.6, zorder=1)
        ax.set_xlabel("Signed coefficient (5 folds shown as dots; ◆ = mean)")
        ax.set_title(title)
        ax.tick_params(axis="y", length=0)

        # Dashed separator above meta rows
        ax.axhline(-0.5, color="#999", lw=0.6, ls=":", zorder=1)

    handles = [
        mpatches.Patch(color=PRO_AD_COL, label="pro-AD"),
        mpatches.Patch(color=PRO_CN_COL, label="pro-CN"),
        plt.scatter([], [], marker="D", color="gray", s=40, label="fold mean"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle("SHAP Per-Fold Signed Coefficients (top-20 latent dims + Age/Sex)",
                 y=1.01)
    fig.tight_layout()
    save_fig(fig, "fig_02_shap_beeswarm.png")


# ─── Figure 3: IG ROI×ROI heatmap ────────────────────────────────────────────
def fig_ig_roi_heatmap() -> None:
    roi_df, roi_to_idx = load_roi_info()
    n_rois = len(roi_df)
    roi_names = roi_df["roi_name_in_tensor"].tolist()
    net_labels = roi_df["network_label_in_tensor"].tolist()

    def _build_matrix(fold_df: pd.DataFrame) -> np.ndarray:
        mat_sum = np.zeros((n_rois, n_rois), dtype=float)
        mat_cnt = np.zeros((n_rois, n_rois), dtype=float)
        for _, row in fold_df.iterrows():
            i = roi_to_idx.get(row["src"])
            j = roi_to_idx.get(row["dst"])
            if i is None or j is None:
                continue
            v = float(row["ig_diff_signed"])
            mat_sum[i, j] += v
            mat_sum[j, i] += v
            mat_cnt[i, j] += 1
            mat_cnt[j, i] += 1
        with np.errstate(invalid="ignore"):
            mat = np.where(mat_cnt > 0, mat_sum / mat_cnt, np.nan)
        return mat

    fold_locked = pd.read_csv(AUDIT_DIR / "ig_fold_edges_locked.csv")
    fold_promoted = pd.read_csv(AUDIT_DIR / "ig_fold_edges_promoted.csv")
    mat_locked = _build_matrix(fold_locked)
    mat_promoted = _build_matrix(fold_promoted)

    # Network block boundaries in sorted ROI order
    net_series = pd.Series(net_labels)
    boundaries = []
    current_net = net_series.iloc[0]
    for idx in range(1, len(net_series)):
        if net_series.iloc[idx] != current_net:
            boundaries.append(idx)
            current_net = net_series.iloc[idx]

    def _net_centers(net_ser):
        centers = {}
        current = net_ser.iloc[0]
        start = 0
        for idx in range(1, len(net_ser)):
            if net_ser.iloc[idx] != current:
                centers[current] = (start + idx - 1) / 2
                start = idx
                current = net_ser.iloc[idx]
        centers[current] = (start + len(net_ser) - 1) / 2
        return centers

    net_centers = _net_centers(net_series)

    vmax = max(
        np.nanmax(np.abs(mat_locked)) if not np.all(np.isnan(mat_locked)) else 0,
        np.nanmax(np.abs(mat_promoted)) if not np.all(np.isnan(mat_promoted)) else 0,
    )
    if vmax == 0:
        vmax = 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, mat, title in zip(
        axes,
        [mat_locked, mat_promoted],
        ["locked_v5p1b  (ld256, β=2.5)",
         "promoted_beta3p75  (ld384, β=3.75)"],
    ):
        mat_plot = mat.copy()
        # Use grey for NaN (unvisited edges)
        cmap = plt.cm.RdBu_r.copy()
        cmap.set_bad("#eeeeee")
        im = ax.imshow(mat_plot, cmap=cmap, norm=norm, aspect="equal",
                       interpolation="nearest")

        # Network block separators
        for b in boundaries:
            ax.axhline(b - 0.5, color="#555", lw=0.4, alpha=0.7)
            ax.axvline(b - 0.5, color="#555", lw=0.4, alpha=0.7)

        # Network labels on diagonal (abbreviated)
        ABBREV = {
            "Visual_Peripheral": "Vis.P", "Visual_Central": "Vis.C",
            "Somatomotor_A": "SM.A", "Somatomotor_B": "SM.B",
            "DorsalAttention_A": "DA.A", "DorsalAttention_B": "DA.B",
            "Salience_VentralAttention_A": "SVA.A",
            "Salience_VentralAttention_B": "SVA.B",
            "Limbic_A_TempPole": "Lim.A", "Limbic_B_OFC": "Lim.B",
            "Control_A": "Ctrl.A", "Control_B": "Ctrl.B",
            "DefaultMode_Core": "DMN.C",
            "DefaultMode_DorsalMedial": "DMN.D",
            "DefaultMode_VentralMedial": "DMN.V",
            "Background/NonCortical": "BG",
        }
        ax.set_xticks(list(net_centers.values()))
        ax.set_xticklabels(
            [ABBREV.get(n, n) for n in net_centers.keys()],
            rotation=45, ha="right", fontsize=6.5,
        )
        ax.set_yticks(list(net_centers.values()))
        ax.set_yticklabels(
            [ABBREV.get(n, n) for n in net_centers.keys()],
            fontsize=6.5,
        )
        ax.set_title(title)

        plt.colorbar(im, ax=ax, shrink=0.7, label="Mean IG diff (AD−CN)",
                     fraction=0.03, pad=0.02)

    fig.suptitle(
        "IG ROI×ROI Saliency Heatmap  (top-20 edges/fold, ≥ 1 fold appearance)\n"
        "Grey = not in any fold's top-20. Red = pro-AD, Blue = pro-CN.",
        y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, "fig_03_ig_roi_heatmap.png")


# ─── Figure 4: Yeo network-pair heatmap ─────────────────────────────────────
def fig_yeo_network_heatmap() -> None:
    fold_locked = pd.read_csv(AUDIT_DIR / "ig_fold_edges_locked.csv")
    fold_promoted = pd.read_csv(AUDIT_DIR / "ig_fold_edges_promoted.csv")

    # Only keep networks that appear in fold edges
    all_nets_l = sorted(
        set(fold_locked["src_network"]) | set(fold_locked["dst_network"])
    )
    all_nets_p = sorted(
        set(fold_promoted["src_network"]) | set(fold_promoted["dst_network"])
    )
    all_nets = sorted(set(all_nets_l) | set(all_nets_p),
                      key=lambda n: YEO_ORDER.index(n) if n in YEO_ORDER else 99)
    net_idx = {n: i for i, n in enumerate(all_nets)}
    n = len(all_nets)

    def _build_net_mat(fold_df: pd.DataFrame, net_idx: dict) -> np.ndarray:
        mat = np.zeros((len(net_idx), len(net_idx)), dtype=float)
        for _, row in fold_df.iterrows():
            i = net_idx.get(row["src_network"])
            j = net_idx.get(row["dst_network"])
            if i is None or j is None:
                continue
            v = abs(float(row["ig_diff_abs"]))
            mat[i, j] += v
            if i != j:
                mat[j, i] += v
        return mat

    mat_locked = _build_net_mat(fold_locked, net_idx)
    mat_promoted = _build_net_mat(fold_promoted, net_idx)

    ABBREV = {
        "Visual_Peripheral": "Vis.P", "Visual_Central": "Vis.C",
        "Somatomotor_A": "SM.A", "Somatomotor_B": "SM.B",
        "DorsalAttention_A": "DA.A", "DorsalAttention_B": "DA.B",
        "Salience_VentralAttention_A": "SVA.A",
        "Salience_VentralAttention_B": "SVA.B",
        "Limbic_A_TempPole": "Lim.A", "Limbic_B_OFC": "Lim.B",
        "Control_A": "Ctrl.A", "Control_B": "Ctrl.B",
        "DefaultMode_Core": "DMN.C",
        "DefaultMode_DorsalMedial": "DMN.D",
        "DefaultMode_VentralMedial": "DMN.V",
        "Background/NonCortical": "BG",
    }
    labels = [ABBREV.get(n, n) for n in all_nets]

    vmax = max(mat_locked.max(), mat_promoted.max())
    if vmax == 0:
        vmax = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, mat, title in zip(
        axes,
        [mat_locked, mat_promoted],
        ["locked_v5p1b  (ld256, β=2.5)",
         "promoted_beta3p75  (ld384, β=3.75)"],
    ):
        # Normalize per-model for visual comparison
        mat_norm = mat / (mat.max() if mat.max() > 0 else 1)
        im = ax.imshow(mat_norm, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal",
                       interpolation="nearest")

        ax.set_xticks(range(len(all_nets)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(all_nets)))
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.set_title(title)

        # Annotate cells above 20% of max
        threshold = 0.20
        for i in range(len(all_nets)):
            for j in range(len(all_nets)):
                if mat_norm[i, j] >= threshold:
                    ax.text(j, i, f"{mat_norm[i,j]:.2f}",
                            ha="center", va="center", fontsize=6,
                            color="white" if mat_norm[i, j] > 0.6 else "black")

        plt.colorbar(im, ax=ax, shrink=0.7,
                     label="ΣIG intensity (normalised per model)",
                     fraction=0.03, pad=0.02)

    fig.suptitle(
        "Yeo-17 Network-Pair IG Intensity  (Σ|IG diff| across all fold top-20 edges)",
        y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, "fig_04_yeo_network_heatmap.png")


# ─── Figure 5: Consensus + near-miss edge table ──────────────────────────────
def fig_consensus_edges_table() -> None:
    fold_locked = pd.read_csv(AUDIT_DIR / "ig_fold_edges_locked.csv")
    fold_promoted = pd.read_csv(AUDIT_DIR / "ig_fold_edges_promoted.csv")
    cons_locked = pd.read_csv(AUDIT_DIR / "ig_consensus_edges_locked.csv")
    cons_promoted = pd.read_csv(AUDIT_DIR / "ig_consensus_edges_promoted.csv")

    near_locked = compute_near_miss_edges(fold_locked, min_folds=2)
    near_promoted = compute_near_miss_edges(fold_promoted, min_folds=2)

    # Remove the consensus edge from locked near-miss (it's already in cons)
    if len(cons_locked) > 0:
        cons_keys_l = set(zip(cons_locked["src"], cons_locked["dst"]))
        near_locked = near_locked[
            ~near_locked.apply(lambda r: (r["src"], r["dst"]) in cons_keys_l, axis=1)
        ]

    def _shorten_net(net: str) -> str:
        abbrev = {
            "DefaultMode_VentralMedial": "DMN.VM",
            "DefaultMode_DorsalMedial": "DMN.DM",
            "DefaultMode_Core": "DMN.C",
            "Limbic_A_TempPole": "Lim.A",
            "Limbic_B_OFC": "Lim.B",
            "Somatomotor_A": "SM.A",
            "Somatomotor_B": "SM.B",
            "DorsalAttention_A": "DA.A",
            "DorsalAttention_B": "DA.B",
            "Salience_VentralAttention_A": "SVA.A",
            "Salience_VentralAttention_B": "SVA.B",
            "Visual_Peripheral": "Vis.P",
            "Visual_Central": "Vis.C",
            "Control_A": "Ctrl.A",
            "Control_B": "Ctrl.B",
            "Background/NonCortical": "BG",
        }
        return abbrev.get(net, net[:8])

    def _rows_from_df(df, model_label, is_consensus=False):
        rows = []
        for _, r in df.iterrows():
            net_a = _shorten_net(r["src_network"])
            net_b = _shorten_net(r["dst_network"])
            freq_str = f"{r['freq']}/5" if is_consensus else f"{r['freq']}/5"
            rows.append([
                model_label,
                f"{r['src']} — {r['dst']}",
                f"{net_a} × {net_b}",
                freq_str,
                r["direction"],
                "★" if is_consensus else "",
            ])
        return rows

    all_rows = []
    if len(cons_locked):
        all_rows += _rows_from_df(cons_locked, "locked", is_consensus=True)
    all_rows += _rows_from_df(near_locked, "locked", is_consensus=False)
    if len(cons_promoted):
        all_rows += _rows_from_df(cons_promoted, "promoted", is_consensus=True)
    all_rows += _rows_from_df(near_promoted, "promoted", is_consensus=False)

    if not all_rows:
        print("  No consensus or near-miss edges; skipping table figure.")
        return

    col_labels = ["Model", "Edge (src — dst)", "Network pair", "Freq", "Direction", "Cons."]
    col_widths = [0.09, 0.34, 0.27, 0.07, 0.10, 0.06]

    n_rows = len(all_rows)
    fig_h = max(3.5, 0.38 * n_rows + 1.0)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=all_rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    # Style header
    for j, _ in enumerate(col_labels):
        cell = table[0, j]
        cell.set_facecolor("#2c2c2c")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_linewidth(0)

    dir_colors = {
        "pro-AD": "#ffd5d5",
        "pro-CN": "#d5e8ff",
        "mixed": "#fffacc",
    }
    model_colors = {
        "locked": "#e8f0fb",
        "promoted": "#fde8e8",
    }
    for i, row in enumerate(all_rows, start=1):
        model = row[0]
        direction = row[4]
        for j in range(len(col_labels)):
            cell = table[i, j]
            cell.set_linewidth(0.2)
            if j == 4:
                cell.set_facecolor(dir_colors.get(direction, "white"))
            elif j == 5 and row[5] == "★":
                cell.set_facecolor("#fffacc")
                cell.set_text_props(fontweight="bold", color="#cc7700")
            else:
                cell.set_facecolor(model_colors.get(model, "white"))

    for j, w in enumerate(col_widths):
        table.auto_set_column_width(j)

    ax.set_title(
        "Consensus (★) and Near-Miss Edges (freq≥2/5) — locked_v5p1b and promoted_beta3p75\n"
        "Consensus: ≥3/5 folds. Near-miss: 2/5 folds. Direction: pro-AD = higher in AD vs CN.",
        fontsize=9, pad=12,
    )
    fig.tight_layout()
    save_fig(fig, "fig_05_consensus_edges_table.png")


# ─── Figure 6: Channel contribution barplot ──────────────────────────────────
def fig_channel_contributions() -> None:
    ch_df = pd.read_csv(AUDIT_DIR / "channel_contributions_summary.csv")
    ch_detail = pd.read_csv(AUDIT_DIR / "channel_contributions_ig.csv")

    channel_order = ["Pearson_Full", "Pearson_OMST", "MI"]
    channel_labels = {
        "Pearson_Full": "Pearson\n(Full)",
        "Pearson_OMST": "Pearson\n(OMST)",
        "MI": "MI",
    }
    channel_colors = {
        "Pearson_Full": "#4e79a7",
        "Pearson_OMST": "#f28e2b",
        "MI": "#59a14f",
    }
    model_order = ["locked_v5p1b", "promoted_beta3p75"]
    model_labels = {
        "locked_v5p1b": "locked\n(ld256, β=2.5)",
        "promoted_beta3p75": "promoted\n(ld384, β=3.75)",
    }
    hatches = ["", "//"]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    n_channels = len(channel_order)
    n_models = len(model_order)
    bar_w = 0.30
    group_gap = 0.15
    group_width = n_models * bar_w + group_gap

    xticks = []
    for ci, ch in enumerate(channel_order):
        x_center = ci * group_width
        xticks.append(x_center + (n_models - 1) * bar_w / 2)
        for mi, model in enumerate(model_order):
            sub = ch_df[(ch_df["candidate"] == model) & (ch_df["channel"] == ch)]
            if len(sub) == 0:
                continue
            mean_val = float(sub["mean_fraction"].values[0])
            std_val = float(sub["std_fraction"].values[0])
            x_pos = x_center + mi * bar_w

            # Get per-fold values for swarm overlay
            fld = ch_detail[(ch_detail["candidate"] == model) &
                             (ch_detail["channel"] == ch)]["fraction"].values

            ax.bar(x_pos, mean_val, width=bar_w * 0.85,
                   color=channel_colors[ch],
                   alpha=0.85 if mi == 0 else 0.55,
                   hatch=hatches[mi],
                   edgecolor="white", linewidth=0.5,
                   yerr=std_val, error_kw={"elinewidth": 1, "capsize": 3,
                                           "ecolor": "#333"})
            # Fold dots
            rng = np.random.default_rng(42 + ci * 10 + mi)
            jitter = rng.uniform(-0.04, 0.04, len(fld))
            ax.scatter(x_pos + jitter, fld, s=16, color="k",
                       alpha=0.6, zorder=5, linewidths=0)

    ax.set_xticks(xticks)
    ax.set_xticklabels([channel_labels[c] for c in channel_order])
    ax.set_ylabel("Fraction of total L1 IG signal")
    ax.set_ylim(0, 0.65)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_title("Channel Contributions to IG Saliency (locked vs promoted)",
                 fontsize=10)

    # Legend for model
    handles_model = [
        mpatches.Patch(color="gray", alpha=0.85, hatch="", label="locked_v5p1b"),
        mpatches.Patch(color="gray", alpha=0.55, hatch="//", label="promoted_beta3p75"),
        plt.scatter([], [], s=16, color="k", alpha=0.6, label="per-fold value"),
    ]
    ax.legend(handles=handles_model, loc="upper right", fontsize=8, frameon=False)

    fig.tight_layout()
    save_fig(fig, "fig_06_channel_contributions.png")


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"Output dir: {OUT_DIR}")

    print("\n[1] SHAP barplot...")
    fig_shap_barplot()

    print("\n[2] SHAP beeswarm...")
    fig_shap_beeswarm()

    print("\n[3] IG ROI×ROI heatmap...")
    fig_ig_roi_heatmap()

    print("\n[4] Yeo network-pair heatmap...")
    fig_yeo_network_heatmap()

    print("\n[5] Consensus edge table...")
    fig_consensus_edges_table()

    print("\n[6] Channel contributions...")
    fig_channel_contributions()

    print("\nDone.")


if __name__ == "__main__":
    main()
