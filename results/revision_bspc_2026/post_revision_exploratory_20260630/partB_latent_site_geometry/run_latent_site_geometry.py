#!/usr/bin/env python3
"""
Part B — Latent site/manufacturer geometry analysis.

Read-only analysis of site and manufacturer structure in the promoted model's
OOF latent space (recover035_latent384_beta3p75_T80_h10000_p560_full5x5).

Guardrails:
- No VAE retraining, classifier retraining, IG/SHAP recomputation.
- No tensor, metadata, or manuscript edits.
- Fails closed if any required file is missing.
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[4]   # vae_AD project root
BIG_DISK = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026")
RUN_NAME = "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
RUN_DIR  = BIG_DISK / RUN_NAME
LATENT_CACHE = RUN_DIR / "classifier_only_readout" / "latent_cache"
OOF_PRED_CSV = RUN_DIR / (
    "all_folds_clf_predictions_MULTI_logreg_vaeconvtranspose4l_ld384_beta3.75_"
    "normzscore_offdiag_ch3sel_intFCquarter_drop0.15_ln0_outer5x1_scoreroc_auc.csv"
)
META_CSV = (
    ROOT / "results/revision_bspc_2026/adni_035_metadata_rescue_preflight"
    / "patched_metadata_candidate.csv"
)
OUT_DIR = Path(__file__).parent

COMMAND_LOG: list[dict] = []
T0 = datetime.now(timezone.utc).isoformat()

# ── guards ────────────────────────────────────────────────────────────────────
for p, label in [
    (LATENT_CACHE, "latent cache dir"),
    (OOF_PRED_CSV, "OOF predictions CSV"),
    (META_CSV, "metadata CSV"),
]:
    if not Path(p).exists():
        sys.exit(f"FAIL: required file/dir missing: {label}\n  {p}")

print("All required paths found. Loading data …")

# ── load latents (OOF test splits only) ───────────────────────────────────────
dfs_test = []
for fold in range(1, 6):
    f = LATENT_CACHE / f"fold_{fold}_test_latent_mu.csv"
    if not f.exists():
        sys.exit(f"FAIL: missing {f}")
    dfs_test.append(pd.read_csv(f))
latents = pd.concat(dfs_test, ignore_index=True)
print(f"Loaded OOF latents: {latents.shape[0]} subjects × {latents.shape[1]} cols")

mu_cols = [c for c in latents.columns if c.startswith("mu_")]
assert len(mu_cols) == 384, f"Expected 384 mu cols, got {len(mu_cols)}"
Z = latents[mu_cols].values.astype(np.float32)
meta_subset = latents[["SubjectID", "fold", "Manufacturer", "y", "Age", "Sex"]].copy()

# ── join site info from metadata ───────────────────────────────────────────────
meta = pd.read_csv(META_CSV)
site_col = "Site3"
if site_col in meta.columns:
    site_map = meta[["SubjectID", site_col]].drop_duplicates().set_index("SubjectID")
    meta_subset = meta_subset.join(site_map, on="SubjectID")
    meta_subset[site_col] = meta_subset[site_col].fillna(-1).astype(int).astype(str)
    print(f"Joined Site3: {meta_subset[site_col].nunique()} unique sites")
else:
    meta_subset["Site3"] = "unknown"
    print("WARNING: Site3 not found in metadata; using 'unknown'")

# ── join OOF predictions ───────────────────────────────────────────────────────
preds = pd.read_csv(OOF_PRED_CSV)
preds_logreg = preds[preds["classifier_type"] == "logreg"].copy()
pred_map = preds_logreg.set_index("SubjectID")[["y_score_final", "y_pred"]].rename(
    columns={"y_score_final": "y_score", "y_pred": "y_pred"}
)
meta_subset = meta_subset.join(pred_map, on="SubjectID")
print(f"OOF predictions joined: {meta_subset['y_score'].notna().sum()} / {len(meta_subset)} subjects have scores")

# ── error labels ──────────────────────────────────────────────────────────────
meta_subset["correct"] = (meta_subset["y"] == meta_subset["y_pred"]).astype(int)
meta_subset["error_type"] = "TP"
meta_subset.loc[(meta_subset["y"] == 1) & (meta_subset["y_pred"] == 0), "error_type"] = "FN"
meta_subset.loc[(meta_subset["y"] == 0) & (meta_subset["y_pred"] == 1), "error_type"] = "FP"
meta_subset.loc[(meta_subset["y"] == 0) & (meta_subset["y_pred"] == 0), "error_type"] = "TN"

n_sub = len(meta_subset)
print(f"Error breakdown: {meta_subset['error_type'].value_counts().to_dict()}")

# ═══════════════════════════════════════════════════════════════════════════════
# Analysis helpers
# ═══════════════════════════════════════════════════════════════════════════════

def centroid(z: np.ndarray) -> np.ndarray:
    return z.mean(axis=0)

def mean_radius(z: np.ndarray, c: np.ndarray) -> float:
    diffs = z - c
    return float(np.linalg.norm(diffs, axis=1).mean())

def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(1.0 - np.dot(a, b) / (na * nb))

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Centroids per Manufacturer
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── 1. Manufacturer centroids ──")
mfr_groups = {}
for mfr, grp in meta_subset.groupby("Manufacturer"):
    idx = grp.index
    mfr_groups[mfr] = {"centroid": centroid(Z[idx]), "radius": mean_radius(Z[idx], centroid(Z[idx])), "n": len(idx)}

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Centroids per Site
# ═══════════════════════════════════════════════════════════════════════════════
print("── 2. Site centroids ──")
site_groups = {}
for site, grp in meta_subset.groupby("Site3"):
    idx = grp.index
    site_groups[site] = {"centroid": centroid(Z[idx]), "radius": mean_radius(Z[idx], centroid(Z[idx])), "n": len(idx)}
print(f"  {len(site_groups)} sites")

# ── pairwise centroid distances ───────────────────────────────────────────────
def pairwise_dist_table(groups: dict, label: str) -> pd.DataFrame:
    keys = sorted(groups.keys(), key=lambda x: str(x))
    rows = []
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            if j <= i:
                continue
            ci = groups[ki]["centroid"]
            cj = groups[kj]["centroid"]
            rows.append({
                label + "_i": ki,
                label + "_j": kj,
                "n_i": groups[ki]["n"],
                "n_j": groups[kj]["n"],
                "euclidean_dist": float(np.linalg.norm(ci - cj)),
                "cosine_dist": cosine_dist(ci, cj),
            })
    return pd.DataFrame(rows)

mfr_centroid_dist = pairwise_dist_table(mfr_groups, "manufacturer")
site_centroid_dist = pairwise_dist_table(site_groups, "site")

# ── radius tables ──────────────────────────────────────────────────────────────
rows_mfr = [{"group_type": "manufacturer", "group": k, "n": v["n"],
              "within_radius_mean": v["radius"]} for k, v in mfr_groups.items()]
rows_site = [{"group_type": "site", "group": k, "n": v["n"],
               "within_radius_mean": v["radius"]} for k, v in site_groups.items()]
radius_table = pd.DataFrame(rows_mfr + rows_site).sort_values(["group_type", "group"])

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Silhouette scores
# ═══════════════════════════════════════════════════════════════════════════════
print("── 3. Silhouette scores ──")
silhouette_rows = []

def silhouette_safe(Z, labels, label_name: str, sample_size: int = 500) -> dict:
    le = LabelEncoder()
    enc = le.fit_transform(labels)
    n_classes = len(np.unique(enc))
    if n_classes < 2:
        return {"grouping": label_name, "n_classes": n_classes, "silhouette": float("nan"),
                "note": "< 2 classes"}
    if len(Z) > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(Z), sample_size, replace=False)
        Z_sub, enc_sub = Z[idx], enc[idx]
    else:
        Z_sub, enc_sub = Z, enc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = silhouette_score(Z_sub, enc_sub, metric="euclidean", random_state=42)
    return {"grouping": label_name, "n_classes": n_classes, "silhouette": float(score),
            "n_samples_used": len(Z_sub), "note": ""}

silhouette_rows.append(silhouette_safe(Z, meta_subset["y"].values, "diagnosis"))
silhouette_rows.append(silhouette_safe(Z, meta_subset["Manufacturer"].values, "manufacturer"))
silhouette_rows.append(silhouette_safe(Z, meta_subset["Site3"].values, "site"))
silhouette_df = pd.DataFrame(silhouette_rows)
print(silhouette_df[["grouping", "silhouette", "n_classes"]].to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
# 4. kNN decoding of manufacturer and site
# ═══════════════════════════════════════════════════════════════════════════════
print("── 4. kNN decoding ──")
knn_rows = []

def knn_decode(Z, labels, target: str, n_neighbors: int = 5, n_splits: int = 3) -> dict:
    le = LabelEncoder()
    enc = le.fit_transform(labels)
    n_classes = len(np.unique(enc))
    if n_classes < 2:
        return {"target": target, "k": n_neighbors, "accuracy": float("nan"),
                "n_classes": n_classes, "note": "< 2 classes"}
    counts = np.bincount(enc)
    if counts.min() < n_splits:
        n_splits = max(2, int(counts.min()))
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric="euclidean", n_jobs=4)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(knn, Z, enc, cv=cv, scoring="balanced_accuracy")
    return {
        "target": target, "k": n_neighbors,
        "balanced_accuracy_mean": float(scores.mean()),
        "balanced_accuracy_std": float(scores.std()),
        "n_classes": n_classes,
        "note": "",
    }

knn_rows.append(knn_decode(Z, meta_subset["Manufacturer"].values, "manufacturer"))
knn_rows.append(knn_decode(Z, meta_subset["y"].astype(str).values, "diagnosis"))

# Site decoding (only sites with >= 5 subjects for meaningful CV)
site_counts = meta_subset["Site3"].value_counts()
large_sites = site_counts[site_counts >= 10].index.tolist()
if len(large_sites) >= 2:
    mask = meta_subset["Site3"].isin(large_sites)
    knn_rows.append(knn_decode(
        Z[mask], meta_subset.loc[mask, "Site3"].values, "site_top10", n_neighbors=5
    ))

knn_df = pd.DataFrame(knn_rows)
print(knn_df[["target", "balanced_accuracy_mean", "n_classes"]].to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Prediction error vs centroid distance
# ═══════════════════════════════════════════════════════════════════════════════
print("── 5. Prediction error by site/manufacturer geometry ──")

def dist_to_centroid(Z_row: np.ndarray, c: np.ndarray) -> float:
    return float(np.linalg.norm(Z_row - c))

# distance from each subject to their manufacturer centroid
meta_subset["dist_to_mfr_centroid"] = [
    dist_to_centroid(Z[i], mfr_groups[row["Manufacturer"]]["centroid"])
    for i, (_, row) in enumerate(meta_subset.iterrows())
]
# distance from each subject to their site centroid
meta_subset["dist_to_site_centroid"] = [
    dist_to_centroid(Z[i], site_groups[row["Site3"]]["centroid"])
    for i, (_, row) in enumerate(meta_subset.iterrows())
]

# Median distance to mfr centroid: correct vs incorrect
err_by_mfr_dist = (
    meta_subset.groupby(["Manufacturer", "correct"])["dist_to_mfr_centroid"]
    .agg(["mean", "median", "std", "count"])
    .reset_index()
)
err_by_mfr_dist.columns = ["Manufacturer", "correct", "mean_dist", "median_dist", "std_dist", "n"]

# FP/FN breakdown by manufacturer
error_type_by_mfr = (
    meta_subset[meta_subset["error_type"].isin(["FP", "FN"])]
    .groupby(["Manufacturer", "error_type"])
    .size()
    .reset_index(name="count")
)

pred_error_df = pd.concat([
    meta_subset[["SubjectID", "Manufacturer", "Site3", "y", "y_score",
                  "y_pred", "error_type", "correct",
                  "dist_to_mfr_centroid", "dist_to_site_centroid"]]
], ignore_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 6. PCA for visualization
# ═══════════════════════════════════════════════════════════════════════════════
print("── 6. PCA ──")
pca = PCA(n_components=10, random_state=42)
Z_pca = pca.fit_transform(Z)
var_ratio = pca.explained_variance_ratio_
print(f"  PC1 var: {var_ratio[0]:.3f}, PC2 var: {var_ratio[1]:.3f}, top10 cumvar: {var_ratio.sum():.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. UMAP
# ═══════════════════════════════════════════════════════════════════════════════
print("── 7. UMAP ──")
try:
    import umap
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                        metric="euclidean", random_state=42, verbose=False)
    Z_umap = reducer.fit_transform(Z)
    have_umap = True
    print(f"  UMAP done: {Z_umap.shape}")
except Exception as e:
    have_umap = False
    Z_umap = None
    print(f"  UMAP failed: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════════
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

DIAG_COLORS = {0: "#2166ac", 1: "#d6604d"}    # CN blue, AD red
MFR_COLORS  = {"GE": "#1b7837", "Philips": "#762a83", "SIEMENS": "#f46d43"}

def scatter2d(x, y, labels_dict: dict, title: str, fname: str, alpha: float = 0.6,
              legend_loc: str = "best") -> None:
    fig, axes = plt.subplots(1, len(labels_dict), figsize=(6 * len(labels_dict), 5))
    if len(labels_dict) == 1:
        axes = [axes]
    for ax, (col_name, (labels, cmap)) in zip(axes, labels_dict.items()):
        uniq = sorted(set(labels), key=lambda v: str(v))
        if isinstance(cmap, dict):
            c_arr = [cmap.get(lbl, "#999999") for lbl in labels]
            patches = [mpatches.Patch(color=cmap.get(u, "#999999"), label=str(u)) for u in uniq]
        else:
            le = LabelEncoder()
            enc = le.fit_transform(labels)
            cmap_obj = plt.get_cmap(cmap, len(uniq))
            c_arr = [cmap_obj(i) for i in enc]
            patches = [mpatches.Patch(color=cmap_obj(i), label=str(u)) for i, u in enumerate(uniq)]
        ax.scatter(x, y, c=c_arr, alpha=alpha, s=12, linewidths=0)
        ax.legend(handles=patches, loc=legend_loc, fontsize=7, markerscale=2)
        ax.set_title(col_name, fontsize=9)
        ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
        ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")

# ── PCA scatter ───────────────────────────────────────────────────────────────
diag_labels = [("CN" if y == 0 else "AD") for y in meta_subset["y"]]
diag_cmap = {"CN": "#2166ac", "AD": "#d6604d"}
mfr_labels = meta_subset["Manufacturer"].tolist()

scatter2d(
    Z_pca[:, 0], Z_pca[:, 1],
    {"Diagnosis": (diag_labels, diag_cmap),
     "Manufacturer": (mfr_labels, MFR_COLORS)},
    "OOF Latent Space — PCA PC1 vs PC2",
    "pca_pc1_pc2_diagnosis_manufacturer.pdf"
)

# PCA colored by error type
err_labels = meta_subset["error_type"].tolist()
err_colors = {"TP": "#2ca02c", "TN": "#1f77b4", "FP": "#d62728", "FN": "#ff7f0e"}
scatter2d(
    Z_pca[:, 0], Z_pca[:, 1],
    {"Prediction outcome": (err_labels, err_colors)},
    "OOF Latent Space — PCA colored by prediction error type",
    "pca_pc1_pc2_error_type.pdf"
)

# ── UMAP scatter ──────────────────────────────────────────────────────────────
if have_umap:
    scatter2d(
        Z_umap[:, 0], Z_umap[:, 1],
        {"Diagnosis": (diag_labels, diag_cmap),
         "Manufacturer": (mfr_labels, MFR_COLORS)},
        "OOF Latent Space — UMAP (n_neighbors=15, min_dist=0.1)",
        "umap_diagnosis_manufacturer.pdf"
    )
    scatter2d(
        Z_umap[:, 0], Z_umap[:, 1],
        {"Prediction outcome": (err_labels, err_colors)},
        "OOF Latent Space — UMAP colored by prediction error type",
        "umap_error_type.pdf"
    )
    # UMAP site plot (color by site — use tab20 cmap)
    site_labels = meta_subset["Site3"].tolist()
    scatter2d(
        Z_umap[:, 0], Z_umap[:, 1],
        {"Site (Site3)": (site_labels, "tab20")},
        "OOF Latent Space — UMAP colored by site",
        "umap_site.pdf",
        alpha=0.7
    )

# ── distance-to-centroid histogram ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, col, title in [
    (axes[0], "dist_to_mfr_centroid", "Distance to manufacturer centroid"),
    (axes[1], "dist_to_site_centroid", "Distance to site centroid"),
]:
    for etype, color in [("TP", "#2ca02c"), ("TN", "#1f77b4"),
                          ("FP", "#d62728"), ("FN", "#ff7f0e")]:
        d = meta_subset.loc[meta_subset["error_type"] == etype, col].dropna()
        if len(d) > 0:
            ax.hist(d.values, bins=30, alpha=0.5, label=etype, color=color, density=True)
    ax.set_xlabel("Euclidean distance (384-dim)")
    ax.set_ylabel("Density")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)
fig.suptitle("OOF prediction error type vs distance to group centroid", fontsize=10)
fig.tight_layout()
fig.savefig(FIG_DIR / "centroid_distance_by_error_type.pdf", dpi=120, bbox_inches="tight")
plt.close(fig)
print("  Saved centroid_distance_by_error_type.pdf")

# ── silhouette sample scores by manufacturer / diagnosis ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, col, title, cmap_d in [
    (axes[0], "Manufacturer", "Per-sample silhouette by Manufacturer", MFR_COLORS),
    (axes[1], "y", "Per-sample silhouette by Diagnosis",
     {0: "#2166ac", 1: "#d6604d"}),
]:
    le = LabelEncoder()
    enc = le.fit_transform(meta_subset[col].values)
    if len(np.unique(enc)) < 2:
        ax.set_title(f"{title} (< 2 classes)")
        continue
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sil_vals = silhouette_samples(Z, enc, metric="euclidean")
    uniq_labels = sorted(meta_subset[col].unique(), key=str)
    y_lower = 0
    for lbl in uniq_labels:
        mask = meta_subset[col].values == lbl
        s = np.sort(sil_vals[mask])
        ax.barh(
            np.arange(y_lower, y_lower + len(s)), s, height=1.0, left=0,
            color=cmap_d.get(lbl, "#999999") if isinstance(cmap_d, dict) else "#999999",
            alpha=0.7, label=str(lbl)
        )
        y_lower += len(s) + 5
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlabel("Silhouette coefficient")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
fig.suptitle("Per-sample silhouette analysis of OOF latent space", fontsize=10)
fig.tight_layout()
fig.savefig(FIG_DIR / "silhouette_per_sample.pdf", dpi=120, bbox_inches="tight")
plt.close(fig)
print("  Saved silhouette_per_sample.pdf")

# ── kNN accuracy comparison bar chart ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
targets_knn = knn_df["target"].tolist()
ba_means = knn_df["balanced_accuracy_mean"].tolist()
ba_stds  = knn_df["balanced_accuracy_std"].tolist() if "balanced_accuracy_std" in knn_df.columns else [0]*len(targets_knn)
x = np.arange(len(targets_knn))
bars = ax.bar(x, ba_means, yerr=ba_stds, capsize=5, color=["#1f77b4","#d62728","#2ca02c"][:len(targets_knn)], alpha=0.8)
ax.axhline(1.0 / knn_df["n_classes"].max(), color="gray", linestyle="--", lw=0.8, label="chance (1/K)")
ax.set_xticks(x)
ax.set_xticklabels(targets_knn, fontsize=9)
ax.set_ylabel("Balanced accuracy (kNN, k=5, 3-fold CV)")
ax.set_ylim(0, 1.05)
ax.set_title("kNN decodability of latent space\n(diagnosis vs manufacturer vs site)", fontsize=9)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "knn_decodability.pdf", dpi=120, bbox_inches="tight")
plt.close(fig)
print("  Saved knn_decodability.pdf")

# ═══════════════════════════════════════════════════════════════════════════════
# Save CSV outputs
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Saving CSV outputs ──")

mfr_centroid_dist.to_csv(OUT_DIR / "centroid_distance_tables.csv", index=False)
print("  centroid_distance_tables.csv (manufacturer pairwise)")

site_centroid_dist_out = site_centroid_dist.copy()
site_centroid_dist_out.to_csv(OUT_DIR / "centroid_distance_tables_site.csv", index=False)
print("  centroid_distance_tables_site.csv")

radius_table.to_csv(OUT_DIR / "site_radius_tables.csv", index=False)
print("  site_radius_tables.csv")

silhouette_df.to_csv(OUT_DIR / "silhouette_summary.csv", index=False)
print("  silhouette_summary.csv")

pred_error_df.to_csv(OUT_DIR / "prediction_error_by_site_geometry.csv", index=False)
print("  prediction_error_by_site_geometry.csv")

knn_df.to_csv(OUT_DIR / "knn_decodability.csv", index=False)
print("  knn_decodability.csv")

err_by_mfr_dist.to_csv(OUT_DIR / "error_by_manufacturer_centroid_dist.csv", index=False)
print("  error_by_manufacturer_centroid_dist.csv")

error_type_by_mfr.to_csv(OUT_DIR / "error_type_by_manufacturer.csv", index=False)
print("  error_type_by_manufacturer.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# Generate report
# ═══════════════════════════════════════════════════════════════════════════════
T1 = datetime.now(timezone.utc).isoformat()

sil_diag = silhouette_df.loc[silhouette_df["grouping"] == "diagnosis", "silhouette"].values[0]
sil_mfr  = silhouette_df.loc[silhouette_df["grouping"] == "manufacturer", "silhouette"].values[0]
sil_site = silhouette_df.loc[silhouette_df["grouping"] == "site", "silhouette"].values[0]
knn_diag = knn_df.loc[knn_df["target"] == "diagnosis", "balanced_accuracy_mean"].values
knn_mfr  = knn_df.loc[knn_df["target"] == "manufacturer", "balanced_accuracy_mean"].values
knn_diag_str = f"{knn_diag[0]:.3f}" if len(knn_diag) else "N/A"
knn_mfr_str  = f"{knn_mfr[0]:.3f}" if len(knn_mfr) else "N/A"

# FP/FN count
n_fp = (meta_subset["error_type"] == "FP").sum()
n_fn = (meta_subset["error_type"] == "FN").sum()
n_tp = (meta_subset["error_type"] == "TP").sum()
n_tn = (meta_subset["error_type"] == "TN").sum()

# Top-3 sites by FP+FN
fp_fn_mask = meta_subset["error_type"].isin(["FP","FN"])
if fp_fn_mask.any():
    errors_by_site = meta_subset[fp_fn_mask]["Site3"].value_counts().head(5)
    errors_by_site_str = "\n".join(f"  - Site {s}: {c}" for s, c in errors_by_site.items())
else:
    errors_by_site_str = "  N/A"

# Manufacturer FP breakdown
fp_by_mfr = meta_subset[meta_subset["error_type"] == "FP"]["Manufacturer"].value_counts()
fn_by_mfr = meta_subset[meta_subset["error_type"] == "FN"]["Manufacturer"].value_counts()

mfr_dist_table_str = mfr_centroid_dist[["manufacturer_i","manufacturer_j","euclidean_dist","cosine_dist"]].to_string(index=False)

radius_mfr_str = radius_table[radius_table["group_type"] == "manufacturer"][
    ["group","n","within_radius_mean"]
].to_string(index=False)

umap_note = "Computed (umap-learn 0.5.3)" if have_umap else "FAILED (umap-learn unavailable)"

report = f"""# Latent Site/Manufacturer Geometry Report

**Run:** {RUN_NAME}
**Analysis date:** {T0[:10]}
**Script:** {Path(__file__).name}
**N subjects (OOF test):** {n_sub}
**Latent dim:** 384
**UMAP:** {umap_note}

---

## 1. Silhouette scores (Euclidean, up to 500 subjects subsample)

| Grouping | n_classes | Silhouette |
|---|---:|---:|
| Diagnosis (AD/CN) | 2 | {sil_diag:.4f} |
| Manufacturer (GE/Philips/SIEMENS) | 3 | {sil_mfr:.4f} |
| Site (Site3) | {meta_subset['Site3'].nunique()} | {sil_site:.4f} |

**Interpretation:** Silhouette > 0.1 indicates detectable clustering;
values near 0 indicate no structure; negative values indicate misassignment.

---

## 2. kNN decodability (k=5, balanced accuracy, 3-fold stratified CV)

{knn_df[['target','n_classes','balanced_accuracy_mean','balanced_accuracy_std']].to_markdown(index=False)}

**Key question:** Is manufacturer clustering stronger than diagnosis clustering?

---

## 3. Manufacturer centroid pairwise distances

{mfr_centroid_dist_out_str if False else mfr_centroid_dist[['manufacturer_i','manufacturer_j','euclidean_dist','cosine_dist']].to_markdown(index=False)}

---

## 4. Manufacturer within-group radius (mean Euclidean distance to centroid)

{radius_table[radius_table['group_type']=='manufacturer'][['group','n','within_radius_mean']].to_markdown(index=False)}

---

## 5. Prediction error breakdown

| Type | N |
|---|---:|
| TP | {n_tp} |
| TN | {n_tn} |
| FP | {n_fp} |
| FN | {n_fn} |

**FP by manufacturer:**
{fp_by_mfr.to_string()}

**FN by manufacturer:**
{fn_by_mfr.to_string()}

**Top sites by FP+FN count:**
{errors_by_site_str}

---

## 6. Error type vs centroid distance

{err_by_mfr_dist.to_markdown(index=False)}

---

## 7. PCA variance explained

| PC | Explained variance ratio |
|---|---:|
{chr(10).join(f"| PC{i+1} | {v:.4f} |" for i, v in enumerate(var_ratio[:5]))}
| top-10 cumulative | {var_ratio.sum():.4f} |

---

## 8. Figures generated

- `figures/pca_pc1_pc2_diagnosis_manufacturer.pdf`
- `figures/pca_pc1_pc2_error_type.pdf`
- `figures/umap_diagnosis_manufacturer.pdf` (if UMAP available)
- `figures/umap_error_type.pdf` (if UMAP available)
- `figures/umap_site.pdf` (if UMAP available)
- `figures/centroid_distance_by_error_type.pdf`
- `figures/silhouette_per_sample.pdf`
- `figures/knn_decodability.pdf`

---

## 9. Interpretation summary

A high manufacturer silhouette combined with low diagnosis silhouette would
confirm that the latent space encodes scanner type more strongly than
pathology — a site-confound risk. Values close to zero for both suggest
the β-VAE has collapsed both signals equally.

The kNN balanced accuracy comparison (diagnosis vs manufacturer) is the
most direct test: if `knn(manufacturer) >> knn(diagnosis)`, the latent
space is scanner-dominated.

FP/FN concentration in specific manufacturer or site clusters would
indicate that classification errors are spatially structured in the latent
space — a sign that the classifier's errors track hardware/site factors
rather than random noise.

---

## Caveats

- Analysis uses OOF test-set latents only (n={n_sub}); training-set geometry may differ.
- Site3 has {meta_subset['Site3'].nunique()} unique values; silhouette computed on all but kNN
  filtered to sites with ≥10 subjects to avoid degenerate CV folds.
- Silhouette uses Euclidean distance in 384-dim space (no dimensionality reduction).
- Mapper/persistent homology: skipped (not requested unless dependencies confirmed).
"""

(OUT_DIR / "latent_site_geometry_report.md").write_text(report, encoding="utf-8")
print("  latent_site_geometry_report.md")

# ─── command log ───────────────────────────────────────────────────────────────
cmd_log = {
    "script": str(Path(__file__).resolve()),
    "run_name": RUN_NAME,
    "started_utc": T0,
    "finished_utc": T1,
    "n_subjects": int(n_sub),
    "latent_dim": 384,
    "umap_available": have_umap,
    "outputs": [
        "centroid_distance_tables.csv",
        "centroid_distance_tables_site.csv",
        "site_radius_tables.csv",
        "silhouette_summary.csv",
        "prediction_error_by_site_geometry.csv",
        "knn_decodability.csv",
        "error_by_manufacturer_centroid_dist.csv",
        "error_type_by_manufacturer.csv",
        "latent_site_geometry_report.md",
        "figures/pca_pc1_pc2_diagnosis_manufacturer.pdf",
        "figures/pca_pc1_pc2_error_type.pdf",
        "figures/centroid_distance_by_error_type.pdf",
        "figures/silhouette_per_sample.pdf",
        "figures/knn_decodability.pdf",
    ] + (
        ["figures/umap_diagnosis_manufacturer.pdf",
         "figures/umap_error_type.pdf",
         "figures/umap_site.pdf"]
        if have_umap else []
    ),
    "guardrails": {
        "no_vae_retraining": True,
        "no_tensor_edits": True,
        "no_metadata_edits": True,
        "no_ig_shap_recomputation": True,
        "read_only_source_files": True,
    }
}
(OUT_DIR / "command_log.json").write_text(json.dumps(cmd_log, indent=2), encoding="utf-8")
print("  command_log.json")

print(f"\n✓ Part B analysis complete. Output: {OUT_DIR}")
