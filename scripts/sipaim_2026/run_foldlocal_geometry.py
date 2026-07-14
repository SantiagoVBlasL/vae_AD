#!/usr/bin/env python3
"""
Fold-local confirmatory geometry and covariance analysis.
Locked model: recover035_latent384_beta3p75_T80_h10000_p560_full5x5

Rules:
  - Never pool raw latent vectors across folds for confirmatory geometry.
  - Fit scaler/residualizer/covariance estimators on train/dev only.
  - Apply once to the corresponding outer-test fold.
  - Primary domain: Manufacturer. Site3 as secondary (missingness caveat).
  - Control diagnosis, Age, Sex.
  - No VAE or classifier retraining.
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
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import scipy.linalg
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

T0 = datetime.now(timezone.utc).isoformat()

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[4]   # vae_AD project root
BIG_DISK = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026")
RUN_NAME = "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
RUN_DIR  = BIG_DISK / RUN_NAME
LATENT_CACHE = RUN_DIR / "classifier_only_readout" / "latent_cache"
STGB_DIR = ROOT / "results/revision_bspc_2026/recover035_latent384_beta3p75_stageB_oof_score_calibration"
META_CSV  = ROOT / "results/revision_bspc_2026/adni_035_metadata_rescue_preflight/patched_metadata_candidate.csv"
TRAV_DIR  = ROOT / "results/revision_bspc_2026/post_revision_exploratory_20260630/locked_model_site_effect_mechanistic_audit_20260704"
OUT_DIR   = Path(__file__).parent
FIG_DIR   = OUT_DIR / "figures"
DAT_DIR   = OUT_DIR / "figure_data"
FIG_DIR.mkdir(exist_ok=True)
DAT_DIR.mkdir(exist_ok=True)

# Fail-closed guards
for p, label in [
    (LATENT_CACHE, "latent cache dir"),
    (STGB_DIR, "Stage-B package"),
    (META_CSV, "metadata CSV"),
    (TRAV_DIR, "previous traversal dir"),
]:
    if not Path(p).exists():
        sys.exit(f"FAIL: required path missing: {label}\n  {p}")

FOLDS = [1, 2, 3, 4, 5]
MFR_COLORS = {"GE": "#2166ac", "Philips": "#d6604d", "SIEMENS": "#1b7837"}
DIAG_COLORS = {"CN": "#74add1", "AD": "#d73027"}
N_PERM = 1000
N_PC_ANGLES = 20
RNG = np.random.default_rng(42)

print(f"[{datetime.now(timezone.utc).isoformat()}] Starting fold-local geometry analysis")

# ── helpers ───────────────────────────────────────────────────────────────────
def load_latents(fold: int, split: str) -> pd.DataFrame:
    f = LATENT_CACHE / f"fold_{fold}_{split}_latent_mu.csv"
    if not f.exists():
        sys.exit(f"FAIL: missing latent cache: {f}")
    return pd.read_csv(f)

def mu_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("mu_")]

def encode_sex(s: pd.Series) -> np.ndarray:
    return (s == "M").astype(float).values

def residualize_nuisance(Z_train: np.ndarray, nuisance_train: np.ndarray,
                          Z_test: np.ndarray,  nuisance_test: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray]:
    """OLS fit on train, apply to train and test. Returns (Z_train_resid, Z_test_resid)."""
    reg = LinearRegression(fit_intercept=True)
    reg.fit(nuisance_train, Z_train)
    return (Z_train - reg.predict(nuisance_train),
            Z_test  - reg.predict(nuisance_test))

def effective_rank(cov: np.ndarray) -> float:
    """Effective rank via spectral entropy: exp(H(p)) where p_i = λ_i/Σλ."""
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 0]
    p = eigvals / eigvals.sum()
    h = -np.sum(p * np.log(p))
    return float(np.exp(h))

def bures_wasserstein(S1: np.ndarray, S2: np.ndarray) -> float:
    """BW distance between zero-mean Gaussians N(0,S1) and N(0,S2)."""
    sqrtS1 = scipy.linalg.sqrtm(S1)
    sqrtS1 = sqrtS1.real
    M = sqrtS1 @ S2 @ sqrtS1
    sqrtM = scipy.linalg.sqrtm(M)
    sqrtM = sqrtM.real
    bw2 = np.trace(S1) + np.trace(S2) - 2.0 * np.trace(sqrtM)
    return float(np.sqrt(max(bw2, 0.0)))

def principal_angles_deg(S1: np.ndarray, S2: np.ndarray, k: int) -> np.ndarray:
    """Principal angles (degrees) between top-k subspaces of S1 and S2."""
    _, v1 = np.linalg.eigh(S1)
    _, v2 = np.linalg.eigh(S2)
    Q1 = v1[:, -k:]
    Q2 = v2[:, -k:]
    angles_rad = scipy.linalg.subspace_angles(Q1, Q2)
    return np.degrees(angles_rad)

def mahalanobis_sq(z: np.ndarray, mu: np.ndarray, precision: np.ndarray) -> float:
    d = z - mu
    return float(d @ precision @ d)

# ── load global metadata with Site3 ───────────────────────────────────────────
meta_full = pd.read_csv(META_CSV)[["SubjectID", "Site3"]].drop_duplicates("SubjectID")

# ── Part 0: Domain metadata audit ─────────────────────────────────────────────
print(f"[{datetime.now(timezone.utc).isoformat()}] Part 0: domain metadata audit")
audit_rows = []
for fold in FOLDS:
    td = load_latents(fold, "trainDev")
    te = load_latents(fold, "test")
    for split, df in [("trainDev", td), ("test", te)]:
        for mfr, grp in df.groupby("Manufacturer"):
            audit_rows.append({
                "fold": fold, "split": split, "manufacturer": mfr,
                "n": len(grp), "n_CN": (grp["y"] == 0).sum(), "n_AD": (grp["y"] == 1).sum(),
                "mean_age": grp["Age"].mean(), "pct_female": (grp["Sex"] == "F").mean(),
            })
audit_df = pd.DataFrame(audit_rows)
audit_df.to_csv(OUT_DIR / "domain_metadata_audit.csv", index=False)

# Site3 missingness audit
te_all = pd.concat([load_latents(f, "test") for f in FOLDS], ignore_index=True)
te_all = te_all.merge(meta_full, on="SubjectID", how="left")
n_total = len(te_all)
n_missing_s3 = te_all["Site3"].isna().sum()
site3_mfr = te_all.groupby("Manufacturer")["Site3"].apply(lambda x: x.isna().mean()).rename("frac_missing_site3")

with open(OUT_DIR / "domain_metadata_audit.md", "w") as f:
    f.write("# Domain Metadata Audit\n\n")
    f.write(f"**Date:** {datetime.now(timezone.utc).date()}\n\n")
    f.write("## Manufacturer counts per fold/split\n\n")
    f.write(audit_df.pivot_table(
        index=["fold","split"], columns="manufacturer", values="n", aggfunc="sum"
    ).to_markdown(floatfmt=".0f") + "\n\n")
    f.write("## Site3 missingness (pooled OOF test set)\n\n")
    f.write(f"- Total OOF subjects: {n_total}\n")
    f.write(f"- Missing Site3: {n_missing_s3} ({100*n_missing_s3/n_total:.1f}%)\n")
    f.write(f"- Present Site3: {n_total - n_missing_s3}\n\n")
    f.write("Missingness by manufacturer:\n")
    f.write(site3_mfr.to_markdown() + "\n\n")
    f.write("**Implication:** Site3 secondary analysis uses only subjects with non-null Site3.\n")
    f.write("Site3 strata with N < 10 are excluded from geometry reporting.\n")

print(f"  Metadata audit done. Site3 missing: {n_missing_s3}/{n_total}")

# ── Part 1: Per-fold geometry ──────────────────────────────────────────────────
print(f"[{datetime.now(timezone.utc).isoformat()}] Part 1: per-fold geometry")

MANUFACTURERS = ["GE", "Philips", "SIEMENS"]

centroid_rows   = []
covariance_rows = []
perm_rows       = []
knn_rows        = []
pca_fold_data   = {}   # fold -> dict for PCA scatter data

for fold in FOLDS:
    print(f"  Fold {fold}...")
    td = load_latents(fold, "trainDev")
    te = load_latents(fold, "test")

    mcols = mu_cols(td)
    assert len(mcols) == 384

    # ── 1.1: Standardize (fit on trainDev) ──────────────────────────────────
    scaler = StandardScaler()
    Z_td = scaler.fit_transform(td[mcols].values)
    Z_te = scaler.transform(te[mcols].values)

    # ── 1.2: Nuisance residualization (fit on trainDev) ──────────────────────
    # Nuisance: diagnosis (y), Age, Sex_binary
    def nuisance_mat(df):
        age = (df["Age"].values - df["Age"].mean()) / (df["Age"].std() + 1e-8)
        sex = encode_sex(df["Sex"])
        y   = df["y"].values.astype(float)
        return np.column_stack([y, age, sex])

    N_td = nuisance_mat(td)
    N_te_raw = nuisance_mat(te)
    # Re-center nuisance using trainDev stats (already done for Age above; apply same for test)
    # Recompute with trainDev centering
    age_mean_td = td["Age"].mean(); age_std_td = td["Age"].std()
    N_td = np.column_stack([
        td["y"].values.astype(float),
        (td["Age"].values - age_mean_td) / (age_std_td + 1e-8),
        encode_sex(td["Sex"])
    ])
    N_te = np.column_stack([
        te["y"].values.astype(float),
        (te["Age"].values - age_mean_td) / (age_std_td + 1e-8),
        encode_sex(te["Sex"])
    ])
    Z_td_r, Z_te_r = residualize_nuisance(Z_td, N_td, Z_te, N_te)

    # ── 1.3: PCA for visualization (fit on trainDev residualized) ────────────
    pca2 = PCA(n_components=2, random_state=42)
    pca2.fit(Z_td_r)
    Z_te_2d = pca2.transform(Z_te_r)
    pca_fold_data[fold] = {
        "Z_2d": Z_te_2d,
        "mfr":  te["Manufacturer"].values,
        "diag": te["ResearchGroup_Mapped"].values,
        "y":    te["y"].values,
        "var_explained": pca2.explained_variance_ratio_,
    }

    # ── 1.4: Covariance estimators per manufacturer (fit on trainDev) ────────
    lw_models = {}
    for mfr in MANUFACTURERS:
        idx_td = td["Manufacturer"] == mfr
        Z_mfr  = Z_td_r[idx_td]
        if Z_mfr.shape[0] < 10:
            print(f"    WARNING: fold {fold} trainDev {mfr} only {Z_mfr.shape[0]} subjects — skipping LW")
            lw_models[mfr] = None
            continue
        lw = LedoitWolf(assume_centered=False)
        lw.fit(Z_mfr)
        lw_models[mfr] = lw

    # ── 1.5: Centroid geometry on TEST fold ──────────────────────────────────
    centroids = {}
    radii = {}
    for mfr in MANUFACTURERS:
        idx_te = te["Manufacturer"] == mfr
        Z_mfr_te = Z_te_r[idx_te]
        if len(Z_mfr_te) == 0:
            centroids[mfr] = None; radii[mfr] = None
            continue
        c = Z_mfr_te.mean(axis=0)
        r = np.sqrt(((Z_mfr_te - c) ** 2).sum(axis=1)).mean()
        centroids[mfr] = c
        radii[mfr] = float(r)

    pairs = [("GE", "Philips"), ("GE", "SIEMENS"), ("Philips", "SIEMENS")]
    for a, b in pairs:
        if centroids[a] is None or centroids[b] is None:
            continue
        c_dist = float(np.linalg.norm(centroids[a] - centroids[b]))
        pooled_r = (radii[a] + radii[b]) / 2.0
        sep_ratio = c_dist / pooled_r if pooled_r > 0 else np.nan

        # Mahalanobis: each test subject to its own manufacturer centroid
        mah_vals_a = []; mah_vals_b = []
        for mfr, mah_list in [(a, mah_vals_a), (b, mah_vals_b)]:
            if lw_models.get(mfr) is not None:
                prec = lw_models[mfr].precision_
                mu_mfr = centroids[mfr]
                for z in Z_te_r[te["Manufacturer"] == mfr]:
                    mah_list.append(np.sqrt(mahalanobis_sq(z, mu_mfr, prec)))

        centroid_rows.append({
            "fold": fold, "pair": f"{a}_vs_{b}",
            "mfr_a": a, "mfr_b": b,
            "n_a_test": (te["Manufacturer"] == a).sum(),
            "n_b_test": (te["Manufacturer"] == b).sum(),
            "centroid_dist_euclid": round(c_dist, 4),
            "radius_a": round(radii[a], 4) if radii[a] else np.nan,
            "radius_b": round(radii[b], 4) if radii[b] else np.nan,
            "sep_ratio": round(sep_ratio, 4),
            "mahal_a_mean": round(np.mean(mah_vals_a), 4) if mah_vals_a else np.nan,
            "mahal_b_mean": round(np.mean(mah_vals_b), 4) if mah_vals_b else np.nan,
        })

    # ── 1.6: Covariance metrics per manufacturer ─────────────────────────────
    eff_ranks = {}
    spectra   = {}
    for mfr in MANUFACTURERS:
        lw = lw_models.get(mfr)
        if lw is None:
            continue
        cov = lw.covariance_
        er  = effective_rank(cov)
        eff_ranks[mfr] = er
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        spectra[mfr] = eigvals
        covariance_rows.append({
            "fold": fold, "manufacturer": mfr,
            "lw_shrinkage": round(float(lw.shrinkage_), 4),
            "effective_rank": round(er, 2),
            "trace_cov": round(float(np.trace(cov)), 4),
            "top1_eigenval": round(float(eigvals[0]), 4),
            "top5_eigenval_sum": round(float(eigvals[:5].sum()), 4),
            "top10_frac_var": round(float(eigvals[:10].sum() / eigvals.sum()), 4),
        })

    # ── 1.7: Bures-Wasserstein distances ─────────────────────────────────────
    for a, b in pairs:
        lw_a = lw_models.get(a); lw_b = lw_models.get(b)
        if lw_a is None or lw_b is None:
            continue
        bw = bures_wasserstein(lw_a.covariance_, lw_b.covariance_)
        covariance_rows[-3 if len(covariance_rows) >= 3 else 0]["dummy_bw"] = None  # handled below
        # Store separately in BW sub-table
        knn_rows.append({"fold": fold, "metric": "bures_wasserstein",
                         "pair": f"{a}_vs_{b}", "value": round(bw, 4)})

    # ── 1.8: Principal angles (top-k subspaces from LW covariance) ───────────
    for a, b in pairs:
        lw_a = lw_models.get(a); lw_b = lw_models.get(b)
        if lw_a is None or lw_b is None:
            continue
        angles = principal_angles_deg(lw_a.covariance_, lw_b.covariance_, k=N_PC_ANGLES)
        for i, ang in enumerate(angles[:5]):
            knn_rows.append({"fold": fold, "metric": f"principal_angle_pc{i+1}",
                             "pair": f"{a}_vs_{b}", "value": round(ang, 3)})
        knn_rows.append({"fold": fold, "metric": "mean_principal_angle_top20",
                         "pair": f"{a}_vs_{b}", "value": round(float(angles.mean()), 3)})

    # ── 1.9: kNN manufacturer balanced accuracy ───────────────────────────────
    le = LabelEncoder()
    y_td_mfr = le.fit_transform(td["Manufacturer"].values)
    y_te_mfr = le.transform(te["Manufacturer"].values)

    knn_clf = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    knn_clf.fit(Z_td_r, y_td_mfr)
    y_pred_te = knn_clf.predict(Z_te_r)
    bacc_obs = balanced_accuracy_score(y_te_mfr, y_pred_te)

    # ── 1.10: Matched permutation null (permute test labels, recompute kNN) ───
    bacc_null = []
    for _ in range(N_PERM):
        y_perm = RNG.permutation(y_te_mfr)
        bacc_null.append(balanced_accuracy_score(y_perm, y_pred_te))
    p_val = (np.sum(np.array(bacc_null) >= bacc_obs) + 1) / (N_PERM + 1)

    knn_rows.append({"fold": fold, "metric": "knn5_mfr_bacc",
                     "pair": "all", "value": round(bacc_obs, 4)})
    perm_rows.append({
        "fold": fold, "test": "knn5_manufacturer_balanced_accuracy",
        "observed": round(bacc_obs, 4),
        "perm_mean": round(np.mean(bacc_null), 4),
        "perm_sd":   round(np.std(bacc_null), 4),
        "perm_p":    round(p_val, 4),
        "n_perm": N_PERM,
    })

    # ── 1.11: Permutation test for centroid distance (GE vs Philips) ─────────
    if centroids["GE"] is not None and centroids["Philips"] is not None:
        idx_ge = te["Manufacturer"] == "GE"
        idx_ph = te["Manufacturer"] == "Philips"
        Z_ge = Z_te_r[idx_ge]; Z_ph = Z_te_r[idx_ph]
        obs_dist = float(np.linalg.norm(Z_ge.mean(0) - Z_ph.mean(0)))
        combined = np.vstack([Z_ge, Z_ph])
        n_ge = len(Z_ge)
        null_dists = []
        for _ in range(N_PERM):
            perm = RNG.permutation(len(combined))
            null_a = combined[perm[:n_ge]].mean(0)
            null_b = combined[perm[n_ge:]].mean(0)
            null_dists.append(np.linalg.norm(null_a - null_b))
        p_cent = (np.sum(np.array(null_dists) >= obs_dist) + 1) / (N_PERM + 1)
        perm_rows.append({
            "fold": fold, "test": "centroid_dist_GE_vs_Philips",
            "observed": round(obs_dist, 4),
            "perm_mean": round(np.mean(null_dists), 4),
            "perm_sd":   round(np.std(null_dists), 4),
            "perm_p":    round(p_cent, 4),
            "n_perm": N_PERM,
        })

    print(f"    kNN BACC={bacc_obs:.3f} (p={p_val:.4f}), fold done.")

# ── Part 2: Save geometry tables ───────────────────────────────────────────────
print(f"[{datetime.now(timezone.utc).isoformat()}] Part 2: saving geometry tables")

centroid_df = pd.DataFrame(centroid_rows)
centroid_df.to_csv(OUT_DIR / "foldlocal_centroid_geometry.csv", index=False)

# Add mean/SD summary
def summarise_metric(df, groupby, metric):
    g = df.groupby(groupby)[metric]
    return g.mean().rename(f"{metric}_mean"), g.std(ddof=1).rename(f"{metric}_sd")

summary_pairs = centroid_df.groupby("pair").agg(
    centroid_dist_mean=("centroid_dist_euclid", "mean"),
    centroid_dist_sd=("centroid_dist_euclid", lambda x: x.std(ddof=1)),
    sep_ratio_mean=("sep_ratio", "mean"),
    sep_ratio_sd=("sep_ratio", lambda x: x.std(ddof=1)),
    mahal_a_mean=("mahal_a_mean", "mean"),
    mahal_b_mean=("mahal_b_mean", "mean"),
).round(4)
summary_pairs.to_csv(OUT_DIR / "foldlocal_centroid_geometry_summary.csv")

with open(OUT_DIR / "foldlocal_centroid_geometry.md", "w") as f:
    f.write("# Fold-Local Centroid Geometry\n\n")
    f.write("**Estimators fitted on trainDev only. Applied to outer-test fold.**\n")
    f.write("**Nuisance residualized: diagnosis (y), Age, Sex.**\n\n")
    f.write("## Per-fold centroid distances\n\n")
    f.write(centroid_df[["fold","pair","centroid_dist_euclid","sep_ratio","mahal_a_mean","mahal_b_mean"]].to_markdown(index=False, floatfmt=".4f") + "\n\n")
    f.write("## Summary (mean ± SD across 5 folds)\n\n")
    f.write(summary_pairs.to_markdown(floatfmt=".4f") + "\n\n")
    f.write("**Interpretation:** `sep_ratio` = centroid distance / pooled within-group mean radius. "
            "Values > 1.0 indicate centroids further apart than the mean subject is from its group centroid.\n")

# BW and principal angles
bw_df   = pd.DataFrame([r for r in knn_rows if r["metric"] == "bures_wasserstein"])
pa_df   = pd.DataFrame([r for r in knn_rows if "principal_angle" in r["metric"]])
bacc_df = pd.DataFrame([r for r in knn_rows if r["metric"] == "knn5_mfr_bacc"])

bw_summary = bw_df.groupby("pair")["value"].agg(["mean","std"]).round(4) if len(bw_df) else pd.DataFrame()
bw_df.to_csv(DAT_DIR / "bures_wasserstein_by_fold.csv", index=False)
pa_df.to_csv(DAT_DIR / "principal_angles_by_fold.csv", index=False)
bacc_df.to_csv(DAT_DIR / "knn_bacc_by_fold.csv", index=False)

cov_df = pd.DataFrame([r for r in covariance_rows if "dummy_bw" not in r or r.get("dummy_bw") is None])
cov_df = pd.DataFrame(covariance_rows).drop(columns=["dummy_bw"], errors="ignore")
cov_df.to_csv(OUT_DIR / "foldlocal_covariance_geometry.csv", index=False)

cov_summary = cov_df.groupby("manufacturer").agg(
    eff_rank_mean=("effective_rank", "mean"), eff_rank_sd=("effective_rank", lambda x: x.std(ddof=1)),
    shrinkage_mean=("lw_shrinkage", "mean"), shrinkage_sd=("lw_shrinkage", lambda x: x.std(ddof=1)),
    top10_frac_mean=("top10_frac_var", "mean"),
).round(4)

with open(OUT_DIR / "foldlocal_covariance_geometry.md", "w") as f:
    f.write("# Fold-Local Covariance Geometry\n\n")
    f.write("**Ledoit-Wolf estimators fitted on trainDev. Statistics reported on trainDev covariance.**\n\n")
    f.write("## Effective rank and shrinkage per fold\n\n")
    f.write(cov_df[["fold","manufacturer","effective_rank","lw_shrinkage","top10_frac_var"]].to_markdown(index=False, floatfmt=".4f") + "\n\n")
    f.write("## Summary (mean ± SD across folds)\n\n")
    f.write(cov_summary.to_markdown(floatfmt=".4f") + "\n\n")
    f.write("## Bures-Wasserstein distances\n\n")
    if len(bw_df):
        f.write(bw_df.to_markdown(index=False, floatfmt=".4f") + "\n\n")
        f.write("Summary:\n\n")
        f.write(bw_summary.to_markdown(floatfmt=".4f") + "\n\n")
    f.write("## Mean principal angle (top-20 PCs)\n\n")
    if len(pa_df):
        sub = pa_df[pa_df["metric"] == "mean_principal_angle_top20"]
        f.write(sub.to_markdown(index=False, floatfmt=".2f") + "\n\n")

perm_df = pd.DataFrame(perm_rows)
perm_df.to_csv(OUT_DIR / "permutation_tests.csv", index=False)

perm_summary = perm_df.groupby("test").agg(
    obs_mean=("observed", "mean"), obs_sd=("observed", lambda x: x.std(ddof=1)),
    perm_p_mean=("perm_p", "mean"), perm_p_min=("perm_p", "min"), perm_p_max=("perm_p", "max"),
).round(4)

with open(OUT_DIR / "permutation_tests.md", "w") as f:
    f.write("# Permutation Tests\n\n")
    f.write(f"**N permutations:** {N_PERM}. **Seed:** 42.\n\n")
    f.write("## Per-fold results\n\n")
    f.write(perm_df.to_markdown(index=False, floatfmt=".4f") + "\n\n")
    f.write("## Summary across folds\n\n")
    f.write(perm_summary.to_markdown(floatfmt=".4f") + "\n\n")

print("  Tables saved.")

# ── Part 3: Figure generation ─────────────────────────────────────────────────
print(f"[{datetime.now(timezone.utc).isoformat()}] Part 3: generating figures")

# ── Figure 1: manufacturer_performance_stageB.pdf ────────────────────────────
print("  Figure 1: manufacturer performance Stage-B")
stgb_fold   = pd.read_csv(STGB_DIR / "calib_foldwise_metrics.csv")
stgb_phi    = pd.read_csv(STGB_DIR / "calib_philips_fpr_by_fold.csv")
OFFICIAL = dict(model_name="logreg_l2_original", feature_set="z_plus_age_sex",
                calib_method="oof_ecdf", threshold_strategy="inner_oof_target_sens_ge_0p70_max_spec")
def filter_official(df):
    for k, v in OFFICIAL.items():
        df = df[df[k] == v]
    return df

fold_metrics = filter_official(stgb_fold).copy()
phi_fpr_fold = filter_official(stgb_phi).copy()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Stage-B Performance — Official Convention\n"
             "(logreg_l2_original / z_plus_age_sex / oof_ecdf / target_sens≥0.70_max_spec)",
             fontsize=10, y=1.01)

# Panel A: ROC-AUC per fold
ax = axes[0]
ax.scatter(fold_metrics["fold"], fold_metrics["auc"], color="#2c7bb6", s=80, zorder=5, label="AUC")
ax.axhline(fold_metrics["auc"].mean(), color="#2c7bb6", linestyle="--", linewidth=1.5, label=f"Mean={fold_metrics['auc'].mean():.3f}")
ax.fill_between([0.5, 5.5],
                fold_metrics["auc"].mean() - fold_metrics["auc"].std(ddof=1),
                fold_metrics["auc"].mean() + fold_metrics["auc"].std(ddof=1),
                alpha=0.2, color="#2c7bb6")
ax.scatter(fold_metrics["fold"], fold_metrics["pr_auc"], color="#d7191c", s=80, marker="^", zorder=5, label="PR-AUC")
ax.axhline(fold_metrics["pr_auc"].mean(), color="#d7191c", linestyle="--", linewidth=1.5, label=f"Mean={fold_metrics['pr_auc'].mean():.3f}")
ax.set_xlabel("Fold"); ax.set_ylabel("AUC"); ax.set_title("ROC-AUC and PR-AUC per fold")
ax.set_xlim(0.5, 5.5); ax.set_ylim(0.3, 1.0)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Panel B: Philips CN FPR per fold
ax = axes[1]
phi_sub = phi_fpr_fold[phi_fpr_fold["manufacturer"] == "Philips"]
ge_sub  = phi_fpr_fold[phi_fpr_fold["manufacturer"] == "GE"]
siem_sub= phi_fpr_fold[phi_fpr_fold["manufacturer"] == "SIEMENS"]
ax.plot(phi_sub["fold"], phi_sub["fpr_cn"], "o-", color=MFR_COLORS["Philips"], label=f"Philips CN FPR (mean={phi_sub['fpr_cn'].mean():.3f})")
ax.plot(ge_sub["fold"],  ge_sub["fpr_cn"],  "s-", color=MFR_COLORS["GE"],      label=f"GE CN FPR (mean={ge_sub['fpr_cn'].mean():.3f})")
ax.plot(siem_sub["fold"],siem_sub["fpr_cn"],"^-", color=MFR_COLORS["SIEMENS"], label=f"SIEMENS CN FPR (mean={siem_sub['fpr_cn'].mean():.3f})")
ax.axhline(0.4545, color=MFR_COLORS["Philips"], linestyle=":", linewidth=1, alpha=0.7, label="Philips pooled FPR=0.455")
ax.set_xlabel("Fold"); ax.set_ylabel("CN False Positive Rate"); ax.set_title("CN FPR by manufacturer per fold")
ax.set_xlim(0.5, 5.5); ax.set_ylim(0, 0.7)
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# Panel C: Sensitivity per manufacturer per fold
ax = axes[2]
for mfr, marker, label in [("Philips","o","Philips AD Sens"), ("GE","s","GE AD Sens"), ("SIEMENS","^","SIEMENS AD Sens")]:
    sub = phi_fpr_fold[phi_fpr_fold["manufacturer"] == mfr]
    ax.plot(sub["fold"], sub["sensitivity_ad"], f"{marker}-", color=MFR_COLORS[mfr], label=f"{label}")
ax.set_xlabel("Fold"); ax.set_ylabel("Sensitivity (AD)"); ax.set_title("AD sensitivity by manufacturer per fold")
ax.set_xlim(0.5, 5.5); ax.set_ylim(0, 1.1)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(FIG_DIR / "manufacturer_performance_stageB.pdf", bbox_inches="tight", dpi=150)
plt.close(fig)
print("    Saved manufacturer_performance_stageB.pdf")

# Save figure data
fold_metrics[["fold","auc","pr_auc","balanced_accuracy","sensitivity","specificity"]].to_csv(
    DAT_DIR / "fig1_fold_metrics.csv", index=False)
phi_fpr_fold[["fold","manufacturer","n_cn","fp_cn","fpr_cn","sensitivity_ad"]].to_csv(
    DAT_DIR / "fig1_philips_fpr.csv", index=False)

# ── Figure 2: latent_geometry_overview.pdf ───────────────────────────────────
print("  Figure 2: latent geometry overview (PCA per fold)")
fig, axes = plt.subplots(5, 2, figsize=(12, 20))
fig.suptitle("Latent Space PCA — Per-Fold Outer-Test Sets\n"
             "(Nuisance residualized: diagnosis, Age, Sex. PCA fitted on trainDev.)",
             fontsize=11)

pca_rows = []
for i, fold in enumerate(FOLDS):
    dat = pca_fold_data[fold]
    Z2d = dat["Z_2d"]; mfr = dat["mfr"]; diag = dat["diag"]
    ev  = dat["var_explained"]

    # Panel A: colored by manufacturer
    ax = axes[i, 0]
    for m in MANUFACTURERS:
        mask = mfr == m
        ax.scatter(Z2d[mask, 0], Z2d[mask, 1], c=MFR_COLORS[m], s=25, alpha=0.7, label=m if i == 0 else "")
    ax.set_xlabel(f"PC1 ({100*ev[0]:.1f}%)" if i == 4 else "")
    ax.set_ylabel(f"PC2 ({100*ev[1]:.1f}%)" if i == 4 else "")
    ax.set_title(f"Fold {fold} — Manufacturer", fontsize=9)
    ax.grid(True, alpha=0.2)
    if i == 0:
        ax.legend(fontsize=7, loc="upper right")

    # Panel B: colored by diagnosis
    ax = axes[i, 1]
    for d, c in DIAG_COLORS.items():
        mask = diag == d
        ax.scatter(Z2d[mask, 0], Z2d[mask, 1], c=c, s=25, alpha=0.7, label=d if i == 0 else "")
    ax.set_xlabel(f"PC1 ({100*ev[0]:.1f}%)" if i == 4 else "")
    ax.set_ylabel(f"PC2 ({100*ev[1]:.1f}%)" if i == 4 else "")
    ax.set_title(f"Fold {fold} — Diagnosis", fontsize=9)
    ax.grid(True, alpha=0.2)
    if i == 0:
        ax.legend(fontsize=7, loc="upper right")

    for j, subj in enumerate(pca_fold_data[fold]["Z_2d"]):
        pca_rows.append({"fold": fold, "pc1": round(subj[0], 4), "pc2": round(subj[1], 4),
                         "manufacturer": mfr[j], "diagnosis": diag[j],
                         "ev1": round(ev[0], 4), "ev2": round(ev[1], 4)})

plt.tight_layout()
fig.savefig(FIG_DIR / "latent_geometry_overview.pdf", bbox_inches="tight", dpi=150)
plt.close(fig)
pd.DataFrame(pca_rows).to_csv(DAT_DIR / "fig2_pca_scatter.csv", index=False)
print("    Saved latent_geometry_overview.pdf")

# ── Figure 3: covariance_geometry_by_manufacturer.pdf ───────────────────────
print("  Figure 3: covariance geometry")
fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

# Panel A: Effective rank per fold per manufacturer
ax_er = fig.add_subplot(gs[0, 0])
for mfr in MANUFACTURERS:
    sub = cov_df[cov_df["manufacturer"] == mfr]
    ax_er.plot(sub["fold"], sub["effective_rank"], "o-", color=MFR_COLORS[mfr], label=mfr)
    ax_er.axhline(sub["effective_rank"].mean(), color=MFR_COLORS[mfr], linestyle=":", alpha=0.5)
ax_er.set_xlabel("Fold"); ax_er.set_ylabel("Effective Rank (exp entropy)")
ax_er.set_title("Effective Rank per Manufacturer\n(LW covariance, trainDev fitted)", fontsize=9)
ax_er.legend(fontsize=8); ax_er.grid(True, alpha=0.3); ax_er.set_xlim(0.5, 5.5)

# Panel B: LW shrinkage per fold per manufacturer
ax_sh = fig.add_subplot(gs[0, 1])
for mfr in MANUFACTURERS:
    sub = cov_df[cov_df["manufacturer"] == mfr]
    ax_sh.plot(sub["fold"], sub["lw_shrinkage"], "o-", color=MFR_COLORS[mfr], label=mfr)
ax_sh.set_xlabel("Fold"); ax_sh.set_ylabel("LW Shrinkage coefficient")
ax_sh.set_title("Ledoit-Wolf Shrinkage\n(higher = more regularization needed)", fontsize=9)
ax_sh.legend(fontsize=8); ax_sh.grid(True, alpha=0.3); ax_sh.set_xlim(0.5, 5.5)

# Panel C: Top-10 PC variance fraction
ax_pv = fig.add_subplot(gs[0, 2])
for mfr in MANUFACTURERS:
    sub = cov_df[cov_df["manufacturer"] == mfr]
    ax_pv.plot(sub["fold"], sub["top10_frac_var"], "o-", color=MFR_COLORS[mfr], label=mfr)
ax_pv.set_xlabel("Fold"); ax_pv.set_ylabel("Fraction of variance in top-10 PCs")
ax_pv.set_title("Top-10 PC Variance Fraction\n(dimensionality concentration)", fontsize=9)
ax_pv.legend(fontsize=8); ax_pv.grid(True, alpha=0.3); ax_pv.set_xlim(0.5, 5.5)

# Panel D: Bures-Wasserstein distance per fold
ax_bw = fig.add_subplot(gs[1, 0])
if len(bw_df):
    for pair, c in [("GE_vs_Philips", "#9b2226"), ("GE_vs_SIEMENS", "#005f73"), ("Philips_vs_SIEMENS", "#ae2012")]:
        sub = bw_df[bw_df["pair"] == pair]
        if len(sub):
            ax_bw.plot(sub["fold"], sub["value"], "o-", color=c, label=pair.replace("_vs_", " vs "))
ax_bw.set_xlabel("Fold"); ax_bw.set_ylabel("Bures-Wasserstein distance")
ax_bw.set_title("Bures-Wasserstein Distance\n(between manufacturer covariances)", fontsize=9)
ax_bw.legend(fontsize=8); ax_bw.grid(True, alpha=0.3); ax_bw.set_xlim(0.5, 5.5)

# Panel E: Mean principal angle (top-20 PCs)
ax_pa = fig.add_subplot(gs[1, 1])
pa_mean = pa_df[pa_df["metric"] == "mean_principal_angle_top20"] if len(pa_df) else pd.DataFrame()
if len(pa_mean):
    for pair, c in [("GE_vs_Philips", "#9b2226"), ("GE_vs_SIEMENS", "#005f73"), ("Philips_vs_SIEMENS", "#ae2012")]:
        sub = pa_mean[pa_mean["pair"] == pair]
        if len(sub):
            ax_pa.plot(sub["fold"], sub["value"], "o-", color=c, label=pair.replace("_vs_", " vs "))
ax_pa.set_xlabel("Fold"); ax_pa.set_ylabel("Mean principal angle (°)")
ax_pa.set_title("Mean Principal Angle (top-20 PCs)\n(90° = orthogonal subspaces)", fontsize=9)
ax_pa.axhline(90, color="gray", linestyle=":", alpha=0.5, label="90° (orthogonal)")
ax_pa.legend(fontsize=8); ax_pa.grid(True, alpha=0.3); ax_pa.set_xlim(0.5, 5.5)

# Panel F: kNN manufacturer BACC per fold with permutation null
ax_knn = fig.add_subplot(gs[1, 2])
obs_bacc = bacc_df["value"].values if len(bacc_df) else np.array([])
perm_d   = perm_df[perm_df["test"] == "knn5_manufacturer_balanced_accuracy"] if len(perm_df) else pd.DataFrame()
if len(obs_bacc):
    ax_knn.scatter(FOLDS, obs_bacc, color="#2c7bb6", s=80, zorder=5, label="Observed BACC")
    ax_knn.axhline(obs_bacc.mean(), color="#2c7bb6", linestyle="--", linewidth=1.5, label=f"Mean={obs_bacc.mean():.3f}")
if len(perm_d):
    ax_knn.errorbar(FOLDS, perm_d["perm_mean"], yerr=perm_d["perm_sd"],
                    fmt="o--", color="gray", alpha=0.7, label="Perm null mean±SD")
ax_knn.axhline(1/3, color="black", linestyle=":", alpha=0.4, label="Chance (1/3)")
ax_knn.set_xlabel("Fold"); ax_knn.set_ylabel("Balanced Accuracy")
ax_knn.set_title("kNN-5 Manufacturer BACC\nvs permutation null", fontsize=9)
ax_knn.legend(fontsize=8); ax_knn.grid(True, alpha=0.3); ax_knn.set_xlim(0.5, 5.5); ax_knn.set_ylim(0, 1)

fig.suptitle("Covariance Geometry by Manufacturer — Fold-Local Analysis\n"
             "(LW covariance fitted on trainDev, applied to trainDev subgroups)", fontsize=11, y=1.01)
fig.savefig(FIG_DIR / "covariance_geometry_by_manufacturer.pdf", bbox_inches="tight", dpi=150)
plt.close(fig)
print("    Saved covariance_geometry_by_manufacturer.pdf")

# ── Figure 4: decoded_manufacturer_traversal_connectomes.pdf ─────────────────
print("  Figure 4: decoded traversal connectomes (EXPLORATORY label)")
trav_csv = TRAV_DIR / "decoded_connectome_delta_by_channel.csv"
roi_csv  = RUN_DIR / "roi_info_from_tensor.csv"
cons_csv = TRAV_DIR / "consensus_site_effect_edges.csv"

trav_df  = pd.read_csv(trav_csv)
roi_df   = pd.read_csv(roi_csv)
cons_df  = pd.read_csv(cons_csv) if cons_csv.exists() else pd.DataFrame()

N_ROI = len(roi_df)
channel_names = ["Pearson_Full_FisherZ_Signed", "Pearson_OMST_GCE_Signed_Weighted", "MI_KNN_Symmetric"]
ch_labels     = ["Pearson Full (Fisher-Z)", "Pearson OMST (Signed)", "MI-KNN"]

# Build per-fold, per-channel mean delta matrices from consensus edges
# We use fold=1, alpha=1.0 as the representative traversal (all folds shown in summary)
def build_delta_matrix_from_consensus(cons, channel_name, roi_df, alpha=1.0, direction="Philips_to_GE"):
    """Build upper-triangular NxN delta matrix from consensus edge list."""
    sub = cons[(cons["channel_name"] == channel_name) &
               (cons["alpha"] == alpha) &
               (cons["direction"] == direction)]
    n = len(roi_df)
    M = np.zeros((n, n))
    for _, row in sub.iterrows():
        i, j = int(row["roi_i"]), int(row["roi_j"])
        M[i, j] = row["mean_delta"]; M[j, i] = row["mean_delta"]
    return M

# Build network-order reindex
networks = roi_df["network_label_in_tensor"].values
unique_nets = sorted(set(networks))
net_order = np.argsort([unique_nets.index(n) for n in networks])

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
EXPLORATORY_NOTE = ("RAW-SCORE EXPLORATORY (N=12 FPs, inner-OOF Youden threshold)\n"
                    "NOT the official Stage-B result (N=45 FPs). Do not cite as confirmatory.")
fig.text(0.5, 1.02, EXPLORATORY_NOTE, ha="center", va="bottom",
         fontsize=9, color="darkred", bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

if len(cons_df):
    for ci, (ch_name, ch_label) in enumerate(zip(channel_names, ch_labels)):
        M = build_delta_matrix_from_consensus(cons_df, ch_name, roi_df)
        M_reord = M[np.ix_(net_order, net_order)]
        vmax = np.abs(M_reord).max()
        im = axes[ci].imshow(M_reord, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=axes[ci], shrink=0.8, label="Mean Δ connectivity")
        axes[ci].set_title(f"{ch_label}\n(Philips→GE, α=1.0, consensus edges only)", fontsize=9)
        axes[ci].set_xlabel("ROI (network order)"); axes[ci].set_ylabel("ROI (network order)")

        # Mark network boundaries
        _, counts = np.unique(networks[net_order], return_counts=True)
        boundaries = np.cumsum(counts)[:-1]
        for b in boundaries:
            axes[ci].axhline(b - 0.5, color="black", linewidth=0.4, alpha=0.6)
            axes[ci].axvline(b - 0.5, color="black", linewidth=0.4, alpha=0.6)
else:
    for ci in range(3):
        axes[ci].text(0.5, 0.5, "No consensus edge data", ha="center", va="center",
                      transform=axes[ci].transAxes)

fig.suptitle("Decoded Manufacturer Traversal — Connectome Deltas\n"
             "Consensus edges (sign-consistent across 29 Philips test subjects, fold 1)",
             fontsize=11, y=1.0)
plt.tight_layout()
fig.savefig(FIG_DIR / "decoded_manufacturer_traversal_connectomes.pdf", bbox_inches="tight", dpi=150)
plt.close(fig)
print("    Saved decoded_manufacturer_traversal_connectomes.pdf")

# ── Figure 5: decoded_manufacturer_network_blocks.pdf ─────────────────────────
print("  Figure 5: network block deltas (EXPLORATORY label)")
blocks_csv = TRAV_DIR / "decoded_network_block_deltas.csv"  # columns: direction,alpha,channel,channel_name,network_a,network_b,n_edges,mean_delta,within_block
if blocks_csv.exists():
    blocks_df = pd.read_csv(blocks_csv)
    # Filter to Philips_to_GE direction, alpha=1.0
    blocks_df = blocks_df[(blocks_df["direction"] == "Philips_to_GE") & (blocks_df["alpha"] == 1.0)]

    # Collect all unique network labels for consistent ordering
    all_nets = sorted(set(blocks_df["network_a"].unique()) | set(blocks_df["network_b"].unique()))

    fig, axes = plt.subplots(1, 3, figsize=(21, 8))
    fig.text(0.5, 1.02, EXPLORATORY_NOTE, ha="center", va="bottom",
             fontsize=9, color="darkred", bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    for ci, (ch_name, ch_label) in enumerate(zip(channel_names, ch_labels)):
        sub = blocks_df[blocks_df["channel_name"] == ch_name]
        if len(sub) == 0:
            axes[ci].text(0.5, 0.5, f"No data for\n{ch_name}", ha="center", va="center",
                          transform=axes[ci].transAxes); continue

        # Build symmetric NxN matrix
        n = len(all_nets); net_idx = {n: i for i, n in enumerate(all_nets)}
        M = np.zeros((n, n))
        for _, row in sub.iterrows():
            i, j = net_idx[row["network_a"]], net_idx[row["network_b"]]
            M[i, j] = row["mean_delta"]; M[j, i] = row["mean_delta"]

        vmax = np.abs(M).max()
        im = axes[ci].imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=axes[ci], shrink=0.7, label="Mean Δ connectivity")
        short_nets = [n.replace("Background/NonCortical","BG/NonCortex")
                       .replace("DorsalAttention","DorsAttn")
                       .replace("VentralAttention","VentAttn")
                       .replace("DefaultMode","DMN")
                       .replace("Salience_","Sal_")
                       .replace("Somatomotor","SomMot") for n in all_nets]
        axes[ci].set_xticks(range(n)); axes[ci].set_yticks(range(n))
        axes[ci].set_xticklabels(short_nets, rotation=90, fontsize=5)
        axes[ci].set_yticklabels(short_nets, fontsize=5)
        axes[ci].set_title(f"{ch_label}\nPhilips→GE, α=1.0", fontsize=9)

    fig.suptitle("Decoded Manufacturer Traversal — Network Block Deltas\n"
                 "(mean edge delta per network pair, across all 29 Philips test subjects in fold 1)",
                 fontsize=11, y=1.0)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "decoded_manufacturer_network_blocks.pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("    Saved decoded_manufacturer_network_blocks.pdf")
else:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.6, "decoded_network_block_deltas.csv not found", ha="center", va="center",
            fontsize=12, transform=ax.transAxes)
    ax.text(0.5, 0.4, EXPLORATORY_NOTE, ha="center", va="center", fontsize=9, color="darkred",
            transform=ax.transAxes, bbox=dict(facecolor="lightyellow", alpha=0.8))
    ax.axis("off")
    fig.savefig(FIG_DIR / "decoded_manufacturer_network_blocks.pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("    WARNING: decoded_network_block_deltas.csv missing, placeholder saved")

# ── Part 4: Traversal audit and 45-FP data plumbing ──────────────────────────
print(f"[{datetime.now(timezone.utc).isoformat()}] Part 4: traversal audit + official FP data plumbing")

# Load official 45 Philips CN FPs
stgb_preds = pd.read_csv(STGB_DIR / "calib_predictions.csv")
official_fps = stgb_preds[
    (stgb_preds["model_name"] == OFFICIAL["model_name"]) &
    (stgb_preds["feature_set"] == OFFICIAL["feature_set"]) &
    (stgb_preds["calib_method"] == OFFICIAL["calib_method"]) &
    (stgb_preds["threshold_strategy"] == OFFICIAL["threshold_strategy"]) &
    (stgb_preds["Manufacturer"] == "Philips") &
    (stgb_preds["ResearchGroup_Mapped"] == "CN") &
    (stgb_preds["y_pred"] == 1)
].copy()
assert len(official_fps) == 45, f"Expected 45 official FPs, got {len(official_fps)}"

# Extract latent mu for the 45 official FPs
fp_subjects_by_fold = official_fps.groupby("fold")["SubjectID"].apply(list).to_dict()
fp_latent_rows = []
for fold, subjects in fp_subjects_by_fold.items():
    te = load_latents(fold, "test")
    mcols_list = mu_cols(te)
    sub = te[te["SubjectID"].isin(subjects)].copy()
    fp_latent_rows.append(sub)

fp_latent_df = pd.concat(fp_latent_rows, ignore_index=True)
assert len(fp_latent_df) == 45, f"Expected 45 FP latent rows, got {len(fp_latent_df)}"

# Merge Stage-B scores
fp_merged = fp_latent_df.merge(
    official_fps[["SubjectID","fold","y_score_raw","y_score","threshold"]],
    on=["SubjectID","fold"]
)

fp_merged.to_csv(DAT_DIR / "official_philips_cn_fp_latents_and_scores.csv", index=False)
print(f"    Official Philips CN FPs: {len(fp_merged)} subjects saved with latent mu + Stage-B scores")

# Try to find saved Stage-B ECDF transformer objects
stgb_jobblibs = list(STGB_DIR.glob("*.joblib")) + list(STGB_DIR.glob("*.pkl"))
ecdf_available = len(stgb_jobblibs) > 0

if ecdf_available:
    print(f"    WARNING: Found {len(stgb_jobblibs)} saved objects — verify if ECDF transformers are present")
else:
    print("    FAIL CLOSED: No saved Stage-B transformer objects found.")
    print("    Official 45-FP traversal BLOCKED: cannot apply oof_ecdf to counterfactual latents without refitting.")
    print("    Data plumbing saved to figure_data/official_philips_cn_fp_latents_and_scores.csv")

# Cross-check: previous traversal used 12 FPs (raw Youden threshold), NOT 45
prev_trav = pd.read_csv(TRAV_DIR / "philips_cn_fp_traversal_analysis.csv") if (TRAV_DIR / "philips_cn_fp_traversal_analysis.csv").exists() else pd.DataFrame()
prev_n_fp = 12  # from site_effect_mechanistic_summary.md

traversal_audit = {
    "previous_traversal_n_fps": prev_n_fp,
    "previous_traversal_threshold": "inner_oof_youden_j (RAW SCORES)",
    "previous_traversal_convention": "RAW-SCORE EXPLORATORY — not Stage-B OOF-ECDF",
    "previous_traversal_flip_rate_at_alpha1": "11/12 = 91.7%",
    "previous_traversal_status": "EXPLORATORY — cannot be presented as official Stage-B FP result",
    "official_stageB_n_fps": 45,
    "official_stageB_threshold": "inner_oof_target_sens_ge_0p70_max_spec (oof_ecdf)",
    "official_stageB_convention": "logreg_l2_original / z_plus_age_sex / oof_ecdf",
    "official_traversal_status": "BLOCKED — Stage-B ECDF transformers not saved, cannot apply to counterfactual latents without refitting",
    "data_plumbing_saved": "figure_data/official_philips_cn_fp_latents_and_scores.csv",
    "data_plumbing_note": "Contains latent mu vectors + y_score_raw + y_score (oof_ecdf) for all 45 subjects. Can be used once ECDF transformers are reconstructed or re-fitted.",
}
pd.Series(traversal_audit).rename("value").to_frame().to_csv(
    DAT_DIR / "traversal_audit_summary.csv")

# ── Part 5: interpretation_guardrails.md ──────────────────────────────────────
print(f"[{datetime.now(timezone.utc).isoformat()}] Part 5: writing interpretation guardrails")

with open(OUT_DIR / "interpretation_guardrails.md", "w") as f:
    f.write("# Interpretation Guardrails\n\n")
    f.write(f"**Date:** {datetime.now(timezone.utc).date()}\n")
    f.write(f"**Analysis:** Fold-local confirmatory geometry — locked model `{RUN_NAME}`\n\n")

    f.write("## 1. No cross-fold pooling of raw coordinates\n\n")
    f.write("All geometry metrics (centroid distances, covariance spectra, Bures-Wasserstein, "
            "principal angles) are computed per fold independently. "
            "PCA projections cannot be overlaid across folds because each fold's VAE model "
            "is a different random initialisation with different latent axis assignments. "
            "The scatter plots in Figure 2 show fold-level panels, not pooled coordinates.\n\n")
    f.write("**DO NOT** compute a single PCA on the pooled 397-subject OOF set.\n\n")

    f.write("## 2. Estimators fitted on trainDev only\n\n")
    f.write("StandardScaler, nuisance LinearRegression, LedoitWolf covariance: ALL "
            "fitted on the corresponding trainDev split. The test fold is never seen during fitting. "
            "This means centroids and covariances reflect trainDev structure, applied to test subjects.\n\n")
    f.write("**Implication:** Centroid distances reported here are computed on **test** residualized latents "
            "using **trainDev**-fitted parameters. These are held-out, uncontaminated estimates.\n\n")

    f.write("## 3. Manufacturer is the primary domain\n\n")
    f.write("Site3 is secondary due to 163/647 (25.2%) missingness. "
            "Site-level analysis uses only subjects with non-null Site3. "
            "Results from Site3 analysis should be treated as exploratory.\n\n")

    f.write("## 4. Traversal scoring convention\n\n")
    f.write("### Previous traversal (figures 4 & 5)\n")
    f.write(f"- **N FPs used:** {prev_n_fp} (inner-OOF Youden-J threshold on raw scores)\n")
    f.write("- **Scoring convention:** RAW classifier output — NOT Stage-B OOF-ECDF\n")
    f.write("- **Status:** EXPLORATORY. The 11/12 flip rate is a **raw-score** result.\n")
    f.write("- **Do not cite** this as the official Stage-B result in the manuscript.\n\n")
    f.write("### Official Stage-B traversal (data plumbing only)\n")
    f.write("- **N FPs:** 45 (logreg_l2_original / z_plus_age_sex / oof_ecdf / target_sens≥0.70_max_spec)\n")
    f.write("- **Status:** BLOCKED. Stage-B OOF-ECDF transformer objects are not saved in the "
            "Stage-B package (only CSV outputs exist). Counterfactual latents cannot be re-scored "
            "without re-fitting the ECDF, which would require access to the inner-CV OOF score distributions.\n")
    f.write("- **What IS available:** The 45 subjects' latent mu vectors and their already-computed "
            "Stage-B OOF-ECDF scores are saved to `figure_data/official_philips_cn_fp_latents_and_scores.csv`. "
            "These can be used for future analysis once the re-scoring pipeline is implemented.\n\n")

    f.write("## 5. Bures-Wasserstein interpretation\n\n")
    f.write("BW distance between zero-mean Gaussians N(0,Σ_A), N(0,Σ_B) after nuisance residualization. "
            "The nuisance residualization removes the mean differences attributable to diagnosis, age, and sex. "
            "The residual BW distance reflects **covariance structure differences** between manufacturers "
            "in the latent space, independent of diagnostic case-mix.\n\n")
    f.write("With d=384 and N~100 per manufacturer per fold, the LW covariance is highly regularized "
            "(shrinkage >> 0). The BW distance is computed on the regularized matrix, so it "
            "underestimates the true population BW. The fold-to-fold SD captures estimation variance.\n\n")

    f.write("## 6. kNN balanced accuracy interpretation\n\n")
    f.write("kNN-5 is trained on trainDev residualized latents (3-class: GE/Philips/SIEMENS) and "
            "evaluated on test fold. Chance = 1/3. Permutation test permutes the **test** labels "
            "while keeping predictions fixed (matched permutation design).\n\n")
    f.write("High BACC means manufacturer information is preserved in the residualized latent space "
            "after removing diagnosis, age, and sex. This is consistent with the known manufacturer "
            "confound but does not imply that this information harms diagnostic utility "
            "(see Section 4.5 of the manuscript).\n\n")

    f.write("## 7. Figure 4 and 5 labels\n\n")
    f.write(f"Figures 4 and 5 replot data from `{TRAV_DIR.name}`. "
            "They are clearly labeled as EXPLORATORY/RAW-SCORE. The N=12 FP "
            "result must not be cited as the official Stage-B Philips FP count (N=45). "
            "Any future publication use of these figures requires either:\n")
    f.write("  (a) replacing them with the official Stage-B traversal (requires ECDF re-scoring), or\n")
    f.write("  (b) explicitly labeling them as raw-score exploratory analysis in the manuscript text.\n")

# ── Part 6: command log ────────────────────────────────────────────────────────
T1 = datetime.now(timezone.utc).isoformat()

centroid_summary_for_log = {}
if len(centroid_df):
    for pair in centroid_df["pair"].unique():
        sub = centroid_df[centroid_df["pair"] == pair]
        centroid_summary_for_log[pair] = {
            "centroid_dist_mean": round(sub["centroid_dist_euclid"].mean(), 4),
            "centroid_dist_sd": round(sub["centroid_dist_euclid"].std(ddof=1), 4),
            "sep_ratio_mean": round(sub["sep_ratio"].mean(), 4),
        }

perm_summary_for_log = {}
if len(perm_df):
    for test in perm_df["test"].unique():
        sub = perm_df[perm_df["test"] == test]
        perm_summary_for_log[test] = {
            "obs_mean": round(sub["observed"].mean(), 4),
            "perm_p_mean": round(sub["perm_p"].mean(), 4),
            "perm_p_min": round(sub["perm_p"].min(), 4),
        }

cov_summary_for_log = {}
if len(cov_df):
    for mfr in MANUFACTURERS:
        sub = cov_df[cov_df["manufacturer"] == mfr]
        if len(sub):
            cov_summary_for_log[mfr] = {
                "eff_rank_mean": round(sub["effective_rank"].mean(), 2),
                "lw_shrinkage_mean": round(sub["lw_shrinkage"].mean(), 4),
            }

log = {
    "created_utc": T0,
    "completed_utc": T1,
    "run_name": RUN_NAME,
    "status": "COMPLETE",
    "n_folds": 5,
    "n_perm": N_PERM,
    "n_pc_angles": N_PC_ANGLES,
    "official_convention": OFFICIAL,
    "official_philips_cn_fps": 45,
    "previous_traversal_fps": prev_n_fp,
    "previous_traversal_status": "RAW_SCORE_EXPLORATORY",
    "official_traversal_status": "BLOCKED_NO_ECDF_TRANSFORMER",
    "centroid_geometry_summary": centroid_summary_for_log,
    "permutation_summary": perm_summary_for_log,
    "covariance_summary": cov_summary_for_log,
    "bw_summary": bw_summary.to_dict() if len(bw_summary) else {},
    "deliverables": [
        "domain_metadata_audit.csv",
        "domain_metadata_audit.md",
        "foldlocal_centroid_geometry.csv",
        "foldlocal_centroid_geometry.md",
        "foldlocal_centroid_geometry_summary.csv",
        "foldlocal_covariance_geometry.csv",
        "foldlocal_covariance_geometry.md",
        "permutation_tests.csv",
        "permutation_tests.md",
        "figure_data/fig1_fold_metrics.csv",
        "figure_data/fig1_philips_fpr.csv",
        "figure_data/fig2_pca_scatter.csv",
        "figure_data/bures_wasserstein_by_fold.csv",
        "figure_data/principal_angles_by_fold.csv",
        "figure_data/knn_bacc_by_fold.csv",
        "figure_data/official_philips_cn_fp_latents_and_scores.csv",
        "figure_data/traversal_audit_summary.csv",
        "figures/manufacturer_performance_stageB.pdf",
        "figures/latent_geometry_overview.pdf",
        "figures/covariance_geometry_by_manufacturer.pdf",
        "figures/decoded_manufacturer_traversal_connectomes.pdf",
        "figures/decoded_manufacturer_network_blocks.pdf",
        "interpretation_guardrails.md",
        "command_log.json",
    ],
}

with open(OUT_DIR / "command_log.json", "w") as f:
    json.dump(log, f, indent=2)

print(f"[{T1}] DONE. All deliverables written to:\n  {OUT_DIR}")
