#!/usr/bin/env python3
"""
Wasserstein distance analysis — locked model latent space
=========================================================
Model   : recover035_latent384_beta3p75_T80_h10000_p560_full5x5
Request : Martín Belzunce (WhatsApp 2026-07-05)
Branch  : exploratory/post-revision-20260630

Fold-local discipline (CRITICAL):
  All estimators fitted on trainDev split of the given fold.
  Raw latent vectors NEVER pooled across folds.
  Results aggregated as mean ± SD across 5 folds.

OASIS note: all 180 OASIS subjects are SIEMENS-scanned.
            Encoder used per fold: fold k's own encoder (existing cached latents).

Wasserstein implementation:
  - 1D W1 : scipy.stats.wasserstein_distance, per latent dimension (384 dims)
  - PCA-W1 : PCA top-10 fitted on trainDev → W1 on each PC → mean (multivariate summary)
  - Gaussian-W2 : closed-form W2 between Gaussians N(μ_A, Σ_A) and N(μ_B, Σ_B)
                  using Bures-Wasserstein formula on LW covariances
  (POT/Sinkhorn not installed; PCA-W1 and Gaussian-W2 are the tractable alternatives)

Tasks:
  1. ADNI manufacturer pairs (GE-Philips, GE-SIEMENS, Philips-SIEMENS)
     on test-fold residualized latents (nuisance: y, Age, Sex)
  2. ADNI trainDev vs OASIS (180 subjects encoded by same fold's encoder)
     residualized by y only (diagnosis; removes AD/CN case-mix without
     over-correcting for cohort-specific age/sex distributions)
  3. Permutation nulls (1000 shuffles, per-fold) for both comparisons
     metric: PCA-W1-mean (10 components)
  4. Top-contributing dimensions to ADNI-vs-OASIS gap (per-dim W1 ranking)
     Note: decoder-direction mapping is follow-up (decoder weights not loaded here)
"""

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from scipy.stats import wasserstein_distance
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ADNI_LATENT_CACHE = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026"
    "/recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
    "/classifier_only_readout/latent_cache"
)
OASIS_LATENT_FILE = Path(
    "/home/diego/proyectos/vae_AD/results/revision_bspc_2026"
    "/promoted_latent384_oasis_vs_adni_latent_distance_audit_20260604"
    "/oasis_fold_latent_mu_runwise164.csv"
)
OUTPUT_DIR = Path(__file__).resolve().parent
N_PERM = 1000
N_PCA = 10
FOLDS = [1, 2, 3, 4, 5]
MANUFACTURER_PAIRS = [("GE", "Philips"), ("GE", "SIEMENS"), ("Philips", "SIEMENS")]
SEED = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
MU_COLS = [f"mu_{i}" for i in range(384)]


def load_adni(fold: int, split: str) -> pd.DataFrame:
    path = ADNI_LATENT_CACHE / f"fold_{fold}_{split}_latent_mu.csv"
    return pd.read_csv(path)


def sex_to_binary(s: pd.Series) -> np.ndarray:
    """Convert 'M'/'F' strings to 0/1 (M=0, F=1)."""
    return (s == "F").astype(float).values


def make_nuisance_matrix(df: pd.DataFrame, age_mean: float, age_std: float) -> np.ndarray:
    """Build [y, age_z, sex_binary] nuisance matrix."""
    sex_b = sex_to_binary(df["Sex"])
    age_z = (df["Age"].values - age_mean) / age_std
    y = df["y"].values.astype(float)
    return np.column_stack([y, age_z, sex_b])


def residualize(Z_train: np.ndarray, N_train: np.ndarray,
                Z_test: np.ndarray, N_test: np.ndarray) -> tuple:
    """OLS residualize Z by N, fitted on train, applied to test."""
    reg = LinearRegression(fit_intercept=True)
    reg.fit(N_train, Z_train)
    return Z_train - reg.predict(N_train), Z_test - reg.predict(N_test)


def residualize_single(Z_ref: np.ndarray, N_ref: np.ndarray,
                       Z_ext: np.ndarray, N_ext: np.ndarray) -> tuple:
    """Residualize ref and external set; OLS fitted on ref."""
    reg = LinearRegression(fit_intercept=True)
    reg.fit(N_ref, Z_ref)
    return Z_ref - reg.predict(N_ref), Z_ext - reg.predict(N_ext)


def w1_per_dim(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.array([wasserstein_distance(A[:, d], B[:, d]) for d in range(A.shape[1])])


def pca_w1_summary(Z_A: np.ndarray, Z_B: np.ndarray, pca: PCA) -> tuple:
    """Project to top-K PCs; compute W1 per PC and return mean + array."""
    pA = pca.transform(Z_A)
    pB = pca.transform(Z_B)
    w1s = np.array([wasserstein_distance(pA[:, k], pB[:, k]) for k in range(pca.n_components)])
    return float(w1s.mean()), w1s


def gaussian_w2(muA: np.ndarray, SA: np.ndarray,
                muB: np.ndarray, SB: np.ndarray) -> float:
    """Closed-form W2 between two Gaussians (Bures-Wasserstein + mean term)."""
    sqrtSA = sqrtm(SA).real
    M = sqrtSA @ SB @ sqrtSA
    sqrtM = sqrtm(M).real
    mean_sq = float(np.dot(muA - muB, muA - muB))
    bw2 = float(np.trace(SA) + np.trace(SB) - 2.0 * np.trace(sqrtM))
    return float(np.sqrt(max(mean_sq + bw2, 0.0)))


def lw_cov(Z: np.ndarray) -> tuple:
    """Fit LedoitWolf; return (covariance, mean, shrinkage)."""
    lw = LedoitWolf(store_precision=False)
    lw.fit(Z)
    return lw.covariance_, lw.location_, float(lw.shrinkage_)


# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
print("[preflight] Checking ADNI latent cache …", flush=True)
for fold in FOLDS:
    for split in ("trainDev", "test"):
        p = ADNI_LATENT_CACHE / f"fold_{fold}_{split}_latent_mu.csv"
        assert p.exists(), f"MISSING: {p}"
print(f"  ✓ ADNI latent cache: {2*len(FOLDS)} files confirmed")

print("[preflight] Checking OASIS latent file …", flush=True)
assert OASIS_LATENT_FILE.exists(), f"MISSING: {OASIS_LATENT_FILE}"
oasis_df_all = pd.read_csv(OASIS_LATENT_FILE)
assert set(oasis_df_all["fold"].unique()) == set(FOLDS), "OASIS fold mismatch"
oasis_manuf = oasis_df_all["Manufacturer"].unique()
print(f"  ✓ OASIS: {len(oasis_df_all)} rows, folds {sorted(oasis_df_all['fold'].unique())}")
print(f"  ✓ OASIS manufacturer(s): {list(oasis_manuf)} (expected: all SIEMENS)")

rng = np.random.default_rng(SEED)

# ===========================================================================
# Task 1 — ADNI manufacturer Wasserstein (per fold, test split)
# ===========================================================================
print("\n[task1] ADNI manufacturer Wasserstein (test split, 5 folds) …", flush=True)

mfr_rows = []
pca_mfr_rows = []  # per-PC per-fold per-pair

for fold in FOLDS:
    td = load_adni(fold, "trainDev")
    te = load_adni(fold, "test")
    Z_td = td[MU_COLS].values
    Z_te = te[MU_COLS].values

    # Scaler fitted on trainDev
    scaler = StandardScaler()
    Z_td_s = scaler.fit_transform(Z_td)
    Z_te_s = scaler.transform(Z_te)

    # Nuisance residualization (y, Age_z, Sex) on trainDev, apply to test
    age_mean, age_std = td["Age"].mean(), td["Age"].std()
    N_td = make_nuisance_matrix(td, age_mean, age_std)
    N_te = make_nuisance_matrix(te, age_mean, age_std)
    Z_td_r, Z_te_r = residualize(Z_td_s, N_td, Z_te_s, N_te)

    # PCA on residualized trainDev
    pca = PCA(n_components=N_PCA, random_state=SEED)
    pca.fit(Z_td_r)

    # Per-manufacturer subsets (test fold)
    mfr_map = {m: Z_te_r[te["Manufacturer"].values == m] for m in ["GE", "Philips", "SIEMENS"]}
    mfr_counts = {m: int((te["Manufacturer"].values == m).sum()) for m in ["GE", "Philips", "SIEMENS"]}

    for mA, mB in MANUFACTURER_PAIRS:
        ZA, ZB = mfr_map[mA], mfr_map[mB]
        nA, nB = len(ZA), len(ZB)
        if nA < 3 or nB < 3:
            print(f"  WARNING fold {fold} {mA}-{mB}: too few subjects ({nA},{nB})")
            continue

        # 1D W1 (all 384 dims)
        w1_dims = w1_per_dim(ZA, ZB)

        # PCA-W1 summary
        pca_mean, pca_w1s = pca_w1_summary(ZA, ZB, pca)
        for pc_idx, w1_pc in enumerate(pca_w1s):
            pca_mfr_rows.append(dict(fold=fold, pair=f"{mA}_vs_{mB}",
                                     pc=pc_idx+1, w1_pc=w1_pc))

        # Gaussian W2 on LW covariances
        SA, muA, shrA = lw_cov(ZA)
        SB, muB, shrB = lw_cov(ZB)
        gw2 = gaussian_w2(muA, SA, muB, SB)

        mfr_rows.append(dict(
            fold=fold,
            pair=f"{mA}_vs_{mB}",
            n_A=nA, n_B=nB,
            manuf_A=mA, manuf_B=mB,
            w1_mean_all_dims=float(w1_dims.mean()),
            w1_max_all_dims=float(w1_dims.max()),
            w1_median_all_dims=float(np.median(w1_dims)),
            pca10_w1_mean=pca_mean,
            gaussian_w2=gw2,
            lw_shrinkage_A=shrA,
            lw_shrinkage_B=shrB,
        ))

    print(f"  fold {fold}: n_GE={mfr_counts['GE']}, n_Philips={mfr_counts['Philips']}, "
          f"n_SIEMENS={mfr_counts['SIEMENS']}", flush=True)

mfr_df = pd.DataFrame(mfr_rows)
mfr_df.to_csv(OUTPUT_DIR / "wasserstein_manufacturer_by_fold.csv", index=False)
print(f"  ✓ wasserstein_manufacturer_by_fold.csv ({len(mfr_df)} rows)")

pca_mfr_df = pd.DataFrame(pca_mfr_rows)
pca_mfr_df.to_csv(OUTPUT_DIR / "wasserstein_manufacturer_pca_by_pc_fold.csv", index=False)

# ===========================================================================
# Task 2 — ADNI trainDev vs OASIS (per fold)
# ===========================================================================
print("\n[task2] ADNI trainDev vs OASIS Wasserstein (5 folds) …", flush=True)

oasis_rows = []
pca_oasis_rows = []
top_dim_rows = []

for fold in FOLDS:
    td = load_adni(fold, "trainDev")
    Z_td = td[MU_COLS].values

    oasis_fold = oasis_df_all[oasis_df_all["fold"] == fold].copy()
    Z_oa = oasis_fold[MU_COLS].values

    n_adni = len(Z_td)
    n_oasis = len(Z_oa)

    # Scaler fitted on ADNI trainDev
    scaler = StandardScaler()
    Z_td_s = scaler.fit_transform(Z_td)
    Z_oa_s = scaler.transform(Z_oa)

    # Residualize by y only (remove AD/CN case-mix; keep cohort-related structure)
    reg_y = LinearRegression(fit_intercept=True)
    reg_y.fit(td["y"].values.reshape(-1, 1), Z_td_s)
    Z_td_r = Z_td_s - reg_y.predict(td["y"].values.reshape(-1, 1))
    Z_oa_r = Z_oa_s - reg_y.predict(oasis_fold["y"].values.reshape(-1, 1))

    # PCA fitted on ADNI trainDev residualized
    pca = PCA(n_components=N_PCA, random_state=SEED)
    pca.fit(Z_td_r)

    # 1D W1 (all 384 dims)
    w1_dims = w1_per_dim(Z_td_r, Z_oa_r)

    # PCA-W1 summary
    pca_mean, pca_w1s = pca_w1_summary(Z_td_r, Z_oa_r, pca)
    for pc_idx, w1_pc in enumerate(pca_w1s):
        pca_oasis_rows.append(dict(fold=fold, comparison="ADNI_trainDev_vs_OASIS",
                                   pc=pc_idx+1, w1_pc=w1_pc))

    # Gaussian W2
    SA, muA, shrA = lw_cov(Z_td_r)
    SB, muB, shrB = lw_cov(Z_oa_r)
    gw2 = gaussian_w2(muA, SA, muB, SB)

    # Top-20 contributing dimensions (highest per-dim W1)
    top20_idx = np.argsort(w1_dims)[::-1][:20]
    for rank, dim_idx in enumerate(top20_idx):
        top_dim_rows.append(dict(fold=fold, rank=rank+1,
                                 latent_dim=int(dim_idx),
                                 w1=float(w1_dims[dim_idx])))

    oasis_rows.append(dict(
        fold=fold,
        comparison="ADNI_trainDev_vs_OASIS",
        n_adni_traindev=n_adni,
        n_oasis=n_oasis,
        oasis_manufacturer="SIEMENS",
        residualization="y_only",
        w1_mean_all_dims=float(w1_dims.mean()),
        w1_max_all_dims=float(w1_dims.max()),
        w1_median_all_dims=float(np.median(w1_dims)),
        pca10_w1_mean=pca_mean,
        gaussian_w2=gw2,
        lw_shrinkage_adni=shrA,
        lw_shrinkage_oasis=shrB,
    ))

    print(f"  fold {fold}: n_ADNI={n_adni}, n_OASIS={n_oasis}, "
          f"pca10_w1_mean={pca_mean:.4f}, gaussian_w2={gw2:.4f}", flush=True)

oasis_df_out = pd.DataFrame(oasis_rows)
oasis_df_out.to_csv(OUTPUT_DIR / "wasserstein_adni_vs_oasis_by_fold.csv", index=False)
print(f"  ✓ wasserstein_adni_vs_oasis_by_fold.csv ({len(oasis_df_out)} rows)")

pca_oasis_df = pd.DataFrame(pca_oasis_rows)
pca_oasis_df.to_csv(OUTPUT_DIR / "wasserstein_adni_vs_oasis_pca_by_pc_fold.csv", index=False)

top_dim_df = pd.DataFrame(top_dim_rows)
top_dim_df.to_csv(OUTPUT_DIR / "wasserstein_top_contributing_dims_adni_vs_oasis.csv", index=False)
print(f"  ✓ wasserstein_top_contributing_dims_adni_vs_oasis.csv ({len(top_dim_df)} rows)")

# ===========================================================================
# Task 3 — Permutation nulls (1000 shuffles, per fold)
# ===========================================================================
print(f"\n[task3] Permutation test ({N_PERM} shuffles) …", flush=True)

perm_rows = []

for fold in FOLDS:
    print(f"  fold {fold} …", flush=True)
    td = load_adni(fold, "trainDev")
    te = load_adni(fold, "test")
    Z_td = td[MU_COLS].values
    Z_te = te[MU_COLS].values

    # Scaler + residualization (consistent with Task 1)
    scaler = StandardScaler()
    Z_td_s = scaler.fit_transform(Z_td)
    Z_te_s = scaler.transform(Z_te)
    age_mean, age_std = td["Age"].mean(), td["Age"].std()
    N_td = make_nuisance_matrix(td, age_mean, age_std)
    N_te = make_nuisance_matrix(te, age_mean, age_std)
    Z_td_r, Z_te_r = residualize(Z_td_s, N_td, Z_te_s, N_te)
    pca = PCA(n_components=N_PCA, random_state=SEED)
    pca.fit(Z_td_r)

    # --- Manufacturer permutation (GE vs Philips as primary pair) ---
    te_labels = te["Manufacturer"].values.copy()
    ge_ph_mask = np.isin(te_labels, ["GE", "Philips"])
    Z_geph = Z_te_r[ge_ph_mask]
    labels_geph = te_labels[ge_ph_mask]

    obs_ge_ph, _ = pca_w1_summary(Z_te_r[te_labels == "GE"],
                                   Z_te_r[te_labels == "Philips"], pca)

    null_ge_ph = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_labels = rng.permutation(labels_geph)
        zA = Z_geph[perm_labels == "GE"]
        zB = Z_geph[perm_labels == "Philips"]
        if len(zA) < 2 or len(zB) < 2:
            null_ge_ph[i] = np.nan
            continue
        null_ge_ph[i], _ = pca_w1_summary(zA, zB, pca)

    p_ge_ph = float((null_ge_ph[~np.isnan(null_ge_ph)] >= obs_ge_ph).mean())
    if p_ge_ph == 0.0:
        p_ge_ph = 1.0 / N_PERM

    # --- ADNI vs OASIS permutation ---
    oasis_fold = oasis_df_all[oasis_df_all["fold"] == fold].copy()
    Z_oa = oasis_fold[MU_COLS].values
    Z_oa_s = scaler.transform(Z_oa)

    # Residualize OASIS by y using ADNI trainDev model
    reg_y = LinearRegression(fit_intercept=True)
    reg_y.fit(td["y"].values.reshape(-1, 1), Z_td_s)
    Z_td_ry = Z_td_s - reg_y.predict(td["y"].values.reshape(-1, 1))
    Z_oa_ry = Z_oa_s - reg_y.predict(oasis_fold["y"].values.reshape(-1, 1))

    pca_oa = PCA(n_components=N_PCA, random_state=SEED)
    pca_oa.fit(Z_td_ry)

    obs_adni_oasis, _ = pca_w1_summary(Z_td_ry, Z_oa_ry, pca_oa)

    n_adni = len(Z_td_ry)
    n_oasis = len(Z_oa_ry)
    Z_pool = np.vstack([Z_td_ry, Z_oa_ry])
    null_adni_oasis = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_idx = rng.permutation(n_adni + n_oasis)
        zA = Z_pool[perm_idx[:n_adni]]
        zB = Z_pool[perm_idx[n_adni:]]
        null_adni_oasis[i], _ = pca_w1_summary(zA, zB, pca_oa)

    p_adni_oasis = float((null_adni_oasis >= obs_adni_oasis).mean())
    if p_adni_oasis == 0.0:
        p_adni_oasis = 1.0 / N_PERM

    perm_rows.append(dict(
        fold=fold,
        comparison="manufacturer_GE_vs_Philips",
        observed_pca10_w1_mean=obs_ge_ph,
        perm_null_mean=float(null_ge_ph[~np.isnan(null_ge_ph)].mean()),
        perm_null_sd=float(null_ge_ph[~np.isnan(null_ge_ph)].std()),
        n_perm=N_PERM,
        p_value=p_ge_ph,
        p_resolution=1.0 / N_PERM,
        permutation_type="shuffle_manufacturer_labels_test_fold_GE_Philips_subset",
    ))
    perm_rows.append(dict(
        fold=fold,
        comparison="ADNI_trainDev_vs_OASIS",
        observed_pca10_w1_mean=obs_adni_oasis,
        perm_null_mean=float(null_adni_oasis.mean()),
        perm_null_sd=float(null_adni_oasis.std()),
        n_perm=N_PERM,
        p_value=p_adni_oasis,
        p_resolution=1.0 / N_PERM,
        permutation_type="shuffle_cohort_labels_pooled_ADNI_trainDev_OASIS",
    ))

    print(f"    GE-Philips: obs={obs_ge_ph:.4f}, p={p_ge_ph:.4f}", flush=True)
    print(f"    ADNI-OASIS: obs={obs_adni_oasis:.4f}, p={p_adni_oasis:.4f}", flush=True)

perm_df = pd.DataFrame(perm_rows)
perm_df.to_csv(OUTPUT_DIR / "wasserstein_permutation_pvalues.csv", index=False)
print(f"  ✓ wasserstein_permutation_pvalues.csv ({len(perm_df)} rows)")

# ===========================================================================
# Aggregate summary statistics
# ===========================================================================
print("\n[summary] Computing cross-fold aggregates …", flush=True)

summary = {}

for pair in [f"{a}_vs_{b}" for a, b in MANUFACTURER_PAIRS]:
    sub = mfr_df[mfr_df["pair"] == pair]
    for metric in ["w1_mean_all_dims", "pca10_w1_mean", "gaussian_w2"]:
        vals = sub[metric].values
        summary[f"mfr_{pair}_{metric}"] = {"mean": float(vals.mean()),
                                             "sd": float(vals.std()),
                                             "folds": vals.tolist()}

for metric in ["w1_mean_all_dims", "pca10_w1_mean", "gaussian_w2"]:
    vals = oasis_df_out[metric].values
    summary[f"adni_vs_oasis_{metric}"] = {"mean": float(vals.mean()),
                                            "sd": float(vals.std()),
                                            "folds": vals.tolist()}

for comparison in ["manufacturer_GE_vs_Philips", "ADNI_trainDev_vs_OASIS"]:
    sub = perm_df[perm_df["comparison"] == comparison]
    summary[f"perm_{comparison}_p"] = {
        "folds": sub["p_value"].tolist(),
        "all_significant_at_0.05": bool((sub["p_value"] < 0.05).all()),
    }

# Write JSON summary
with open(OUTPUT_DIR / "wasserstein_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("  ✓ wasserstein_summary.json")

# ===========================================================================
# Task 4 — Dimension-to-decoder mapping check
# ===========================================================================
print("\n[task4] Dimension-to-decoder mapping …", flush=True)
# Check if per-dimension direction vectors exist from prior mechanistic audit
prior_mfr_dir = Path(
    "/home/diego/proyectos/vae_AD/results/revision_bspc_2026"
    "/post_revision_exploratory_20260630"
    "/locked_model_site_effect_mechanistic_audit_20260704"
)
dim_mapping_status = "FOLLOW_UP_REQUIRED"
dim_mapping_note = (
    "The decoded_connectome_delta_by_channel.csv maps DIRECTIONS (ridge-regression β vectors) "
    "to FC deltas, not individual latent dimensions. The per-dimension β weights would need "
    "the actual direction vector (384-dim array) from the audit script, not saved in CSV outputs. "
    "Top-contributing dimensions by per-dim W1 are saved in "
    "wasserstein_top_contributing_dims_adni_vs_oasis.csv for future cross-reference."
)
print(f"  Status: {dim_mapping_status}")
print(f"  Note: {dim_mapping_note[:100]}…")

# ===========================================================================
# command_log.json
# ===========================================================================
log = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "script": str(Path(__file__).name),
    "model": "recover035_latent384_beta3p75_T80_h10000_p560_full5x5",
    "request_source": "Martín Belzunce WhatsApp 2026-07-05",
    "status": "COMPLETE",
    "fold_local_discipline": "ENFORCED",
    "adni_latent_cache": str(ADNI_LATENT_CACHE),
    "oasis_latent_file": str(OASIS_LATENT_FILE),
    "oasis_note": "All 180 OASIS subjects are SIEMENS-scanned (TrioTim). Each fold uses that fold's own encoder.",
    "n_folds": len(FOLDS),
    "n_perm": N_PERM,
    "n_pca_components": N_PCA,
    "wasserstein_methods": {
        "1d_w1": "scipy.stats.wasserstein_distance per latent dimension (384 dims)",
        "pca10_w1_mean": "PCA top-10 (fitted on trainDev) → W1 per PC → mean",
        "gaussian_w2": "closed-form W2 between N(mu_A, LW_A) and N(mu_B, LW_B) via Bures-Wasserstein",
        "sinkhorn": "NOT computed — POT library not installed",
    },
    "residualization": {
        "manufacturer_comparison": "y (diagnosis) + Age_z + Sex (OLS on trainDev, applied to test fold)",
        "adni_vs_oasis": "y (diagnosis) only — preserve cohort-related distributional structure",
    },
    "task4_dim_decoder_mapping": dim_mapping_status,
    "task4_note": dim_mapping_note,
    "deliverables": [
        "wasserstein_manufacturer_by_fold.csv",
        "wasserstein_manufacturer_pca_by_pc_fold.csv",
        "wasserstein_adni_vs_oasis_by_fold.csv",
        "wasserstein_adni_vs_oasis_pca_by_pc_fold.csv",
        "wasserstein_top_contributing_dims_adni_vs_oasis.csv",
        "wasserstein_permutation_pvalues.csv",
        "wasserstein_summary.json",
        "wasserstein_summary_across_folds.md",
        "command_log.json",
    ],
    "guardrails": {
        "no_retraining": True,
        "no_new_oasis_inference": True,
        "fold_local_enforced": True,
        "no_cross_fold_pooling": True,
        "no_manuscript_edits": True,
    },
}

with open(OUTPUT_DIR / "command_log.json", "w") as f:
    json.dump(log, f, indent=2)

print("\n[done] All deliverables written to:")
print(f"  {OUTPUT_DIR}")
print("  Remaining: wasserstein_summary_across_folds.md (written by separate step)")
