#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/ood_policy_adni.py — Principled ADNI-derived OOD policy family

Computes ADNI train-dev baseline distributions for BOTH OOD metrics
(recon_error + Mahalanobis distance), defines a family of OOD gates,
selects a non-degenerate DEFAULT, and produces:

  A1) Tables/adni_ood_distribution.csv         — ADNI baseline stats
  A2) Updated covid_ood_quadrants_{clf}.csv     — new OOD flags + quadrants
  A3) Calibrated OOD scores (z-std + z-robust)
  A4) Tables/ood_policy_sensitivity_summary.csv — sensitivity report
  C)  Figures/fig_ood_baseline_shift.png        — ADNI vs COVID KDE
  D)  QC_SUMMARY_OOD.md                        — provenance + summary

Usage:
    conda activate vae_ad
    python scripts/ood_policy_adni.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.is_dir():
    sys.path.insert(0, str(SRC_DIR))
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import mahalanobis
from scipy.stats import median_abs_deviation
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ═══════════════════════════════════════════════════════════════════════
#  Paths & config
# ═══════════════════════════════════════════════════════════════════════
RESULTS_DIR = PROJECT_ROOT / "results" / "vae_3channels_beta65_pro"
OUTPUT_DIR  = RESULTS_DIR / "inference_covid_paper_output"
TABLES_DIR  = OUTPUT_DIR / "Tables"
FIGURES_DIR = OUTPUT_DIR / "Figures"

for d in [TABLES_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

run_config: Dict[str, Any] = json.loads(
    (RESULTS_DIR / "run_config.json").read_text()
)
args_cfg = run_config["args"]

CHANNELS_TO_USE = args_cfg["channels_to_use"]           # [1, 0, 2]
NORM_MODE       = args_cfg["norm_mode"]                  # zscore_offdiag
LATENT_DIM      = int(args_cfg["latent_dim"])            # 256
N_FOLDS         = int(args_cfg.get("outer_folds", 5))
FOLD_IDS        = list(range(1, N_FOLDS + 1))
CLASSIFIER_TYPES = ["logreg"]                            # primary

ADNI_TENSOR_REL = args_cfg["global_tensor_path"]

# Youden threshold (computed in notebook §6)
ADNI_THRESHOLDS_PATH = TABLES_DIR / "adni_derived_thresholds.csv"

print("=" * 72)
print("  ood_policy_adni.py — ADNI-derived OOD policy family")
print("=" * 72)


# ═══════════════════════════════════════════════════════════════════════
#  Helpers (mirrored from inference to avoid import issues)
# ═══════════════════════════════════════════════════════════════════════

def apply_normalization_params(data: np.ndarray, norm_params: list) -> np.ndarray:
    n_subj, n_ch, n_rois, _ = data.shape
    off_diag = ~np.eye(n_rois, dtype=bool)
    out = data.copy()
    for c in range(n_ch):
        p = norm_params[c]
        mode = p.get("mode", "zscore_offdiag")
        if p.get("no_scale", False):
            continue
        ch = data[:, c, :, :]
        scaled = ch.copy()
        if mode == "zscore_offdiag" and p["std"] > 1e-9:
            scaled[:, off_diag] = (ch[:, off_diag] - p["mean"]) / p["std"]
        elif mode == "minmax_offdiag":
            rng = p.get("max", 1.0) - p.get("min", 0.0)
            if rng > 1e-9:
                scaled[:, off_diag] = (ch[:, off_diag] - p["min"]) / rng
        out[:, c, :, :] = scaled
        out[:, c, ~off_diag] = 0.0
    return out


def compute_recon_error_offdiag(X_true, X_recon):
    n_rois = X_true.shape[-1]
    off = ~np.eye(n_rois, dtype=bool)
    return np.abs(X_true - X_recon)[:, :, off].mean(axis=-1).mean(axis=1)


def encode_and_reconstruct(vae, tensor_norm, device, bs=64):
    n = tensor_norm.shape[0]
    lats, recs = [], []
    with torch.no_grad():
        for s in range(0, n, bs):
            e = min(s + bs, n)
            batch = torch.from_numpy(tensor_norm[s:e]).float().to(device)
            recon_x, mu, _, z = vae(batch)
            lats.append(mu.cpu().numpy())
            recs.append(recon_x.cpu().numpy())
    return np.concatenate(lats), np.concatenate(recs)


def compute_mahalanobis_pca(
    target_latent: np.ndarray,
    baseline_latent: np.ndarray,
    pca_var_threshold: float = 0.95,
    regularisation: float = 1e-6,
):
    """
    Project into PCA space that captures `pca_var_threshold` of baseline
    variance, then compute Mahalanobis distances. This avoids degenerate
    distances when d >> n (256-d latent, ~146 baseline subjects).

    Returns: (distances, n_pca_components, pca_object)
    """
    n_baseline = baseline_latent.shape[0]
    d = baseline_latent.shape[1]
    max_k = min(n_baseline - 1, d)

    pca = PCA(n_components=max_k, random_state=SEED)
    pca.fit(baseline_latent)

    cum_var = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cum_var, pca_var_threshold) + 1)
    k = max(k, 2)  # ensure at least 2 components
    k = min(k, max_k)

    # Project both sets into k-d PCA space
    base_proj = pca.transform(baseline_latent)[:, :k]
    targ_proj = pca.transform(target_latent)[:, :k]

    mu = base_proj.mean(axis=0)
    cov = np.cov(base_proj, rowvar=False)
    cov += regularisation * np.eye(k)
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)

    dists = np.array([mahalanobis(z, mu, cov_inv) for z in targ_proj])
    return dists, k, pca


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "N/A"


def _assign_quadrant(df, score_col, ood_col, t_primary, q_prefix):
    """Assign quadrant labels given score threshold + boolean OOD flag."""
    q_col = f"quadrant_{q_prefix}"
    df[q_col] = "CN-like_InDist"
    df.loc[(df[score_col] >= t_primary) & (df[ood_col] == 0), q_col] = "AD-like_InDist"
    df.loc[(df[score_col] >= t_primary) & (df[ood_col] == 1), q_col] = "AD-like_OOD"
    df.loc[(df[score_col] <  t_primary) & (df[ood_col] == 1), q_col] = "CN-like_OOD"
    return df


# ═══════════════════════════════════════════════════════════════════════
#  Load ADNI tensor & derive Youden threshold
# ═══════════════════════════════════════════════════════════════════════
print("\n[1/7] Loading data …")

adni_npz = np.load(PROJECT_ROOT / ADNI_TENSOR_REL, allow_pickle=True)
adni_tensor_full = adni_npz["global_tensor_data"]
adni_tensor = adni_tensor_full[:, CHANNELS_TO_USE, :, :]
N_ROIS = adni_tensor.shape[2]
print(f"  ADNI tensor: {adni_tensor.shape}")

# Youden threshold
if ADNI_THRESHOLDS_PATH.exists():
    _th = pd.read_csv(ADNI_THRESHOLDS_PATH)
    _youden = _th[_th["policy"] == "Youden"]
    t_primary = float(_youden["threshold"].iloc[0])
    print(f"  Youden threshold: {t_primary:.4f}")
else:
    t_primary = 0.5
    print("  ⚠ ADNI thresholds not found; using 0.5")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

# Import VAE
from betavae_xai.models import ConvolutionalVAE


# ═══════════════════════════════════════════════════════════════════════
#  A1) Compute ADNI baseline distributions (recon + mdist) per fold
#      AND COVID PCA-Mahalanobis per fold (for consistent comparison)
# ═══════════════════════════════════════════════════════════════════════
print("\n[2/7] Computing ADNI baseline distributions per fold …")
print("  (Mahalanobis via PCA to handle d=256 >> n≈146)")

N_CH = len(CHANNELS_TO_USE)

# Load COVID tensor for per-fold encoding
COVID_TENSOR_REL = (
    "data/COVID_AAL3_Tensor_v1_AAL3_131ROIs_OMST_GCE_Signed_"
    "GrangerLag1_ChNorm_ROIreorderedYeo17_ParallelTuned/"
    "GLOBAL_TENSOR_from_COVID_AAL3_Tensor_v1_AAL3_131ROIs_OMST_GCE_"
    "Signed_GrangerLag1_ChNorm_ROIreorderedYeo17_ParallelTuned.npz"
)
covid_npz = np.load(PROJECT_ROOT / COVID_TENSOR_REL, allow_pickle=True)
covid_tensor_full = covid_npz["global_tensor_data"]
covid_ids_tensor = np.array([str(s) for s in covid_npz["subject_ids"]])
covid_tensor = covid_tensor_full[:, CHANNELS_TO_USE, :, :]
N_COVID = covid_tensor.shape[0]
print(f"  COVID tensor: {covid_tensor.shape}")

adni_dist_rows = []
# Per-fold COVID PCA-mdist (will median-aggregate later)
covid_mdist_per_fold = {sid: [] for sid in covid_ids_tensor}

for fold_i in FOLD_IDS:
    fold_dir = RESULTS_DIR / f"fold_{fold_i}"
    assert fold_dir.exists(), f"Missing {fold_dir}"

    # Artefacts
    norm_params = joblib.load(fold_dir / "vae_norm_params.joblib")
    td_idx = np.load(fold_dir / "train_dev_tensor_idx.npy")

    # VAE model
    vae = ConvolutionalVAE(
        input_channels=N_CH,
        latent_dim=LATENT_DIM,
        image_size=N_ROIS,
        final_activation=args_cfg["vae_final_activation"],
        intermediate_fc_dim_config=args_cfg["intermediate_fc_dim_vae"],
        dropout_rate=float(args_cfg["dropout_rate_vae"]),
        use_layernorm_fc=bool(args_cfg.get("use_layernorm_vae_fc", False)),
        num_conv_layers_encoder=int(args_cfg["num_conv_layers_encoder"]),
        decoder_type=args_cfg["decoder_type"],
    ).to(device)
    vae_path = fold_dir / f"vae_model_fold_{fold_i}.pt"
    vae.load_state_dict(
        torch.load(vae_path, map_location=device, weights_only=True)
    )
    vae.eval()

    # Select and normalise ADNI train-dev
    adni_sel = adni_tensor[td_idx]
    adni_norm = apply_normalization_params(adni_sel, norm_params)

    # Forward pass — ADNI baseline
    baseline_latent, baseline_recon = encode_and_reconstruct(
        vae, adni_norm, device
    )

    # Recon error — ADNI
    recon_err = compute_recon_error_offdiag(adni_norm, baseline_recon)

    # PCA-Mahalanobis — ADNI baseline (self-distances)
    mdist_adni, n_pca, pca_obj = compute_mahalanobis_pca(
        baseline_latent, baseline_latent, pca_var_threshold=0.95
    )

    n_td = len(td_idx)

    for metric_name, values in [("recon_error", recon_err),
                                 ("latent_distance", mdist_adni)]:
        mad_val = float(median_abs_deviation(values, nan_policy="omit"))
        adni_dist_rows.append({
            "fold": fold_i,
            "metric": metric_name,
            "n_baseline": n_td,
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "median": float(np.median(values)),
            "mad": mad_val,
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        })
        if metric_name == "latent_distance":
            adni_dist_rows[-1]["n_pca_components"] = n_pca

    # Forward pass — COVID (same VAE, same norm)
    covid_norm = apply_normalization_params(covid_tensor, norm_params)
    covid_latent, _ = encode_and_reconstruct(vae, covid_norm, device)

    # PCA-Mahalanobis — COVID (same PCA transform + Gaussian as ADNI)
    mdist_covid, _, _ = compute_mahalanobis_pca(
        covid_latent, baseline_latent, pca_var_threshold=0.95
    )

    for local_i, sid in enumerate(covid_ids_tensor):
        covid_mdist_per_fold[sid].append(mdist_covid[local_i])

    print(f"  Fold {fold_i}: n_ADNI={n_td}, PCA k={n_pca}, "
          f"recon P95={np.percentile(recon_err, 95):.4f}, "
          f"mdist P95={np.percentile(mdist_adni, 95):.1f}, "
          f"COVID mdist median={np.median(mdist_covid):.1f}")

    # Clean up GPU memory
    del vae
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# COVID PCA-mdist ensemble (median across folds)
covid_mdist_pca_ens = pd.DataFrame({
    "SubjectID": covid_ids_tensor,
    "mdist_pca_ensemble": [
        np.median(covid_mdist_per_fold[sid]) for sid in covid_ids_tensor
    ],
})


# Ensemble medians across folds
adni_dist_df = pd.DataFrame(adni_dist_rows)

for metric_name in ["recon_error", "latent_distance"]:
    _sub = adni_dist_df[adni_dist_df["metric"] == metric_name]
    ens_row = {
        "fold": "ensemble_median",
        "metric": metric_name,
        "n_baseline": int(_sub["n_baseline"].median()),
    }
    for col in ["mean", "std", "median", "mad", "p90", "p95", "p99",
                "min", "max"]:
        ens_row[col] = float(_sub[col].median())
    adni_dist_rows.append(ens_row)

adni_dist_df = pd.DataFrame(adni_dist_rows)
adni_dist_df.to_csv(TABLES_DIR / "adni_ood_distribution.csv", index=False)
print(f"\n  → Saved: Tables/adni_ood_distribution.csv "
      f"({len(adni_dist_df)} rows)")

# Extract ensemble thresholds
def _ens_val(metric, stat):
    row = adni_dist_df[
        (adni_dist_df["fold"] == "ensemble_median") &
        (adni_dist_df["metric"] == metric)
    ]
    return float(row[stat].iloc[0])

RECON_P90 = _ens_val("recon_error", "p90")
RECON_P95 = _ens_val("recon_error", "p95")
RECON_P99 = _ens_val("recon_error", "p99")
RECON_MEAN = _ens_val("recon_error", "mean")
RECON_STD  = _ens_val("recon_error", "std")
RECON_MED  = _ens_val("recon_error", "median")
RECON_MAD  = _ens_val("recon_error", "mad")

MDIST_P90 = _ens_val("latent_distance", "p90")
MDIST_P95 = _ens_val("latent_distance", "p95")
MDIST_P99 = _ens_val("latent_distance", "p99")
MDIST_MEAN = _ens_val("latent_distance", "mean")
MDIST_STD  = _ens_val("latent_distance", "std")
MDIST_MED  = _ens_val("latent_distance", "median")
MDIST_MAD  = _ens_val("latent_distance", "mad")

print(f"\n  ADNI ensemble thresholds:")
print(f"    recon  — P90={RECON_P90:.4f}  P95={RECON_P95:.4f}  "
      f"P99={RECON_P99:.4f}")
print(f"    mdist  — P90={MDIST_P90:.1f}  P95={MDIST_P95:.1f}  "
      f"P99={MDIST_P99:.1f}")


# ═══════════════════════════════════════════════════════════════════════
#  A2-A3) Load COVID data, compute OOD flags + calibrated scores
# ═══════════════════════════════════════════════════════════════════════
print("\n[3/7] Applying OOD policies to COVID data …")

for clf in CLASSIFIER_TYPES:
    q_path = TABLES_DIR / f"covid_ood_quadrants_{clf}.csv"
    assert q_path.exists(), f"Missing {q_path}"
    quad = pd.read_csv(q_path)

    # Ensure required columns
    assert "recon_error_ensemble" in quad.columns
    assert "y_score_ensemble" in quad.columns

    recon = quad["recon_error_ensemble"].values
    score = quad["y_score_ensemble"].values

    # Merge PCA-mdist (computed above per fold, ensemble median)
    quad = quad.merge(covid_mdist_pca_ens, on="SubjectID", how="left")
    mdist = quad["mdist_pca_ensemble"].values

    # Keep original latent_distance_ensemble for backward compat
    # (it's the raw 256-d Mahalanobis from the inference script)
    # The new PCA-based mdist is used for OOD policy.

    # ── A2: OOD flags ────────────────────────────────────────────
    for pct, p_label in [(90, "p90"), (95, "p95"), (99, "p99")]:
        recon_thr = _ens_val("recon_error", p_label)
        mdist_thr = _ens_val("latent_distance", p_label)

        f_recon = f"ood_flag_adni_recon_{p_label}"
        f_mdist = f"ood_flag_adni_mdist_{p_label}"
        f_joint = f"ood_flag_adni_joint_{p_label}"

        quad[f_recon] = (recon >= recon_thr).astype(int)
        quad[f_mdist] = (mdist >= mdist_thr).astype(int)
        quad[f_joint] = ((recon >= recon_thr) & (mdist >= mdist_thr)).astype(int)

    # ── A3: Calibrated OOD scores ────────────────────────────────
    # Standard z-scores
    recon_z = (recon - RECON_MEAN) / max(RECON_STD, 1e-9)
    mdist_z = (mdist - MDIST_MEAN) / max(MDIST_STD, 1e-9)
    quad["recon_z_std"] = recon_z
    quad["mdist_z_std"] = mdist_z
    quad["ood_score_max_z"] = np.maximum(recon_z, mdist_z)
    quad["ood_score_joint_z"] = np.sqrt(recon_z**2 + mdist_z**2)

    # Robust z-scores (median / MAD)
    k_mad = 1.4826  # consistency constant for normal
    recon_z_r = (recon - RECON_MED) / max(k_mad * RECON_MAD, 1e-9)
    mdist_z_r = (mdist - MDIST_MED) / max(k_mad * MDIST_MAD, 1e-9)
    quad["recon_z_robust"] = recon_z_r
    quad["mdist_z_robust"] = mdist_z_r
    quad["ood_score_max_z_robust"] = np.maximum(recon_z_r, mdist_z_r)
    quad["ood_score_joint_z_robust"] = np.sqrt(recon_z_r**2 + mdist_z_r**2)

    # ── Choose DEFAULT policy ────────────────────────────────────
    # Preferred: joint_p95 → fall back if OOD rate > 50%
    DEFAULT_GATE = "ood_flag_adni_joint_p95"
    ood_rate = quad[DEFAULT_GATE].mean()
    chosen_gate = "joint_p95"

    if ood_rate > 0.50:
        DEFAULT_GATE = "ood_flag_adni_joint_p99"
        ood_rate = quad[DEFAULT_GATE].mean()
        chosen_gate = "joint_p99"
        if ood_rate > 0.50:
            # Last resort: use recon_p99 only
            DEFAULT_GATE = "ood_flag_adni_recon_p99"
            ood_rate = quad[DEFAULT_GATE].mean()
            chosen_gate = "recon_p99"

    print(f"\n  {clf.upper()} — DEFAULT OOD gate: {chosen_gate} "
          f"(OOD rate: {ood_rate:.1%}, n={quad[DEFAULT_GATE].sum()}/{len(quad)})")

    # ── A2/B: Assign quadrants for all policies ──────────────────
    quad["ood_flag_adni_default"] = quad[DEFAULT_GATE]

    for suffix in ["adni_recon_p90", "adni_recon_p95", "adni_recon_p99",
                    "adni_mdist_p90", "adni_mdist_p95", "adni_mdist_p99",
                    "adni_joint_p90", "adni_joint_p95", "adni_joint_p99",
                    "adni_default"]:
        flag_col = f"ood_flag_{suffix}"
        if flag_col in quad.columns:
            _assign_quadrant(quad, "y_score_ensemble", flag_col,
                             t_primary, suffix)

    # ── Save updated quadrant table ──────────────────────────────
    quad.to_csv(q_path, index=False)
    print(f"  → Updated: {q_path.name} ({len(quad.columns)} cols)")

    # Print DEFAULT quadrant counts
    q_col = "quadrant_adni_default"
    print(f"\n  Quadrant counts ({q_col}):")
    for q in ["AD-like_InDist", "AD-like_OOD", "CN-like_InDist", "CN-like_OOD"]:
        n = (quad[q_col] == q).sum()
        print(f"    {q:20s}: {n:4d} ({100*n/len(quad):.1f}%)")


# ═══════════════════════════════════════════════════════════════════════
#  A4) Sensitivity summary
# ═══════════════════════════════════════════════════════════════════════
print("\n[4/7] Sensitivity report …")

sens_rows = []
# Reload the saved table (it has all flags)
for clf in CLASSIFIER_TYPES:
    quad = pd.read_csv(TABLES_DIR / f"covid_ood_quadrants_{clf}.csv")
    n_total = len(quad)

    for metric in ["recon", "mdist", "joint"]:
        for pct in ["p90", "p95", "p99"]:
            flag_col = f"ood_flag_adni_{metric}_{pct}"
            if flag_col not in quad.columns:
                continue
            n_ood = int(quad[flag_col].sum())
            rate = n_ood / n_total
            # Also break down by AD-like vs CN-like
            ad_mask = quad["y_score_ensemble"] >= t_primary
            n_ad_ood = int(quad.loc[ad_mask, flag_col].sum())
            n_cn_ood = int(quad.loc[~ad_mask, flag_col].sum())

            sens_rows.append({
                "classifier": clf,
                "metric": metric,
                "percentile": pct,
                "flag_col": flag_col,
                "n_total": n_total,
                "n_ood": n_ood,
                "ood_rate": rate,
                "n_ad_like": int(ad_mask.sum()),
                "n_ad_like_ood": n_ad_ood,
                "n_cn_like": int((~ad_mask).sum()),
                "n_cn_like_ood": n_cn_ood,
                "is_default": (flag_col == DEFAULT_GATE),
            })

sens_df = pd.DataFrame(sens_rows)
sens_df.to_csv(TABLES_DIR / "ood_policy_sensitivity_summary.csv", index=False)
print(f"  → Saved: Tables/ood_policy_sensitivity_summary.csv "
      f"({len(sens_df)} rows)")
print(sens_df[["metric", "percentile", "n_ood", "ood_rate",
               "is_default"]].to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════
#  C) Diagnostics figure: ADNI baseline vs COVID
# ═══════════════════════════════════════════════════════════════════════
print("\n[5/7] Creating figure: ADNI baseline vs COVID …")

# Collect ADNI per-subject baseline values (median across folds)
import collections
_recon_per_subj = collections.defaultdict(list)
_mdist_per_subj = collections.defaultdict(list)

for fold_i in FOLD_IDS:
    fold_dir = RESULTS_DIR / f"fold_{fold_i}"
    norm_params = joblib.load(fold_dir / "vae_norm_params.joblib")
    td_idx = np.load(fold_dir / "train_dev_tensor_idx.npy")

    vae = ConvolutionalVAE(
        input_channels=N_CH,
        latent_dim=LATENT_DIM,
        image_size=N_ROIS,
        final_activation=args_cfg["vae_final_activation"],
        intermediate_fc_dim_config=args_cfg["intermediate_fc_dim_vae"],
        dropout_rate=float(args_cfg["dropout_rate_vae"]),
        use_layernorm_fc=bool(args_cfg.get("use_layernorm_vae_fc", False)),
        num_conv_layers_encoder=int(args_cfg["num_conv_layers_encoder"]),
        decoder_type=args_cfg["decoder_type"],
    ).to(device)
    vae.load_state_dict(
        torch.load(fold_dir / f"vae_model_fold_{fold_i}.pt",
                    map_location=device, weights_only=True)
    )
    vae.eval()

    adni_sel = adni_tensor[td_idx]
    adni_norm = apply_normalization_params(adni_sel, norm_params)
    latent, recon = encode_and_reconstruct(vae, adni_norm, device)
    recon_err = compute_recon_error_offdiag(adni_norm, recon)
    mdist_vals, _, _ = compute_mahalanobis_pca(latent, latent,
                                                pca_var_threshold=0.95)

    for local_i, global_idx in enumerate(td_idx):
        _recon_per_subj[global_idx].append(recon_err[local_i])
        _mdist_per_subj[global_idx].append(mdist_vals[local_i])

    del vae
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

adni_recon_baseline = np.array([np.median(v) for v in _recon_per_subj.values()])
adni_mdist_baseline = np.array([np.median(v) for v in _mdist_per_subj.values()])

# COVID values (PCA-mdist and recon)
covid_recon = pd.read_csv(TABLES_DIR / "covid_recon_error_ensemble.csv")[
    "recon_error_ensemble"
].values
covid_mdist_fig = covid_mdist_pca_ens["mdist_pca_ensemble"].values

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel A: Recon error
ax = axes[0]
ax.hist(adni_recon_baseline, bins=40, density=True, alpha=0.55,
        color="#3498DB", edgecolor="white", label="ADNI train-dev")
ax.hist(covid_recon, bins=40, density=True, alpha=0.55,
        color="#E74C3C", edgecolor="white", label="COVID cohort")
for pct, ls, col, lbl in [
    (RECON_P95, "--", "#2980B9", f"P95={RECON_P95:.4f}"),
    (RECON_P99, "-.", "#1B4F72", f"P99={RECON_P99:.4f}"),
]:
    ax.axvline(pct, color=col, ls=ls, lw=1.4, label=f"ADNI {lbl}")
ax.set_xlabel("Reconstruction Error (MAE off-diag)")
ax.set_ylabel("Density")
ax.set_title("A. Recon Error — ADNI baseline vs COVID")
ax.legend(fontsize=8)

# Panel B: PCA-Mahalanobis distance
ax = axes[1]
ax.hist(adni_mdist_baseline, bins=40, density=True, alpha=0.55,
        color="#3498DB", edgecolor="white", label="ADNI train-dev")
ax.hist(covid_mdist_fig, bins=40, density=True, alpha=0.55,
        color="#E74C3C", edgecolor="white", label="COVID cohort")
for pct, ls, col, lbl in [
    (MDIST_P95, "--", "#2980B9", f"P95={MDIST_P95:.1f}"),
    (MDIST_P99, "-.", "#1B4F72", f"P99={MDIST_P99:.1f}"),
]:
    ax.axvline(pct, color=col, ls=ls, lw=1.4, label=f"ADNI {lbl}")
ax.set_xlabel("PCA-Mahalanobis Distance (latent space, 95% var)")
ax.set_ylabel("Density")
ax.set_title("B. PCA-Mahalanobis — ADNI baseline vs COVID")
ax.legend(fontsize=8)

fig.suptitle("OOD Baseline Shift: ADNI Train-Dev vs COVID Cohort",
             fontsize=12, y=1.02)
plt.tight_layout()
fig_path = FIGURES_DIR / "fig_ood_baseline_shift.png"
fig.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
assert fig_path.stat().st_size > 0
print(f"  → Saved: Figures/fig_ood_baseline_shift.png "
      f"({fig_path.stat().st_size / 1024:.0f} KB)")


# ═══════════════════════════════════════════════════════════════════════
#  D) QC summary
# ═══════════════════════════════════════════════════════════════════════
print("\n[6/7] Writing QC_SUMMARY_OOD.md …")

qc_lines = [
    "# QC Summary — OOD Policy (ADNI-derived)",
    "",
    f"**Timestamp:** {datetime.now(timezone.utc).isoformat()}",
    f"**Git hash:** {_git_hash()}",
    f"**Script:** scripts/ood_policy_adni.py",
    "",
    "## ADNI Baseline",
    "",
    f"- ADNI tensor: {adni_tensor.shape}",
    f"- Train-dev per fold: ~{_ens_val('recon_error', 'n_baseline'):.0f}",
    f"- Folds: {N_FOLDS}",
    "",
    "### Ensemble Thresholds (median across folds)",
    "",
    "| Metric | Mean | Std | Median | MAD | P90 | P95 | P99 |",
    "|--------|------|-----|--------|-----|-----|-----|-----|",
    f"| recon_error | {RECON_MEAN:.4f} | {RECON_STD:.4f} | "
    f"{RECON_MED:.4f} | {RECON_MAD:.4f} | "
    f"{RECON_P90:.4f} | {RECON_P95:.4f} | {RECON_P99:.4f} |",
    f"| latent_distance | {MDIST_MEAN:.1f} | {MDIST_STD:.1f} | "
    f"{MDIST_MED:.1f} | {MDIST_MAD:.1f} | "
    f"{MDIST_P90:.1f} | {MDIST_P95:.1f} | {MDIST_P99:.1f} |",
    "",
    f"## Chosen DEFAULT Gate: **{chosen_gate}**",
    "",
    f"OOD rate under DEFAULT: {ood_rate:.1%} ({quad[DEFAULT_GATE].sum()}/{len(quad)})",
    "",
    "### OOD Rates by Policy",
    "",
    "| Policy | n_OOD | Rate |",
    "|--------|-------|------|",
]

for _, row in sens_df.iterrows():
    tag = " **[DEFAULT]**" if row["is_default"] else ""
    qc_lines.append(
        f"| {row['metric']}_{row['percentile']} | "
        f"{row['n_ood']} | {row['ood_rate']:.1%} |"
        f"{tag}"
    )

qc_lines.extend([
    "",
    f"### Quadrant Counts (quadrant_adni_default, t_primary={t_primary:.4f})",
    "",
])

q_col = "quadrant_adni_default"
for q in ["AD-like_InDist", "AD-like_OOD", "CN-like_InDist", "CN-like_OOD"]:
    n = int((quad[q_col] == q).sum())
    qc_lines.append(f"- **{q}**: {n} ({100*n/len(quad):.1f}%)")

qc_path = OUTPUT_DIR / "QC_SUMMARY_OOD.md"
qc_path.write_text("\n".join(qc_lines) + "\n")
print(f"  → Saved: {qc_path.name}")


# ═══════════════════════════════════════════════════════════════════════
#  Final summary
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("  SUMMARY")
print("=" * 72)
print(f"\n  ADNI ensemble thresholds:")
print(f"    recon_error     — P90={RECON_P90:.4f}  P95={RECON_P95:.4f}  "
      f"P99={RECON_P99:.4f}")
print(f"    latent_distance — P90={MDIST_P90:.1f}  P95={MDIST_P95:.1f}  "
      f"P99={MDIST_P99:.1f}")
print(f"\n  COVID OOD rates (n={len(quad)}):")
for _, row in sens_df.iterrows():
    tag = " ← DEFAULT" if row["is_default"] else ""
    print(f"    {row['metric']:6s}_{row['percentile']:3s}: "
          f"{row['n_ood']:4d} ({row['ood_rate']:5.1%}){tag}")

print(f"\n  Chosen DEFAULT: {chosen_gate} "
      f"(OOD = {quad[DEFAULT_GATE].sum()}/{len(quad)}, "
      f"{quad[DEFAULT_GATE].mean():.1%})")

print(f"\n  quadrant_adni_default:")
for q in ["AD-like_InDist", "AD-like_OOD", "CN-like_InDist", "CN-like_OOD"]:
    n = int((quad[q_col] == q).sum())
    print(f"    {q:20s}: {n:4d} ({100*n/len(quad):.1f}%)")

print(f"\n  Output files:")
print(f"    Tables/adni_ood_distribution.csv")
print(f"    Tables/ood_policy_sensitivity_summary.csv")
print(f"    Tables/covid_ood_quadrants_logreg.csv (updated)")
print(f"    Figures/fig_ood_baseline_shift.png")
print(f"    QC_SUMMARY_OOD.md")
print("\n✓ Done.")
