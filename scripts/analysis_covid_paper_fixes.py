#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/analysis_covid_paper_fixes.py  —  v3.2 paper fixes

Fixes A–E for publishable COVID→AD transfer analysis:

  A) Fix tautological enrichment: full-universe (8515 edges) edge-level
     tests + Fisher enrichment with top 5% from ALL edges + permutation
     sanity check (5 000 permutations).
  B) Sign-agreement metric for within-COVID analysis + Spearman
     edge–score association across all 8515 edges.
  C) ADNI-derived OOD thresholds (P90, P95) from train-dev recon errors
     per fold → ensemble median; update quadrant labels.
  D) Guard against 0-byte figure saves (check + re-save if needed).
  E) QC summary text file.

Usage:
    conda activate vae_ad
    python scripts/analysis_covid_paper_fixes.py
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Bootstrap: add src/ to sys.path ────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.is_dir():
    sys.path.insert(0, str(SRC_DIR))
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import (
    binomtest,
    fisher_exact,
    mannwhitneyu,
    spearmanr,
)
from statsmodels.stats.multitest import multipletests

SEED = 42
np.random.seed(SEED)

# ═══════════════════════════════════════════════════════════════════════
#  Paths & config
# ═══════════════════════════════════════════════════════════════════════
RESULTS_DIR = PROJECT_ROOT / "results" / "vae_3channels_beta65_pro"
OUTPUT_DIR  = RESULTS_DIR / "inference_covid_paper_output"
TABLES_DIR  = OUTPUT_DIR / "Tables"
FIGURES_DIR = OUTPUT_DIR / "Figures"

run_config: Dict[str, Any] = json.loads(
    (RESULTS_DIR / "run_config.json").read_text()
)
args_cfg = run_config["args"]

CHANNELS_TO_USE = args_cfg["channels_to_use"]            # [1, 0, 2]
NORM_MODE       = args_cfg["norm_mode"]                   # "zscore_offdiag"
LATENT_DIM      = int(args_cfg["latent_dim"])             # 256

N_FOLDS  = int(args_cfg.get("outer_folds", 5))
FOLD_IDS = list(range(1, N_FOLDS + 1))

COVID_TENSOR_REL = (
    "data/COVID_AAL3_Tensor_v1_AAL3_131ROIs_OMST_GCE_Signed_"
    "GrangerLag1_ChNorm_ROIreorderedYeo17_ParallelTuned/"
    "GLOBAL_TENSOR_from_COVID_AAL3_Tensor_v1_AAL3_131ROIs_OMST_GCE_"
    "Signed_GrangerLag1_ChNorm_ROIreorderedYeo17_ParallelTuned.npz"
)
ADNI_TENSOR_REL = args_cfg["global_tensor_path"]

print("=" * 72)
print("  analysis_covid_paper_fixes.py — v3.2  (A-E)")
print("=" * 72)


# ═══════════════════════════════════════════════════════════════════════
#  Helpers (mirrored from inference script to avoid import issues)
# ═══════════════════════════════════════════════════════════════════════

def apply_normalization_params(
    data: np.ndarray, norm_params: list,
) -> np.ndarray:
    """Apply saved per-channel normalisation parameters."""
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
    """Per-subject mean absolute off-diagonal recon error, averaged over channels."""
    n_rois = X_true.shape[-1]
    off = ~np.eye(n_rois, dtype=bool)
    return np.abs(X_true - X_recon)[:, :, off].mean(axis=-1).mean(axis=1)


def encode_and_reconstruct(vae, tensor_norm, device, bs=64):
    """Forward pass: returns (latent_mu, reconstruction)."""
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


def normalize_roi_name(s: str) -> str:
    """Aggressive normalisation: casefold, strip punctuation, collapse spaces."""
    s = str(s).strip().casefold()
    s = re.sub(r"[\(\)\[\]\{\},;:'\".]+", " ", s)
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def fisher_enrichment(hit_set, ref_set, universe_size):
    """
    Fisher exact test for over-representation.
    hit_set, ref_set: sets of (i,j) edge tuples.
    Returns: (overlap_n, odds_ratio, p_value).
    """
    overlap = hit_set & ref_set
    a = len(overlap)
    b = len(hit_set - ref_set)
    c = len(ref_set - hit_set)
    d = universe_size - a - b - c
    table = [[a, b], [c, d]]
    odds, p = fisher_exact(table, alternative="greater")
    return a, odds, p


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "N/A"


# ═══════════════════════════════════════════════════════════════════════
#  Load data
# ═══════════════════════════════════════════════════════════════════════
print("\n[1/6] Loading data …")

# COVID tensor + IDs
covid_npz = np.load(PROJECT_ROOT / COVID_TENSOR_REL, allow_pickle=True)
covid_tensor_full = covid_npz["global_tensor_data"]       # (194, C_full, 131, 131)
covid_ids = np.array([str(s) for s in covid_npz["subject_ids"]])
covid_ch_names = list(covid_npz["channel_names"])
covid_tensor = covid_tensor_full[:, CHANNELS_TO_USE, :, :]
N_SUBJ, N_CH, N_ROIS, _ = covid_tensor.shape
N_EDGES_UNIVERSE = N_ROIS * (N_ROIS - 1) // 2

print(f"  COVID tensor: {covid_tensor.shape}")
print(f"  Universe: {N_ROIS} ROIs → {N_EDGES_UNIVERSE} edges (upper triangle)")

# ADNI tensor
adni_npz = np.load(PROJECT_ROOT / ADNI_TENSOR_REL, allow_pickle=True)
adni_tensor_full = adni_npz["global_tensor_data"]
adni_tensor = adni_tensor_full[:, CHANNELS_TO_USE, :, :]
print(f"  ADNI tensor: {adni_tensor.shape}")

# Clinical metadata
covid_meta_path = PROJECT_ROOT / "data" / "SubjectsData_AAL3_COVID.csv"
covid_meta = pd.read_csv(covid_meta_path)
# Detect merge ID column
_merge_col = "ID" if "ID" in covid_meta.columns else covid_meta.columns[0]
covid_meta["_merge_id"] = covid_meta[_merge_col].astype(str)

# Ensemble predictions
ensemble = pd.read_csv(TABLES_DIR / "covid_predictions_ensemble.csv")
ensemble["SubjectID"] = ensemble["SubjectID"].astype(str)

# Recon errors (COVID)
recon_ens = pd.read_csv(TABLES_DIR / "covid_recon_error_ensemble.csv")
recon_ens["SubjectID"] = recon_ens["SubjectID"].astype(str)

# Latent distances (optional)
dist_ens_path = TABLES_DIR / "covid_latent_distance_ensemble.csv"
dist_ens = pd.read_csv(dist_ens_path) if dist_ens_path.exists() else None
if dist_ens is not None:
    dist_ens["SubjectID"] = dist_ens["SubjectID"].astype(str)

# Consensus edges (AD signature) — already mapped to tensor indices
ce = pd.read_csv(TABLES_DIR / "consensus_edges_mapped.csv")
N_SIG_EDGES = len(ce)
print(f"  Signature edges: {N_SIG_EDGES}")

# ROI info
roi_info = pd.read_csv(RESULTS_DIR / "roi_info_from_tensor.csv")
_roi_col = "roi_name_in_tensor"
roi_info["_norm"] = roi_info[_roi_col].map(normalize_roi_name)
roi_names = roi_info[_roi_col].values

# Build signature edge set (canonical i<j)
sig_edge_set = set(zip(
    np.minimum(ce["i"].values, ce["j"].values),
    np.maximum(ce["i"].values, ce["j"].values),
))

# Threshold policies
thresh_path = TABLES_DIR / "adni_derived_thresholds.csv"
thresh_df = pd.read_csv(thresh_path)
t_youden = float(
    thresh_df.loc[thresh_df["policy"] == "Youden", "threshold"].iloc[0]
)
print(f"  Youden threshold: {t_youden:.4f}")

# Score map for all subjects
_ens_logreg = ensemble[ensemble["classifier"] == "logreg"]
score_map = _ens_logreg.set_index("SubjectID")["y_score_ensemble"].to_dict()

# ── Resolve Long-COVID vs Control groups ───────────────────────
_group_col = "ResearchGroup"
if _group_col not in covid_meta.columns:
    for _c in covid_meta.columns:
        if "group" in _c.lower():
            _group_col = _c
            break

covid_set = set(
    covid_meta.loc[
        covid_meta[_group_col].str.contains("COVID|Long", case=False, na=False),
        "_merge_id",
    ]
)
ctrl_set = set(
    covid_meta.loc[
        covid_meta[_group_col].str.contains("Control|CTRL|HC", case=False, na=False),
        "_merge_id",
    ]
)
# Fallback: if groups not clearly separated, treat non-control as COVID
if len(covid_set) == 0:
    covid_set = set(covid_ids) - ctrl_set
print(f"  Long-COVID n={len(covid_set)}, Controls n={len(ctrl_set)}")

# Boolean masks aligned to covid_ids
mask_covid = np.array([sid in covid_set for sid in covid_ids])
mask_ctrl  = np.array([sid in ctrl_set  for sid in covid_ids])

# Within-COVID: AD-like vs CN-like (by Youden)
covid_scores = np.array([score_map.get(sid, np.nan) for sid in covid_ids])
mask_adlike = mask_covid & (covid_scores >= t_youden)
mask_cnlike = mask_covid & (covid_scores < t_youden) & np.isfinite(covid_scores)

n_ad       = int(mask_adlike.sum())
n_cn       = int(mask_cnlike.sum())
n1_covid   = int(mask_covid.sum())
n2_ctrl    = int(mask_ctrl.sum())
print(f"  Within Long-COVID: AD-like={n_ad}, CN-like={n_cn}")

ch_names_sel = [covid_ch_names[c] for c in CHANNELS_TO_USE]

# Upper-triangle indices (full universe)
_triu_i, _triu_j = np.triu_indices(N_ROIS, k=1)

print("[1/6] Done.\n")


# ═══════════════════════════════════════════════════════════════════════
# ▶ A) FIX: Full-universe edge-level tests + non-tautological enrichment
# ═══════════════════════════════════════════════════════════════════════
print("=" * 72)
print("[2/6] A) Full-universe edge tests + enrichment")
print("=" * 72)

N_PERM = 5_000

for ci, ch_idx in enumerate(CHANNELS_TO_USE):
    ch_name = ch_names_sel[ci]
    tag = f"ch{ch_idx}"
    print(f"\n  ── Channel {ch_idx} ({ch_name[:50]}) ──")
    X = covid_tensor[:, ci, :, :]      # NB: ci (position) not ch_idx (name)

    # ── COVID vs CONTROL: Cliff's δ + MWU for ALL 8515 edges ────
    v_cov = X[mask_covid][:, _triu_i, _triu_j]    # (n1, 8515)
    v_ctr = X[mask_ctrl ][:, _triu_i, _triu_j]    # (n2, 8515)

    n_e = v_cov.shape[1]
    assert n_e == N_EDGES_UNIVERSE, f"Expected {N_EDGES_UNIVERSE}, got {n_e}"

    p_raw   = np.empty(n_e)
    delta   = np.empty(n_e)

    for k in range(n_e):
        u, pv = mannwhitneyu(
            v_cov[:, k], v_ctr[:, k],
            alternative="two-sided", method="asymptotic",
        )
        p_raw[k] = pv
        delta[k] = 2 * u / (n1_covid * n2_ctrl) - 1   # Cliff's δ

    _, q_fdr, _, _ = multipletests(p_raw, method="fdr_bh")

    edge_df = pd.DataFrame({
        "i": _triu_i, "j": _triu_j,
        "roi_i": roi_names[_triu_i], "roi_j": roi_names[_triu_j],
        "ch_idx": ch_idx,
        "mean_COVID": v_cov.mean(axis=0),
        "mean_CONTROL": v_ctr.mean(axis=0),
        "cliffs_delta": delta,
        "abs_delta": np.abs(delta),
        "p_raw": p_raw,
        "q_fdr": q_fdr,
        "sig_fdr05": q_fdr < 0.05,
    })
    edge_df = edge_df.sort_values("abs_delta", ascending=False)
    out_path = TABLES_DIR / f"covid_vs_control_all_edges_tests_{tag}.csv"
    edge_df.to_csv(out_path, index=False)
    n_sig = int(edge_df["sig_fdr05"].sum())
    print(f"    ALL edges tested: {n_e}")
    print(f"    FDR<0.05 significant: {n_sig}")
    print(f"    |δ| median={np.median(np.abs(delta)):.4f}, "
          f"max={np.max(np.abs(delta)):.4f}")

    # ── Fisher enrichment: top 5% by |δ| vs AD-signature ────────
    top_pct = max(1, int(0.05 * n_e))    # ≈ 426
    _top = edge_df.head(top_pct)
    hit_set = set(zip(_top["i"].values, _top["j"].values))

    overlap, odds_ratio, p_fisher = fisher_enrichment(
        hit_set, sig_edge_set, N_EDGES_UNIVERSE,
    )

    # FDR-sig set too (if any)
    if n_sig > 0:
        _sigdf = edge_df[edge_df["sig_fdr05"]]
        sig_set_fdr = set(zip(_sigdf["i"].values, _sigdf["j"].values))
        ov2, or2, p2 = fisher_enrichment(
            sig_set_fdr, sig_edge_set, N_EDGES_UNIVERSE,
        )
    else:
        ov2, or2, p2 = 0, np.nan, np.nan

    print(f"    Enrichment (top 5% |δ| ∩ sig): overlap={overlap}/{top_pct}, "
          f"OR={odds_ratio:.2f}, p={p_fisher:.4g}")
    if n_sig > 0:
        print(f"    Enrichment (FDR<0.05 ∩ sig):    overlap={ov2}/{n_sig}, "
              f"OR={or2:.2f}, p={p2:.4g}")

    # ── Permutation sanity check ────────────────────────────────
    rng = np.random.RandomState(SEED)
    perm_overlaps = np.empty(N_PERM, dtype=int)
    _all_edges_arr = np.column_stack([_triu_i, _triu_j])

    for pi in range(N_PERM):
        idx_perm = rng.choice(n_e, size=len(hit_set), replace=False)
        perm_set = set(map(tuple, _all_edges_arr[idx_perm]))
        perm_overlaps[pi] = len(perm_set & sig_edge_set)

    perm_p    = float((perm_overlaps >= overlap).mean())
    perm_mean = float(perm_overlaps.mean())
    perm_std  = float(perm_overlaps.std())
    perm_95   = float(np.percentile(perm_overlaps, 95))

    print(f"    Permutation ({N_PERM}): observed={overlap}, "
          f"null mean={perm_mean:.1f}±{perm_std:.1f}, "
          f"null 95th={perm_95:.0f}, p_perm={perm_p:.4g}")

    # Save enrichment results
    enrich_rows = [
        {
            "ch_idx": ch_idx, "test": "top5pct_by_abs_delta_UNIVERSE",
            "n_hit": len(hit_set), "n_sig_ref": N_SIG_EDGES,
            "universe": N_EDGES_UNIVERSE,
            "overlap": overlap, "odds_ratio": odds_ratio, "p_fisher": p_fisher,
            "n_perm": N_PERM, "perm_mean_overlap": perm_mean,
            "perm_std_overlap": perm_std, "perm_95th": perm_95,
            "p_perm": perm_p,
        },
        {
            "ch_idx": ch_idx, "test": "fdr05_sig_UNIVERSE",
            "n_hit": n_sig, "n_sig_ref": N_SIG_EDGES,
            "universe": N_EDGES_UNIVERSE,
            "overlap": ov2, "odds_ratio": or2, "p_fisher": p2,
            "n_perm": np.nan, "perm_mean_overlap": np.nan,
            "perm_std_overlap": np.nan, "perm_95th": np.nan,
            "p_perm": np.nan,
        },
    ]
    enrich_df = pd.DataFrame(enrich_rows)
    enrich_path = TABLES_DIR / f"enrichment_signature_vs_top5pct_universe_{tag}.csv"
    enrich_df.to_csv(enrich_path, index=False)
    print(f"    → {out_path.name}")
    print(f"    → {enrich_path.name}")

print("\n  ✓ A) Complete.\n")


# ═══════════════════════════════════════════════════════════════════════
# ▶ B) Sign-agreement + Spearman edge–score association
# ═══════════════════════════════════════════════════════════════════════
print("=" * 72)
print("[3/6] B) Sign-agreement + Spearman edge–score association")
print("=" * 72)

_ei_sig = ce["i"].values
_ej_sig = ce["j"].values

for ci, ch_idx in enumerate(CHANNELS_TO_USE):
    ch_name = ch_names_sel[ci]
    tag = f"ch{ch_idx}"
    print(f"\n  ── Channel {ch_idx} ({ch_name[:50]}) ──")

    X = covid_tensor[:, ci, :, :]

    # ── B1: Sign agreement for SIGNATURE edges (within-COVID) ───
    v_ad = X[mask_adlike][:, _ei_sig, _ej_sig]
    v_cn = X[mask_cnlike][:, _ei_sig, _ej_sig]

    mean_diff = v_ad.mean(axis=0) - v_cn.mean(axis=0)
    sign_observed = np.sign(mean_diff)
    sign_signature = (
        np.sign(ce["w_signed"].values) if "w_signed" in ce.columns else None
    )

    if sign_signature is not None:
        # Remove edges where either sign is exactly 0
        valid = (sign_observed != 0) & (sign_signature != 0)
        n_valid = int(valid.sum())
        agree = int((sign_observed[valid] == sign_signature[valid]).sum())
        pct_agree = 100.0 * agree / n_valid if n_valid > 0 else 0.0

        # Binomial test: H0 = 50% agreement by chance
        binom_res = binomtest(agree, n_valid, 0.5, alternative="greater")
        p_binom = binom_res.pvalue

        sign_df = pd.DataFrame({
            "ch_idx": [ch_idx],
            "n_edges": [N_SIG_EDGES],
            "n_valid_sign": [n_valid],
            "n_agree": [agree],
            "pct_agree": [pct_agree],
            "p_binomial": [p_binom],
            "interpretation": [
                "above chance" if p_binom < 0.05 else "not significant"
            ],
        })
        sign_path = TABLES_DIR / f"within_covid_sign_agreement_{tag}.csv"
        sign_df.to_csv(sign_path, index=False)
        print(f"    Sign agreement: {agree}/{n_valid} ({pct_agree:.1f}%), "
              f"p_binom={p_binom:.4g}")
    else:
        print("    ⚠ w_signed not in consensus_edges_mapped → skip sign test")

    # ── B2: Spearman (edge value, score) for ALL 8515 edges ─────
    # Within Long-COVID only
    X_covid_only = X[mask_covid]
    scores_within = covid_scores[mask_covid]
    valid_s = np.isfinite(scores_within)
    X_val = X_covid_only[valid_s]
    sc_val = scores_within[valid_s]

    v_all = X_val[:, _triu_i, _triu_j]                # (n, 8515)
    n_e = v_all.shape[1]
    rho_arr = np.empty(n_e)
    p_sp    = np.empty(n_e)

    for k in range(n_e):
        r, p = spearmanr(v_all[:, k], sc_val)
        rho_arr[k] = r
        p_sp[k]    = p

    _, q_sp, _, _ = multipletests(p_sp, method="fdr_bh")

    assoc_df = pd.DataFrame({
        "i": _triu_i, "j": _triu_j,
        "roi_i": roi_names[_triu_i], "roi_j": roi_names[_triu_j],
        "ch_idx": ch_idx,
        "spearman_rho": rho_arr,
        "abs_rho": np.abs(rho_arr),
        "p_spearman": p_sp,
        "q_fdr": q_sp,
        "sig_fdr05": q_sp < 0.05,
    })
    assoc_df = assoc_df.sort_values("abs_rho", ascending=False)
    assoc_path = TABLES_DIR / f"within_covid_edge_score_assoc_all_edges_{tag}.csv"
    assoc_df.to_csv(assoc_path, index=False)

    n_sp_sig = int(assoc_df["sig_fdr05"].sum())
    print(f"    Spearman |ρ| median={np.median(np.abs(rho_arr)):.4f}, "
          f"max={np.max(np.abs(rho_arr)):.4f}")
    print(f"    FDR<0.05 edges: {n_sp_sig}/{n_e}")

    # Fisher enrichment: top 5% by |ρ| vs AD-signature
    top_rho_n = max(1, int(0.05 * n_e))
    _top_rho = assoc_df.head(top_rho_n)
    rho_hit = set(zip(_top_rho["i"].values, _top_rho["j"].values))
    ov_r, or_r, p_r = fisher_enrichment(rho_hit, sig_edge_set, N_EDGES_UNIVERSE)
    print(f"    Enrichment (top 5% |ρ| ∩ sig): overlap={ov_r}/{top_rho_n}, "
          f"OR={or_r:.2f}, p={p_r:.4g}")
    print(f"    → {assoc_path.name}")

print("\n  ✓ B) Complete.\n")


# ═══════════════════════════════════════════════════════════════════════
# ▶ C) ADNI-derived OOD thresholds
# ═══════════════════════════════════════════════════════════════════════
print("=" * 72)
print("[4/6] C) ADNI-derived OOD thresholds")
print("=" * 72)

from betavae_xai.models import ConvolutionalVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

adni_recon_p90_per_fold: List[float] = []
adni_recon_p95_per_fold: List[float] = []
adni_recon_stats: List[Dict[str, Any]] = []

for fold_idx in FOLD_IDS:
    fold_dir = RESULTS_DIR / f"fold_{fold_idx}"
    vae_path  = fold_dir / f"vae_model_fold_{fold_idx}.pt"
    norm_path = fold_dir / "vae_norm_params.joblib"
    td_path   = fold_dir / "train_dev_tensor_idx.npy"

    if not all(p.exists() for p in [vae_path, norm_path, td_path]):
        print(f"  ⚠ Fold {fold_idx}: missing artifacts, skipping")
        continue

    norm_params = joblib.load(norm_path)
    td_idx = np.load(td_path)

    # Select ADNI train-dev subjects (already channel-selected)
    adni_sel = adni_tensor[td_idx]

    # Normalise using fold's saved parameters
    adni_norm = apply_normalization_params(adni_sel, norm_params)

    # Load VAE
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

    state_dict = torch.load(vae_path, map_location=device, weights_only=True)
    vae.load_state_dict(state_dict)
    vae.eval()

    # Forward pass → reconstruction error on train-dev
    _, recon = encode_and_reconstruct(vae, adni_norm, device)
    recon_err = compute_recon_error_offdiag(adni_norm, recon)

    p90 = float(np.percentile(recon_err, 90))
    p95 = float(np.percentile(recon_err, 95))
    adni_recon_p90_per_fold.append(p90)
    adni_recon_p95_per_fold.append(p95)
    adni_recon_stats.append({
        "fold": fold_idx,
        "n_train_dev": len(td_idx),
        "recon_mean": float(recon_err.mean()),
        "recon_std": float(recon_err.std()),
        "recon_p90": p90,
        "recon_p95": p95,
        "recon_min": float(recon_err.min()),
        "recon_max": float(recon_err.max()),
    })
    print(f"    Fold {fold_idx}: n={len(td_idx)}, "
          f"recon mean={recon_err.mean():.4f}, P90={p90:.4f}, P95={p95:.4f}")

    # Free GPU memory
    del vae, adni_norm, recon
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Ensemble: median across folds
adni_p90_ens: Optional[float] = None
adni_p95_ens: Optional[float] = None
old_ood_count: Any = "N/A"
new_ood_count: Any = "N/A"

if adni_recon_p90_per_fold:
    adni_p90_ens = float(np.median(adni_recon_p90_per_fold))
    adni_p95_ens = float(np.median(adni_recon_p95_per_fold))
    print(f"\n  ADNI-derived OOD thresholds (median across folds):")
    print(f"    P90 = {adni_p90_ens:.6f}")
    print(f"    P95 = {adni_p95_ens:.6f}")

    # Save per-fold + ensemble stats
    adni_stats_df = pd.DataFrame(adni_recon_stats)
    adni_stats_df.loc[len(adni_stats_df)] = {
        "fold": "ensemble_median",
        "n_train_dev": np.nan,
        "recon_mean": np.nan, "recon_std": np.nan,
        "recon_p90": adni_p90_ens, "recon_p95": adni_p95_ens,
        "recon_min": np.nan, "recon_max": np.nan,
    }
    adni_stats_path = TABLES_DIR / "adni_ood_recon_distribution.csv"
    adni_stats_df.to_csv(adni_stats_path, index=False)
    print(f"  → {adni_stats_path.name}")

    # ── Update COVID quadrant table ──────────────────────────────
    quad_path = TABLES_DIR / "covid_ood_quadrants_logreg.csv"
    quad_df = pd.read_csv(quad_path)
    quad_df["SubjectID"] = quad_df["SubjectID"].astype(str)

    # New OOD flags
    quad_df["ood_flag_adni_p90"] = (
        quad_df["recon_error_ensemble"] >= adni_p90_ens
    ).astype(int)
    quad_df["ood_flag_adni_p95"] = (
        quad_df["recon_error_ensemble"] >= adni_p95_ens
    ).astype(int)
    quad_df["adni_ood_p90_threshold"] = adni_p90_ens
    quad_df["adni_ood_p95_threshold"] = adni_p95_ens

    _sc = "y_score_ensemble"

    # --- Quadrant based on ADNI P90 ---
    quad_df["quadrant_adni_p90"] = "CN-like_InDist"
    quad_df.loc[
        (quad_df[_sc] >= t_youden) & (quad_df["ood_flag_adni_p90"] == 0),
        "quadrant_adni_p90",
    ] = "AD-like_InDist"
    quad_df.loc[
        (quad_df[_sc] >= t_youden) & (quad_df["ood_flag_adni_p90"] == 1),
        "quadrant_adni_p90",
    ] = "AD-like_OOD"
    quad_df.loc[
        (quad_df[_sc] < t_youden) & (quad_df["ood_flag_adni_p90"] == 1),
        "quadrant_adni_p90",
    ] = "CN-like_OOD"

    # --- Quadrant based on ADNI P95 ---
    quad_df["quadrant_adni_p95"] = "CN-like_InDist"
    quad_df.loc[
        (quad_df[_sc] >= t_youden) & (quad_df["ood_flag_adni_p95"] == 0),
        "quadrant_adni_p95",
    ] = "AD-like_InDist"
    quad_df.loc[
        (quad_df[_sc] >= t_youden) & (quad_df["ood_flag_adni_p95"] == 1),
        "quadrant_adni_p95",
    ] = "AD-like_OOD"
    quad_df.loc[
        (quad_df[_sc] < t_youden) & (quad_df["ood_flag_adni_p95"] == 1),
        "quadrant_adni_p95",
    ] = "CN-like_OOD"

    quad_df.to_csv(quad_path, index=False)
    print(f"  Updated → {quad_path.name}")

    # Quadrant summary
    print("\n  ADNI-P90 quadrant summary:")
    for q in ["AD-like_InDist", "AD-like_OOD", "CN-like_InDist", "CN-like_OOD"]:
        n = int((quad_df["quadrant_adni_p90"] == q).sum())
        print(f"    {q:20s}: {n}")

    old_ood_count = (
        int(quad_df["ood_flag"].sum()) if "ood_flag" in quad_df.columns else "N/A"
    )
    new_ood_count = int(quad_df["ood_flag_adni_p90"].sum())
    print(f"\n  Comparison: old COVID-P90 OOD={old_ood_count}, "
          f"new ADNI-P90 OOD={new_ood_count}")
else:
    print("  ⚠ No fold data available — ADNI OOD skipped.")

print("\n  ✓ C) Complete.\n")


# ═══════════════════════════════════════════════════════════════════════
# ▶ D) Guard against 0-byte figure saves
# ═══════════════════════════════════════════════════════════════════════
print("=" * 72)
print("[5/6] D) Checking figure file integrity")
print("=" * 72)

_zero_byte: List[str] = []
_ok = 0
for fig_path in sorted(FIGURES_DIR.glob("*.png")):
    sz = fig_path.stat().st_size
    if sz == 0:
        _zero_byte.append(fig_path.name)
    else:
        _ok += 1

if _zero_byte:
    print(f"  ⚠ Found {len(_zero_byte)} zero-byte figures:")
    for fn in _zero_byte:
        print(f"      {fn}")
    print("  → Re-execute the notebook to regenerate these.")
else:
    print(f"  All {_ok} figures OK (>0 bytes).")

print("\n  ✓ D) Complete.\n")


# ═══════════════════════════════════════════════════════════════════════
# ▶ E) QC summary file
# ═══════════════════════════════════════════════════════════════════════
print("=" * 72)
print("[6/6] E) Writing QC summary")
print("=" * 72)

qc_lines = [
    "=" * 72,
    "  QC Summary — analysis_covid_paper_fixes.py v3.2",
    f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
    f"  Git hash:  {_git_hash()}",
    "=" * 72,
    "",
    "── Data dimensions ────────────────────────────────────────────────",
    f"  COVID tensor shape    : {tuple(covid_tensor.shape)}",
    f"  ADNI tensor shape     : {tuple(adni_tensor.shape)}",
    f"  N ROIs                : {N_ROIS}",
    f"  N edges (universe)    : {N_EDGES_UNIVERSE}",
    f"  N signature edges     : {N_SIG_EDGES}",
    f"  Channels used         : {CHANNELS_TO_USE} → {ch_names_sel}",
    "",
    "── Cohort ─────────────────────────────────────────────────────────",
    f"  Total subjects        : {len(covid_ids)}",
    f"  Long-COVID            : {n1_covid}",
    f"  Controls              : {n2_ctrl}",
    f"  Within COVID AD-like  : {n_ad}",
    f"  Within COVID CN-like  : {n_cn}",
    f"  Youden threshold      : {t_youden:.6f}",
    "",
    "── A) Enrichment (COVID vs CTRL, full universe) ──────────────────",
]

for ci, ch_idx in enumerate(CHANNELS_TO_USE):
    tag = f"ch{ch_idx}"
    ep = TABLES_DIR / f"enrichment_signature_vs_top5pct_universe_{tag}.csv"
    if ep.exists():
        _e = pd.read_csv(ep)
        _row = _e[_e["test"] == "top5pct_by_abs_delta_UNIVERSE"].iloc[0]
        qc_lines.append(
            f"  ch{ch_idx}: overlap={int(_row['overlap'])}/{int(_row['n_hit'])}, "
            f"OR={_row['odds_ratio']:.2f}, p_fisher={_row['p_fisher']:.4g}, "
            f"p_perm={_row['p_perm']:.4g}"
        )

qc_lines += [
    "",
    "── B) Sign-agreement (within-COVID, signature edges) ─────────────",
]
for ci, ch_idx in enumerate(CHANNELS_TO_USE):
    tag = f"ch{ch_idx}"
    sp = TABLES_DIR / f"within_covid_sign_agreement_{tag}.csv"
    if sp.exists():
        _s = pd.read_csv(sp)
        r = _s.iloc[0]
        qc_lines.append(
            f"  ch{ch_idx}: {int(r['n_agree'])}/{int(r['n_valid_sign'])} "
            f"({r['pct_agree']:.1f}%), p_binom={r['p_binomial']:.4g}"
        )
    ap = TABLES_DIR / f"within_covid_edge_score_assoc_all_edges_{tag}.csv"
    if ap.exists():
        _a = pd.read_csv(ap)
        n_sp = int(_a["sig_fdr05"].sum())
        qc_lines.append(
            f"         Spearman: {n_sp}/{len(_a)} FDR-sig, "
            f"|ρ| max={_a['abs_rho'].max():.4f}"
        )

qc_lines += [
    "",
    "── C) ADNI-derived OOD thresholds ────────────────────────────────",
]
if adni_p90_ens is not None:
    qc_lines.append(f"  ADNI P90 (ensemble median): {adni_p90_ens:.6f}")
    qc_lines.append(f"  ADNI P95 (ensemble median): {adni_p95_ens:.6f}")
    for row in adni_recon_stats:
        qc_lines.append(
            f"    Fold {row['fold']}: n={row['n_train_dev']}, "
            f"P90={row['recon_p90']:.6f}, P95={row['recon_p95']:.6f}"
        )
    qc_lines.append(
        f"  OOD count: old (COVID P90)={old_ood_count}, "
        f"new (ADNI P90)={new_ood_count}"
    )
else:
    qc_lines.append("  Not computed (missing fold artifacts)")

qc_lines += [
    "",
    "── D) Figure integrity ────────────────────────────────────────────",
]
if _zero_byte:
    qc_lines.append(f"  ⚠ {len(_zero_byte)} zero-byte figures")
    for fn in _zero_byte:
        qc_lines.append(f"      {fn}")
else:
    qc_lines.append(f"  All {_ok} figures OK")

qc_lines += [
    "",
    "── Acceptance checklist ────────────────────────────────────────────",
    f"  [{'x' if N_EDGES_UNIVERSE == 8515 else ' '}] Universe = 8515 edges",
    f"  [{'x' if N_SIG_EDGES > 0 else ' '}] Signature edges mapped = {N_SIG_EDGES}",
    f"  [{'x' if not _zero_byte else ' '}] All figures >0 bytes",
    f"  [{'x' if adni_p90_ens is not None else ' '}] ADNI OOD thresholds computed",
    "",
    f"  SEED: {SEED}",
    "=" * 72,
]

qc_text = "\n".join(qc_lines)
qc_path = OUTPUT_DIR / "qc_summary_v3.2.txt"
qc_path.write_text(qc_text, encoding="utf-8")
print(qc_text)
print(f"\n  → {qc_path.name}")

print("\n" + "=" * 72)
print("  All fixes (A–E) complete.")
print("=" * 72)
