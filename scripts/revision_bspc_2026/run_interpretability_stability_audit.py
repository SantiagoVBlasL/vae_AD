#!/usr/bin/env python3
"""
Interpretability stability audit: locked_v5p1b (ld256 beta2.5) vs
recover035_latent384_beta3p75 OOF-logitz (ld384 beta3.75).

Read-only. No retraining. No model selection. No threshold fitting.

Tasks:
  1. SHAP latent rankings via LR coefficients (exact for linear models)
  2. IG backprojection (input connectome → classifier score) for both models
  3. Consensus edges: overlap, Yeo-17 network analysis, sign consistency
  4. Channel contributions from IG
  5. Recommendation: can promoted replace locked for interpretation?

Output: results/revision_bspc_2026/oof_logitz_interpretability_stability_audit/
"""

from __future__ import annotations
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
import joblib
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from betavae_xai.models.convolutional_vae import ConvolutionalVAE
from betavae_xai.interpretability.interpret_fold import (
    apply_normalization_params,
    _load_global_and_merge,
    _subset_cnad,
    generate_saliency_ig_classifier_score,
)
from betavae_xai.interpretability.composite_edge_shap import (
    extract_logreg_latent_weights,
)

# ─── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "revision_bspc_2026"

LOCKED_RUN = RES / "adni_v5_1_batch20260514b_ch1_0_2_mfrsplit_3840_final_candidate"
LOCKED_LATENT = RES / "adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep" / "latent_cache"
LOCKED_META = Path("/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
                   "adni_expanded_v5_1_batch20260514b_no_pybandpass/"
                   "training_ready_metadata_v5_1_batch20260514b_no_pybandpass.csv")

PROMOTED_RUN = RES / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
PROMOTED_LATENT = PROMOTED_RUN / "classifier_only_readout" / "latent_cache"
PROMOTED_META = RES / "adni_035_metadata_rescue_preflight" / "patched_metadata_candidate.csv"

GLOBAL_TENSOR = Path("/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
                     "adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors/"
                     "GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz")

ROI_INFO_CSV = PROMOTED_RUN / "roi_info_from_tensor.csv"
OLD_CONSENSUS_CSV = (ROOT / "results" / "vae_3channels_beta65_pro" /
                     "interpretability_paper_output" / "tables" /
                     "consensus_set_logreg_integrated_gradients_top50.csv")

OUT_DIR = RES / "oof_logitz_interpretability_stability_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS = [1, 0, 2]
CHANNEL_NAMES = ["Pearson_Full", "Pearson_OMST", "MI"]
N_ROIS = 131
FOLDS = [1, 2, 3, 4, 5]
LOCKED_DIM = 256
PROMOTED_DIM = 384
TOP_K_LATENT = 20
TOP_K_EDGES = 20
IG_STEPS = 50
SEED = 42
CONSENSUS_MIN_FOLDS = 3


# ─── VAE builder ──────────────────────────────────────────────────────────────
def build_vae(fold_dir: Path, latent_dim: int, n_channels: int,
              device: torch.device) -> ConvolutionalVAE:
    ckpt = fold_dir / f"vae_model_fold_{fold_dir.name.split('_')[1]}.pt"
    vae = ConvolutionalVAE(
        input_channels=n_channels,
        latent_dim=latent_dim,
        image_size=N_ROIS,
        dropout_rate=0.15,
        use_layernorm_fc=False,
        num_conv_layers_encoder=4,
        decoder_type="convtranspose",
        intermediate_fc_dim_config="quarter",
        final_activation="tanh",
        num_groups=16,
    ).to(device)
    state = torch.load(ckpt, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    vae.load_state_dict(state, strict=True)
    vae.eval()
    return vae


# ─── SHAP via LR coefficients ─────────────────────────────────────────────────
def compute_shap_importance(run_dir: Path, latent_dim: int,
                            meta_cols: List[str]) -> pd.DataFrame:
    """
    For each fold: extract LR coef in raw (μ) space.
    Returns DataFrame: fold | latent_dim_idx | coef_raw | abs_coef | rank_in_fold
    Plus metadata coef rows.
    """
    rows = []
    for fold in FOLDS:
        fold_dir = run_dir / f"fold_{fold}"
        pipe_path = fold_dir / f"classifier_logreg_raw_pipeline_fold_{fold}.joblib"
        fc_path = fold_dir / "feature_columns.json"
        if not pipe_path.exists() or not fc_path.exists():
            print(f"  [SHAP] Missing pipeline for fold {fold} at {run_dir.name}")
            continue
        pipe = joblib.load(pipe_path)
        with open(fc_path) as f:
            fc = json.load(f)
        feature_columns = fc["final_feature_columns"]
        w_latent, bias, _, _, status = extract_logreg_latent_weights(
            pipe, latent_dim, meta_cols, feature_columns
        )
        if status != "ok" or w_latent is None:
            print(f"  [SHAP] fold {fold}: {status}")
            continue
        # Also extract metadata coef directly from model
        model = pipe.named_steps["model"]
        coef_proc = model.coef_.ravel()
        for i in range(latent_dim):
            rows.append({
                "fold": fold,
                "feature_type": "latent",
                "feature": f"latent_{i}",
                "coef_raw": float(w_latent[i]),
                "abs_coef": float(abs(w_latent[i])),
            })
        # Add metadata coef from processed space (these are standardized)
        meta_start = latent_dim
        for j, col in enumerate(meta_cols):
            col_idx = meta_start + j
            if col_idx < len(coef_proc):
                rows.append({
                    "fold": fold,
                    "feature_type": "metadata",
                    "feature": col,
                    "coef_raw": float(coef_proc[col_idx]),
                    "abs_coef": float(abs(coef_proc[col_idx])),
                })
    df = pd.DataFrame(rows)
    # Add fold-level rank
    def _rank(grp):
        grp = grp.copy()
        grp["rank_in_fold"] = grp["abs_coef"].rank(ascending=False).astype(int)
        return grp
    df = df.groupby("fold", group_keys=False).apply(_rank)
    return df


# ─── IG backprojection ────────────────────────────────────────────────────────
def run_ig_for_fold(fold_dir: Path, fold: int, latent_dim: int,
                    meta_cols: List[str], feature_columns: List[str],
                    tensor_all: np.ndarray, cnad: pd.DataFrame,
                    device: torch.device) -> Optional[Dict]:
    """Run IG for one fold. Returns attribution maps and metadata."""
    pipe_path = fold_dir / f"classifier_logreg_raw_pipeline_fold_{fold}.joblib"
    norm_path = fold_dir / "vae_norm_params.joblib"
    test_idx_path = fold_dir / "test_indices.npy"
    if not all(p.exists() for p in [pipe_path, norm_path, test_idx_path]):
        print(f"  [IG] Missing files for fold {fold}")
        return None
    pipe = joblib.load(pipe_path)
    norm_params = joblib.load(norm_path)
    test_idx = np.load(test_idx_path)
    test_df = cnad.iloc[test_idx].copy()
    gidx = test_df["tensor_idx"].values
    labels = (test_df["ResearchGroup_Mapped"] == "AD").astype(int).values
    tens = tensor_all[gidx][:, CHANNELS, :, :]
    tens = apply_normalization_params(tens, norm_params)
    tens_t = torch.from_numpy(tens).float()
    x_ad = tens_t[labels == 1]
    x_cn = tens_t[labels == 0]
    vae = build_vae(fold_dir, latent_dim, len(CHANNELS), device)
    # CN median train baseline (leakage-safe: uses training fold)
    train_idx = np.setdiff1d(np.arange(len(cnad)), test_idx)
    train_df = cnad.iloc[train_idx]
    cn_train_idx = train_df[train_df["ResearchGroup_Mapped"] == "CN"]["tensor_idx"].values
    cn_train_tens = tensor_all[cn_train_idx][:, CHANNELS, :, :]
    cn_train_tens = apply_normalization_params(cn_train_tens, norm_params)
    baseline = torch.from_numpy(np.median(cn_train_tens, axis=0)).float()
    sal_ad_s, sal_ad_a, st_ad = generate_saliency_ig_classifier_score(
        vae, pipe, x_ad, device, latent_dim, meta_cols, feature_columns,
        baseline=baseline, n_steps=IG_STEPS,
    )
    sal_cn_s, sal_cn_a, st_cn = generate_saliency_ig_classifier_score(
        vae, pipe, x_cn, device, latent_dim, meta_cols, feature_columns,
        baseline=baseline, n_steps=IG_STEPS,
    )
    if sal_ad_s is None or sal_cn_s is None:
        print(f"  [IG] fold {fold}: AD={st_ad} CN={st_cn}")
        return None
    diff_signed = sal_ad_s - sal_cn_s
    return {
        "fold": fold,
        "sal_ad_signed": sal_ad_s,
        "sal_cn_signed": sal_cn_s,
        "diff_signed": diff_signed,
        "n_ad": int(x_ad.shape[0]),
        "n_cn": int(x_cn.shape[0]),
    }


# ─── Edge analysis ────────────────────────────────────────────────────────────
def ig_map_to_top_edges(diff_signed: np.ndarray, roi_names: List[str],
                         network_labels: List[str], k: int = TOP_K_EDGES
                         ) -> pd.DataFrame:
    """
    Project multi-channel IG diff map to top-k edges.
    diff_signed: (n_channels, n_rois, n_rois)
    Aggregation: sum across channels (preserving sign), then take upper triangle.
    """
    agg = diff_signed.sum(axis=0)  # (n_rois, n_rois)
    # Symmetrize
    agg_sym = (agg + agg.T) / 2.0
    n = agg_sym.shape[0]
    rows, cols = np.triu_indices(n, k=1)
    values = agg_sym[rows, cols]
    abs_values = np.abs(values)
    top_idx = np.argsort(abs_values)[::-1][:k]
    records = []
    for i in top_idx:
        r, c = rows[i], cols[i]
        records.append({
            "edge_key": f"({roi_names[r]!r}, {roi_names[c]!r})",
            "src": roi_names[r],
            "dst": roi_names[c],
            "src_network": network_labels[r],
            "dst_network": network_labels[c],
            "ig_diff_signed": float(values[i]),
            "ig_diff_abs": float(abs_values[top_idx[list(top_idx).index(i)]]),
            "sign": 1 if values[i] > 0 else -1,
        })
    return pd.DataFrame(records)


def consensus_edges(per_fold_dfs: List[pd.DataFrame],
                    min_folds: int = CONSENSUS_MIN_FOLDS) -> pd.DataFrame:
    """Find edges appearing in ≥ min_folds folds. Report frequency and sign consistency."""
    from collections import defaultdict
    edge_folds = defaultdict(list)
    edge_signs = defaultdict(list)
    edge_meta = {}
    for fold_df in per_fold_dfs:
        for _, row in fold_df.iterrows():
            k = row["edge_key"]
            edge_folds[k].append(row["fold"])
            edge_signs[k].append(row["sign"])
            if k not in edge_meta:
                edge_meta[k] = {
                    "src": row["src"],
                    "dst": row["dst"],
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
        consistent_sign = 1 if sign_sum > 0 else (-1 if sign_sum < 0 else 0)
        sign_consistency = abs(sign_sum) / freq
        meta = edge_meta[k]
        records.append({
            "edge_key": k,
            "src": meta["src"],
            "dst": meta["dst"],
            "src_network": meta["src_network"],
            "dst_network": meta["dst_network"],
            "freq": freq,
            "pi": round(freq / 5, 2),
            "sign": consistent_sign,
            "sign_consistency": round(sign_consistency, 2),
            "direction": "pro-AD" if consistent_sign > 0 else "pro-CN",
            "folds": sorted(folds),
        })
    if not records:
        return pd.DataFrame(columns=["edge_key", "src", "dst", "src_network", "dst_network",
                                     "freq", "pi", "sign", "sign_consistency", "direction", "folds"])
    df = pd.DataFrame(records).sort_values(["freq", "sign_consistency"],
                                            ascending=[False, False])
    return df.reset_index(drop=True)


def channel_contributions_from_ig(ig_results: List[Dict]) -> pd.DataFrame:
    """Per-channel L1 abs contribution fractions from IG diff maps."""
    rows = []
    for res in ig_results:
        if res is None:
            continue
        diff = res["diff_signed"]
        l1 = np.abs(diff).sum(axis=(1, 2))
        total = l1.sum() + 1e-12
        for i, ch in enumerate(CHANNEL_NAMES):
            rows.append({
                "fold": res["fold"],
                "channel": ch,
                "l1_abs": float(l1[i]),
                "fraction": float(l1[i] / total),
            })
    return pd.DataFrame(rows)


# ─── network analysis ─────────────────────────────────────────────────────────
def network_pair_analysis(consensus_df: pd.DataFrame) -> pd.DataFrame:
    """Count network pairs in consensus edges."""
    pairs = []
    for _, row in consensus_df.iterrows():
        a, b = sorted([row["src_network"], row["dst_network"]])
        pairs.append({
            "network_pair": f"{a} — {b}",
            "net_a": a,
            "net_b": b,
            "edge_key": row["edge_key"],
            "direction": row["direction"],
            "freq": row["freq"],
        })
    if not pairs:
        return pd.DataFrame()
    df = pd.DataFrame(pairs)
    summary = df.groupby("network_pair").agg(
        n_edges=("edge_key", "count"),
        mean_freq=("freq", "mean"),
        directions=("direction", lambda x: "/".join(sorted(set(x)))),
    ).reset_index().sort_values("n_edges", ascending=False)
    return summary


def dmn_limbic_visual_involvement(consensus_df: pd.DataFrame) -> Dict:
    """Check DMN, Limbic, Visual involvement in consensus edges."""
    if consensus_df.empty:
        return {}
    dmn_kws = ["DefaultMode"]
    limbic_kws = ["Limbic"]
    visual_kws = ["Visual"]

    def _match(row, kws):
        return any(k in row["src_network"] or k in row["dst_network"] for k in kws)

    n_dmn = int(consensus_df.apply(lambda r: _match(r, dmn_kws), axis=1).sum())
    n_limbic = int(consensus_df.apply(lambda r: _match(r, limbic_kws), axis=1).sum())
    n_visual = int(consensus_df.apply(lambda r: _match(r, visual_kws), axis=1).sum())
    n_total = len(consensus_df)
    return {
        "n_consensus_edges": n_total,
        "dmn_edges": n_dmn,
        "dmn_frac": round(n_dmn / n_total, 2) if n_total else 0,
        "limbic_edges": n_limbic,
        "limbic_frac": round(n_limbic / n_total, 2) if n_total else 0,
        "visual_edges": n_visual,
        "visual_frac": round(n_visual / n_total, 2) if n_total else 0,
    }


# ─── rank correlation ─────────────────────────────────────────────────────────
def shap_rank_correlation(df_locked: pd.DataFrame, df_promoted: pd.DataFrame,
                           top_k: int = TOP_K_LATENT) -> Dict:
    """
    Cross-model rank correlation of mean latent importance.
    Latent dims are NOT aligned 1:1 (different latent_dim), so we compare
    by the fraction of top-K importance that falls in AD-network-linked dims,
    and by fold-mean SHAP profile shape.
    """
    from scipy import stats

    # Pool across folds: mean abs coef per latent dim for each model
    def _mean_abs(df):
        lat = df[df["feature_type"] == "latent"].copy()
        return lat.groupby("feature")["abs_coef"].mean().reset_index()

    lock_mean = _mean_abs(df_locked).sort_values("abs_coef", ascending=False).reset_index(drop=True)
    prom_mean = _mean_abs(df_promoted).sort_values("abs_coef", ascending=False).reset_index(drop=True)

    # Fraction of total importance in top-K dims
    def _topk_frac(df_mean, k):
        total = df_mean["abs_coef"].sum()
        topk_sum = df_mean.head(k)["abs_coef"].sum()
        return float(topk_sum / total) if total > 0 else 0.0

    # Fold-level rank stability: Spearman corr of top-K dim indices across folds
    def _fold_rank_stability(df):
        lat = df[df["feature_type"] == "latent"]
        fold_ranks = {}
        for fold, grp in lat.groupby("fold"):
            top = grp.nlargest(top_k, "abs_coef")["feature"].tolist()
            fold_ranks[fold] = top
        # Pairwise overlap of top-K sets
        folds = list(fold_ranks.keys())
        overlaps = []
        for i in range(len(folds)):
            for j in range(i + 1, len(folds)):
                s1 = set(fold_ranks[folds[i]])
                s2 = set(fold_ranks[folds[j]])
                overlaps.append(len(s1 & s2) / top_k)
        return float(np.mean(overlaps)) if overlaps else 0.0

    lock_topk_frac = _topk_frac(lock_mean, top_k)
    prom_topk_frac = _topk_frac(prom_mean, top_k)
    lock_stability = _fold_rank_stability(df_locked)
    prom_stability = _fold_rank_stability(df_promoted)

    # Cross-model rank comparison: since latent dims are NOT aligned,
    # compare the signed direction of top importance dims (pro-AD fraction)
    def _pro_ad_frac_topk(df_mean, df_raw, k):
        top_feats = set(df_mean.head(k)["feature"].tolist())
        lat = df_raw[(df_raw["feature_type"] == "latent") & (df_raw["feature"].isin(top_feats))]
        mean_coef = lat.groupby("feature")["coef_raw"].mean()
        n_pos = int((mean_coef > 0).sum())
        return float(n_pos / len(mean_coef)) if len(mean_coef) > 0 else 0.0

    lock_proadfrac = _pro_ad_frac_topk(lock_mean, df_locked, top_k)
    prom_proadfrac = _pro_ad_frac_topk(prom_mean, df_promoted, top_k)

    # Age/Sex contribution
    def _meta_share(df):
        total_abs = df.groupby("fold")["abs_coef"].sum()
        meta_abs = df[df["feature_type"] == "metadata"].groupby("fold")["abs_coef"].sum()
        ratio = (meta_abs / total_abs).dropna()
        return float(ratio.mean()), float(ratio.std(ddof=1))

    lock_meta_mean, lock_meta_std = _meta_share(df_locked)
    prom_meta_mean, prom_meta_std = _meta_share(df_promoted)

    # Age vs Sex split
    def _agesex(df):
        meta = df[df["feature_type"] == "metadata"]
        out = {}
        for feat in ["Age", "Sex"]:
            sub = meta[meta["feature"] == feat]
            if len(sub):
                out[feat] = float(sub["abs_coef"].mean())
        return out

    return {
        "top_k": top_k,
        "locked_topk_importance_fraction": round(lock_topk_frac, 4),
        "promoted_topk_importance_fraction": round(prom_topk_frac, 4),
        "locked_fold_rank_stability_jaccard": round(lock_stability, 4),
        "promoted_fold_rank_stability_jaccard": round(prom_stability, 4),
        "locked_topk_proAD_fraction": round(lock_proadfrac, 4),
        "promoted_topk_proAD_fraction": round(prom_proadfrac, 4),
        "locked_meta_share_mean": round(lock_meta_mean, 4),
        "locked_meta_share_std": round(lock_meta_std, 4),
        "promoted_meta_share_mean": round(prom_meta_mean, 4),
        "promoted_meta_share_std": round(prom_meta_std, 4),
        "locked_agesex": _agesex(df_locked),
        "promoted_agesex": _agesex(df_promoted),
    }


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. Load global tensor and metadata ────────────────────────────────────
    print("\n[1] Loading global tensor...")
    tensor_all, merged_locked = _load_global_and_merge(GLOBAL_TENSOR, LOCKED_META)
    _, merged_promoted = _load_global_and_merge(GLOBAL_TENSOR, PROMOTED_META)
    cnad_locked = _subset_cnad(merged_locked)
    cnad_promoted = _subset_cnad(merged_promoted)
    print(f"  Tensor shape: {tensor_all.shape}")
    print(f"  locked CNAD: {len(cnad_locked)} | promoted CNAD: {len(cnad_promoted)}")

    # ROI info
    roi_df = pd.read_csv(ROI_INFO_CSV)
    roi_names = roi_df["roi_name_in_tensor"].astype(str).tolist()
    network_labels = roi_df["network_label_in_tensor"].astype(str).tolist()
    print(f"  ROIs: {len(roi_names)}")

    # ── 2. SHAP latent importance ─────────────────────────────────────────────
    print("\n[2] Computing SHAP latent importance (LR coefficients)...")
    meta_cols = ["Age", "Sex"]
    shap_locked = compute_shap_importance(LOCKED_RUN, LOCKED_DIM, meta_cols)
    shap_promoted = compute_shap_importance(PROMOTED_RUN, PROMOTED_DIM, meta_cols)
    shap_locked.to_csv(OUT_DIR / "shap_latent_importance_locked.csv", index=False)
    shap_promoted.to_csv(OUT_DIR / "shap_latent_importance_promoted.csv", index=False)

    # Top-K summary
    def _topk_summary(df, latent_dim, k=TOP_K_LATENT):
        lat = df[df["feature_type"] == "latent"]
        mean_abs = lat.groupby("feature")["abs_coef"].mean().reset_index()
        mean_abs["latent_idx"] = mean_abs["feature"].str.extract(r"latent_(\d+)").astype(int)
        mean_abs = mean_abs.sort_values("abs_coef", ascending=False).head(k)
        mean_coef = lat.groupby("feature")["coef_raw"].mean()
        mean_abs["mean_coef_signed"] = mean_abs["feature"].map(mean_coef)
        mean_abs["direction"] = mean_abs["mean_coef_signed"].apply(
            lambda x: "pro-AD" if x > 0 else "pro-CN"
        )
        return mean_abs[["latent_idx", "feature", "abs_coef", "mean_coef_signed", "direction"]]

    top_locked = _topk_summary(shap_locked, LOCKED_DIM)
    top_promoted = _topk_summary(shap_promoted, PROMOTED_DIM)
    top_locked.to_csv(OUT_DIR / f"shap_top{TOP_K_LATENT}_locked.csv", index=False)
    top_promoted.to_csv(OUT_DIR / f"shap_top{TOP_K_LATENT}_promoted.csv", index=False)
    print(f"  Locked top-5: {top_locked['feature'].head(5).tolist()}")
    print(f"  Promoted top-5: {top_promoted['feature'].head(5).tolist()}")

    rank_stats = shap_rank_correlation(shap_locked, shap_promoted)
    with open(OUT_DIR / "shap_rank_stats.json", "w") as f:
        json.dump(rank_stats, f, indent=2)
    print(f"  Locked meta share: {rank_stats['locked_meta_share_mean']:.4f} ± "
          f"{rank_stats['locked_meta_share_std']:.4f}")
    print(f"  Promoted meta share: {rank_stats['promoted_meta_share_mean']:.4f} ± "
          f"{rank_stats['promoted_meta_share_std']:.4f}")

    # ── 3. IG backprojection ──────────────────────────────────────────────────
    print("\n[3] Running IG backprojection...")
    ig_locked_results = []
    ig_promoted_results = []

    for fold in FOLDS:
        print(f"  Fold {fold}/5 — locked...", end=" ")
        fc_path = LOCKED_RUN / f"fold_{fold}" / "feature_columns.json"
        with open(fc_path) as f:
            fc_locked = json.load(f)["final_feature_columns"]
        res = run_ig_for_fold(
            LOCKED_RUN / f"fold_{fold}", fold, LOCKED_DIM, meta_cols,
            fc_locked, tensor_all, cnad_locked, device
        )
        ig_locked_results.append(res)
        print("done" if res else "FAILED")

        print(f"  Fold {fold}/5 — promoted...", end=" ")
        fc_path2 = PROMOTED_RUN / f"fold_{fold}" / "feature_columns.json"
        with open(fc_path2) as f:
            fc_promoted = json.load(f)["final_feature_columns"]
        res2 = run_ig_for_fold(
            PROMOTED_RUN / f"fold_{fold}", fold, PROMOTED_DIM, meta_cols,
            fc_promoted, tensor_all, cnad_promoted, device
        )
        ig_promoted_results.append(res2)
        print("done" if res2 else "FAILED")

    # ── 4. Per-fold edge tables ───────────────────────────────────────────────
    print("\n[4] Projecting IG maps to edges...")
    fold_edges_locked = []
    fold_edges_promoted = []
    for res in ig_locked_results:
        if res is None:
            continue
        df = ig_map_to_top_edges(res["diff_signed"], roi_names, network_labels)
        df["fold"] = res["fold"]
        fold_edges_locked.append(df)
    for res in ig_promoted_results:
        if res is None:
            continue
        df = ig_map_to_top_edges(res["diff_signed"], roi_names, network_labels)
        df["fold"] = res["fold"]
        fold_edges_promoted.append(df)

    pd.concat(fold_edges_locked, ignore_index=True).to_csv(
        OUT_DIR / "ig_fold_edges_locked.csv", index=False)
    pd.concat(fold_edges_promoted, ignore_index=True).to_csv(
        OUT_DIR / "ig_fold_edges_promoted.csv", index=False)

    # ── 5. Consensus edges ────────────────────────────────────────────────────
    print("\n[5] Computing consensus edges...")
    cons_locked = consensus_edges(fold_edges_locked)
    cons_promoted = consensus_edges(fold_edges_promoted)
    cons_locked.to_csv(OUT_DIR / "ig_consensus_edges_locked.csv", index=False)
    cons_promoted.to_csv(OUT_DIR / "ig_consensus_edges_promoted.csv", index=False)
    print(f"  Locked consensus edges (≥{CONSENSUS_MIN_FOLDS}/5): {len(cons_locked)}")
    print(f"  Promoted consensus edges (≥{CONSENSUS_MIN_FOLDS}/5): {len(cons_promoted)}")

    # Overlap between models
    lock_keys = set(cons_locked["edge_key"].tolist())
    prom_keys = set(cons_promoted["edge_key"].tolist())
    overlap_keys = lock_keys & prom_keys

    # Comparison with old beta65_pro consensus
    old_cons = pd.read_csv(OLD_CONSENSUS_CSV)
    old_keys = set(old_cons["edge_key"].tolist())

    overlap_with_old_locked = lock_keys & old_keys
    overlap_with_old_promoted = prom_keys & old_keys

    edge_overlap_stats = {
        "locked_n_consensus": len(lock_keys),
        "promoted_n_consensus": len(prom_keys),
        "overlap_locked_promoted": len(overlap_keys),
        "overlap_keys": sorted(overlap_keys),
        "old_consensus_n": len(old_keys),
        "locked_overlap_with_old": len(overlap_with_old_locked),
        "promoted_overlap_with_old": len(overlap_with_old_promoted),
        "locked_overlap_with_old_keys": sorted(overlap_with_old_locked),
        "promoted_overlap_with_old_keys": sorted(overlap_with_old_promoted),
    }
    with open(OUT_DIR / "edge_overlap_stats.json", "w") as f:
        json.dump(edge_overlap_stats, f, indent=2)
    print(f"  Overlap locked↔promoted: {len(overlap_keys)}")
    print(f"  Locked overlap with old model: {len(overlap_with_old_locked)}")
    print(f"  Promoted overlap with old model: {len(overlap_with_old_promoted)}")

    # ── 6. Channel contributions ──────────────────────────────────────────────
    print("\n[6] Channel contributions from IG...")
    ch_locked = channel_contributions_from_ig(ig_locked_results)
    ch_promoted = channel_contributions_from_ig(ig_promoted_results)
    ch_locked["candidate"] = "locked_v5p1b"
    ch_promoted["candidate"] = "promoted_beta3p75"
    ch_all = pd.concat([ch_locked, ch_promoted], ignore_index=True)
    ch_all.to_csv(OUT_DIR / "channel_contributions_ig.csv", index=False)
    # Summary
    ch_summary = ch_all.groupby(["candidate", "channel"])["fraction"].agg(
        ["mean", "std"]).round(4).reset_index()
    ch_summary.columns = ["candidate", "channel", "mean_fraction", "std_fraction"]
    ch_summary.to_csv(OUT_DIR / "channel_contributions_summary.csv", index=False)
    print(ch_summary.to_string(index=False))

    # Channel dist stats comparison from fold dist_norm.csv files
    dist_rows = []
    for candidate, run_dir in [("locked_v5p1b", LOCKED_RUN),
                                ("promoted_beta3p75", PROMOTED_RUN)]:
        for fold in FOLDS:
            fold_dir = run_dir / f"fold_{fold}"
            dist_path = fold_dir / f"fold_{fold}_dist_norm.csv"
            if dist_path.exists():
                df = pd.read_csv(dist_path)
                df["fold"] = fold
                df["candidate"] = candidate
                dist_rows.append(df)
    if dist_rows:
        dist_df = pd.concat(dist_rows, ignore_index=True)
        dist_df.to_csv(OUT_DIR / "channel_dist_norm_comparison.csv", index=False)
        dist_summary = dist_df.groupby(["candidate", "channel"])["std"].agg(["mean", "std"]).round(4)
        print("\n  Channel dist std comparison:")
        print(dist_summary.to_string())

    # ── 7. Network analysis ───────────────────────────────────────────────────
    print("\n[7] Network analysis...")
    net_locked = network_pair_analysis(cons_locked)
    net_promoted = network_pair_analysis(cons_promoted)
    net_locked["candidate"] = "locked_v5p1b"
    net_promoted["candidate"] = "promoted_beta3p75"
    net_all = pd.concat([net_locked, net_promoted], ignore_index=True)
    net_all.to_csv(OUT_DIR / "network_pair_analysis.csv", index=False)

    involve_locked = dmn_limbic_visual_involvement(cons_locked)
    involve_promoted = dmn_limbic_visual_involvement(cons_promoted)
    print(f"  Locked:   DMN={involve_locked.get('dmn_edges',0)}, "
          f"Limbic={involve_locked.get('limbic_edges',0)}, "
          f"Visual={involve_locked.get('visual_edges',0)}")
    print(f"  Promoted: DMN={involve_promoted.get('dmn_edges',0)}, "
          f"Limbic={involve_promoted.get('limbic_edges',0)}, "
          f"Visual={involve_promoted.get('visual_edges',0)}")

    # ── 8. Write final report ─────────────────────────────────────────────────
    _write_final_report(
        rank_stats, edge_overlap_stats, involve_locked, involve_promoted,
        cons_locked, cons_promoted, top_locked, top_promoted,
        ch_summary, net_locked, net_promoted,
    )

    # ── command log ──────────────────────────────────────────────────────────
    log_data = {
        "script": "scripts/revision_bspc_2026/run_interpretability_stability_audit.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "locked_run": str(LOCKED_RUN),
        "promoted_run": str(PROMOTED_RUN),
        "device": str(device),
        "ig_steps": IG_STEPS,
        "top_k_latent": TOP_K_LATENT,
        "top_k_edges": TOP_K_EDGES,
        "consensus_min_folds": CONSENSUS_MIN_FOLDS,
        "training_launched": False,
        "threshold_fitting": False,
        "tensor_modification": False,
    }
    with open(OUT_DIR / "command_log.json", "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"\nDone. Output: {OUT_DIR}")


# ─── report writer ────────────────────────────────────────────────────────────
def _write_final_report(rank_stats, edge_overlap_stats, involve_locked, involve_promoted,
                         cons_locked, cons_promoted, top_locked, top_promoted,
                         ch_summary, net_locked, net_promoted):
    lock_n = edge_overlap_stats["locked_n_consensus"]
    prom_n = edge_overlap_stats["promoted_n_consensus"]
    overlap_n = edge_overlap_stats["overlap_locked_promoted"]
    old_n = edge_overlap_stats["old_consensus_n"]
    old_lock_n = edge_overlap_stats["locked_overlap_with_old"]
    old_prom_n = edge_overlap_stats["promoted_overlap_with_old"]

    # Sign consistency check
    def _sign_consistent(df):
        if df.empty:
            return "N/A (no consensus edges)"
        n_consistent = int((df["sign_consistency"] == 1.0).sum())
        return f"{n_consistent}/{len(df)} fully sign-consistent"

    def _pro_ad_cn(df):
        if df.empty:
            return "N/A"
        n_ad = int((df["direction"] == "pro-AD").sum())
        n_cn = int((df["direction"] == "pro-CN").sum())
        return f"{n_ad} pro-AD / {n_cn} pro-CN"

    # Channel summary for report
    def _ch_row(candidate, ch):
        sub = ch_summary[(ch_summary["candidate"] == candidate) &
                          (ch_summary["channel"] == ch)]
        if sub.empty:
            return "N/A"
        return f"{sub.iloc[0]['mean_fraction']:.3f} ± {sub.iloc[0]['std_fraction']:.3f}"

    # Top latent dims for report
    def _fmt_top(df, n=5):
        top = df.head(n)
        return ", ".join(f"{r['feature']} ({r['direction']})" for _, r in top.iterrows())

    # Recommendation logic
    consensus_overlap_frac = overlap_n / max(lock_n, prom_n, 1)
    meta_diff = abs(rank_stats["locked_meta_share_mean"] - rank_stats["promoted_meta_share_mean"])
    dmn_consitent = (involve_locked.get("dmn_edges", 0) > 0 and
                      involve_promoted.get("dmn_edges", 0) > 0)
    sign_consistent_lock = "sign_consistency" in cons_locked.columns and (
        cons_locked["sign_consistency"] == 1.0).all() if not cons_locked.empty else False

    if consensus_overlap_frac >= 0.5 and meta_diff < 0.05:
        verdict = "REPLACE"
        rationale = ("Consensus edge overlap ≥50%, Age/Sex contribution profiles match. "
                     "The promoted model can replace the locked model as the interpretability reference.")
    elif consensus_overlap_frac >= 0.3 or meta_diff < 0.03:
        verdict = "SPLIT"
        rationale = ("Partial consensus overlap or small meta divergence. "
                     "Recommended: promoted model as classification primary, "
                     "locked model as legacy interpretability reference until full IG audit is complete.")
    else:
        verdict = "KEEP_LOCKED"
        rationale = ("Insufficient consensus edge overlap or significant meta contribution divergence. "
                     "Keep locked model as interpretability reference.")

    lines = [
        "# Interpretability Stability Audit: locked_v5p1b vs beta3p75 OOF-logitz",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Models compared",
        "- **A (locked)**: locked_v5p1b, latent_dim=256, beta=2.5, channels=[1,0,2]",
        "- **B (promoted)**: recover035_latent384_beta3p75 oof_logitz, latent_dim=384, beta=3.75, channels=[1,0,2]",
        "",
        "## 1. SHAP latent rankings (exact linear SHAP via LR coefficients)",
        "",
        f"Top-{TOP_K_LATENT} latent dims concentrated: "
        f"locked={rank_stats['locked_topk_importance_fraction']:.3f} | "
        f"promoted={rank_stats['promoted_topk_importance_fraction']:.3f} of total abs coef mass",
        "",
        f"Fold-level top-{TOP_K_LATENT} rank stability (mean pairwise Jaccard):",
        f"  locked={rank_stats['locked_fold_rank_stability_jaccard']:.3f} | "
        f"promoted={rank_stats['promoted_fold_rank_stability_jaccard']:.3f}",
        "",
        f"Pro-AD fraction in top-{TOP_K_LATENT} dims:",
        f"  locked={rank_stats['locked_topk_proAD_fraction']:.3f} | "
        f"promoted={rank_stats['promoted_topk_proAD_fraction']:.3f}",
        "",
        "**Locked top-5 dims**: " + _fmt_top(top_locked),
        "**Promoted top-5 dims**: " + _fmt_top(top_promoted),
        "",
        "## 2. Age/Sex contribution",
        "",
        f"| Model | Mean meta share | Std |",
        f"|---|---|---|",
        f"| locked_v5p1b | {rank_stats['locked_meta_share_mean']:.4f} | "
        f"{rank_stats['locked_meta_share_std']:.4f} |",
        f"| beta3p75_logitz | {rank_stats['promoted_meta_share_mean']:.4f} | "
        f"{rank_stats['promoted_meta_share_std']:.4f} |",
        "",
        f"Locked Age/Sex: {rank_stats['locked_agesex']}",
        f"Promoted Age/Sex: {rank_stats['promoted_agesex']}",
        "",
        "Note: Latent dims are not aligned across models (different latent_dim). "
        f"Rank correlation is not directly interpretable. Meta share comparison is the "
        "most reliable cross-model stability metric for classification bias.",
        "",
        "## 3. IG backprojection — consensus edges",
        "",
        f"Consensus criterion: edge in top-{TOP_K_EDGES} per fold, appearing in "
        f"≥{CONSENSUS_MIN_FOLDS}/5 folds.",
        "",
        f"| Model | n_consensus_edges |",
        f"|---|---|",
        f"| locked_v5p1b | {lock_n} |",
        f"| beta3p75_logitz | {prom_n} |",
        "",
        f"Cross-model overlap (locked ∩ promoted): **{overlap_n}** edges",
        f"Overlap fraction (of larger set): {consensus_overlap_frac:.2f}",
        "",
        f"Old beta65_pro consensus (n={old_n}) overlap:",
        f"  locked overlap with old: {old_lock_n}/{old_n}",
        f"  promoted overlap with old: {old_prom_n}/{old_n}",
        "",
        "### Sign consistency",
        f"  locked: {_sign_consistent(cons_locked)} | {_pro_ad_cn(cons_locked)}",
        f"  promoted: {_sign_consistent(cons_promoted)} | {_pro_ad_cn(cons_promoted)}",
        "",
        "## 4. Channel contributions (IG diff maps, mean fraction across folds)",
        "",
        "| Candidate | Pearson_Full | Pearson_OMST | MI |",
        "|---|---|---|---|",
        f"| locked_v5p1b | {_ch_row('locked_v5p1b','Pearson_Full')} | "
        f"{_ch_row('locked_v5p1b','Pearson_OMST')} | "
        f"{_ch_row('locked_v5p1b','MI')} |",
        f"| beta3p75_logitz | {_ch_row('promoted_beta3p75','Pearson_Full')} | "
        f"{_ch_row('promoted_beta3p75','Pearson_OMST')} | "
        f"{_ch_row('promoted_beta3p75','MI')} |",
        "",
        "## 5. Yeo-17 network involvement",
        "",
        "### locked_v5p1b consensus edges",
        f"DMN={involve_locked.get('dmn_edges',0)}/{lock_n}, "
        f"Limbic={involve_locked.get('limbic_edges',0)}/{lock_n}, "
        f"Visual={involve_locked.get('visual_edges',0)}/{lock_n}",
        "",
        "### beta3p75_logitz consensus edges",
        f"DMN={involve_promoted.get('dmn_edges',0)}/{prom_n}, "
        f"Limbic={involve_promoted.get('limbic_edges',0)}/{prom_n}, "
        f"Visual={involve_promoted.get('visual_edges',0)}/{prom_n}",
        "",
        "## Decision",
        "",
        f"**Verdict: {verdict}**",
        "",
        rationale,
        "",
        "### Context",
        "- Latent dims not aligned: direct dim-level rank correlation not possible.",
        "- OOF logit-z calibration is a post-processing step; it does not change the VAE "
          "or LR model parameters, only the score scale.",
        "- The promoted model has a larger latent space (384 vs 256) and higher beta (3.75 vs 2.5), "
          "which may concentrate information differently.",
        "- The IG backprojection uses the raw LR pipeline (Optuna-selected C), consistent "
          "with the trained classifier.",
        "- Old beta65_pro consensus edges are from a different model and dataset version; "
          "overlap is indicative, not definitive.",
        "",
        "## Read-only guarantee",
        "Did not retrain VAE. Did not fit thresholds. Did not modify tensors, metadata, or existing outputs.",
    ]

    with open(OUT_DIR / "final_report.md", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
