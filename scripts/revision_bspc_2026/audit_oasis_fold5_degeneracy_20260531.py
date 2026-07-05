#!/usr/bin/env python3
"""
Read-only fold-5 OASIS degeneracy audit.

Diagnoses why fold 5 of adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5
produces near-constant probabilities (range 0.474–0.520, std≈0.010) for all 120 OASIS
subjects, while folds 1–4 produce dispersed probabilities (std≈0.07–0.12).

Sections
  A. Per-fold probability statistics and per-fold AUC/PR-AUC
     - Leave-one-fold-out (LOFO) ensemble AUCs (diagnostic only)
  B. Latent diagnostics: encode OASIS with each fold VAE, compute mu statistics
     - Active units, norm distribution, mean/std per dim
     - Compare OASIS mu statistics to ADNI test set
  C. Input normalization diagnostics: compare fold-specific norm params
     - Apply each fold norm to OASIS; compare off-diagonal x_norm statistics
  D. Reconstruction diagnostics: ELBO / reconstruction loss per fold
  E. Classifier diagnostics: logreg intercept, coefficient L2 norm, decision_function range

Constraints
  - No training, no threshold fitting on OASIS, no model selection
  - All OASIS metrics are informational / diagnostic
  - Uses existing subject_scores.csv for section A
  - Runs VAE inference for sections B–D

Usage
  python scripts/revision_bspc_2026/audit_oasis_fold5_degeneracy_20260531.py [--device cpu]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "revision_bspc_2026"
if SRC_DIR.is_dir():
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from betavae_xai.data.preprocessing import apply_normalization_params
from sklearn.metrics import roc_auc_score, average_precision_score

RESULTS_DIR = PROJECT_ROOT / "results" / "revision_bspc_2026"
ADNI_RUN_DIR = RESULTS_DIR / "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5"
TENSOR_DIR = RESULTS_DIR / "oasis_next_60cn_60ad_tensor_build_pilot_parity_runwise_20260531"
SPLIT_CSV = RESULTS_DIR / "oasis_60cn_60ad_calibration_test_protocol" / "split_calibration_test.csv"
SCORES_CSV = (
    RESULTS_DIR
    / "oasis_next_60cn_60ad_external_scoring_pilot_parity_runwise_horizon4480_20260531"
    / "subject_scores.csv"
)
OUTPUT_DIR = RESULTS_DIR / "oasis_fold5_degeneracy_audit_20260531"

TENSOR_CANDIDATES = {
    "runwise_140TR_pilot_parity": "tensor_runwise_140TR_pilot_parity.npz",
    "runwise164_pilot_parity": "tensor_runwise164_pilot_parity.npz",
}
SELECTED_CHANNEL_NAMES = [
    "Pearson_Full_FisherZ_Signed",
    "Pearson_OMST_GCE_Signed_Weighted",
    "MI_KNN_Symmetric",
]
N_FOLDS = 5
LATENT_DIM = 256
BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def require(path: Path, label: str = "") -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _md_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".4f")


def write(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"  -> {path}")


# ---------------------------------------------------------------------------
# Model helpers (reuse scoring script logic)
# ---------------------------------------------------------------------------
def _load_vae_helpers():
    from run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep import encode_mu, make_model
    return encode_mu, make_model


def _run_cfg() -> Dict[str, Any]:
    cfg_raw = json.loads((ADNI_RUN_DIR / "run_config.json").read_text())
    return cfg_raw.get("args", {})


def _model_cfg() -> Dict[str, Any]:
    cfg = _run_cfg()
    return {
        "latent_dim": LATENT_DIM,
        "dropout_rate_vae": float(cfg.get("dropout_rate_vae", 0.15)),
        "vae_final_activation": str(cfg.get("vae_final_activation", "tanh")),
        "intermediate_fc_dim_vae": cfg.get("intermediate_fc_dim_vae", "quarter"),
        "use_layernorm_vae_fc": bool(cfg.get("use_layernorm_vae_fc", False)),
        "num_conv_layers_encoder": int(cfg.get("num_conv_layers_encoder", 4)),
        "decoder_type": str(cfg.get("decoder_type", "convtranspose")),
        "encoder_norm_mode": str(cfg.get("encoder_norm_mode", "groupnorm")),
        "vae_dropout_scope": str(cfg.get("vae_dropout_scope", "legacy_all")),
        "vae_block_order": str(cfg.get("vae_block_order", "legacy_act_norm")),
        "vae_conditioning_mode": "none",
        "vae_conditioning_vars": "none",
    }


def load_vae_fold(fold: int, device: torch.device):
    encode_mu, make_model = _load_vae_helpers()
    fold_dir = ADNI_RUN_DIR / f"fold_{fold}"
    checkpoint = fold_dir / f"vae_model_fold_{fold}.pt"
    require(checkpoint, f"fold {fold} VAE checkpoint")
    model = make_model(_model_cfg(), image_size=16, n_channels=3, device=device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def encode_oasis_fold(
    fold: int,
    x_oasis: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return mu (N, D) and logvar (N, D) for OASIS subjects using fold VAE."""
    encode_mu, make_model = _load_vae_helpers()
    fold_dir = ADNI_RUN_DIR / f"fold_{fold}"
    checkpoint = fold_dir / f"vae_model_fold_{fold}.pt"
    norm_path = fold_dir / "vae_norm_params.joblib"
    require(checkpoint, f"fold {fold} VAE checkpoint")
    require(norm_path, f"fold {fold} norm params")

    norm_params = joblib.load(norm_path)
    x_norm = apply_normalization_params(x_oasis, norm_params)

    model = make_model(_model_cfg(), image_size=x_norm.shape[-1], n_channels=x_norm.shape[1], device=device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Encode collecting both mu and logvar
    all_mu = []
    all_logvar = []
    all_recon_loss = []
    with torch.no_grad():
        for start in range(0, x_norm.shape[0], BATCH_SIZE):
            xb = torch.from_numpy(x_norm[start:start + BATCH_SIZE]).float().to(device)
            mu_b, logvar_b = model.encode(xb)
            all_mu.append(mu_b.detach().cpu().numpy())
            all_logvar.append(logvar_b.detach().cpu().numpy())
            # Reconstruction loss (MSE per sample)
            recon_b, _, _, _ = model(xb)
            recon_loss_b = ((recon_b - xb) ** 2).mean(dim=(1, 2, 3))
            all_recon_loss.append(recon_loss_b.detach().cpu().numpy())

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    mu = np.concatenate(all_mu, axis=0)
    logvar = np.concatenate(all_logvar, axis=0)
    recon_loss = np.concatenate(all_recon_loss, axis=0)
    return mu, logvar, recon_loss, norm_params, x_norm


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_tensor(candidate: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    npz_path = TENSOR_DIR / TENSOR_CANDIDATES[candidate]
    require(npz_path, f"tensor {candidate}")
    data = np.load(npz_path, allow_pickle=True)
    tensor = np.asarray(data["global_tensor_data"], dtype=np.float32)
    subject_ids = data["subject_ids"].astype(str)
    channel_names = data["channel_names"].astype(str).tolist()
    return tensor, subject_ids, channel_names


def select_channels(tensor: np.ndarray, tensor_channels: List[str], target: List[str]) -> np.ndarray:
    idx_map = {n: i for i, n in enumerate(tensor_channels)}
    return tensor[:, [idx_map[n] for n in target], :, :]


def load_split_meta() -> pd.DataFrame:
    require(SPLIT_CSV, "split CSV")
    df = pd.read_csv(SPLIT_CSV)
    df["y"] = (df["diagnosis"].isin(["AD", "AD_DEMENTIA"])).astype(int)
    # sex may be int (1=M, 2=F) or string
    sex_raw = df["sex"]
    if pd.api.types.is_numeric_dtype(sex_raw):
        df["Sex"] = sex_raw.map({1: "M", 2: "F"}).fillna("UNKNOWN")
    else:
        df["Sex"] = sex_raw.astype(str).str.upper().str.strip()
    df["Age"] = pd.to_numeric(df["age_at_MR"], errors="coerce")
    return df


def align_to_split(
    subject_ids: np.ndarray,
    split_df: pd.DataFrame,
) -> Tuple[np.ndarray, pd.DataFrame]:
    id_to_row = {sid: i for i, sid in enumerate(subject_ids)}
    split_ids = split_df["subject_id"].astype(str).tolist()
    missing = [s for s in split_ids if s not in id_to_row]
    if missing:
        raise RuntimeError(f"{len(missing)} split subjects not found in tensor: {missing[:5]}")
    indices = np.array([id_to_row[s] for s in split_ids], dtype=int)
    meta = split_df.reset_index(drop=True).copy()
    return indices, meta


# ---------------------------------------------------------------------------
# Section A: Per-fold probability statistics and per-fold AUC/PR-AUC
# ---------------------------------------------------------------------------
def section_a(scores_df: pd.DataFrame, output_dir: Path) -> Dict[str, Any]:
    """Analyse existing subject_scores.csv — no inference needed."""
    print("\n=== SECTION A: Per-fold probability statistics ===")
    rows = []
    lofo_rows = []

    for cand in scores_df["candidate"].unique():
        sub = scores_df[scores_df["candidate"] == cand].copy()
        y = sub["y"].values

        # Per-fold stats and AUC
        fold_probs: Dict[int, np.ndarray] = {}
        for fold in range(1, N_FOLDS + 1):
            col = f"prob_fold{fold}"
            p = sub[col].values
            fold_probs[fold] = p
            cn_p = p[y == 0]
            ad_p = p[y == 1]
            try:
                fold_auc = float(roc_auc_score(y, p))
            except Exception:
                fold_auc = float("nan")
            try:
                fold_prauc = float(average_precision_score(y, p))
            except Exception:
                fold_prauc = float("nan")
            rows.append({
                "candidate": cand,
                "fold": fold,
                "n": len(p),
                "prob_min": float(p.min()),
                "prob_max": float(p.max()),
                "prob_mean": float(p.mean()),
                "prob_std": float(p.std()),
                "CN_mean": float(cn_p.mean()),
                "CN_std": float(cn_p.std()),
                "AD_mean": float(ad_p.mean()),
                "AD_std": float(ad_p.std()),
                "CN_vs_AD_diff": float(ad_p.mean() - cn_p.mean()),
                "fold_auc": fold_auc,
                "fold_prauc": fold_prauc,
            })

        # Leave-one-fold-out ensemble AUC (diagnostic only)
        ens_p_all = np.mean([fold_probs[f] for f in range(1, N_FOLDS + 1)], axis=0)
        try:
            ens_auc = float(roc_auc_score(y, ens_p_all))
            ens_prauc = float(average_precision_score(y, ens_p_all))
        except Exception:
            ens_auc = ens_prauc = float("nan")
        lofo_rows.append({
            "candidate": cand, "excluded_fold": "none",
            "n_folds_used": N_FOLDS, "auc": ens_auc, "pr_auc": ens_prauc,
        })

        for held_out in range(1, N_FOLDS + 1):
            kept = [f for f in range(1, N_FOLDS + 1) if f != held_out]
            lofo_p = np.mean([fold_probs[f] for f in kept], axis=0)
            try:
                lofo_auc = float(roc_auc_score(y, lofo_p))
                lofo_prauc = float(average_precision_score(y, lofo_p))
            except Exception:
                lofo_auc = lofo_prauc = float("nan")
            lofo_rows.append({
                "candidate": cand,
                "excluded_fold": held_out,
                "n_folds_used": len(kept),
                "auc": lofo_auc,
                "pr_auc": lofo_prauc,
            })

    stats_df = pd.DataFrame(rows)
    lofo_df = pd.DataFrame(lofo_rows)

    stats_df.to_csv(output_dir / "A_fold_prob_stats.csv", index=False)
    lofo_df.to_csv(output_dir / "A_lofo_ensemble_auc.csv", index=False)

    print(stats_df[["candidate","fold","prob_min","prob_max","prob_std","CN_mean","AD_mean","CN_vs_AD_diff","fold_auc"]].to_string(index=False))
    print("\nLeave-one-fold-out ensemble AUC (diagnostic, not for reporting):")
    print(lofo_df.to_string(index=False))

    return {"fold_prob_stats": stats_df, "lofo": lofo_df}


# ---------------------------------------------------------------------------
# Section B: Latent diagnostics
# ---------------------------------------------------------------------------
def active_units(mu: np.ndarray, threshold: float = 0.01) -> int:
    """Count dims with variance > threshold (Au metric)."""
    return int((mu.var(axis=0) > threshold).sum())


def section_b(
    x_oasis_by_cand: Dict[str, np.ndarray],
    split_meta: pd.DataFrame,
    device: torch.device,
    output_dir: Path,
) -> None:
    print("\n=== SECTION B: Latent diagnostics (OASIS mu vs ADNI test) ===")
    rows_oasis = []
    rows_cmp = []

    # Pick one candidate for latent diagnostics (use first — folds are the same VAE)
    cand_primary = list(x_oasis_by_cand.keys())[0]
    x_for_latent = x_oasis_by_cand[cand_primary]
    y = split_meta["y"].values

    for fold in range(1, N_FOLDS + 1):
        print(f"  Encoding fold {fold} OASIS ...", end=" ", flush=True)
        mu, logvar, recon_loss, norm_params, x_norm = encode_oasis_fold(fold, x_for_latent, device)
        print(f"mu shape={mu.shape}")

        # Per-dim stats
        mu_var = mu.var(axis=0)
        mu_mean = mu.mean(axis=0)
        std_per_dim = mu.std(axis=0)
        mu_norm = np.linalg.norm(mu, axis=1)

        au = active_units(mu, threshold=0.01)
        au_strict = active_units(mu, threshold=0.1)

        # KL per dim: -0.5 * (1 + logvar - exp(logvar) - mu^2)
        # Average over subjects, sum over dims
        sigma2 = np.exp(logvar)
        kl_per_dim = 0.5 * (sigma2 + mu ** 2 - 1 - logvar)
        kl_mean_per_dim = kl_per_dim.mean(axis=0)

        row = {
            "fold": fold,
            "candidate": cand_primary,
            "n_subjects": len(mu),
            "active_units_0p01": au,
            "active_units_0p1": au_strict,
            "mu_mean_global": float(mu_mean.mean()),
            "mu_std_global": float(mu.std()),
            "mu_var_mean_across_dims": float(mu_var.mean()),
            "mu_var_max_across_dims": float(mu_var.max()),
            "mu_norm_mean": float(mu_norm.mean()),
            "mu_norm_std": float(mu_norm.std()),
            "mu_norm_min": float(mu_norm.min()),
            "mu_norm_max": float(mu_norm.max()),
            "recon_loss_mean": float(recon_loss.mean()),
            "recon_loss_std": float(recon_loss.std()),
            "recon_loss_CN_mean": float(recon_loss[y == 0].mean()),
            "recon_loss_AD_mean": float(recon_loss[y == 1].mean()),
            "kl_total_mean_across_subjects": float(kl_mean_per_dim.sum()),
            "kl_per_dim_mean": float(kl_mean_per_dim.mean()),
            "kl_per_dim_max": float(kl_mean_per_dim.max()),
            "n_dims_kl_gt_0p1": int((kl_mean_per_dim > 0.1).sum()),
            "n_dims_kl_gt_1p0": int((kl_mean_per_dim > 1.0).sum()),
        }
        rows_oasis.append(row)

        # Compare to ADNI test latent info (from saved CSVs)
        adni_latent_csv = ADNI_RUN_DIR / f"fold_{fold}" / f"fold_{fold}_test_latent_info_per_dim.csv"
        adni_summary_csv = ADNI_RUN_DIR / f"fold_{fold}" / f"fold_{fold}_test_latent_info_summary.csv"
        if adni_summary_csv.exists():
            adni_sum = pd.read_csv(adni_summary_csv)
            adni_y_row = adni_sum[adni_sum["variable"] == "Y_target"]
            adni_mi_sum = float(adni_y_row["mi_sum_nats"].values[0]) if len(adni_y_row) > 0 else float("nan")
            adni_n_active = int(adni_y_row["n_active"].values[0]) if len(adni_y_row) > 0 else -1
        else:
            adni_mi_sum = float("nan")
            adni_n_active = -1

        rows_cmp.append({
            "fold": fold,
            "oasis_au_0p01": au,
            "oasis_au_0p1": au_strict,
            "oasis_mu_var_mean": float(mu_var.mean()),
            "oasis_mu_norm_mean": float(mu_norm.mean()),
            "oasis_kl_total_mean": float(kl_mean_per_dim.sum()),
            "adni_test_n_active": adni_n_active,
            "adni_test_mi_sum_Y": adni_mi_sum,
            "oasis_recon_loss_mean": float(recon_loss.mean()),
        })

        # Save per-dim stats for fold 5 and fold 1 (key comparison)
        if fold in (1, 5):
            dim_df = pd.DataFrame({
                "dim": np.arange(LATENT_DIM),
                "mu_mean": mu_mean,
                "mu_std": std_per_dim,
                "mu_var": mu_var,
                "kl_mean": kl_mean_per_dim,
            })
            dim_df.to_csv(output_dir / f"B_latent_per_dim_fold{fold}.csv", index=False)

    oasis_df = pd.DataFrame(rows_oasis)
    cmp_df = pd.DataFrame(rows_cmp)
    oasis_df.to_csv(output_dir / "B_latent_oasis_summary.csv", index=False)
    cmp_df.to_csv(output_dir / "B_latent_oasis_vs_adni_cmp.csv", index=False)
    print("\nOASIS latent summary:")
    print(oasis_df[["fold","active_units_0p01","active_units_0p1","mu_var_mean_across_dims",
                     "mu_norm_mean","kl_total_mean_across_subjects","n_dims_kl_gt_0p1","n_dims_kl_gt_1p0",
                     "recon_loss_mean"]].to_string(index=False))
    print("\nOASIS vs ADNI comparison:")
    print(cmp_df.to_string(index=False))


# ---------------------------------------------------------------------------
# Section C: Input normalization diagnostics
# ---------------------------------------------------------------------------
def section_c(
    x_oasis_by_cand: Dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    print("\n=== SECTION C: Input normalization diagnostics ===")
    cand_primary = list(x_oasis_by_cand.keys())[0]
    x_raw = x_oasis_by_cand[cand_primary]

    rows_norm = []
    rows_params = []

    for fold in range(1, N_FOLDS + 1):
        norm_path = ADNI_RUN_DIR / f"fold_{fold}" / "vae_norm_params.joblib"
        norm_params = joblib.load(norm_path)
        for item in norm_params:
            rows_params.append({
                "fold": fold,
                "channel": item["original_name"],
                "mode": item["mode"],
                "mean": float(item["mean"]),
                "std": float(item["std"]),
            })

        x_norm = apply_normalization_params(x_raw, norm_params)

        n_roi = x_raw.shape[-1]
        mask = ~np.eye(n_roi, dtype=bool)
        for ch_idx, ch_name in enumerate(SELECTED_CHANNEL_NAMES):
            x_ch_norm = x_norm[:, ch_idx, :, :]
            off_diag = x_ch_norm[:, mask]
            rows_norm.append({
                "fold": fold,
                "channel": ch_name,
                "x_norm_mean": float(off_diag.mean()),
                "x_norm_std": float(off_diag.std()),
                "x_norm_min": float(off_diag.min()),
                "x_norm_max": float(off_diag.max()),
                "x_norm_abs_gt3": float((np.abs(off_diag) > 3).mean()),
                "x_norm_abs_gt5": float((np.abs(off_diag) > 5).mean()),
            })

    params_df = pd.DataFrame(rows_params)
    norm_df = pd.DataFrame(rows_norm)
    params_df.to_csv(output_dir / "C_norm_params.csv", index=False)
    norm_df.to_csv(output_dir / "C_x_norm_stats.csv", index=False)

    print("\nNorm params across folds:")
    print(params_df.to_string(index=False))
    print("\nOASIS x_norm statistics by fold and channel:")
    print(norm_df.to_string(index=False))


# ---------------------------------------------------------------------------
# Section D: Reconstruction diagnostics — already captured in section B
# ---------------------------------------------------------------------------
def section_d_from_b(b_csv: Path, output_dir: Path) -> None:
    print("\n=== SECTION D: Reconstruction diagnostics (from Section B) ===")
    df = pd.read_csv(b_csv)
    cols = ["fold", "recon_loss_mean", "recon_loss_std", "recon_loss_CN_mean", "recon_loss_AD_mean"]
    print(df[cols].to_string(index=False))
    # Re-save focused view
    df[cols].to_csv(output_dir / "D_recon_loss_summary.csv", index=False)


# ---------------------------------------------------------------------------
# Section E: Classifier diagnostics
# ---------------------------------------------------------------------------
def section_e(split_meta: pd.DataFrame, output_dir: Path) -> None:
    print("\n=== SECTION E: Classifier diagnostics ===")
    rows = []
    for fold in range(1, N_FOLDS + 1):
        fold_dir = ADNI_RUN_DIR / f"fold_{fold}"
        pipeline_path = fold_dir / f"classifier_logreg_raw_pipeline_fold_{fold}.joblib"
        require(pipeline_path, f"fold {fold} logreg pipeline")
        pipeline = joblib.load(pipeline_path)

        # Extract the logistic regression step
        logreg = None
        if hasattr(pipeline, "named_steps"):
            for name, step in pipeline.named_steps.items():
                if hasattr(step, "coef_"):
                    logreg = step
                    break
        if logreg is None and hasattr(pipeline, "best_estimator_"):
            logreg = pipeline.best_estimator_

        if logreg is None:
            print(f"  fold {fold}: could not extract logreg step from {type(pipeline)}")
            coef_l2 = float("nan")
            intercept = float("nan")
            C_val = float("nan")
        else:
            coef = logreg.coef_.ravel()
            coef_l2 = float(np.linalg.norm(coef))
            intercept = float(logreg.intercept_[0] if logreg.intercept_.ndim > 0 else logreg.intercept_)
            C_val = float(getattr(logreg, "C", float("nan")))

        # Decision function range on ADNI test set (from saved predictions)
        test_pred_path = fold_dir / "test_predictions_logreg.csv"
        if test_pred_path.exists():
            test_pred_df = pd.read_csv(test_pred_path)
            prob_col = None
            for col in ["prob_1", "prob_AD", "prob", "prob_positive"]:
                if col in test_pred_df.columns:
                    prob_col = col
                    break
            if prob_col is None:
                # Try to find any prob column
                prob_cols = [c for c in test_pred_df.columns if "prob" in c.lower()]
                if prob_cols:
                    prob_col = prob_cols[-1]
            if prob_col:
                adni_probs = test_pred_df[prob_col].values
                adni_prob_min = float(adni_probs.min())
                adni_prob_max = float(adni_probs.max())
                adni_prob_std = float(adni_probs.std())
            else:
                adni_prob_min = adni_prob_max = adni_prob_std = float("nan")
        else:
            adni_prob_min = adni_prob_max = adni_prob_std = float("nan")

        rows.append({
            "fold": fold,
            "logreg_C": C_val,
            "coef_l2_norm": coef_l2,
            "intercept": intercept,
            "adni_test_prob_min": adni_prob_min,
            "adni_test_prob_max": adni_prob_max,
            "adni_test_prob_std": adni_prob_std,
        })

    clf_df = pd.DataFrame(rows)
    clf_df.to_csv(output_dir / "E_classifier_diagnostics.csv", index=False)
    print(clf_df.to_string(index=False))


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------
def write_summary(output_dir: Path) -> None:
    a_stats = pd.read_csv(output_dir / "A_fold_prob_stats.csv")
    b_latent = pd.read_csv(output_dir / "B_latent_oasis_summary.csv")
    b_cmp = pd.read_csv(output_dir / "B_latent_oasis_vs_adni_cmp.csv")
    c_norm = pd.read_csv(output_dir / "C_x_norm_stats.csv")
    d_recon = pd.read_csv(output_dir / "D_recon_loss_summary.csv")
    e_clf = pd.read_csv(output_dir / "E_classifier_diagnostics.csv")
    lofo = pd.read_csv(output_dir / "A_lofo_ensemble_auc.csv")

    lines = [
        "# Fold-5 OASIS Degeneracy Audit",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Context",
        "- Model: `adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5` (latent_dim=256)",
        "- Fold 5 produces near-constant OASIS probabilities (range ≈0.474–0.520, std≈0.010)",
        "- Folds 1–4 produce dispersed probabilities (std≈0.07–0.12)",
        "- This collapses the ensemble toward ~0.50 and degrades AUC",
        "",
        "## A. Per-fold probability statistics",
        "",
    ]

    cand_primary = a_stats["candidate"].iloc[0]
    sub_a = a_stats[a_stats["candidate"] == cand_primary]
    lines.append(_md_table(sub_a[["fold","prob_min","prob_max","prob_std","CN_mean","AD_mean","CN_vs_AD_diff","fold_auc"]]))
    lines.append("")
    lines.append("### Leave-one-fold-out ensemble AUC (diagnostic only)")
    lines.append(_md_table(lofo[lofo["candidate"] == cand_primary]))
    lines.append("")

    lines += [
        "## B. Latent diagnostics (OASIS)",
        "",
        _md_table(b_latent[["fold","active_units_0p01","active_units_0p1","mu_var_mean_across_dims",
                              "mu_norm_mean","kl_total_mean_across_subjects","n_dims_kl_gt_0p1","n_dims_kl_gt_1p0"]]),
        "",
        "### OASIS vs ADNI comparison",
        "",
        _md_table(b_cmp),
        "",
    ]

    lines += [
        "## C. Input normalization — off-diagonal x_norm statistics by fold",
        "",
        _md_table(c_norm),
        "",
    ]

    lines += [
        "## D. Reconstruction loss by fold",
        "",
        _md_table(d_recon),
        "",
    ]

    lines += [
        "## E. Classifier diagnostics",
        "",
        _md_table(e_clf),
        "",
    ]

    lines += [
        "## Interpretation",
        "",
        "Fill in after reviewing sections A–E.",
    ]

    write("\n".join(lines), output_dir / "summary.md")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUTPUT_DIR}")
    print(f"Device: {device}")

    # Load existing scores (section A — no inference needed)
    require(SCORES_CSV, "subject_scores.csv")
    scores_df = pd.read_csv(SCORES_CSV)
    # Use full 120-subject set (both calibration + locked_test)
    print(f"Loaded scores: {len(scores_df)} rows, candidates={scores_df['candidate'].unique().tolist()}")

    split_meta = load_split_meta()

    # Load OASIS tensors (for sections B–D, channel-selected)
    x_oasis_by_cand: Dict[str, np.ndarray] = {}
    for cand, fname in TENSOR_CANDIDATES.items():
        tensor, subject_ids, channel_names = load_tensor(cand)
        x_sel = select_channels(tensor, channel_names, SELECTED_CHANNEL_NAMES)
        # Align to full 120-subject split order
        idx, _ = align_to_split(subject_ids, split_meta)
        x_oasis_by_cand[cand] = x_sel[idx]
        print(f"Tensor {cand}: shape={x_sel.shape} -> aligned {x_oasis_by_cand[cand].shape}")

    # --- Run sections ---
    section_a(scores_df, OUTPUT_DIR)
    section_b(x_oasis_by_cand, split_meta, device, OUTPUT_DIR)
    section_c(x_oasis_by_cand, OUTPUT_DIR)
    section_d_from_b(OUTPUT_DIR / "B_latent_oasis_summary.csv", OUTPUT_DIR)
    section_e(split_meta, OUTPUT_DIR)
    write_summary(OUTPUT_DIR)

    log = {
        "script": __file__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "adni_run_dir": str(ADNI_RUN_DIR),
        "tensor_dir": str(TENSOR_DIR),
        "split_csv": str(SPLIT_CSV),
        "scores_csv": str(SCORES_CSV),
        "output_dir": str(OUTPUT_DIR),
        "device": str(device),
        "no_training": True,
        "no_oasis_model_selection": True,
        "latent_inference_candidate": list(TENSOR_CANDIDATES.keys())[0],
    }
    (OUTPUT_DIR / "command_log.json").write_text(json.dumps(log, indent=2))
    print(f"\nDone. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
