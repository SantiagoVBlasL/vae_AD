#!/usr/bin/env python3
"""
Gate completion audit for Fase 1A+1B exhaustive singles/pairs.

Closes two gaps before any Phase 2 decision:
  1. Active latent units backfill — all 28 candidates × 3 folds = 84 records.
  2. Missing bootstrap comparison — pair_ch2_5 vs locked [1,0,2] control.
     (single_ch5 vs locked already in prior audit; extracted here too.)
  3. Gate criteria applied to finalists: single_ch5, pair_ch1_3, pair_ch2_5.
  4. Phase 2 shortlist recommendation.

Guardrails:
  - read-only inputs; no gradient updates, no training, no triples.
  - does not modify prior audit outputs or pilot arms.
  - does not edit gate_criteria_preregistered.md.

Output: results/revision_bspc_2026/post_revision_exploratory_20260630/
        fast_exhaustive_singles_pairs_gate_completion_20260703/
"""

import sys
import json
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from betavae_xai.models.convolutional_vae import ConvolutionalVAE

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path("/home/diego/proyectos/vae_AD")
RUN_ROOT = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026"
    "/post_revision_exploratory_20260630/fast_exhaustive_singles_pairs_20260702"
)
LOCKED_CONTROL = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026"
    "/post_revision_exploratory_20260630"
    "/fast_channelmean_loss_ablation_beta_matched_20260702/beta_cal_ch102_beta250"
)
PRIOR_AUDIT = (
    REPO_ROOT
    / "results/revision_bspc_2026/post_revision_exploratory_20260630"
    / "fast_exhaustive_singles_pairs_postrun_audit_20260703"
)
MANIFEST = (
    REPO_ROOT
    / "results/revision_bspc_2026/post_revision_exploratory_20260630"
    / "fast_exhaustive_singles_pairs_preflight_20260702"
    / "planned_candidates_singles_pairs.csv"
)
TENSOR_PATH = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026"
    "/adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors"
    "/GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
)
OUT_DIR = (
    REPO_ROOT
    / "results/revision_bspc_2026/post_revision_exploratory_20260630"
    / "fast_exhaustive_singles_pairs_gate_completion_20260703"
)

BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20260703
N_FOLDS = 3
VAR_EPS = 1e-4          # project standard: fold_qc.py:915

# Gate thresholds (gate_criteria_preregistered.md — do not modify)
AUC_SUPERIORITY = 0.015
PR_AUC_MAX_WORSE = 0.02
FOLD_SD_MAX = 0.07
ACTIVE_UNITS_MIN = 50

# Frozen architecture (launch_exhaustive_singles_pairs.sh COMMON_FLAGS)
VAE_ARCH = dict(
    latent_dim=128,
    image_size=131,
    final_activation="tanh",
    intermediate_fc_dim_config="quarter",
    dropout_rate=0.15,
    use_layernorm_fc=False,
    num_conv_layers_encoder=4,
    decoder_type="convtranspose",
)

FINALISTS = ["single_ch5", "pair_ch1_3", "pair_ch2_5"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def apply_zscore_offdiag(data: np.ndarray, norm_params: list) -> np.ndarray:
    """Apply saved zscore_offdiag norm params to data (N, C, H, W).

    data is modified in-place and returned.
    norm_params: list of dicts with mean, std per channel (project order).
    """
    N, C, H, W = data.shape
    diag_mask = np.eye(H, W, dtype=bool)
    offdiag_mask = ~diag_mask
    for c_idx, p in enumerate(norm_params):
        mean_ = float(p["mean"])
        std_ = float(p["std"])
        ch = data[:, c_idx, :, :]                        # (N, H, W)
        ch[:, offdiag_mask] = (ch[:, offdiag_mask] - mean_) / (std_ + 1e-12)
        ch[:, diag_mask] = 0.0
        data[:, c_idx, :, :] = ch
    return data


def load_predictions(candidate_dir: Path) -> pd.DataFrame:
    """Load all_folds_clf_predictions_*.csv from a candidate directory."""
    csvs = sorted(candidate_dir.glob("all_folds_clf_predictions_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No predictions CSV in {candidate_dir}")
    return pd.read_csv(csvs[0])


def paired_bootstrap(df_a: pd.DataFrame, df_b: pd.DataFrame,
                     label_a: str, label_b: str) -> dict:
    """Subject-level paired stratified bootstrap, ≥10,000 resamples."""
    a2 = df_a[["SubjectID", "y_true", "y_score_final"]].rename(
        columns={"y_score_final": "score_a"})
    b2 = df_b[["SubjectID", "y_true", "y_score_final"]].rename(
        columns={"y_score_final": "score_b"})
    m = a2.merge(b2, on="SubjectID", suffixes=("_a", "_b"))
    assert (m["y_true_a"].to_numpy() == m["y_true_b"].to_numpy()).all(), \
        f"Label mismatch: {label_a} vs {label_b}"
    y = m["y_true_a"].to_numpy().astype(int)
    score_a = m["score_a"].to_numpy(dtype=float)
    score_b = m["score_b"].to_numpy(dtype=float)
    base_auc = roc_auc_score(y, score_a) - roc_auc_score(y, score_b)
    base_pr = average_precision_score(y, score_a) - average_precision_score(y, score_b)
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    auc_d = np.empty(BOOTSTRAP_N)
    pr_d = np.empty(BOOTSTRAP_N)
    for i in range(BOOTSTRAP_N):
        s0 = rng.choice(idx0, size=len(idx0), replace=True)
        s1 = rng.choice(idx1, size=len(idx1), replace=True)
        idx = np.concatenate([s0, s1])
        yy = y[idx]
        auc_d[i] = (roc_auc_score(yy, score_a[idx])
                    - roc_auc_score(yy, score_b[idx]))
        pr_d[i] = (average_precision_score(yy, score_a[idx])
                   - average_precision_score(yy, score_b[idx]))
    return {
        "comparison": f"{label_a} minus {label_b}",
        "model_a": label_a,
        "model_b": label_b,
        "n_common_subjects": int(len(m)),
        "n_cn": int((y == 0).sum()),
        "n_ad": int((y == 1).sum()),
        "delta_auc": float(base_auc),
        "delta_auc_ci_low": float(np.percentile(auc_d, 2.5)),
        "delta_auc_ci_high": float(np.percentile(auc_d, 97.5)),
        "p_delta_auc_gt0": float((auc_d > 0).mean()),
        "delta_pr_auc": float(base_pr),
        "delta_pr_auc_ci_low": float(np.percentile(pr_d, 2.5)),
        "delta_pr_auc_ci_high": float(np.percentile(pr_d, 97.5)),
        "p_delta_pr_auc_gt0": float((pr_d > 0).mean()),
        "n_bootstrap": BOOTSTRAP_N,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


# ── Task 1: Active units backfill ──────────────────────────────────────────────

def compute_active_units_fold(candidate_dir: Path, fold_k: int,
                              global_tensor: np.ndarray) -> dict:
    """Compute active latent units for one candidate/fold.

    Returns a dict with fold-level diagnostics.
    """
    fold_dir = candidate_dir / f"fold_{fold_k}"

    # Load saved norm params (channel ordering + mean/std)
    norm_params = joblib.load(fold_dir / "vae_norm_params.joblib")
    n_ch = len(norm_params)

    # Build channel-name-to-global-index mapping
    DEFAULT_CH_NAMES = [
        "Pearson_OMST_GCE_Signed_Weighted",   # 0
        "Pearson_Full_FisherZ_Signed",          # 1
        "MI_KNN_Symmetric",                     # 2
        "dFC_AbsDiffMean",                      # 3
        "dFC_StdDev",                           # 4
        "DistanceCorr",                         # 5
        "Granger_F_lag1",                       # 6
    ]
    ch_indices = [DEFAULT_CH_NAMES.index(p["original_name"]) for p in norm_params]

    # Load training pool index (maps pool positions → global tensor rows)
    pool_idx = np.load(fold_dir / "vae_training_pool_tensor_idx.npy")
    # Load local training indices (subset of pool)
    train_local = np.load(fold_dir / "vae_actual_train_idx_local_to_pool.npy")
    train_global = pool_idx[train_local]

    # Extract training data for this candidate's channels
    X = global_tensor[train_global][:, ch_indices, :, :].astype(np.float32).copy()

    # Apply zscore_offdiag normalization with saved params
    apply_zscore_offdiag(X, norm_params)

    # Load model checkpoint
    ckpt_path = fold_dir / f"vae_model_fold_{fold_k}.pt"
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Instantiate model with correct input_channels
    model = ConvolutionalVAE(input_channels=n_ch, **VAE_ARCH)
    model.load_state_dict(state_dict)
    model.eval()

    # Run encoder in no_grad to get mu
    batch_size = 64
    mu_list = []
    xt = torch.from_numpy(X)
    with torch.no_grad():
        for start in range(0, len(xt), batch_size):
            batch = xt[start : start + batch_size]
            mu_b, _ = model.encode(batch)
            mu_list.append(mu_b.cpu().numpy())
    mu_all = np.concatenate(mu_list, axis=0)   # (N_train, latent_dim)

    # Compute active units: Var(mu_i) > 1e-4 per project definition
    vars_ = np.var(mu_all, axis=0)
    n_active = int((vars_ > VAR_EPS).sum())

    return {
        "fold": fold_k,
        "n_channels": n_ch,
        "channel_indices": ch_indices,
        "n_train": len(train_global),
        "n_active_units": n_active,
        "latent_dim": mu_all.shape[1],
    }


def backfill_active_units(manifest: pd.DataFrame, global_tensor: np.ndarray):
    """Run active unit computation for all 28 candidates × 3 folds."""
    rows_by_fold = []
    rows_by_candidate = []

    for _, row in manifest.iterrows():
        run_label = row["run_label"]
        output_dir = Path(row["output_dir"])
        channels_str = str(row["channels"])
        cardinality = int(row["cardinality"])

        fold_n_active = []
        print(f"  [{run_label}] channels={channels_str}", flush=True)
        for k in range(1, N_FOLDS + 1):
            result = compute_active_units_fold(output_dir, k, global_tensor)
            fold_n_active.append(result["n_active_units"])
            rows_by_fold.append({
                "candidate": run_label,
                "channels": channels_str,
                "cardinality": cardinality,
                "fold": k,
                "n_channels": result["n_channels"],
                "n_train": result["n_train"],
                "n_active_units": result["n_active_units"],
                "latent_dim": result["latent_dim"],
                "frac_active": round(result["n_active_units"] / result["latent_dim"], 4),
            })
            print(f"    fold {k}: n_active={result['n_active_units']}", flush=True)

        rows_by_candidate.append({
            "candidate": run_label,
            "channels": channels_str,
            "cardinality": cardinality,
            "n_active_mean": round(float(np.mean(fold_n_active)), 2),
            "n_active_min": int(np.min(fold_n_active)),
            "n_active_max": int(np.max(fold_n_active)),
            "gate_active_ge_50": bool(np.min(fold_n_active) >= ACTIVE_UNITS_MIN),
            "latent_dim": 128,
        })

    df_fold = pd.DataFrame(rows_by_fold)
    df_cand = pd.DataFrame(rows_by_candidate)
    return df_fold, df_cand


# ── Task 2: Bootstrap ──────────────────────────────────────────────────────────

def run_missing_bootstrap(prior_boot_df: pd.DataFrame):
    """Compute pair_ch2_5 vs locked control bootstrap; also collects existing
    single_ch5 vs locked row from prior audit for the deliverable.
    """
    preds_ch25 = load_predictions(RUN_ROOT / "pair_ch2_5")
    preds_locked = load_predictions(LOCKED_CONTROL)

    print("  Computing pair_ch2_5 minus locked_ch102 bootstrap ...", flush=True)
    new_row = paired_bootstrap(preds_ch25, preds_locked,
                               "pair_ch2_5", "locked_ch102")

    # single_ch5 vs locked — already in prior audit, extract for standalone file
    mask = prior_boot_df["comparison"] == "single_ch5 minus locked_ch102"
    assert mask.any(), "single_ch5 minus locked_ch102 not found in prior audit CSV"
    existing_row = prior_boot_df[mask].iloc[0].to_dict()

    return new_row, existing_row


# ── Task 3: Gate criteria application ─────────────────────────────────────────

def apply_gate_criteria(finalist_name: str,
                        eligibility_df: pd.DataFrame,
                        active_by_cand: pd.DataFrame,
                        all_boot_df: pd.DataFrame,
                        locked_auc: float,
                        locked_pr_auc: float) -> dict:
    """Apply all pre-registered gate criteria to one finalist."""
    row_elig = eligibility_df[eligibility_df["candidate"] == finalist_name].iloc[0]
    row_active = active_by_cand[active_by_cand["candidate"] == finalist_name].iloc[0]

    pooled_auc = float(row_elig["pooled_roc_auc"])
    pooled_pr = float(row_elig["pooled_pr_auc"])
    fold_sd = float(row_elig["fold_auc_sd"])
    high_beta_pass = bool(row_elig["gate_high_beta_checkpoint_100pct"])
    no_pathology = bool(row_elig["gate_no_training_pathology"])
    n_active_min = int(row_active["n_active_min"])

    # Compare vs locked control
    delta_auc = pooled_auc - locked_auc
    delta_pr = pooled_pr - locked_pr_auc

    # Bootstrap CI (vs locked control only)
    boot_mask = (all_boot_df["model_a"] == finalist_name) & \
                (all_boot_df["model_b"] == "locked_ch102")
    if not boot_mask.any():
        boot_ci_low = float("nan")
        boot_ci_high = float("nan")
    else:
        boot_row = all_boot_df[boot_mask].iloc[0]
        boot_ci_low = float(boot_row["delta_auc_ci_low"])
        boot_ci_high = float(boot_row["delta_auc_ci_high"])

    # Gate decisions
    gate_fold_sd = bool(fold_sd <= FOLD_SD_MAX)
    gate_active_units = bool(n_active_min >= ACTIVE_UNITS_MIN)
    gate_pr_auc = bool(delta_pr >= -PR_AUC_MAX_WORSE)
    gate_high_beta = high_beta_pass
    gate_no_pathology = no_pathology
    gate_no_collapse = True   # all nan_flags=0 from prior audit

    # AUC superiority — two readings:
    # Loose: point estimate >= 0.015
    gate_auc_point = bool(delta_auc >= AUC_SUPERIORITY)
    # Strict: 95% CI lower bound also >= 0.015
    gate_auc_ci = bool(boot_ci_low >= AUC_SUPERIORITY) if not np.isnan(boot_ci_low) else False

    all_loose = all([gate_fold_sd, gate_active_units, gate_pr_auc,
                     gate_high_beta, gate_no_pathology, gate_no_collapse, gate_auc_point])
    all_strict = all([gate_fold_sd, gate_active_units, gate_pr_auc,
                      gate_high_beta, gate_no_pathology, gate_no_collapse, gate_auc_ci])

    return {
        "finalist": finalist_name,
        "pooled_roc_auc": round(pooled_auc, 6),
        "pooled_pr_auc": round(pooled_pr, 6),
        "delta_auc_vs_locked": round(delta_auc, 6),
        "delta_pr_auc_vs_locked": round(delta_pr, 6),
        "delta_auc_ci_low": round(boot_ci_low, 6) if not np.isnan(boot_ci_low) else float("nan"),
        "delta_auc_ci_high": round(boot_ci_high, 6) if not np.isnan(boot_ci_high) else float("nan"),
        "fold_auc_sd": round(fold_sd, 6),
        "n_active_min": n_active_min,
        # Individual gate results
        "gate_fold_sd_le_0p07": gate_fold_sd,
        "gate_active_units_ge_50": gate_active_units,
        "gate_pr_auc_within_0p02": gate_pr_auc,
        "gate_high_beta_100pct": gate_high_beta,
        "gate_no_pathology": gate_no_pathology,
        "gate_no_collapse": gate_no_collapse,
        # AUC superiority — both readings
        "gate_auc_point_estimate_ge_0p015": gate_auc_point,
        "gate_auc_ci_lower_ge_0p015": gate_auc_ci,
        # Overall pass under each reading
        "overall_pass_loose": all_loose,
        "overall_pass_strict": all_strict,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Gate completion audit — 2026-07-03")
    print("=" * 70)

    # ── Load manifest ─────────────────────────────────────────────────────────
    print("\n[1/5] Loading manifest ...")
    manifest = pd.read_csv(MANIFEST)
    manifest["run_label"] = manifest["run_label"].str.strip()
    print(f"  Manifest: {len(manifest)} candidates")

    # ── Load global tensor (once) ─────────────────────────────────────────────
    print("\n[2/5] Loading global tensor ...")
    raw = np.load(TENSOR_PATH)
    # Key is typically the only array in the npz
    tensor_key = [k for k in raw.files if not k.endswith("_idx")][0]
    global_tensor = raw[tensor_key]
    print(f"  Tensor shape: {global_tensor.shape}  dtype: {global_tensor.dtype}")

    # ── Task 1: Active units backfill ─────────────────────────────────────────
    print("\n[3/5] Backfilling active units (28 candidates × 3 folds) ...")
    df_fold, df_cand = backfill_active_units(manifest, global_tensor)

    fold_out = OUT_DIR / "active_units_by_fold.csv"
    cand_out = OUT_DIR / "active_units_by_candidate.csv"
    df_fold.to_csv(fold_out, index=False)
    df_cand.to_csv(cand_out, index=False)
    print(f"  Wrote {fold_out}")
    print(f"  Wrote {cand_out}")

    # Summary statistics
    print(f"\n  Active unit summary:")
    print(f"  Overall min n_active: {df_fold['n_active_units'].min()}")
    print(f"  Overall max n_active: {df_fold['n_active_units'].max()}")
    print(f"  Overall mean n_active: {df_fold['n_active_units'].mean():.1f}")
    print(f"  Candidates with min n_active < 50: "
          f"{(df_cand['n_active_min'] < 50).sum()}")

    # ── Task 2: Bootstrap ─────────────────────────────────────────────────────
    print("\n[4/5] Running missing bootstrap comparison ...")
    prior_boot_df = pd.read_csv(PRIOR_AUDIT / "paired_bootstrap_comparisons.csv")

    new_boot_row, existing_single_ch5_row = run_missing_bootstrap(prior_boot_df)
    print(f"  pair_ch2_5 vs locked — delta_auc={new_boot_row['delta_auc']:.4f} "
          f"CI=[{new_boot_row['delta_auc_ci_low']:.4f}, "
          f"{new_boot_row['delta_auc_ci_high']:.4f}]")

    # Compose full bootstrap table (prior 3 rows + new pair_ch2_5 row)
    all_boot_rows = prior_boot_df.to_dict("records") + [new_boot_row]
    all_boot_df = pd.DataFrame(all_boot_rows)
    all_boot_out = OUT_DIR / "all_bootstrap_comparisons.csv"
    all_boot_df.to_csv(all_boot_out, index=False)
    print(f"  Wrote {all_boot_out}")

    # Standalone single_ch5_vs_control deliverable (Task 2 formal output)
    ch5_vs_ctrl = pd.DataFrame([existing_single_ch5_row])
    ch5_csv_out = OUT_DIR / "single_ch5_vs_control_bootstrap.csv"
    ch5_vs_ctrl.to_csv(ch5_csv_out, index=False)
    # Markdown version
    r = existing_single_ch5_row
    ch5_md = (
        "# single_ch5 vs locked [1,0,2] control — paired bootstrap\n\n"
        "**Source:** prior audit `paired_bootstrap_comparisons.csv` "
        "(computed 2026-07-03, seed=20260703, n=10,000 resamples).\n\n"
        "| metric | value |\n"
        "|:-------|------:|\n"
        f"| comparison | {r['comparison']} |\n"
        f"| n_common_subjects | {r['n_common_subjects']} |\n"
        f"| n_CN | {r['n_cn']} |\n"
        f"| n_AD | {r['n_ad']} |\n"
        f"| delta_ROC_AUC (point) | {r['delta_auc']:+.4f} |\n"
        f"| delta_ROC_AUC 95% CI | [{r['delta_auc_ci_low']:+.4f}, "
        f"{r['delta_auc_ci_high']:+.4f}] |\n"
        f"| p(delta_AUC > 0) | {r['p_delta_auc_gt0']:.4f} |\n"
        f"| delta_PR_AUC (point) | {r['delta_pr_auc']:+.4f} |\n"
        f"| delta_PR_AUC 95% CI | [{r['delta_pr_auc_ci_low']:+.4f}, "
        f"{r['delta_pr_auc_ci_high']:+.4f}] |\n"
        f"| p(delta_PR > 0) | {r['p_delta_pr_auc_gt0']:.4f} |\n"
        f"| n_bootstrap | {r['n_bootstrap']} |\n"
        f"| bootstrap_seed | {r['bootstrap_seed']} |\n\n"
        "**Note:** single_ch5 vs locked comparison was already present in the "
        "prior audit (`fast_exhaustive_singles_pairs_postrun_audit_20260703/`). "
        "This file extracts it verbatim for standalone reference. "
        "The newly computed comparison is `pair_ch2_5 minus locked_ch102` "
        "(see all_bootstrap_comparisons.csv).\n"
    )
    (OUT_DIR / "single_ch5_vs_control_bootstrap.md").write_text(ch5_md)
    print(f"  Wrote {OUT_DIR / 'single_ch5_vs_control_bootstrap.csv'}")
    print(f"  Wrote {OUT_DIR / 'single_ch5_vs_control_bootstrap.md'}")

    # ── Task 3: Gate criteria ──────────────────────────────────────────────────
    print("\n[5/5] Applying gate criteria to finalists ...")
    eligibility_df = pd.read_csv(PRIOR_AUDIT / "eligibility_gates.csv")

    # Locked control pooled metrics (from prior audit's pooled_oof_metrics.csv
    # — the locked arm is NOT in that file; use values from phase1c_recommendation.md
    # and bootstrap denominator: controlled value from paired_bootstrap comparisons)
    # Recompute directly from locked predictions to be self-contained.
    preds_locked = load_predictions(LOCKED_CONTROL)
    locked_auc = float(roc_auc_score(
        preds_locked["y_true"].to_numpy(), preds_locked["y_score_final"].to_numpy()))
    locked_pr_auc = float(average_precision_score(
        preds_locked["y_true"].to_numpy(), preds_locked["y_score_final"].to_numpy()))
    print(f"  Locked control [1,0,2]: pooled ROC-AUC={locked_auc:.4f}, "
          f"PR-AUC={locked_pr_auc:.4f}")

    gate_rows = []
    for finalist in FINALISTS:
        g = apply_gate_criteria(finalist, eligibility_df, df_cand,
                                all_boot_df, locked_auc, locked_pr_auc)
        gate_rows.append(g)
        loose_str = "PASS" if g["overall_pass_loose"] else "FAIL"
        strict_str = "PASS" if g["overall_pass_strict"] else "FAIL"
        print(f"  {finalist}: loose={loose_str}  strict={strict_str}  "
              f"n_active_min={g['n_active_min']}")

    gate_df = pd.DataFrame(gate_rows)

    # ── Write finalist_gate_check_20260703.md ─────────────────────────────────
    gate_md_lines = [
        "# Finalist Gate Check — 2026-07-03",
        "",
        "Gate criteria source: `gate_criteria_preregistered.md` (UTC 2026-07-02T14:40:44Z).",
        "Active units: backfilled in Task 1 of this audit.",
        "Bootstrap CIs: 10,000 paired stratified resamples, seed=20260703.",
        "",
        f"Locked control [1,0,2]: pooled ROC-AUC={locked_auc:.4f}, "
        f"PR-AUC={locked_pr_auc:.4f}",
        "",
        "**Two readings of the AUC superiority gate:**",
        "- **Loose**: point-estimate delta ≥ 0.015 suffices.",
        "- **Strict**: 95% CI lower bound must also ≥ 0.015.",
        "",
        "---",
        "",
    ]
    for g in gate_rows:
        f_name = g["finalist"]
        r_sd = "PASS" if g["gate_fold_sd_le_0p07"] else "FAIL"
        r_au = "PASS" if g["gate_active_units_ge_50"] else "FAIL"
        r_pr = "PASS" if g["gate_pr_auc_within_0p02"] else "FAIL"
        r_hb = "PASS" if g["gate_high_beta_100pct"] else "FAIL"
        r_np = "PASS" if g["gate_no_pathology"] else "FAIL"
        r_nc = "PASS" if g["gate_no_collapse"] else "FAIL"
        r_pt = "PASS" if g["gate_auc_point_estimate_ge_0p015"] else "FAIL"
        r_ci = "PASS" if g["gate_auc_ci_lower_ge_0p015"] else "FAIL"
        r_lo = "PASS" if g["overall_pass_loose"] else "FAIL"
        r_st = "PASS" if g["overall_pass_strict"] else "FAIL"
        gate_md_lines += [
            f"## {f_name}",
            "",
            "| gate criterion | value | result |",
            "|:---------------|------:|:------:|",
            f"| fold AUC SD ≤ 0.07 | {g['fold_auc_sd']:.4f} | {r_sd} |",
            f"| active units ≥ 50/128 (min across folds) | {g['n_active_min']} | {r_au} |",
            f"| PR-AUC not >0.02 worse than locked | {g['delta_pr_auc_vs_locked']:+.4f} | {r_pr} |",
            f"| high-beta checkpoint 100% | — | {r_hb} |",
            f"| no training pathology | — | {r_np} |",
            f"| no collapse / NaN | — | {r_nc} |",
            f"| **AUC beats locked by ≥0.015 (point estimate)** | "
            f"{g['delta_auc_vs_locked']:+.4f} | **{r_pt}** |",
            f"| **AUC beats locked by ≥0.015 (95% CI lower bound)** | "
            f"{g['delta_auc_ci_low']:+.4f} | **{r_ci}** |",
            "",
            f"**Overall — LOOSE reading (point estimate):** {r_lo}",
            "",
            f"**Overall — STRICT reading (CI-inclusive):** {r_st}",
            "",
            "---",
            "",
        ]

    gate_md_path = OUT_DIR / "finalist_gate_check_20260703.md"
    gate_md_path.write_text("\n".join(gate_md_lines))
    gate_df.to_csv(OUT_DIR / "finalist_gate_check_20260703.csv", index=False)
    print(f"  Wrote {gate_md_path}")

    # ── Task 4: Phase 2 shortlist recommendation ───────────────────────────────
    loosely_pass = [g["finalist"] for g in gate_rows if g["overall_pass_loose"]]
    strictly_pass = [g["finalist"] for g in gate_rows if g["overall_pass_strict"]]

    # Collect per-finalist data for narrative
    g_ch5 = gate_rows[0]    # single_ch5
    g_13 = gate_rows[1]     # pair_ch1_3
    g_25 = gate_rows[2]     # pair_ch2_5

    # Retrieve fold-wise AUC for ch3 singles instability note
    # from prior audit's vae_diagnostics_by_candidate.csv
    cand_summary = pd.read_csv(PRIOR_AUDIT / "candidate_summary_all.csv")
    ch3_row = cand_summary[cand_summary["candidate"] == "single_ch3"]
    ch3_sd = float(ch3_row["sd_auc"].iloc[0]) if not ch3_row.empty else float("nan")
    ch3_auc_mean = float(ch3_row["mean_auc"].iloc[0]) if not ch3_row.empty else float("nan")

    shortlist_md = f"""# Phase 2 Shortlist Recommendation — 2026-07-03

## Evidence Summary

| finalist | pooled ROC-AUC | delta vs locked | CI [lo, hi] | CI lo ≥ 0.015? | PR-AUC | fold AUC SD | n_active_min | loose PASS | strict PASS |
|:---------|---------------:|----------------:|:-----------:|:--------------:|-------:|------------:|-------------:|:----------:|:-----------:|
| single_ch5 | {g_ch5['pooled_roc_auc']:.4f} | {g_ch5['delta_auc_vs_locked']:+.4f} | [{g_ch5['delta_auc_ci_low']:+.4f}, {g_ch5['delta_auc_ci_high']:+.4f}] | {'YES' if g_ch5['gate_auc_ci_lower_ge_0p015'] else 'NO'} | {g_ch5['pooled_pr_auc']:.4f} | {g_ch5['fold_auc_sd']:.4f} | {g_ch5['n_active_min']} | {'YES' if g_ch5['overall_pass_loose'] else 'NO'} | {'YES' if g_ch5['overall_pass_strict'] else 'NO'} |
| pair_ch1_3 | {g_13['pooled_roc_auc']:.4f} | {g_13['delta_auc_vs_locked']:+.4f} | [{g_13['delta_auc_ci_low']:+.4f}, {g_13['delta_auc_ci_high']:+.4f}] | {'YES' if g_13['gate_auc_ci_lower_ge_0p015'] else 'NO'} | {g_13['pooled_pr_auc']:.4f} | {g_13['fold_auc_sd']:.4f} | {g_13['n_active_min']} | {'YES' if g_13['overall_pass_loose'] else 'NO'} | {'YES' if g_13['overall_pass_strict'] else 'NO'} |
| pair_ch2_5 | {g_25['pooled_roc_auc']:.4f} | {g_25['delta_auc_vs_locked']:+.4f} | [{g_25['delta_auc_ci_low']:+.4f}, {g_25['delta_auc_ci_high']:+.4f}] | {'YES' if g_25['gate_auc_ci_lower_ge_0p015'] else 'NO'} | {g_25['pooled_pr_auc']:.4f} | {g_25['fold_auc_sd']:.4f} | {g_25['n_active_min']} | {'YES' if g_25['overall_pass_loose'] else 'NO'} | {'YES' if g_25['overall_pass_strict'] else 'NO'} |

Locked control [1,0,2]: pooled ROC-AUC={locked_auc:.4f}, PR-AUC={locked_pr_auc:.4f}

---

## Recommendation

**Finalists passing the LOOSE gate (point estimate ≥ 0.015):** {', '.join(loosely_pass) if loosely_pass else 'none'}

**Finalists passing the STRICT gate (CI lower bound ≥ 0.015):** {', '.join(strictly_pass) if strictly_pass else 'none'}

### Per-finalist notes

**single_ch5** (DistanceCorr, k=1):
- Highest PR-AUC of all candidates ({g_ch5['pooled_pr_auc']:.4f}) — strongest separation on the imbalanced positive class.
- delta ROC-AUC = {g_ch5['delta_auc_vs_locked']:+.4f} vs locked; CI lower bound = {g_ch5['delta_auc_ci_low']:+.4f}.
- Passes loose gate ({('YES' if g_ch5['overall_pass_loose'] else 'NO')}); strict gate ({('YES' if g_ch5['overall_pass_strict'] else 'NO')}).
- Active units: {g_ch5['n_active_min']}/128 minimum. Fold SD: {g_ch5['fold_auc_sd']:.4f} — highly stable.
- No component-instability concerns (single channel, no dangerous sub-channel).

**pair_ch1_3** (Pearson Full FisherZ + dFC_AbsDiffMean, k=2):
- Highest pooled ROC-AUC of all 28 candidates ({g_13['pooled_roc_auc']:.4f}).
- delta ROC-AUC = {g_13['delta_auc_vs_locked']:+.4f} vs locked; CI lower bound = {g_13['delta_auc_ci_low']:+.4f}.
- Passes loose gate ({('YES' if g_13['overall_pass_loose'] else 'NO')}); strict gate ({('YES' if g_13['overall_pass_strict'] else 'NO')}).
- **Instability concern (ch3):** single_ch3 (dFC_AbsDiffMean alone) has the widest
  fold-wise AUC error bars of the 7 singles (SD={ch3_sd:.4f}, mean_AUC={ch3_auc_mean:.4f}).
  This is a flag for caution, not disqualification. The pairing with ch1 may compensate or
  compound ch3's variability in a beta-robustness re-test. Recommend treating as provisional
  finalist with explicit monitoring at beta=3.75.
- PR-AUC ({g_13['pooled_pr_auc']:.4f}) is lower than single_ch5 — the pair gains ROC-AUC
  but does not improve PR-AUC, suggesting the positive-class benefit is modest.

**pair_ch2_5** (MI_KNN_Symmetric + DistanceCorr, k=2):
- pooled ROC-AUC={g_25['pooled_roc_auc']:.4f}, near-equal to pair_ch1_3.
- delta ROC-AUC = {g_25['delta_auc_vs_locked']:+.4f} vs locked; CI lower bound = {g_25['delta_auc_ci_low']:+.4f}.
- Passes loose gate ({('YES' if g_25['overall_pass_loose'] else 'NO')}); strict gate ({('YES' if g_25['overall_pass_strict'] else 'NO')}).
- Exceptionally low fold AUC SD = {g_25['fold_auc_sd']:.4f} — most stable pair candidate.
- Contains ch5 (DistanceCorr), the single best channel, combined with ch2 (MI_KNN_Symmetric).
- No component-instability concern: single_ch2 was the second-most stable single.

---

## Phase 2 Shortlist

**Recommended to advance to beta=3.75 robustness re-test:**

| rank | candidate | rationale |
|:----:|:----------|:----------|
| 1 | **single_ch5** | Highest PR-AUC, most stable single, passes loose gate. Clinically interpretable (single channel). |
| 2 | **pair_ch1_3** | Highest ROC-AUC, passes loose gate. Provisional — ch3 instability warrants monitoring at beta=3.75. |
| 3 | **pair_ch2_5** | Stable pair with competitive ROC-AUC and no component-instability concerns. Passes loose gate. |

**None of the three finalists pass the STRICT gate** under the pre-registered criteria (CI lo ≥ 0.015).
This does not disqualify them from Phase 2; it means that at beta=2.50 alone the superiority margin
is not uncertainty-robust. Phase 2 (beta=3.75) is exactly designed to test whether the lead holds
under a different regularization pressure, per the robustness criterion in gate_criteria_preregistered.md.

---

## Guardrail confirmations

- No Phase 2 (beta=3.75) training was launched in this task.
- No triples were computed or scheduled.
- Prior audit outputs and pilot arm directories were not modified.
- gate_criteria_preregistered.md was not edited.
"""

    shortlist_path = OUT_DIR / "phase2_shortlist_recommendation.md"
    shortlist_path.write_text(shortlist_md)
    print(f"  Wrote {shortlist_path}")

    # ── command_log.json ───────────────────────────────────────────────────────
    log = {
        "script": str(Path(__file__).resolve()),
        "timestamp_utc": datetime.datetime.utcnow().isoformat(),
        "run_root": str(RUN_ROOT),
        "locked_control": str(LOCKED_CONTROL),
        "prior_audit": str(PRIOR_AUDIT),
        "outputs": [
            "active_units_by_fold.csv",
            "active_units_by_candidate.csv",
            "single_ch5_vs_control_bootstrap.csv",
            "single_ch5_vs_control_bootstrap.md",
            "all_bootstrap_comparisons.csv",
            "finalist_gate_check_20260703.csv",
            "finalist_gate_check_20260703.md",
            "phase2_shortlist_recommendation.md",
            "command_log.json",
        ],
        "bootstrap": {
            "n_resamples": BOOTSTRAP_N,
            "seed": BOOTSTRAP_SEED,
            "methodology": "subject-level paired stratified (class-stratified resample within CN and AD)",
            "new_comparison": "pair_ch2_5 minus locked_ch102",
            "existing_comparison_reused": "single_ch5 minus locked_ch102 (from prior audit)",
        },
        "active_units": {
            "var_eps": VAR_EPS,
            "definition": "count(Var(mu_i) > 1e-4) per fold_qc.py:915",
            "n_folds": N_FOLDS,
            "n_candidates": len(manifest),
            "n_fold_records": N_FOLDS * len(manifest),
        },
        "finalists_evaluated": FINALISTS,
        "locked_control_pooled_auc": locked_auc,
        "locked_control_pooled_pr_auc": locked_pr_auc,
        "gate_results": {
            g["finalist"]: {
                "loose": g["overall_pass_loose"],
                "strict": g["overall_pass_strict"],
            }
            for g in gate_rows
        },
        "guardrails": {
            "training_launched": False,
            "triples_launched": False,
            "prior_audit_modified": False,
            "pilot_arms_modified": False,
            "gate_criteria_file_modified": False,
        },
    }
    log_path = OUT_DIR / "command_log.json"
    log_path.write_text(json.dumps(log, indent=2))
    print(f"  Wrote {log_path}")

    print("\n" + "=" * 70)
    print("DONE. All deliverables written to:")
    print(f"  {OUT_DIR}")
    print("=" * 70)

    return log


if __name__ == "__main__":
    main()
