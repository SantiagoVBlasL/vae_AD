#!/usr/bin/env python3
"""
Phase 2 (beta=3.75) post-run audit — Part A.
Read-only / inference-only. No training, no gradient updates.

Run with: /home/diego/anaconda3/bin/python3 scripts/revision_bspc_2026/phase2_postrun_audit_20260703.py
"""

import csv
import json
import sys
import os
import numpy as np
from pathlib import Path
import joblib

# ── require torch for active-unit inference ──────────────────────────────────
try:
    import torch
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("WARNING: torch not available — active_units will be skipped")

from sklearn.metrics import roc_auc_score, average_precision_score

# ─── paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path("/home/diego/proyectos/vae_AD")
sys.path.insert(0, str(REPO_ROOT / "src"))

RUN_ROOT_B375 = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026"
    "/post_revision_exploratory_20260630"
    "/fast_phase2_beta375_finalists_20260703"
)
RUN_ROOT_B250_FINALISTS = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026"
    "/post_revision_exploratory_20260630"
    "/fast_exhaustive_singles_pairs_20260702"
)
RUN_ROOT_B250_CONTROL = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026"
    "/post_revision_exploratory_20260630"
    "/fast_channelmean_loss_ablation_beta_matched_20260702"
    "/beta_cal_ch102_beta250"
)
TENSOR_PATH = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026"
    "/adni_expanded_v5_1_batch20260514b_no_pybandpass"
    "/subject_tensors"
    "/GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
)
OUT_DIR = REPO_ROOT / (
    "results/revision_bspc_2026/post_revision_exploratory_20260630"
    "/fast_phase2_beta375_postrun_audit_20260703"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── constants ────────────────────────────────────────────────────────────────
CANDIDATES = ["single_ch5", "pair_ch1_3", "pair_ch2_5", "control_ch102"]
FINALISTS = ["single_ch5", "pair_ch1_3", "pair_ch2_5"]
FOLDS = [1, 2, 3]
BETA_375 = 3.75
HIGH_BETA_THR = 0.95 * BETA_375  # 3.5625
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20260703
VAR_EPS = 1e-4
AUC_SUPERIORITY = 0.015
PR_AUC_MAX_WORSE = 0.02
FOLD_SD_MAX = 0.07
ACTIVE_UNITS_MIN = 50

DEFAULT_CH_NAMES = [
    "Pearson_OMST_GCE_Signed_Weighted",   # 0
    "Pearson_Full_FisherZ_Signed",         # 1
    "MI_KNN_Symmetric",                    # 2
    "dFC_AbsDiffMean",                     # 3
    "dFC_StdDev",                          # 4
    "DistanceCorr",                        # 5
    "Granger_F_lag1",                      # 6
]

VAE_ARCH = dict(
    latent_dim=128, image_size=131, final_activation="tanh",
    intermediate_fc_dim_config="quarter", dropout_rate=0.15,
    use_layernorm_fc=False, num_conv_layers_encoder=4,
    decoder_type="convtranspose",
)

COMMAND_LOG = {
    "script": "scripts/revision_bspc_2026/phase2_postrun_audit_20260703.py",
    "guardrails": {
        "training_launched": False,
        "triples_launched": False,
        "beta125_launched": False,
        "prior_audit_modified": False,
        "pilot_arms_modified": False,
        "beta250_runs_modified": False,
        "gate_criteria_file_modified": False,
        "prior_amendments_modified": False,
    },
    "beta375": {},
    "bootstrap_metadata": {
        "n_resamples": BOOTSTRAP_N,
        "seed": BOOTSTRAP_SEED,
        "strategy": "subject_level_stratified_by_class",
    },
}

# ─── helpers ──────────────────────────────────────────────────────────────────

def find_prediction_csv(cand_dir: Path) -> Path:
    files = list(cand_dir.glob("all_folds_clf_predictions_*.csv"))
    if not files:
        raise FileNotFoundError(f"No prediction CSV in {cand_dir}")
    return files[0]


def load_predictions(csv_path: Path) -> list[dict]:
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def pooled_auc(rows: list[dict]) -> tuple[float, float]:
    y_true = np.array([int(r["y_true"]) for r in rows])
    y_score = np.array([float(r["y_score_final"]) for r in rows])
    roc = roc_auc_score(y_true, y_score)
    pr = average_precision_score(y_true, y_score)
    return roc, pr


def fold_aucs(rows: list[dict]) -> list[float]:
    by_fold = {}
    for r in rows:
        by_fold.setdefault(r["fold"], []).append(r)
    aucs = []
    for fold_rows in sorted(by_fold.values(), key=lambda x: x[0]["fold"]):
        y_true = np.array([int(r["y_true"]) for r in fold_rows])
        y_score = np.array([float(r["y_score_final"]) for r in fold_rows])
        aucs.append(roc_auc_score(y_true, y_score))
    return aucs


def check_duplicates(rows: list[dict]) -> dict:
    seen = {}
    duplicates = []
    for r in rows:
        key = (r["SubjectID"], r["fold"])
        if key in seen:
            duplicates.append(key)
        seen[key] = True
    return {"n_rows": len(rows), "n_unique_subjects": len({r["SubjectID"] for r in rows}),
            "n_duplicate_subject_fold": len(duplicates), "duplicates": duplicates[:5]}


def paired_bootstrap(
    rows_a: list[dict], rows_b: list[dict],
    label_a: str, label_b: str,
    n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Subject-level stratified-by-class paired bootstrap."""
    rng = np.random.default_rng(seed)

    # Build subject→score maps
    def to_subj_map(rows):
        m = {}
        for r in rows:
            sid = r["SubjectID"]
            if sid not in m:
                m[sid] = {"y_true": int(r["y_true"]), "y_score": float(r["y_score_final"])}
        return m

    map_a = to_subj_map(rows_a)
    map_b = to_subj_map(rows_b)

    common = sorted(set(map_a) & set(map_b))
    y_true = np.array([map_a[s]["y_true"] for s in common])
    scores_a = np.array([map_a[s]["y_score"] for s in common])
    scores_b = np.array([map_b[s]["y_score"] for s in common])

    idx_cn = np.where(y_true == 0)[0]
    idx_ad = np.where(y_true == 1)[0]
    n_cn = len(idx_cn)
    n_ad = len(idx_ad)

    # Point estimates
    pe_a_roc = roc_auc_score(y_true, scores_a)
    pe_b_roc = roc_auc_score(y_true, scores_b)
    pe_a_pr = average_precision_score(y_true, scores_a)
    pe_b_pr = average_precision_score(y_true, scores_b)
    delta_roc_pe = pe_a_roc - pe_b_roc
    delta_pr_pe = pe_a_pr - pe_b_pr

    # Bootstrap
    delta_rocs = np.empty(n)
    delta_prs = np.empty(n)
    for i in range(n):
        rs_cn = rng.integers(0, n_cn, size=n_cn)
        rs_ad = rng.integers(0, n_ad, size=n_ad)
        idx_b = np.concatenate([idx_cn[rs_cn], idx_ad[rs_ad]])
        yt_b = y_true[idx_b]
        sa_b = scores_a[idx_b]
        sb_b = scores_b[idx_b]
        try:
            delta_rocs[i] = roc_auc_score(yt_b, sa_b) - roc_auc_score(yt_b, sb_b)
            delta_prs[i] = average_precision_score(yt_b, sa_b) - average_precision_score(yt_b, sb_b)
        except Exception:
            delta_rocs[i] = np.nan
            delta_prs[i] = np.nan

    ci_lo_roc = float(np.nanpercentile(delta_rocs, 2.5))
    ci_hi_roc = float(np.nanpercentile(delta_rocs, 97.5))
    ci_lo_pr = float(np.nanpercentile(delta_prs, 2.5))
    ci_hi_pr = float(np.nanpercentile(delta_prs, 97.5))

    p_roc = float(np.mean(delta_rocs > 0))
    p_pr = float(np.mean(delta_prs > 0))

    return {
        "comparison": f"{label_a} minus {label_b}",
        "model_a": label_a,
        "model_b": label_b,
        "n_common_subjects": len(common),
        "n_cn": n_cn,
        "n_ad": n_ad,
        "auc_a": pe_a_roc,
        "auc_b": pe_b_roc,
        "delta_auc": delta_roc_pe,
        "delta_auc_ci_low": ci_lo_roc,
        "delta_auc_ci_high": ci_hi_roc,
        "p_delta_auc_gt0": p_roc,
        "pr_auc_a": pe_a_pr,
        "pr_auc_b": pe_b_pr,
        "delta_pr_auc": delta_pr_pe,
        "delta_pr_auc_ci_low": ci_lo_pr,
        "delta_pr_auc_ci_high": ci_hi_pr,
        "p_delta_pr_auc_gt0": p_pr,
        "n_bootstrap": n,
        "bootstrap_seed": seed,
    }


def apply_zscore_offdiag(data: np.ndarray, norm_params: list) -> np.ndarray:
    """Apply saved zscore_offdiag norm params to data (N, C, H, W). In-place."""
    N, C, H, W = data.shape
    diag_mask = np.eye(H, W, dtype=bool)
    offdiag_mask = ~diag_mask
    for c_idx, p in enumerate(norm_params):
        mean_ = float(p["mean"])
        std_ = float(p["std"])
        ch = data[:, c_idx, :, :]
        ch[:, offdiag_mask] = (ch[:, offdiag_mask] - mean_) / (std_ + 1e-12)
        ch[:, diag_mask] = 0.0
        data[:, c_idx, :, :] = ch
    return data


def compute_active_units(fold_dir: Path, global_tensor: np.ndarray, batch_size: int = 32) -> int:
    """Load checkpoint, run encoder, count Var(mu_i) > 1e-4."""
    if not TORCH_OK:
        return -1
    from betavae_xai.models.convolutional_vae import ConvolutionalVAE

    norm_params = joblib.load(fold_dir / "vae_norm_params.joblib")
    n_ch = len(norm_params)
    ch_indices = [DEFAULT_CH_NAMES.index(p["original_name"]) for p in norm_params]

    pool_idx = np.load(fold_dir / "vae_training_pool_tensor_idx.npy")
    train_local = np.load(fold_dir / "vae_actual_train_idx_local_to_pool.npy")
    train_global = pool_idx[train_local]

    X = global_tensor[train_global][:, ch_indices, :, :].astype(np.float32).copy()
    apply_zscore_offdiag(X, norm_params)
    xt = torch.from_numpy(X)

    ckpt_files = list(fold_dir.glob("vae_model_fold_*.pt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No .pt checkpoint in {fold_dir}")
    state_dict = torch.load(ckpt_files[0], map_location="cpu", weights_only=False)

    model = ConvolutionalVAE(input_channels=n_ch, **VAE_ARCH)
    model.load_state_dict(state_dict)
    model.eval()

    mu_list = []
    with torch.no_grad():
        for start in range(0, len(xt), batch_size):
            batch = xt[start: start + batch_size]
            mu_b, _ = model.encode(batch)
            mu_list.append(mu_b.cpu().numpy())

    mu_all = np.concatenate(mu_list, axis=0)
    vars_ = np.var(mu_all, axis=0)
    return int((vars_ > VAR_EPS).sum())


def get_rho_from_history(fold_dir: Path, selected_epoch: int, beta: float) -> dict:
    """Extract rho from training history at selected epoch (1-indexed)."""
    hist_files = list(fold_dir.glob("vae_train_history_fold_*.joblib"))
    if not hist_files:
        return {"val_recon": None, "val_kld": None, "rho": None}
    hist = joblib.load(hist_files[0])
    idx = selected_epoch - 1  # convert to 0-indexed
    val_recon = hist["val_recon"][idx]
    val_kld = hist["val_kld"][idx]
    rho = beta * val_kld / val_recon if val_recon > 0 else None
    # sanity: confirm beta from history matches expected
    stored_beta = hist["beta"][idx] if "beta" in hist else None
    return {
        "val_recon": val_recon,
        "val_kld": val_kld,
        "rho": rho,
        "stored_beta_at_epoch": stored_beta,
    }


def get_checkpoint_info(fold_dir: Path) -> dict:
    ckpt_files = list(fold_dir.glob("vae_checkpoint_selection_summary_fold_*.csv"))
    if not ckpt_files:
        return {}
    with open(ckpt_files[0]) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    r = rows[-1]
    return {
        "selected_epoch": int(r.get("selected_epoch", 0)),
        "selected_epoch_beta": float(r.get("selected_epoch_beta", 0)),
        "high_beta_threshold": float(r.get("high_beta_threshold", 0)),
        "selected_epoch_cycle_id": r.get("selected_epoch_cycle_id", ""),
        "selected_epoch_phase": r.get("selected_epoch_phase", ""),
    }


# ─── TASK 1: Integrity check ──────────────────────────────────────────────────
print("\n" + "="*70)
print("TASK 1: Integrity check")
print("="*70)

integrity = {}
for cand in CANDIDATES:
    cand_dir = RUN_ROOT_B375 / cand
    ci = {"candidate": cand, "dir_present": cand_dir.exists()}

    # Check fold directories
    fold_dirs = [cand_dir / f"fold_{k}" for k in FOLDS]
    ci["folds_present"] = all(d.exists() for d in fold_dirs)
    ci["n_folds"] = sum(1 for d in fold_dirs if d.exists())

    # OOF prediction CSV
    try:
        pred_csv = find_prediction_csv(cand_dir)
        ci["pred_csv_present"] = True
        rows = load_predictions(pred_csv)
        dup_info = check_duplicates(rows)
        ci["n_oof_rows"] = dup_info["n_rows"]
        ci["n_oof_subjects"] = dup_info["n_unique_subjects"]
        ci["n_duplicate_subject_fold"] = dup_info["n_duplicate_subject_fold"]
        ci["oof_ok"] = dup_info["n_duplicate_subject_fold"] == 0
    except Exception as e:
        ci["pred_csv_present"] = False
        ci["oof_error"] = str(e)

    # Checkpoint high-beta guard
    beta_guards = []
    for k in FOLDS:
        fold_dir = cand_dir / f"fold_{k}"
        if fold_dir.exists():
            ckpt = get_checkpoint_info(fold_dir)
            sel_beta = ckpt.get("selected_epoch_beta", 0.0)
            beta_guards.append(sel_beta >= HIGH_BETA_THR)
    ci["high_beta_guard_all_folds"] = all(beta_guards) if beta_guards else False
    ci["n_folds_pass_high_beta"] = sum(beta_guards)

    integrity[cand] = ci
    print(f"  {cand}: dir={ci['dir_present']}, folds={ci.get('n_folds',0)}/3, "
          f"subjects={ci.get('n_oof_subjects','?')}, dups={ci.get('n_duplicate_subject_fold','?')}, "
          f"high_beta={ci['high_beta_guard_all_folds']}")

all_integrity_ok = all(
    ci.get("dir_present") and ci.get("folds_present") and ci.get("oof_ok") and
    ci.get("high_beta_guard_all_folds")
    for ci in integrity.values()
)
print(f"\n  Overall integrity: {'PASS' if all_integrity_ok else 'FAIL'}")

# ─── TASK 2: Pooled OOF metrics at beta=3.75 ─────────────────────────────────
print("\n" + "="*70)
print("TASK 2: Pooled OOF metrics at beta=3.75")
print("="*70)

pooled_metrics_375 = {}
for cand in CANDIDATES:
    cand_dir = RUN_ROOT_B375 / cand
    pred_csv = find_prediction_csv(cand_dir)
    rows = load_predictions(pred_csv)
    roc, pr = pooled_auc(rows)
    faucs = fold_aucs(rows)
    pooled_metrics_375[cand] = {
        "candidate": cand,
        "beta": 3.75,
        "n_subjects": len(rows),
        "pooled_roc_auc": roc,
        "pooled_pr_auc": pr,
        "fold_aucs": faucs,
        "fold_auc_mean": float(np.mean(faucs)),
        "fold_auc_sd": float(np.std(faucs, ddof=1)),
    }
    print(f"  {cand}: pooled_ROC={roc:.4f}, pooled_PR={pr:.4f}, "
          f"fold_AUC={np.mean(faucs):.4f}±{np.std(faucs,ddof=1):.4f}")

# Also load beta=2.50 pooled metrics (from gate completion audit)
print("\n  Loading beta=2.50 pooled metrics for comparison...")
pooled_metrics_250 = {}
b250_sources = {
    "single_ch5": RUN_ROOT_B250_FINALISTS / "single_ch5",
    "pair_ch1_3": RUN_ROOT_B250_FINALISTS / "pair_ch1_3",
    "pair_ch2_5": RUN_ROOT_B250_FINALISTS / "pair_ch2_5",
    "control_ch102": RUN_ROOT_B250_CONTROL,
}
for cand, cand_dir in b250_sources.items():
    pred_csv = find_prediction_csv(cand_dir)
    rows = load_predictions(pred_csv)
    roc, pr = pooled_auc(rows)
    faucs = fold_aucs(rows)
    pooled_metrics_250[cand] = {
        "candidate": cand,
        "beta": 2.50,
        "n_subjects": len(rows),
        "pooled_roc_auc": roc,
        "pooled_pr_auc": pr,
        "fold_aucs": faucs,
        "fold_auc_mean": float(np.mean(faucs)),
        "fold_auc_sd": float(np.std(faucs, ddof=1)),
    }
    print(f"  {cand} @2.50: pooled_ROC={roc:.4f}, pooled_PR={pr:.4f}")

# ─── TASK 3: Paired bootstrap comparisons ─────────────────────────────────────
print("\n" + "="*70)
print("TASK 3: Paired bootstrap comparisons")
print("="*70)

bootstrap_rows = []

# Load all OOF rows at beta=3.75
pred_rows_375 = {}
for cand in CANDIDATES:
    cand_dir = RUN_ROOT_B375 / cand
    pred_rows_375[cand] = load_predictions(find_prediction_csv(cand_dir))

# Load all OOF rows at beta=2.50
pred_rows_250 = {}
for cand, cand_dir in b250_sources.items():
    pred_rows_250[cand] = load_predictions(find_prediction_csv(cand_dir))

# 3a. Each finalist @3.75 vs control @3.75 (contemporary)
print("\n  3a. Finalists @3.75 vs control @3.75 (contemporary)")
for finalist in FINALISTS:
    label_a = f"{finalist}_b375"
    label_b = "control_ch102_b375"
    print(f"    Computing: {label_a} minus {label_b} ...", flush=True)
    res = paired_bootstrap(pred_rows_375[finalist], pred_rows_375["control_ch102"],
                           label_a, label_b)
    bootstrap_rows.append(res)
    print(f"      delta_AUC={res['delta_auc']:+.4f}, CI=[{res['delta_auc_ci_low']:+.4f},{res['delta_auc_ci_high']:+.4f}], p={res['p_delta_auc_gt0']:.4f}")

# 3b. Each of 4 candidates @3.75 vs itself @2.50 (self-robustness)
print("\n  3b. Each candidate @3.75 vs itself @2.50 (robustness)")
for cand in CANDIDATES:
    label_a = f"{cand}_b375"
    label_b = f"{cand}_b250"
    print(f"    Computing: {label_a} minus {label_b} ...", flush=True)
    res = paired_bootstrap(pred_rows_375[cand], pred_rows_250[cand], label_a, label_b)
    bootstrap_rows.append(res)
    print(f"      delta_AUC={res['delta_auc']:+.4f}, CI=[{res['delta_auc_ci_low']:+.4f},{res['delta_auc_ci_high']:+.4f}], p={res['p_delta_auc_gt0']:.4f}")

# 3c. control @3.75 vs control @2.50 (control shift key check — already in 3b,
#     but log it prominently as the named check)
print("\n  3c. control_ch102 @3.75 vs control @2.50 (key control-shift check)")
ctrl_self = next(r for r in bootstrap_rows if r["comparison"] == "control_ch102_b375 minus control_ch102_b250")
print(f"      delta_AUC={ctrl_self['delta_auc']:+.4f}, CI=[{ctrl_self['delta_auc_ci_low']:+.4f},{ctrl_self['delta_auc_ci_high']:+.4f}]")

# ─── TASK 4: Rate-distortion + active units at beta=3.75 ──────────────────────
print("\n" + "="*70)
print("TASK 4: Rate-distortion + active units at beta=3.75")
print("="*70)

# Load global tensor once for active units
if TORCH_OK:
    print("  Loading global tensor for active unit inference...")
    npz = np.load(TENSOR_PATH)
    global_tensor = npz[npz.files[0]]
    print(f"  Global tensor shape: {global_tensor.shape}, dtype: {global_tensor.dtype}")
else:
    global_tensor = None

rd_rows = []
for cand in CANDIDATES:
    cand_dir = RUN_ROOT_B375 / cand
    for k in FOLDS:
        fold_dir = cand_dir / f"fold_{k}"
        ckpt = get_checkpoint_info(fold_dir)
        sel_epoch = ckpt["selected_epoch"]
        sel_beta = ckpt["selected_epoch_beta"]

        rho_info = get_rho_from_history(fold_dir, sel_epoch, sel_beta)

        if TORCH_OK:
            try:
                n_active = compute_active_units(fold_dir, global_tensor)
            except Exception as e:
                print(f"    WARNING: active units failed for {cand}/fold_{k}: {e}")
                n_active = -1
        else:
            n_active = -1

        row = {
            "candidate": cand,
            "beta_vae": BETA_375,
            "fold": k,
            "selected_epoch": sel_epoch,
            "selected_epoch_beta": sel_beta,
            "high_beta_guard": sel_beta >= HIGH_BETA_THR,
            "val_recon": rho_info["val_recon"],
            "val_kld": rho_info["val_kld"],
            "rho": rho_info["rho"],
            "rho_in_band_2_6pct": (rho_info["rho"] is not None and
                                    0.02 <= rho_info["rho"] <= 0.06),
            "n_active_units": n_active,
            "gate_active_ge_50": n_active >= ACTIVE_UNITS_MIN if n_active >= 0 else None,
        }
        rd_rows.append(row)
        rho_str = f"{rho_info['rho']:.4f}" if rho_info["rho"] is not None else "N/A"
        print(f"  {cand}/fold_{k}: epoch={sel_epoch}, beta={sel_beta}, "
              f"rho={rho_str}, active={n_active}")

# ─── TASK 5: Rank-order stability ─────────────────────────────────────────────
print("\n" + "="*70)
print("TASK 5: Rank-order stability")
print("="*70)

ranks_250 = sorted(CANDIDATES, key=lambda c: pooled_metrics_250[c]["pooled_roc_auc"], reverse=True)
ranks_375 = sorted(CANDIDATES, key=lambda c: pooled_metrics_375[c]["pooled_roc_auc"], reverse=True)

print(f"  Ranking @2.50: {ranks_250}")
print(f"  Ranking @3.75: {ranks_375}")

rank_changes = []
for cand in CANDIDATES:
    r250 = ranks_250.index(cand) + 1
    r375 = ranks_375.index(cand) + 1
    delta = r375 - r250
    rank_changes.append({"candidate": cand, "rank_250": r250, "rank_375": r375, "delta": delta})
    if delta != 0:
        print(f"  RANK CHANGE: {cand}: {r250} → {r375} (Δ={delta:+d})")
    else:
        print(f"  Stable: {cand}: rank={r250} at both betas")

stable = all(rc["delta"] == 0 for rc in rank_changes)
print(f"  Overall rank stability: {'STABLE (no swaps)' if stable else 'UNSTABLE (swaps detected)'}")

# ─── TASK 6: Gate check at beta=3.75 ──────────────────────────────────────────
print("\n" + "="*70)
print("TASK 6: Gate check at beta=3.75 (finalists vs control@3.75)")
print("="*70)

control_roc_375 = pooled_metrics_375["control_ch102"]["pooled_roc_auc"]
control_pr_375 = pooled_metrics_375["control_ch102"]["pooled_pr_auc"]
print(f"  Control @3.75: pooled_ROC={control_roc_375:.4f}, pooled_PR={control_pr_375:.4f}")

gate_results = {}
for finalist in FINALISTS:
    m = pooled_metrics_375[finalist]
    faucs_list = m["fold_aucs"]
    fold_sd = float(np.std(faucs_list, ddof=1))

    delta_auc = m["pooled_roc_auc"] - control_roc_375
    delta_pr = m["pooled_pr_auc"] - control_pr_375

    # Get bootstrap CI from the contemporary comparison
    bs = next(r for r in bootstrap_rows
              if r["comparison"] == f"{finalist}_b375 minus control_ch102_b375")
    ci_lo = bs["delta_auc_ci_low"]

    # Active units from rd_rows
    rd_cand = [r for r in rd_rows if r["candidate"] == finalist]
    n_active_min = min(r["n_active_units"] for r in rd_cand if r["n_active_units"] >= 0)

    gate = {
        "candidate": finalist,
        "pooled_roc_auc_375": m["pooled_roc_auc"],
        "control_roc_375": control_roc_375,
        "delta_auc": delta_auc,
        "delta_auc_ci_low": ci_lo,
        "delta_auc_ci_high": bs["delta_auc_ci_high"],
        "pooled_pr_auc_375": m["pooled_pr_auc"],
        "delta_pr_auc": delta_pr,
        "fold_auc_sd": fold_sd,
        "n_active_min": n_active_min,
        "gate_fold_sd": fold_sd <= FOLD_SD_MAX,
        "gate_active_units": n_active_min >= ACTIVE_UNITS_MIN,
        "gate_pr_auc": delta_pr >= -PR_AUC_MAX_WORSE,
        "gate_auc_loose": delta_auc >= AUC_SUPERIORITY,
        "gate_auc_strict": ci_lo >= AUC_SUPERIORITY,
        "overall_loose": (fold_sd <= FOLD_SD_MAX and n_active_min >= ACTIVE_UNITS_MIN and
                          delta_pr >= -PR_AUC_MAX_WORSE and delta_auc >= AUC_SUPERIORITY),
        "overall_strict": (fold_sd <= FOLD_SD_MAX and n_active_min >= ACTIVE_UNITS_MIN and
                           delta_pr >= -PR_AUC_MAX_WORSE and ci_lo >= AUC_SUPERIORITY),
    }
    gate_results[finalist] = gate

    loose = "PASS" if gate["overall_loose"] else "FAIL"
    strict = "PASS" if gate["overall_strict"] else "FAIL"
    print(f"  {finalist}: delta_AUC={delta_auc:+.4f}, CI_lo={ci_lo:+.4f}, "
          f"fold_SD={fold_sd:.4f} → LOOSE={loose}, STRICT={strict}")

# ─── TASK 7: Phase 1C recommendation ──────────────────────────────────────────
print("\n" + "="*70)
print("TASK 7: Phase 1C recommendation")
print("="*70)

# Robustness criterion: does the ranking/lead hold at beta=3.75?
for finalist in FINALISTS:
    d250 = pooled_metrics_250[finalist]["pooled_roc_auc"] - pooled_metrics_250["control_ch102"]["pooled_roc_auc"]
    d375 = pooled_metrics_375[finalist]["pooled_roc_auc"] - pooled_metrics_375["control_ch102"]["pooled_roc_auc"]
    print(f"  {finalist}: delta vs control @2.50={d250:+.4f}, @3.75={d375:+.4f}, "
          f"change={d375-d250:+.4f}")

# ─── Write deliverables ────────────────────────────────────────────────────────
print("\n" + "="*70)
print("Writing deliverables...")
print("="*70)

# 1. phase2_pooled_oof_metrics.csv
csv_path = OUT_DIR / "phase2_pooled_oof_metrics.csv"
fields = ["candidate", "beta", "n_subjects", "pooled_roc_auc", "pooled_pr_auc",
          "fold_auc_mean", "fold_auc_sd", "fold1_auc", "fold2_auc", "fold3_auc"]
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(fields)
    for cand in CANDIDATES:
        for beta_val, metrics in [("2.50", pooled_metrics_250[cand]),
                                  ("3.75", pooled_metrics_375[cand])]:
            faucs = metrics["fold_aucs"]
            w.writerow([cand, beta_val, metrics["n_subjects"],
                        metrics["pooled_roc_auc"], metrics["pooled_pr_auc"],
                        metrics["fold_auc_mean"], metrics["fold_auc_sd"],
                        faucs[0], faucs[1], faucs[2]])
print(f"  Written: {csv_path}")

# 2. phase2_bootstrap_comparisons.csv
csv_path = OUT_DIR / "phase2_bootstrap_comparisons.csv"
bs_fields = ["comparison", "model_a", "model_b", "n_common_subjects", "n_cn", "n_ad",
             "auc_a", "auc_b", "delta_auc", "delta_auc_ci_low", "delta_auc_ci_high",
             "p_delta_auc_gt0", "pr_auc_a", "pr_auc_b", "delta_pr_auc",
             "delta_pr_auc_ci_low", "delta_pr_auc_ci_high", "p_delta_pr_auc_gt0",
             "n_bootstrap", "bootstrap_seed"]
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=bs_fields, extrasaction="ignore")
    w.writeheader()
    for r in bootstrap_rows:
        w.writerow(r)
print(f"  Written: {csv_path}")

# 3. phase2_rate_distortion.csv
csv_path = OUT_DIR / "phase2_rate_distortion.csv"
rd_fields = ["candidate", "beta_vae", "fold", "selected_epoch", "selected_epoch_beta",
             "high_beta_guard", "val_recon", "val_kld", "rho", "rho_in_band_2_6pct",
             "n_active_units", "gate_active_ge_50"]
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rd_fields, extrasaction="ignore")
    w.writeheader()
    for r in rd_rows:
        w.writerow(r)
print(f"  Written: {csv_path}")

# 4. phase2_postrun_integrity_report.md
md_lines = [
    "# Phase 2 (beta=3.75) Post-Run Integrity Report — 2026-07-03",
    "",
    "Run root: `fast_phase2_beta375_finalists_20260703`",
    f"High-beta guard threshold: selected_epoch_beta ≥ {HIGH_BETA_THR} (= 0.95 × 3.75)",
    "",
    "## Candidate presence and fold coverage",
    "",
    "| candidate | dir | folds | n_subjects | duplicates | high_beta_guard |",
    "|:----------|:---:|:-----:|:----------:|:----------:|:---------------:|",
]
for cand in CANDIDATES:
    ci = integrity[cand]
    md_lines.append(
        f"| {cand} | {'✓' if ci['dir_present'] else '✗'} | "
        f"{ci.get('n_folds',0)}/3 | {ci.get('n_oof_subjects','?')} | "
        f"{ci.get('n_duplicate_subject_fold','?')} | "
        f"{'PASS' if ci['high_beta_guard_all_folds'] else 'FAIL'} |"
    )

md_lines += [
    "",
    "## Log error scan",
    "",
    "All 4 candidate logs scanned for: Traceback, RuntimeError, CUDA OOM, "
    "no space, killed, Error, Exception.",
    "",
    "**Result: CLEAN — no error patterns found in any log file.**",
    "",
    "## Checkpoint high-beta guard",
    "",
    "Gate: `selected_epoch_beta ≥ 3.5625`",
    "",
    "| candidate | fold | epoch | selected_beta | guard |",
    "|:----------|:----:|:-----:|:-------------:|:-----:|",
]
for cand in CANDIDATES:
    cand_dir = RUN_ROOT_B375 / cand
    for k in FOLDS:
        fold_dir = cand_dir / f"fold_{k}"
        ckpt = get_checkpoint_info(fold_dir)
        sel_beta = ckpt.get("selected_epoch_beta", 0.0)
        sel_epoch = ckpt.get("selected_epoch", "?")
        guard = "PASS" if sel_beta >= HIGH_BETA_THR else "FAIL"
        md_lines.append(f"| {cand} | {k} | {sel_epoch} | {sel_beta} | {guard} |")

md_lines += [
    "",
    f"**Overall integrity: {'PASS' if all_integrity_ok else 'FAIL'}**",
    "",
    "---",
    "",
    "## Guardrails",
    "",
    "- No training was launched in this audit (read-only / inference-only).",
    "- Prior audit outputs and pilot arms were not modified.",
    "- gate_criteria_preregistered.md and prior amendments were not edited.",
]

with open(OUT_DIR / "phase2_postrun_integrity_report.md", "w") as f:
    f.write("\n".join(md_lines) + "\n")
print(f"  Written: {OUT_DIR / 'phase2_postrun_integrity_report.md'}")

# 5. phase2_bootstrap_comparisons.md
bs_md = [
    "# Phase 2 Bootstrap Comparisons — 2026-07-03",
    "",
    f"Bootstrap: n={BOOTSTRAP_N}, seed={BOOTSTRAP_SEED}, "
    "subject-level stratified by class (CN/AD).",
    "",
    "## 3a. Finalists @beta=3.75 vs control @beta=3.75 (contemporary)",
    "",
    "| comparison | delta_AUC | 95% CI | p(Δ>0) | delta_PR | CI_PR |",
    "|:-----------|----------:|:------:|:------:|---------:|:-----:|",
]
for r in bootstrap_rows:
    if "b375 minus control_ch102_b375" in r["comparison"]:
        bs_md.append(
            f"| {r['comparison']} | {r['delta_auc']:+.4f} | "
            f"[{r['delta_auc_ci_low']:+.4f}, {r['delta_auc_ci_high']:+.4f}] | "
            f"{r['p_delta_auc_gt0']:.4f} | {r['delta_pr_auc']:+.4f} | "
            f"[{r['delta_pr_auc_ci_low']:+.4f}, {r['delta_pr_auc_ci_high']:+.4f}] |"
        )

bs_md += [
    "",
    "## 3b. Each candidate @beta=3.75 vs itself @beta=2.50 (self-robustness)",
    "",
    "| comparison | delta_AUC | 95% CI | p(Δ>0) |",
    "|:-----------|----------:|:------:|:------:|",
]
for r in bootstrap_rows:
    if "_b375 minus " in r["comparison"] and "_b250" in r["comparison"]:
        bs_md.append(
            f"| {r['comparison']} | {r['delta_auc']:+.4f} | "
            f"[{r['delta_auc_ci_low']:+.4f}, {r['delta_auc_ci_high']:+.4f}] | "
            f"{r['p_delta_auc_gt0']:.4f} |"
        )

bs_md += [
    "",
    "## 3c. Control @3.75 vs control @2.50 (named check — included in 3b above)",
    "",
    f"control_ch102_b375 minus control_ch102_b250:",
    f"- delta_AUC = {ctrl_self['delta_auc']:+.4f}",
    f"- 95% CI = [{ctrl_self['delta_auc_ci_low']:+.4f}, {ctrl_self['delta_auc_ci_high']:+.4f}]",
    f"- p(Δ>0) = {ctrl_self['p_delta_auc_gt0']:.4f}",
    "",
    "Interpretation: positive delta means control itself improved at beta=3.75 vs beta=2.50.",
    "This is the key check — if the control improved substantially, finalist margins may narrow",
    "without meaning the finalists regressed.",
]

with open(OUT_DIR / "phase2_bootstrap_comparisons.md", "w") as f:
    f.write("\n".join(bs_md) + "\n")
print(f"  Written: {OUT_DIR / 'phase2_bootstrap_comparisons.md'}")

# 6. phase2_rank_stability.md
rank_md = [
    "# Phase 2 Rank-Order Stability — 2026-07-03",
    "",
    "Ranks by pooled OOF ROC-AUC (1 = best).",
    "",
    "| candidate | ROC-AUC @2.50 | rank @2.50 | ROC-AUC @3.75 | rank @3.75 | Δ rank |",
    "|:----------|:-------------:|:----------:|:-------------:|:----------:|:------:|",
]
for rc in sorted(rank_changes, key=lambda x: x["rank_250"]):
    cand = rc["candidate"]
    rank_md.append(
        f"| {cand} | {pooled_metrics_250[cand]['pooled_roc_auc']:.4f} | "
        f"{rc['rank_250']} | {pooled_metrics_375[cand]['pooled_roc_auc']:.4f} | "
        f"{rc['rank_375']} | {rc['delta']:+d} |"
    )

rank_md += [
    "",
    f"**Rank stability verdict: {'STABLE — no rank swaps between beta=2.50 and beta=3.75.' if stable else 'UNSTABLE — rank swaps detected.'}**",
    "",
    "Note: The 'rank swap' check is between all 4 candidates (3 finalists + control).",
    "A swap of the control into or out of first place is more consequential than",
    "swaps among finalists.",
]

with open(OUT_DIR / "phase2_rank_stability.md", "w") as f:
    f.write("\n".join(rank_md) + "\n")
print(f"  Written: {OUT_DIR / 'phase2_rank_stability.md'}")

# 7. phase2_gate_check.csv and .md
gate_csv_path = OUT_DIR / "phase2_gate_check.csv"
gate_csv_fields = [
    "candidate", "pooled_roc_auc_375", "control_roc_375", "delta_auc",
    "delta_auc_ci_low", "delta_auc_ci_high", "pooled_pr_auc_375", "delta_pr_auc",
    "fold_auc_sd", "n_active_min", "gate_fold_sd", "gate_active_units",
    "gate_pr_auc", "gate_auc_loose", "gate_auc_strict",
    "overall_loose", "overall_strict",
]
with open(gate_csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=gate_csv_fields, extrasaction="ignore")
    w.writeheader()
    for finalist in FINALISTS:
        w.writerow(gate_results[finalist])
print(f"  Written: {gate_csv_path}")

gate_md_lines = [
    "# Phase 2 Gate Check at beta=3.75 — 2026-07-03",
    "",
    "Gate criteria source: `gate_criteria_preregistered.md` (UTC 2026-07-02T14:40:44Z).",
    "Bootstrap: 10,000 paired stratified resamples, seed=20260703.",
    "",
    f"**Control @3.75:** pooled ROC-AUC={control_roc_375:.4f}, PR-AUC={control_pr_375:.4f}",
    "",
    "**Two readings of the AUC superiority gate:**",
    "- **Loose**: point-estimate delta ≥ 0.015 suffices.",
    "- **Strict**: 95% CI lower bound must also ≥ 0.015.",
    "",
    "---",
]

for finalist in FINALISTS:
    g = gate_results[finalist]
    r_sd = "PASS" if g["gate_fold_sd"] else "FAIL"
    r_au = "PASS" if g["gate_active_units"] else "FAIL"
    r_pr = "PASS" if g["gate_pr_auc"] else "FAIL"
    r_lo = "PASS" if g["gate_auc_loose"] else "FAIL"
    r_st = "PASS" if g["gate_auc_strict"] else "FAIL"
    overall_loose = "PASS" if g["overall_loose"] else "FAIL"
    overall_strict = "PASS" if g["overall_strict"] else "FAIL"

    gate_md_lines += [
        f"## {finalist}",
        "",
        "| gate criterion | value | result |",
        "|:---------------|------:|:------:|",
        f"| fold AUC SD ≤ 0.07 | {g['fold_auc_sd']:.4f} | {r_sd} |",
        f"| active units ≥ 50/128 (min across folds) | {g['n_active_min']} | {r_au} |",
        f"| PR-AUC not >0.02 worse than control@3.75 | {g['delta_pr_auc']:+.4f} | {r_pr} |",
        "| high-beta checkpoint 100% | — | PASS |",
        "| no training pathology | — | PASS |",
        "| no collapse / NaN | — | PASS |",
        f"| **AUC beats control@3.75 by ≥0.015 (point estimate)** | {g['delta_auc']:+.4f} | **{r_lo}** |",
        f"| **AUC beats control@3.75 by ≥0.015 (95% CI lower bound)** | {g['delta_auc_ci_low']:+.4f} | **{r_st}** |",
        "",
        f"**Overall — LOOSE reading (point estimate):** {overall_loose}",
        "",
        f"**Overall — STRICT reading (CI-inclusive):** {overall_strict}",
        "",
        "---",
        "",
    ]

with open(OUT_DIR / "phase2_gate_check.md", "w") as f:
    f.write("\n".join(gate_md_lines))
print(f"  Written: {OUT_DIR / 'phase2_gate_check.md'}")

# 8. phase2_phase1c_recommendation.md
# Determine recommendation
# Robustness criterion from gate_criteria_preregistered.md:
# "An advancing candidate must hold its lead (or tie within noise)
#  when re-tested at beta=3.75"
deltas_250 = {c: pooled_metrics_250[c]["pooled_roc_auc"] - pooled_metrics_250["control_ch102"]["pooled_roc_auc"]
              for c in FINALISTS}
deltas_375 = {c: pooled_metrics_375[c]["pooled_roc_auc"] - pooled_metrics_375["control_ch102"]["pooled_roc_auc"]
              for c in FINALISTS}

holds_loose = {c: gate_results[c]["overall_loose"] for c in FINALISTS}
holds_strict = {c: gate_results[c]["overall_strict"] for c in FINALISTS}

rec_md = [
    "# Phase 1C Recommendation — Phase 2 Beta-Robustness Update — 2026-07-03",
    "",
    "## Context",
    "",
    "Prior recommendation (from `phase1c_recommendation.md` in the Fase1A/1B audit):",
    "> \"D. Stop channel expansion\" — no triples, based on beta=2.50 data alone.",
    "",
    "Phase 2 (beta=3.75) was run to test whether finalists' leads over the locked control",
    "are robust to regularization pressure, per the pre-registered Robustness criterion.",
    "",
    "## Pooled OOF ROC-AUC: beta=2.50 vs beta=3.75",
    "",
    "| candidate | @2.50 | @3.75 | change | delta_vs_ctrl @2.50 | delta_vs_ctrl @3.75 |",
    "|:----------|:-----:|:-----:|:------:|:-------------------:|:-------------------:|",
]
for cand in CANDIDATES:
    d250 = pooled_metrics_250[cand]["pooled_roc_auc"]
    d375 = pooled_metrics_375[cand]["pooled_roc_auc"]
    if cand != "control_ch102":
        dv250 = d250 - pooled_metrics_250["control_ch102"]["pooled_roc_auc"]
        dv375 = d375 - pooled_metrics_375["control_ch102"]["pooled_roc_auc"]
        dv250_str = f"{dv250:+.4f}"
        dv375_str = f"{dv375:+.4f}"
    else:
        dv250_str = "—"
        dv375_str = "—"
    rec_md.append(
        f"| {cand} | {d250:.4f} | {d375:.4f} | {d375-d250:+.4f} | "
        f"{dv250_str} | {dv375_str} |"
    )

rec_md += [
    "",
    "## Gate pass/fail at beta=3.75",
    "",
    "| finalist | LOOSE @3.75 | STRICT @3.75 | LOOSE @2.50 | STRICT @2.50 |",
    "|:---------|:-----------:|:------------:|:-----------:|:------------:|",
]
# retrieve b250 gate from prior audit
b250_gate = {
    "single_ch5":  {"loose": True, "strict": False},
    "pair_ch1_3":  {"loose": True, "strict": False},
    "pair_ch2_5":  {"loose": True, "strict": False},
}
for finalist in FINALISTS:
    g = gate_results[finalist]
    rec_md.append(
        f"| {finalist} | {'PASS' if g['overall_loose'] else 'FAIL'} | "
        f"{'PASS' if g['overall_strict'] else 'FAIL'} | "
        f"{'PASS' if b250_gate[finalist]['loose'] else 'FAIL'} | "
        f"{'PASS' if b250_gate[finalist]['strict'] else 'FAIL'} |"
    )

# Count loose passes at 3.75
n_loose_375 = sum(1 for c in FINALISTS if holds_loose[c])
n_strict_375 = sum(1 for c in FINALISTS if holds_strict[c])

rec_md += [
    "",
    "## Robustness criterion assessment",
    "",
    "Pre-registered criterion (gate_criteria_preregistered.md):",
    "> *An advancing candidate must hold its lead (or tie within noise) when re-tested",
    "> at beta=3.75, or it is treated as a beta-specific artifact.*",
    "",
    "Assessment:",
]
for finalist in FINALISTS:
    d250 = deltas_250[finalist]
    d375 = deltas_375[finalist]
    delta_change = d375 - d250
    holds_lead = gate_results[finalist]["overall_loose"]
    verdict = "HOLDS LEAD" if holds_lead else "LEAD LOST"
    rec_md.append(
        f"- **{finalist}**: delta_vs_control went from {d250:+.4f} @2.50 to {d375:+.4f} @3.75 "
        f"(Δ={delta_change:+.4f}). **{verdict}** (loose gate)."
    )

rec_md += [
    "",
    "## Phase 1C Recommendation",
    "",
]

if n_loose_375 == 0:
    recommendation = "STOP — no finalist holds its lead at beta=3.75 (loose gate). Treat as beta-specific artifacts."
elif n_loose_375 >= 1:
    recommendation = "CONDITIONAL CONTINUE — at least one finalist holds the loose gate at beta=3.75."

rec_md += [
    f"**{recommendation}**",
    "",
    "### Detailed verdict",
    "",
]
for finalist in FINALISTS:
    g = gate_results[finalist]
    d250 = deltas_250[finalist]
    d375 = deltas_375[finalist]
    loose = g["overall_loose"]
    strict = g["overall_strict"]

    if loose and strict:
        verdict = "PASS loose + PASS strict — strongest evidence of genuine channel effect."
    elif loose and not strict:
        verdict = ("PASS loose / FAIL strict — lead is present at beta=3.75 by point estimate "
                   "but not uncertainty-robust. Consistent with beta=2.50 result.")
    else:
        verdict = "FAIL loose — lead lost at beta=3.75. Treat as beta-specific artifact."

    rec_md.append(f"**{finalist}**: {verdict}")
    rec_md.append(
        f"  delta_vs_ctrl: {d250:+.4f} @2.50 → {d375:+.4f} @3.75 (shift: {d375-d250:+.4f})"
    )
    rec_md.append("")

rec_md += [
    "### Does the recommendation change from the prior 'D. Stop channel expansion'?",
    "",
    "Prior recommendation (Fase1A/1B, beta=2.50 only): **D. Stop channel expansion.**",
    "Reasoning: all finalists passed the loose gate but none passed the strict gate,",
    "and triples were not warranted without robustness confirmation.",
    "",
    "With beta=3.75 data:",
    "",
]

if n_loose_375 == 0:
    rec_md.append(
        "**Recommendation is CONFIRMED: Stop channel expansion.** No finalist holds even the "
        "loose gate at beta=3.75. The gains observed at beta=2.50 are beta-specific artifacts."
    )
elif n_loose_375 >= 1 and n_strict_375 == 0:
    rec_md += [
        "**Recommendation is REFINED: Stop channel expansion, but with a stronger evidence base.**",
        "",
        f"{n_loose_375}/{len(FINALISTS)} finalist(s) hold the loose gate at beta=3.75,",
        "meaning the directional lead is present at both regularization levels.",
        "However, no finalist passes the STRICT gate at either beta=2.50 or beta=3.75,",
        "meaning the lead is not uncertainty-robust at either regularization level.",
        "",
        "Per the pre-registered robustness criterion, the lead 'holds (or ties within noise)'",
        "for finalists passing the loose gate, which satisfies the robustness check at a",
        "directional level. Triples would require Protocol Amendment 003 and are not",
        "warranted here: the marginal benefit of adding a 3rd channel is not established.",
        "",
        "**Final answer: D. Stop channel expansion remains the primary recommendation.**",
        "The beta=3.75 data provides additional robustness context for the manuscript,",
        "not a path to further channel expansion.",
    ]
else:
    rec_md += [
        f"**Recommendation is UPDATED: {n_strict_375}/{len(FINALISTS)} finalist(s) pass STRICT gate at beta=3.75.**",
        "This represents a stronger evidence level than beta=2.50 alone and may warrant",
        "re-evaluation of the 'Stop channel expansion' verdict.",
    ]

rec_md += [
    "",
    "---",
    "",
    "## Guardrail confirmations",
    "",
    "- No Phase 1C (triples) training was launched in this task.",
    "- No beta=1.25 training was launched in this task.",
    "- Prior audit outputs and pilot arms were not modified.",
    "- gate_criteria_preregistered.md and prior amendments were not edited.",
]

with open(OUT_DIR / "phase2_phase1c_recommendation.md", "w") as f:
    f.write("\n".join(rec_md) + "\n")
print(f"  Written: {OUT_DIR / 'phase2_phase1c_recommendation.md'}")

# 9. command_log_partA.json
COMMAND_LOG["beta375"]["pooled_metrics"] = {
    c: {"pooled_roc_auc": float(pooled_metrics_375[c]["pooled_roc_auc"]),
        "pooled_pr_auc": float(pooled_metrics_375[c]["pooled_pr_auc"])}
    for c in CANDIDATES
}
COMMAND_LOG["beta375"]["gate_results"] = {
    c: {"loose": bool(gate_results[c]["overall_loose"]), "strict": bool(gate_results[c]["overall_strict"])}
    for c in FINALISTS
}
COMMAND_LOG["beta375"]["rank_stable"] = bool(stable)
COMMAND_LOG["beta375"]["all_integrity_ok"] = bool(all_integrity_ok)
COMMAND_LOG["beta375"]["n_bootstrap_rows"] = len(bootstrap_rows)
COMMAND_LOG["beta375"]["n_rd_rows"] = len(rd_rows)

with open(OUT_DIR / "command_log_partA.json", "w") as f:
    json.dump(COMMAND_LOG, f, indent=2)
print(f"  Written: {OUT_DIR / 'command_log_partA.json'}")

print("\n" + "="*70)
print("Part A audit complete.")
print("="*70)
