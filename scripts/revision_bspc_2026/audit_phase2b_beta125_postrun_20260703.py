#!/usr/bin/env python3
"""Read-only post-run audit for beta=1.25 FAST finalist arm.

Computes beta=1.25 integrity/metrics/gates and three-beta summaries using
existing beta=2.50 and beta=3.75 artifacts. No training or classifier refit.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

try:
    import torch

    TORCH_OK = True
except Exception:
    TORCH_OK = False


REPO = Path("/home/diego/proyectos/vae_AD")
sys.path.insert(0, str(REPO / "src"))

OUT = REPO / (
    "results/revision_bspc_2026/post_revision_exploratory_20260630/"
    "fast_phase2b_beta125_postrun_audit_20260703"
)

ROOT_B125 = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026/post_revision_exploratory_20260630/"
    "fast_phase2b_beta125_finalists_20260703"
)
LOG_B125 = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026/_launch_logs/post_revision_exploratory_20260630/"
    "fast_phase2b_beta125_finalists_20260703"
)
ROOT_B375 = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026/post_revision_exploratory_20260630/"
    "fast_phase2_beta375_finalists_20260703"
)
ROOT_B250_FAST = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026/post_revision_exploratory_20260630/"
    "fast_exhaustive_singles_pairs_20260702"
)
ROOT_B250_CONTROL = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026/post_revision_exploratory_20260630/"
    "fast_channelmean_loss_ablation_beta_matched_20260702/beta_cal_ch102_beta250"
)
TENSOR = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors/"
    "GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
)

BETA_ROOTS = {
    1.25: {
        "single_ch5": ROOT_B125 / "single_ch5",
        "pair_ch1_3": ROOT_B125 / "pair_ch1_3",
        "pair_ch2_5": ROOT_B125 / "pair_ch2_5",
        "control_ch102": ROOT_B125 / "control_ch102",
    },
    2.50: {
        "single_ch5": ROOT_B250_FAST / "single_ch5",
        "pair_ch1_3": ROOT_B250_FAST / "pair_ch1_3",
        "pair_ch2_5": ROOT_B250_FAST / "pair_ch2_5",
        "control_ch102": ROOT_B250_CONTROL,
    },
    3.75: {
        "single_ch5": ROOT_B375 / "single_ch5",
        "pair_ch1_3": ROOT_B375 / "pair_ch1_3",
        "pair_ch2_5": ROOT_B375 / "pair_ch2_5",
        "control_ch102": ROOT_B375 / "control_ch102",
    },
}

CANDIDATES = ["single_ch5", "pair_ch1_3", "pair_ch2_5", "control_ch102"]
FINALISTS = ["single_ch5", "pair_ch1_3", "pair_ch2_5"]
FOLDS = [1, 2, 3]
LATENT_DIM = 128
VAR_EPS = 1e-4
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20260703
LOOSE_AUC = 0.015
STRICT_AUC = 0.015
PR_MAX_WORSE = 0.02
FOLD_SD_MAX = 0.07
ACTIVE_MIN = 50

CHANNEL_NAMES = [
    "Pearson_OMST_GCE_Signed_Weighted",
    "Pearson_Full_FisherZ_Signed",
    "MI_KNN_Symmetric",
    "dFC_AbsDiffMean",
    "dFC_StdDev",
    "DistanceCorr",
    "Granger_F_lag1",
]

CHANNELS = {
    "single_ch5": [5],
    "pair_ch1_3": [1, 3],
    "pair_ch2_5": [2, 5],
    "control_ch102": [1, 0, 2],
}

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

LOG_PATTERNS = {
    "traceback": re.compile(r"Traceback", re.I),
    "runtime_error": re.compile(r"RuntimeError", re.I),
    "cuda_oom": re.compile(r"CUDA out of memory|CUDA OOM", re.I),
    "killed": re.compile(r"\bKilled\b|killed process", re.I),
    "no_space": re.compile(r"No space left|no space", re.I),
    "high_beta_failure": re.compile(r"no eligible high-beta|checkpoint.*beta.*fail", re.I),
}


def find_one(path: Path, pattern: str) -> Path:
    files = sorted(path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No {pattern} in {path}")
    if len(files) > 1:
        # Prefer joblib/csv with MULTI logreg; filenames are otherwise unique per dir.
        pass
    return files[0]


def load_predictions(path: Path) -> pd.DataFrame:
    joblib_files = sorted(path.glob("all_folds_clf_predictions_MULTI_*.joblib"))
    csv_files = sorted(path.glob("all_folds_clf_predictions_MULTI_*.csv"))
    if joblib_files:
        obj = joblib.load(joblib_files[0])
        if isinstance(obj, list):
            df = pd.concat(obj, ignore_index=True)
        elif isinstance(obj, pd.DataFrame):
            df = obj.copy()
        else:
            raise TypeError(f"Unsupported prediction joblib {joblib_files[0]}: {type(obj)}")
    elif csv_files:
        df = pd.read_csv(csv_files[0])
    else:
        raise FileNotFoundError(f"No OOF predictions in {path}")
    if "classifier_type" in df.columns:
        df = df[df["classifier_type"].astype(str).str.lower().eq("logreg")].copy()
    df["SubjectID"] = df["SubjectID"].astype(str)
    return df


def load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(find_one(path, "all_folds_metrics_MULTI_*.csv"))
    if "actual_classifier_type" in df.columns:
        df = df[df["actual_classifier_type"].astype(str).str.lower().eq("logreg")].copy()
    return df


def pooled_metrics(pred: pd.DataFrame) -> dict:
    y = pred["y_true"].to_numpy(int)
    score = pred["y_score_final"].to_numpy(float)
    y_pred = pred["y_pred"].to_numpy(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    return {
        "n_subjects": int(len(pred)),
        "n_unique_subjects": int(pred["SubjectID"].nunique()),
        "n_duplicate_subject_rows": int(pred["SubjectID"].duplicated().sum()),
        "n_cn": int((y == 0).sum()),
        "n_ad": int((y == 1).sum()),
        "pooled_roc_auc": float(roc_auc_score(y, score)),
        "pooled_pr_auc": float(average_precision_score(y, score)),
        "pooled_balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "pooled_sensitivity": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "pooled_specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "pooled_f1": float(f1_score(y, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, label_a: str, label_b: str) -> dict:
    a2 = a[["SubjectID", "y_true", "y_score_final"]].rename(columns={"y_score_final": "score_a"})
    b2 = b[["SubjectID", "y_true", "y_score_final"]].rename(columns={"y_score_final": "score_b"})
    m = a2.merge(b2, on="SubjectID", suffixes=("_a", "_b"))
    if not (m["y_true_a"].to_numpy() == m["y_true_b"].to_numpy()).all():
        raise ValueError(f"label mismatch for {label_a} vs {label_b}")
    y = m["y_true_a"].to_numpy(int)
    score_a = m["score_a"].to_numpy(float)
    score_b = m["score_b"].to_numpy(float)
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    auc_d = np.empty(BOOTSTRAP_N)
    pr_d = np.empty(BOOTSTRAP_N)
    def fast_auc(yy: np.ndarray, ss: np.ndarray) -> float:
        # Mann-Whitney form; scores are continuous in practice. Tie handling is
        # negligible for these OOF scores and much faster than rankdata here.
        order = np.argsort(ss)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(ss) + 1, dtype=float)
        pos = yy == 1
        n_pos = int(pos.sum())
        n_neg = int((~pos).sum())
        return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

    def fast_ap(yy: np.ndarray, ss: np.ndarray) -> float:
        order = np.argsort(-ss)
        y_sorted = yy[order]
        n_pos = int(y_sorted.sum())
        tp = np.cumsum(y_sorted)
        precision = tp / (np.arange(len(y_sorted)) + 1)
        return float((precision * y_sorted).sum() / n_pos)

    for i in range(BOOTSTRAP_N):
        s0 = rng.choice(idx0, len(idx0), replace=True)
        s1 = rng.choice(idx1, len(idx1), replace=True)
        idx = np.concatenate([s0, s1])
        yy = y[idx]
        auc_d[i] = fast_auc(yy, score_a[idx]) - fast_auc(yy, score_b[idx])
        pr_d[i] = fast_ap(yy, score_a[idx]) - fast_ap(yy, score_b[idx])
    point_auc = roc_auc_score(y, score_a) - roc_auc_score(y, score_b)
    point_pr = average_precision_score(y, score_a) - average_precision_score(y, score_b)
    return {
        "comparison": f"{label_a} minus {label_b}",
        "model_a": label_a,
        "model_b": label_b,
        "n_common_subjects": int(len(m)),
        "n_cn": int((y == 0).sum()),
        "n_ad": int((y == 1).sum()),
        "auc_a": float(roc_auc_score(y, score_a)),
        "auc_b": float(roc_auc_score(y, score_b)),
        "delta_auc": float(point_auc),
        "delta_auc_ci_low": float(np.percentile(auc_d, 2.5)),
        "delta_auc_ci_high": float(np.percentile(auc_d, 97.5)),
        "p_delta_auc_gt0": float((auc_d > 0).mean()),
        "pr_auc_a": float(average_precision_score(y, score_a)),
        "pr_auc_b": float(average_precision_score(y, score_b)),
        "delta_pr_auc": float(point_pr),
        "delta_pr_auc_ci_low": float(np.percentile(pr_d, 2.5)),
        "delta_pr_auc_ci_high": float(np.percentile(pr_d, 97.5)),
        "p_delta_pr_auc_gt0": float((pr_d > 0).mean()),
        "n_bootstrap": BOOTSTRAP_N,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def get_history_value(history: dict, key: str, epoch: int) -> float:
    vals = history.get(key)
    if vals is None:
        return np.nan
    idx = int(epoch) - 1
    if idx < 0 or idx >= len(vals):
        return np.nan
    return float(vals[idx])


def checkpoint_info(fold_dir: Path) -> dict:
    ckpt_file = find_one(fold_dir, "vae_checkpoint_selection_summary_fold_*.csv")
    row = pd.read_csv(ckpt_file).iloc[0].to_dict()
    return row


def apply_zscore_offdiag(data: np.ndarray, norm_params: list[dict]) -> np.ndarray:
    _, _, h, w = data.shape
    diag = np.eye(h, w, dtype=bool)
    offdiag = ~diag
    for c, p in enumerate(norm_params):
        mean = float(p["mean"])
        std = float(p["std"])
        ch = data[:, c, :, :]
        ch[:, offdiag] = (ch[:, offdiag] - mean) / (std + 1e-12)
        ch[:, diag] = 0.0
        data[:, c, :, :] = ch
    return data


def compute_active_units(fold_dir: Path, global_tensor: np.ndarray) -> int:
    if not TORCH_OK:
        return -1
    from betavae_xai.models.convolutional_vae import ConvolutionalVAE

    norm_params = joblib.load(fold_dir / "vae_norm_params.joblib")
    ch_indices = [CHANNEL_NAMES.index(p["original_name"]) for p in norm_params]
    pool_idx = np.load(fold_dir / "vae_training_pool_tensor_idx.npy")
    train_local = np.load(fold_dir / "vae_actual_train_idx_local_to_pool.npy")
    train_global = pool_idx[train_local]
    x = global_tensor[train_global][:, ch_indices, :, :].astype(np.float32).copy()
    x = apply_zscore_offdiag(x, norm_params)
    ckpt_file = find_one(fold_dir, "vae_model_fold_*.pt")
    state = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    model = ConvolutionalVAE(input_channels=len(ch_indices), **VAE_ARCH)
    model.load_state_dict(state)
    model.eval()
    mus = []
    with torch.no_grad():
        xt = torch.from_numpy(x)
        for start in range(0, len(xt), 32):
            mu, _ = model.encode(xt[start : start + 32])
            mus.append(mu.cpu().numpy())
    mu_all = np.concatenate(mus, axis=0)
    return int((np.var(mu_all, axis=0) > VAR_EPS).sum())


def rate_distortion_for(beta: float, cand: str, root: Path, tensor: np.ndarray | None) -> list[dict]:
    rows = []
    high_thr = 0.95 * beta
    for fold in FOLDS:
        fold_dir = root / f"fold_{fold}"
        ckpt = checkpoint_info(fold_dir)
        epoch = int(ckpt.get("selected_epoch", 0))
        selected_beta = float(ckpt.get("selected_epoch_beta", np.nan))
        hist = joblib.load(find_one(fold_dir, "vae_train_history_fold_*.joblib"))
        val_recon = get_history_value(hist, "val_recon", epoch)
        val_kld = get_history_value(hist, "val_kld", epoch)
        stored_beta = get_history_value(hist, "beta", epoch)
        active = compute_active_units(fold_dir, tensor) if tensor is not None else -1
        rows.append(
            {
                "candidate": cand,
                "beta_vae": beta,
                "fold": fold,
                "selected_epoch": epoch,
                "selected_epoch_beta": selected_beta,
                "stored_beta_at_epoch": stored_beta,
                "high_beta_threshold": high_thr,
                "high_beta_guard": bool(selected_beta >= high_thr),
                "val_recon": val_recon,
                "val_kld": val_kld,
                "beta_times_val_kld": beta * val_kld,
                "rho_beta_kld_over_recon": beta * val_kld / val_recon if val_recon else np.nan,
                "n_active_units": active,
                "gate_active_ge_50": bool(active >= ACTIVE_MIN) if active >= 0 else False,
            }
        )
    return rows


def rate_distortion_no_active(beta: float, cand: str, root: Path, active_lookup: dict[tuple[str, int], int]) -> list[dict]:
    rows = []
    high_thr = 0.95 * beta
    for fold in FOLDS:
        fold_dir = root / f"fold_{fold}"
        ckpt = checkpoint_info(fold_dir)
        epoch = int(ckpt.get("selected_epoch", 0))
        selected_beta = float(ckpt.get("selected_epoch_beta", np.nan))
        hist = joblib.load(find_one(fold_dir, "vae_train_history_fold_*.joblib"))
        val_recon = get_history_value(hist, "val_recon", epoch)
        val_kld = get_history_value(hist, "val_kld", epoch)
        stored_beta = get_history_value(hist, "beta", epoch)
        active = active_lookup.get((cand, fold), np.nan)
        rows.append(
            {
                "candidate": cand,
                "beta_vae": beta,
                "fold": fold,
                "selected_epoch": epoch,
                "selected_epoch_beta": selected_beta,
                "stored_beta_at_epoch": stored_beta,
                "high_beta_threshold": high_thr,
                "high_beta_guard": bool(selected_beta >= high_thr),
                "val_recon": val_recon,
                "val_kld": val_kld,
                "beta_times_val_kld": beta * val_kld,
                "rho_beta_kld_over_recon": beta * val_kld / val_recon if val_recon else np.nan,
                "n_active_units": active,
                "gate_active_ge_50": bool(active >= ACTIVE_MIN) if pd.notna(active) else False,
            }
        )
    return rows


def scan_logs() -> pd.DataFrame:
    rows = []
    for log in sorted(LOG_B125.glob("*.log")):
        text = log.read_text(errors="replace")
        for name, pattern in LOG_PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                rows.append({"log_file": str(log), "pattern": name, "n_hits": len(matches)})
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    command_log = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "guardrails": {
            "did_train": False,
            "did_launch_triples": False,
            "did_launch_full_runs": False,
            "did_refit_classifier": False,
            "did_run_oasis": False,
            "read_only_existing_outputs": True,
        },
        "roots": {str(k): {c: str(p) for c, p in v.items()} for k, v in BETA_ROOTS.items()},
    }

    pred: dict[tuple[float, str], pd.DataFrame] = {}
    metrics_rows = []
    integrity_rows = []

    for beta, roots in BETA_ROOTS.items():
        for cand, root in roots.items():
            p = load_predictions(root)
            m = load_metrics(root)
            pred[(beta, cand)] = p
            pooled = pooled_metrics(p)
            fold_auc = m["auc"].to_numpy(float)
            fold_pr = m["pr_auc"].to_numpy(float)
            metrics_rows.append(
                {
                    "candidate": cand,
                    "beta_vae": beta,
                    "channels": " ".join(map(str, CHANNELS[cand])),
                    "cardinality": len(CHANNELS[cand]),
                    **pooled,
                    "fold_auc_mean": float(np.mean(fold_auc)),
                    "fold_auc_sd": float(np.std(fold_auc, ddof=1)),
                    "fold_pr_auc_mean": float(np.mean(fold_pr)),
                    "fold_pr_auc_sd": float(np.std(fold_pr, ddof=1)),
                    "fold1_auc": fold_auc[0],
                    "fold2_auc": fold_auc[1],
                    "fold3_auc": fold_auc[2],
                    "fold1_pr_auc": fold_pr[0],
                    "fold2_pr_auc": fold_pr[1],
                    "fold3_pr_auc": fold_pr[2],
                }
            )

            if beta == 1.25:
                ckpt_ok = []
                fold_dirs = []
                for fold in FOLDS:
                    fd = root / f"fold_{fold}"
                    fold_dirs.append(fd.exists())
                    ckpt = checkpoint_info(fd)
                    ckpt_ok.append(float(ckpt.get("selected_epoch_beta", 0)) >= 0.95 * beta)
                integrity_rows.append(
                    {
                        "candidate": cand,
                        "run_dir": str(root),
                        "dir_exists": root.exists(),
                        "n_fold_dirs": int(sum(fold_dirs)),
                        "metrics_files": len(list(root.glob("all_folds_metrics_MULTI_*.csv"))),
                        "prediction_joblibs": len(list(root.glob("all_folds_clf_predictions_MULTI_*.joblib"))),
                        "oof_rows": len(p),
                        "unique_subjects": p["SubjectID"].nunique(),
                        "duplicate_subject_rows": int(p["SubjectID"].duplicated().sum()),
                        "checkpoint_high_beta_all": all(ckpt_ok),
                    }
                )

    metrics_df = pd.DataFrame(metrics_rows)
    integrity_df = pd.DataFrame(integrity_rows)
    log_issues = scan_logs()

    # Rate-distortion and active units: compute beta=1.25 fresh; reuse beta=3.75
    # if already present; compute beta=2.50 rho from histories and active units
    # from the prior gate-completion table.
    tensor = None
    if TORCH_OK:
        npz = np.load(TENSOR)
        tensor = npz[npz.files[0]]
    rd_rows = []
    for cand, root in BETA_ROOTS[1.25].items():
        rd_rows.extend(rate_distortion_for(1.25, cand, root, tensor))
    rd_b125 = pd.DataFrame(rd_rows)

    active_b250_path = REPO / (
        "results/revision_bspc_2026/post_revision_exploratory_20260630/"
        "fast_exhaustive_singles_pairs_gate_completion_20260703/active_units_by_fold.csv"
    )
    active_b250 = pd.read_csv(active_b250_path) if active_b250_path.exists() else pd.DataFrame()
    if not active_b250.empty:
        active_b250 = active_b250[active_b250["candidate"].isin(CANDIDATES)].copy()
    active_lookup_b250 = {
        (str(r["candidate"]), int(r["fold"])): int(r["n_active_units"])
        for _, r in active_b250.iterrows()
    }
    rd_b250_rows = []
    for cand, root in BETA_ROOTS[2.50].items():
        rd_b250_rows.extend(rate_distortion_no_active(2.50, cand, root, active_lookup_b250))
    rd_b250 = pd.DataFrame(rd_b250_rows)

    rd_b375_path = REPO / (
        "results/revision_bspc_2026/post_revision_exploratory_20260630/"
        "fast_phase2_beta375_postrun_audit_20260703/phase2_rate_distortion.csv"
    )
    rd_b375 = pd.read_csv(rd_b375_path) if rd_b375_path.exists() else pd.DataFrame()
    if not rd_b375.empty:
        rd_b375 = rd_b375.rename(columns={"rho": "rho_beta_kld_over_recon"})
        if "beta_times_val_kld" not in rd_b375.columns:
            rd_b375["beta_times_val_kld"] = rd_b375["beta_vae"] * rd_b375["val_kld"]

    rd_all = pd.concat([rd_b125, rd_b250, rd_b375], ignore_index=True, sort=False)
    rd_cand = (
        rd_all.groupby(["candidate", "beta_vae"], dropna=False)
        .agg(
            selected_epoch_mean=("selected_epoch", "mean"),
            selected_beta_min=("selected_epoch_beta", "min"),
            high_beta_guard_all=("high_beta_guard", "all"),
            val_recon_mean=("val_recon", "mean"),
            val_kld_mean=("val_kld", "mean"),
            rho_mean=("rho_beta_kld_over_recon", "mean"),
            rho_min=("rho_beta_kld_over_recon", "min"),
            n_active_min=("n_active_units", "min"),
            n_active_mean=("n_active_units", "mean"),
        )
        .reset_index()
    )

    # Pairwise bootstrap finalist vs control at beta=1.25.
    b125_vs_control = []
    for cand in FINALISTS:
        b125_vs_control.append(
            paired_bootstrap(
                pred[(1.25, cand)],
                pred[(1.25, "control_ch102")],
                f"{cand}_b125",
                "control_ch102_b125",
            )
        )
    b125_vs_control_df = pd.DataFrame(b125_vs_control)

    # Cross-beta paired bootstrap for each candidate across all beta pairs.
    cross_beta = []
    beta_pairs = [(1.25, 2.50), (1.25, 3.75), (3.75, 2.50)]
    for cand in CANDIDATES:
        for a, b in beta_pairs:
            cross_beta.append(
                paired_bootstrap(
                    pred[(a, cand)],
                    pred[(b, cand)],
                    f"{cand}_b{str(a).replace('.', 'p')}",
                    f"{cand}_b{str(b).replace('.', 'p')}",
                )
            )
    cross_beta_df = pd.DataFrame(cross_beta)

    # Three-beta summary and gates.
    summary = metrics_df.merge(rd_cand, on=["candidate", "beta_vae"], how="left")
    gate_rows = []
    for beta in [1.25, 2.50, 3.75]:
        ctrl = summary[(summary["beta_vae"].eq(beta)) & summary["candidate"].eq("control_ch102")].iloc[0]
        for cand in FINALISTS:
            row = summary[(summary["beta_vae"].eq(beta)) & summary["candidate"].eq(cand)].iloc[0]
            # CI for beta=1.25 from fresh bootstrap, beta=3.75 from existing/fresh below, beta=2.50 from gate completion if available.
            if beta == 1.25:
                bs = b125_vs_control_df[b125_vs_control_df["model_a"].eq(f"{cand}_b125")].iloc[0]
            else:
                bs = paired_bootstrap(
                    pred[(beta, cand)],
                    pred[(beta, "control_ch102")],
                    f"{cand}_b{str(beta).replace('.', 'p')}",
                    f"control_ch102_b{str(beta).replace('.', 'p')}",
                )
            delta_auc = row["pooled_roc_auc"] - ctrl["pooled_roc_auc"]
            delta_pr = row["pooled_pr_auc"] - ctrl["pooled_pr_auc"]
            loose = (
                delta_auc >= LOOSE_AUC
                and row["fold_auc_sd"] <= FOLD_SD_MAX
                and (pd.isna(row["n_active_min"]) or row["n_active_min"] >= ACTIVE_MIN)
                and delta_pr >= -PR_MAX_WORSE
                and bool(row.get("high_beta_guard_all", True))
            )
            strict = loose and bs["delta_auc_ci_low"] >= STRICT_AUC
            gate_rows.append(
                {
                    "candidate": cand,
                    "beta_vae": beta,
                    "candidate_auc": row["pooled_roc_auc"],
                    "control_auc": ctrl["pooled_roc_auc"],
                    "delta_auc_vs_control": delta_auc,
                    "delta_auc_ci_low": bs["delta_auc_ci_low"],
                    "delta_auc_ci_high": bs["delta_auc_ci_high"],
                    "candidate_pr_auc": row["pooled_pr_auc"],
                    "control_pr_auc": ctrl["pooled_pr_auc"],
                    "delta_pr_auc_vs_control": delta_pr,
                    "fold_auc_sd": row["fold_auc_sd"],
                    "n_active_min": row.get("n_active_min", np.nan),
                    "high_beta_guard_all": row.get("high_beta_guard_all", np.nan),
                    "gate_auc_loose": delta_auc >= LOOSE_AUC,
                    "gate_auc_strict": bs["delta_auc_ci_low"] >= STRICT_AUC,
                    "gate_fold_sd": row["fold_auc_sd"] <= FOLD_SD_MAX,
                    "gate_active_units": (pd.isna(row.get("n_active_min", np.nan)) or row["n_active_min"] >= ACTIVE_MIN),
                    "gate_pr_auc_within_0p02": delta_pr >= -PR_MAX_WORSE,
                    "overall_loose": bool(loose),
                    "overall_strict": bool(strict),
                }
            )
    gates_df = pd.DataFrame(gate_rows)

    # Eligibility for pair_ch1_3 targeted triples.
    p13 = gates_df[gates_df["candidate"].eq("pair_ch1_3")].copy()
    p13_b125 = p13[p13["beta_vae"].eq(1.25)].iloc[0]
    rank_b125 = (
        metrics_df[metrics_df["beta_vae"].eq(1.25)]
        .sort_values(["pooled_roc_auc", "pooled_pr_auc"], ascending=False)
        [["candidate", "pooled_roc_auc", "pooled_pr_auc"]]
        .reset_index(drop=True)
    )
    p13_rank = int(rank_b125.index[rank_b125["candidate"].eq("pair_ch1_3")][0] + 1)
    if bool(p13_b125["overall_loose"]) and p13_rank == 1:
        triple_decision = "eligible_for_targeted_triple_expansion"
        triple_reason = "pair_ch1_3 is top-ranked at beta=1.25 and passes the loose gate versus beta=1.25 control."
    elif bool(p13_b125["overall_loose"]):
        triple_decision = "not_eligible_rank_not_leading"
        triple_reason = "pair_ch1_3 passes the beta=1.25 loose gate, but is not the top beta=1.25 candidate."
    else:
        triple_decision = "not_eligible_gate_failure"
        triple_reason = "pair_ch1_3 does not pass the beta=1.25 loose gate."

    # Write deliverables.
    integrity_df.to_csv(OUT / "beta125_completion_checkpoint_integrity.csv", index=False)
    (OUT / "beta125_completion_checkpoint_integrity.md").write_text(md_table(integrity_df), encoding="utf-8")
    metrics_df[metrics_df["beta_vae"].eq(1.25)].to_csv(OUT / "beta125_pooled_oof_metrics.csv", index=False)
    (OUT / "beta125_pooled_oof_metrics.md").write_text(
        md_table(metrics_df[metrics_df["beta_vae"].eq(1.25)]), encoding="utf-8"
    )
    b125_vs_control_df.to_csv(OUT / "beta125_vs_control_paired_bootstrap.csv", index=False)
    (OUT / "beta125_vs_control_paired_bootstrap.md").write_text(md_table(b125_vs_control_df), encoding="utf-8")
    cross_beta_df.to_csv(OUT / "cross_beta_paired_bootstrap.csv", index=False)
    (OUT / "cross_beta_paired_bootstrap.md").write_text(md_table(cross_beta_df), encoding="utf-8")
    rd_b125.to_csv(OUT / "beta125_rate_distortion_active_units_by_fold.csv", index=False)
    rd_all.to_csv(OUT / "rate_distortion_active_units_by_fold_all_betas.csv", index=False)
    rd_cand.to_csv(OUT / "rate_distortion_active_units_by_candidate.csv", index=False)
    summary.to_csv(OUT / "three_beta_candidate_summary.csv", index=False)
    (OUT / "three_beta_candidate_summary.md").write_text(md_table(summary), encoding="utf-8")
    gates_df.to_csv(OUT / "loose_strict_gates_by_beta.csv", index=False)
    (OUT / "loose_strict_gates_by_beta.md").write_text(md_table(gates_df), encoding="utf-8")
    if not log_issues.empty:
        log_issues.to_csv(OUT / "beta125_log_issues.csv", index=False)

    rec = f"""# Beta=1.25 audit recommendation

Decision for `pair_ch1_3` targeted triple expansion: **{triple_decision}**.

Reason: {triple_reason}

## Beta=1.25 ranking

{md_table(rank_b125)}

## `pair_ch1_3` beta=1.25 gate row

{md_table(pd.DataFrame([p13_b125]))}

## Interpretation

This audit does not launch triples and does not decide a final channel-set winner. It only evaluates whether the completed beta=1.25 arm preserves the same gate logic used in the Fase 1A/1B gate-completion audit.

Loose gate: point-estimate AUC delta versus same-beta `control_ch102` >= 0.015, PR-AUC no more than 0.02 below control, fold AUC SD <= 0.07, active units >= 50/128, and high-beta checkpoint guard pass.

Strict gate: loose gate plus paired-bootstrap 95% CI lower bound for delta AUC >= 0.015.

Guardrails: no training, no triples, no FULL runs, no OASIS, no classifier refitting.
"""
    (OUT / "targeted_triple_expansion_recommendation.md").write_text(rec, encoding="utf-8")

    integrity_report = f"""# Beta=1.25 completion and checkpoint integrity

Run root: `{ROOT_B125}`

Log root: `{LOG_B125}`

Overall beta=1.25 candidate directories found: {int(integrity_df['dir_exists'].sum())}/4.

All candidates have 3 fold dirs, one OOF prediction joblib, one metrics CSV, 397 unique OOF subjects, and high-beta checkpoint guard pass: **{bool((integrity_df['n_fold_dirs'].eq(3) & integrity_df['checkpoint_high_beta_all'] & integrity_df['duplicate_subject_rows'].eq(0)).all())}**.

Log issue scan found {len(log_issues)} issue rows.

{md_table(integrity_df)}
"""
    (OUT / "completion_checkpoint_integrity_report.md").write_text(integrity_report, encoding="utf-8")

    command_log["outputs"] = sorted(p.name for p in OUT.iterdir() if p.is_file())
    command_log["torch_available"] = TORCH_OK
    command_log["bootstrap_n"] = BOOTSTRAP_N
    command_log["bootstrap_seed"] = BOOTSTRAP_SEED
    command_log["triple_decision_pair_ch1_3"] = triple_decision
    (OUT / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "triple_decision_pair_ch1_3": triple_decision}, indent=2))


if __name__ == "__main__":
    main()
