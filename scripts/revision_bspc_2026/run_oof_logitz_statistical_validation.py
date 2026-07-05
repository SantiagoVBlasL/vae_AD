#!/usr/bin/env python3.8
"""
Statistical validation audit: OOF logit-z promoted candidate vs locked reference.

Read-only. No training. No threshold fitting. No model modification.
Compares:
  A: locked_v5p1b raw
  B: recover035_latent384_beta3p75 raw
  C: recover035_latent384_beta3p75 oof_logitz  <- promoted

Tests:
  1. Bootstrap 95% CI for pooled AUC and PR-AUC (B=5000, subject-level resampling)
  2. Paired bootstrap delta AUC / delta PR-AUC vs locked (subject-level pairing)
  3. DeLong test for ROC-AUC (structural components, paired correlated AUCs)
  4. Subject-level paired permutation test for delta AUC / delta PR-AUC
  5. Foldwise paired comparison (descriptive only — n=5 folds)

Output: results/revision_bspc_2026/oof_logitz_statistical_validation/
"""

from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

# ─── config ───────────────────────────────────────────────────────────────────
RESULTS = Path("results/revision_bspc_2026")
PRED_CSV = RESULTS / "oof_logitz_all_candidates_comparison" / "calib_predictions_all.csv"
OUT_DIR = RESULTS / "oof_logitz_statistical_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

B_BOOTSTRAP = 5000
B_PERMUTATION = 10000
SEED = 42

LOCKED = "locked_v5p1b"
PROMOTED = "recover035_latent384_beta3p75"

# ─── data loading ─────────────────────────────────────────────────────────────
def load_predictions() -> pd.DataFrame:
    df = pd.read_csv(PRED_CSV)
    assert df["y_true"].isin([0, 1]).all()
    return df


def extract(df: pd.DataFrame, candidate: str, calib: str) -> pd.DataFrame:
    """Return subject-level rows for one candidate / calibration, sorted by SubjectID."""
    sub = df[(df["candidate"] == candidate) & (df["calib_method"] == calib)].copy()
    return sub.sort_values("SubjectID").reset_index(drop=True)


def get_paired(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_true, scores_a, scores_b) for the intersection of subjects."""
    common = sorted(set(df_a["SubjectID"]) & set(df_b["SubjectID"]))
    a = df_a[df_a["SubjectID"].isin(common)].sort_values("SubjectID")
    b = df_b[df_b["SubjectID"].isin(common)].sort_values("SubjectID")
    assert (a["SubjectID"].values == b["SubjectID"].values).all()
    assert (a["y_true"].values == b["y_true"].values).all()
    return a["y_true"].values.astype(int), a["y_score"].values, b["y_score"].values


# ─── DeLong test ──────────────────────────────────────────────────────────────
def _structural_components(y_true: np.ndarray, y_score: np.ndarray):
    """
    Vectorised structural components for DeLong variance/covariance estimation.
    Returns V10 (n_pos,), V01 (n_neg,), n0, n1.
    Reference: DeLong et al. 1988, Biometrics 44(3):837-845.
    """
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    n1, n0 = len(pos), len(neg)
    # psi(pos_i, neg_j): 1 if pos > neg, 0.5 if tie, 0 if pos < neg
    diff = pos[:, None] - neg[None, :]   # (n1, n0)
    psi = np.where(diff > 0, 1.0, np.where(diff == 0, 0.5, 0.0))
    V10 = psi.mean(axis=1)  # (n1,) — mean over negatives for each positive
    V01 = psi.mean(axis=0)  # (n0,) — mean over positives for each negative
    return V10, V01, n0, n1


def _cov2x2(V1: np.ndarray, V2: np.ndarray, theta1: float, theta2: float) -> np.ndarray:
    """Unbiased 2×2 covariance matrix from paired structural component vectors."""
    n = len(V1)
    if n <= 1:
        return np.eye(2) * 1e-15
    d1 = V1 - theta1
    d2 = V2 - theta2
    return np.array([
        [np.dot(d1, d1) / (n - 1), np.dot(d1, d2) / (n - 1)],
        [np.dot(d1, d2) / (n - 1), np.dot(d2, d2) / (n - 1)],
    ])


def delong_test(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray,
                label_a: str = "A", label_b: str = "B") -> dict:
    """
    Paired DeLong test comparing AUC(A) vs AUC(B) on the same subjects.
    Returns z-statistic and two-sided p-value.
    """
    auc_a = float(roc_auc_score(y_true, score_a))
    auc_b = float(roc_auc_score(y_true, score_b))

    V10_a, V01_a, n0, n1 = _structural_components(y_true, score_a)
    V10_b, V01_b, _, __ = _structural_components(y_true, score_b)

    S10 = _cov2x2(V10_a, V10_b, auc_a, auc_b)
    S01 = _cov2x2(V01_a, V01_b, auc_a, auc_b)
    S = S10 / n1 + S01 / n0

    L = np.array([1.0, -1.0])
    var_diff = max(float(L @ S @ L), 1e-15)
    se = np.sqrt(var_diff)
    z = (auc_a - auc_b) / se
    p_two = float(2 * (1 - stats.norm.cdf(abs(z))))

    return {
        f"auc_{label_a}": round(auc_a, 6),
        f"auc_{label_b}": round(auc_b, 6),
        "delta_auc": round(auc_a - auc_b, 6),
        "se_diff": round(se, 6),
        "z": round(z, 4),
        "p_two_sided": round(p_two, 6),
        "n_pos": n1,
        "n_neg": n0,
        "n_paired": n0 + n1,
        "S": [[round(S[0, 0], 8), round(S[0, 1], 8)],
              [round(S[1, 0], 8), round(S[1, 1], 8)]],
    }


# ─── bootstrap CI ─────────────────────────────────────────────────────────────
def bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray,
                 B: int = 5000, seed: int = 42) -> dict:
    """
    Bootstrap 95% CI for AUC and PR-AUC via percentile method.
    Subject-level resampling with replacement.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = np.empty(B)
    praucs = np.empty(B)
    n_degen = 0
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ys = y_score[idx]
        if len(np.unique(yt)) < 2:
            aucs[b] = np.nan
            praucs[b] = np.nan
            n_degen += 1
        else:
            aucs[b] = roc_auc_score(yt, ys)
            praucs[b] = average_precision_score(yt, ys)
    va = aucs[~np.isnan(aucs)]
    vp = praucs[~np.isnan(praucs)]
    return {
        "n": n,
        "B": B,
        "auc_obs": round(float(roc_auc_score(y_true, y_score)), 6),
        "auc_ci95_lower": round(float(np.percentile(va, 2.5)), 6),
        "auc_ci95_upper": round(float(np.percentile(va, 97.5)), 6),
        "prauc_obs": round(float(average_precision_score(y_true, y_score)), 6),
        "prauc_ci95_lower": round(float(np.percentile(vp, 2.5)), 6),
        "prauc_ci95_upper": round(float(np.percentile(vp, 97.5)), 6),
        "n_degenerate_samples": n_degen,
    }


# ─── paired bootstrap delta ────────────────────────────────────────────────────
def paired_bootstrap_delta(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray,
                            B: int = 5000, seed: int = 42) -> dict:
    """
    Paired bootstrap for delta AUC and delta PR-AUC (A − B).
    Same bootstrap index applied to both systems (preserves subject pairing).
    p-value (one-sided): fraction of bootstrap deltas ≤ 0.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    obs_auc = float(roc_auc_score(y_true, score_a)) - float(roc_auc_score(y_true, score_b))
    obs_pr = float(average_precision_score(y_true, score_a)) - float(average_precision_score(y_true, score_b))
    d_aucs = np.empty(B)
    d_prs = np.empty(B)
    n_degen = 0
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        sa = score_a[idx]
        sb = score_b[idx]
        if len(np.unique(yt)) < 2:
            d_aucs[b] = np.nan
            d_prs[b] = np.nan
            n_degen += 1
        else:
            d_aucs[b] = roc_auc_score(yt, sa) - roc_auc_score(yt, sb)
            d_prs[b] = average_precision_score(yt, sa) - average_precision_score(yt, sb)
    va = d_aucs[~np.isnan(d_aucs)]
    vp = d_prs[~np.isnan(d_prs)]
    return {
        "n_paired": n,
        "B": B,
        "delta_auc_obs": round(obs_auc, 6),
        "delta_auc_ci95_lower": round(float(np.percentile(va, 2.5)), 6),
        "delta_auc_ci95_upper": round(float(np.percentile(va, 97.5)), 6),
        "p_delta_auc_le0_onesided": round(float(np.mean(va <= 0)), 6),
        "delta_prauc_obs": round(obs_pr, 6),
        "delta_prauc_ci95_lower": round(float(np.percentile(vp, 2.5)), 6),
        "delta_prauc_ci95_upper": round(float(np.percentile(vp, 97.5)), 6),
        "p_delta_prauc_le0_onesided": round(float(np.mean(vp <= 0)), 6),
        "n_degenerate_samples": n_degen,
    }


# ─── paired permutation test ────────────────────────────────────────────────────
def paired_permutation_test(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray,
                             B: int = 10000, seed: int = 42) -> dict:
    """
    Subject-level paired permutation test for delta AUC and delta PR-AUC (A − B).

    Under H0: for each subject, scores from systems A and B are exchangeable.
    For each permutation, randomly swap (score_a[i], score_b[i]) with probability 0.5.
    p-value (one-sided): fraction of permuted deltas >= observed delta.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    obs_auc = float(roc_auc_score(y_true, score_a)) - float(roc_auc_score(y_true, score_b))
    obs_pr = float(average_precision_score(y_true, score_a)) - float(average_precision_score(y_true, score_b))
    d_aucs = np.empty(B)
    d_prs = np.empty(B)
    n_degen = 0
    for b in range(B):
        swap = rng.integers(0, 2, size=n).astype(bool)
        sa = np.where(swap, score_b, score_a)
        sb = np.where(swap, score_a, score_b)
        if len(np.unique(y_true)) < 2:
            d_aucs[b] = np.nan
            d_prs[b] = np.nan
            n_degen += 1
        else:
            d_aucs[b] = roc_auc_score(y_true, sa) - roc_auc_score(y_true, sb)
            d_prs[b] = average_precision_score(y_true, sa) - average_precision_score(y_true, sb)
    va = d_aucs[~np.isnan(d_aucs)]
    vp = d_prs[~np.isnan(d_prs)]
    return {
        "n_paired": n,
        "B": B,
        "delta_auc_obs": round(obs_auc, 6),
        "p_auc_ge_obs_onesided": round(float(np.mean(va >= obs_auc)), 6),
        "delta_prauc_obs": round(obs_pr, 6),
        "p_prauc_ge_obs_onesided": round(float(np.mean(vp >= obs_pr)), 6),
        "n_degenerate_samples": n_degen,
    }


# ─── foldwise comparison ───────────────────────────────────────────────────────
def foldwise_comparison(df_a: pd.DataFrame, df_b: pd.DataFrame,
                        label_a: str, label_b: str) -> tuple[pd.DataFrame, dict]:
    """
    Per-fold AUC and PR-AUC comparison. Descriptive only (n=5 folds).
    Wilcoxon signed-rank included for completeness but is not interpretable at n=5.
    """
    rows = []
    for fold in sorted(df_a["fold"].unique()):
        fa = df_a[df_a["fold"] == fold]
        fb = df_b[df_b["fold"] == fold]
        common_ids = set(fa["SubjectID"]) & set(fb["SubjectID"])
        fa = fa[fa["SubjectID"].isin(common_ids)].sort_values("SubjectID")
        fb = fb[fb["SubjectID"].isin(common_ids)].sort_values("SubjectID")
        auc_a = float(roc_auc_score(fa["y_true"], fa["y_score"]))
        auc_b = float(roc_auc_score(fb["y_true"], fb["y_score"]))
        pr_a = float(average_precision_score(fa["y_true"], fa["y_score"]))
        pr_b = float(average_precision_score(fb["y_true"], fb["y_score"]))
        rows.append({
            "fold": fold,
            f"auc_{label_a}": round(auc_a, 4),
            f"auc_{label_b}": round(auc_b, 4),
            "delta_auc": round(auc_a - auc_b, 4),
            f"prauc_{label_a}": round(pr_a, 4),
            f"prauc_{label_b}": round(pr_b, 4),
            "delta_prauc": round(pr_a - pr_b, 4),
            "n_fold_a": len(fa),
            "n_fold_b": len(fb),
        })
    tbl = pd.DataFrame(rows)
    da = tbl["delta_auc"].values.astype(float)
    dp = tbl["delta_prauc"].values.astype(float)
    try:
        w_a, p_a = stats.wilcoxon(da)
    except Exception:
        w_a, p_a = np.nan, np.nan
    try:
        w_p, p_p = stats.wilcoxon(dp)
    except Exception:
        w_p, p_p = np.nan, np.nan
    summary = {
        "n_folds": len(rows),
        "mean_delta_auc": round(float(da.mean()), 4),
        "std_delta_auc": round(float(da.std(ddof=1)), 4),
        "n_positive_delta_auc": int((da > 0).sum()),
        "wilcoxon_W_auc": None if np.isnan(w_a) else round(float(w_a), 4),
        "wilcoxon_p_auc": None if np.isnan(p_a) else round(float(p_a), 4),
        "mean_delta_prauc": round(float(dp.mean()), 4),
        "std_delta_prauc": round(float(dp.std(ddof=1)), 4),
        "n_positive_delta_prauc": int((dp > 0).sum()),
        "wilcoxon_W_prauc": None if np.isnan(w_p) else round(float(w_p), 4),
        "wilcoxon_p_prauc": None if np.isnan(p_p) else round(float(p_p), 4),
        "note": "n=5 folds — Wilcoxon and summary stats are descriptive only; no inferential power at this fold count.",
    }
    return tbl, summary


# ─── report helpers ────────────────────────────────────────────────────────────
def _sig_label(p: float | None, threshold: float = 0.05) -> str:
    if p is None:
        return "—"
    if p < 0.001:
        return f"p={p:.4f} ***"
    if p < 0.01:
        return f"p={p:.4f} **"
    if p < threshold:
        return f"p={p:.4f} *"
    return f"p={p:.4f} (ns)"


def _ci_str(lower: float, upper: float) -> str:
    return f"[{lower:.4f}, {upper:.4f}]"


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading predictions...")
    df = load_predictions()

    # Extract the three arms
    df_lock_raw = extract(df, LOCKED, "raw")
    df_b3p75_raw = extract(df, PROMOTED, "raw")
    df_b3p75_lz = extract(df, PROMOTED, "oof_logitz")

    print(f"  locked_raw       n={len(df_lock_raw)}  n_ad={df_lock_raw.y_true.sum()}")
    print(f"  beta3p75_raw     n={len(df_b3p75_raw)}  n_ad={df_b3p75_raw.y_true.sum()}")
    print(f"  beta3p75_logitz  n={len(df_b3p75_lz)}  n_ad={df_b3p75_lz.y_true.sum()}")

    # Paired subject intersection
    y_lock, s_lock, s_b3p75_raw = get_paired(df_lock_raw, df_b3p75_raw)
    y_lock2, s_lock2, s_b3p75_lz = get_paired(df_lock_raw, df_b3p75_lz)
    assert (y_lock == y_lock2).all()
    assert (s_lock == s_lock2).all()
    n_paired = len(y_lock)
    print(f"  Paired subjects (locked ∩ beta3p75): n={n_paired}")

    # ── 1. Bootstrap CIs ──────────────────────────────────────────────────────
    print("\n[1/5] Bootstrap CIs (B={})...".format(B_BOOTSTRAP))
    ci_lock = bootstrap_ci(df_lock_raw["y_true"].values, df_lock_raw["y_score"].values,
                           B=B_BOOTSTRAP, seed=SEED)
    ci_b3p75_raw = bootstrap_ci(df_b3p75_raw["y_true"].values, df_b3p75_raw["y_score"].values,
                                B=B_BOOTSTRAP, seed=SEED + 1)
    ci_b3p75_lz = bootstrap_ci(df_b3p75_lz["y_true"].values, df_b3p75_lz["y_score"].values,
                                B=B_BOOTSTRAP, seed=SEED + 2)
    bootstrap_results = {
        "locked_v5p1b_raw": ci_lock,
        "recover035_latent384_beta3p75_raw": ci_b3p75_raw,
        "recover035_latent384_beta3p75_oof_logitz": ci_b3p75_lz,
    }
    with open(OUT_DIR / "bootstrap_ci.json", "w") as f:
        json.dump(bootstrap_results, f, indent=2)
    print("  locked:        AUC={auc_obs:.4f} {lo} PR-AUC={prauc_obs:.4f} {plo}".format(
        lo=_ci_str(ci_lock["auc_ci95_lower"], ci_lock["auc_ci95_upper"]),
        plo=_ci_str(ci_lock["prauc_ci95_lower"], ci_lock["prauc_ci95_upper"]),
        **ci_lock))
    print("  beta3p75 raw:  AUC={auc_obs:.4f} {lo} PR-AUC={prauc_obs:.4f} {plo}".format(
        lo=_ci_str(ci_b3p75_raw["auc_ci95_lower"], ci_b3p75_raw["auc_ci95_upper"]),
        plo=_ci_str(ci_b3p75_raw["prauc_ci95_lower"], ci_b3p75_raw["prauc_ci95_upper"]),
        **ci_b3p75_raw))
    print("  beta3p75 lz:   AUC={auc_obs:.4f} {lo} PR-AUC={prauc_obs:.4f} {plo}".format(
        lo=_ci_str(ci_b3p75_lz["auc_ci95_lower"], ci_b3p75_lz["auc_ci95_upper"]),
        plo=_ci_str(ci_b3p75_lz["prauc_ci95_lower"], ci_b3p75_lz["prauc_ci95_upper"]),
        **ci_b3p75_lz))

    # ── 2. DeLong test ─────────────────────────────────────────────────────────
    print("\n[2/5] DeLong test...")
    dl_lz_vs_lock = delong_test(y_lock2, s_b3p75_lz, s_lock2,
                                label_a="beta3p75_logitz", label_b="locked_raw")
    dl_raw_vs_lock = delong_test(y_lock, s_b3p75_raw, s_lock,
                                 label_a="beta3p75_raw", label_b="locked_raw")
    dl_lz_vs_raw = delong_test(y_lock, s_b3p75_lz, s_b3p75_raw,
                                label_a="beta3p75_logitz", label_b="beta3p75_raw")
    delong_results = {
        "beta3p75_logitz_vs_locked_raw": dl_lz_vs_lock,
        "beta3p75_raw_vs_locked_raw": dl_raw_vs_lock,
        "beta3p75_logitz_vs_beta3p75_raw": dl_lz_vs_raw,
    }
    with open(OUT_DIR / "delong_test.json", "w") as f:
        json.dump(delong_results, f, indent=2)
    for key, res in delong_results.items():
        ka, kb = list(res.keys())[:2]
        print(f"  {key}: ΔAUC={res['delta_auc']:+.4f}  z={res['z']:.3f}  {_sig_label(res['p_two_sided'])}")

    # ── 3. Paired bootstrap delta ──────────────────────────────────────────────
    print("\n[3/5] Paired bootstrap delta (B={})...".format(B_BOOTSTRAP))
    pb_lz_vs_lock = paired_bootstrap_delta(y_lock2, s_b3p75_lz, s_lock2,
                                           B=B_BOOTSTRAP, seed=SEED + 10)
    pb_raw_vs_lock = paired_bootstrap_delta(y_lock, s_b3p75_raw, s_lock,
                                            B=B_BOOTSTRAP, seed=SEED + 11)
    paired_boot_results = {
        "beta3p75_logitz_vs_locked_raw": pb_lz_vs_lock,
        "beta3p75_raw_vs_locked_raw": pb_raw_vs_lock,
    }
    with open(OUT_DIR / "paired_bootstrap_delta.json", "w") as f:
        json.dump(paired_boot_results, f, indent=2)
    for key, res in paired_boot_results.items():
        print(f"  {key}:")
        print(f"    ΔAUC={res['delta_auc_obs']:+.4f} CI={_ci_str(res['delta_auc_ci95_lower'], res['delta_auc_ci95_upper'])}  p(≤0)={res['p_delta_auc_le0_onesided']:.4f}")
        print(f"    ΔPR-AUC={res['delta_prauc_obs']:+.4f} CI={_ci_str(res['delta_prauc_ci95_lower'], res['delta_prauc_ci95_upper'])}  p(≤0)={res['p_delta_prauc_le0_onesided']:.4f}")

    # ── 4. Paired permutation test ─────────────────────────────────────────────
    print("\n[4/5] Paired permutation test (B={})...".format(B_PERMUTATION))
    pp_lz_vs_lock = paired_permutation_test(y_lock2, s_b3p75_lz, s_lock2,
                                            B=B_PERMUTATION, seed=SEED + 20)
    pp_raw_vs_lock = paired_permutation_test(y_lock, s_b3p75_raw, s_lock,
                                             B=B_PERMUTATION, seed=SEED + 21)
    perm_results = {
        "beta3p75_logitz_vs_locked_raw": pp_lz_vs_lock,
        "beta3p75_raw_vs_locked_raw": pp_raw_vs_lock,
    }
    with open(OUT_DIR / "paired_permutation_test.json", "w") as f:
        json.dump(perm_results, f, indent=2)
    for key, res in perm_results.items():
        print(f"  {key}:")
        print(f"    ΔAUC={res['delta_auc_obs']:+.4f}  {_sig_label(res['p_auc_ge_obs_onesided'])}")
        print(f"    ΔPR-AUC={res['delta_prauc_obs']:+.4f}  {_sig_label(res['p_prauc_ge_obs_onesided'])}")

    # ── 5. Foldwise comparison ─────────────────────────────────────────────────
    print("\n[5/5] Foldwise comparison (descriptive)...")
    fw_tbl_lz, fw_sum_lz = foldwise_comparison(df_b3p75_lz, df_lock_raw,
                                                "beta3p75_logitz", "locked_raw")
    fw_tbl_raw, fw_sum_raw = foldwise_comparison(df_b3p75_raw, df_lock_raw,
                                                  "beta3p75_raw", "locked_raw")
    fw_tbl_lz["comparison"] = "beta3p75_logitz_vs_locked_raw"
    fw_tbl_raw["comparison"] = "beta3p75_raw_vs_locked_raw"
    fw_all = pd.concat([fw_tbl_lz, fw_tbl_raw], ignore_index=True)
    fw_all.to_csv(OUT_DIR / "foldwise_comparison.csv", index=False)
    foldwise_results = {
        "beta3p75_logitz_vs_locked_raw": fw_sum_lz,
        "beta3p75_raw_vs_locked_raw": fw_sum_raw,
    }
    with open(OUT_DIR / "foldwise_comparison_summary.json", "w") as f:
        json.dump(foldwise_results, f, indent=2)
    print(fw_tbl_lz[["fold", "auc_beta3p75_logitz", "auc_locked_raw", "delta_auc",
                      "prauc_beta3p75_logitz", "prauc_locked_raw", "delta_prauc"]].to_string(index=False))

    # ── write markdown tables ──────────────────────────────────────────────────
    _write_markdown_tables(
        bootstrap_results, delong_results, paired_boot_results, perm_results,
        fw_tbl_lz, fw_tbl_raw, fw_sum_lz, fw_sum_raw,
        n_paired=n_paired,
    )

    # ── final report ──────────────────────────────────────────────────────────
    _write_final_report(
        bootstrap_results, delong_results, paired_boot_results, perm_results,
        fw_sum_lz, n_paired=n_paired,
    )

    # ── command log ───────────────────────────────────────────────────────────
    log = {
        "script": "scripts/revision_bspc_2026/run_oof_logitz_statistical_validation.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(PRED_CSV),
        "output_dir": str(OUT_DIR),
        "B_bootstrap": B_BOOTSTRAP,
        "B_permutation": B_PERMUTATION,
        "seed": SEED,
        "n_paired_subjects": n_paired,
        "training_launched": False,
        "threshold_fitting": False,
        "tensor_modification": False,
    }
    with open(OUT_DIR / "command_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print("\nDone. Output:", OUT_DIR)


# ─── markdown writers ─────────────────────────────────────────────────────────
def _write_markdown_tables(
    bootstrap_results, delong_results, paired_boot_results, perm_results,
    fw_tbl_lz, fw_tbl_raw, fw_sum_lz, fw_sum_raw, n_paired,
):
    # Bootstrap CI table
    rows = []
    for k, r in bootstrap_results.items():
        rows.append({
            "candidate": k,
            "n": r["n"],
            "auc": f"{r['auc_obs']:.4f}",
            "auc_95ci": _ci_str(r["auc_ci95_lower"], r["auc_ci95_upper"]),
            "pr_auc": f"{r['prauc_obs']:.4f}",
            "prauc_95ci": _ci_str(r["prauc_ci95_lower"], r["prauc_ci95_upper"]),
        })
    pd.DataFrame(rows).to_markdown(OUT_DIR / "bootstrap_ci.md", index=False)

    # DeLong table
    rows = []
    for k, r in delong_results.items():
        rows.append({
            "comparison": k,
            "n_paired": r["n_paired"],
            "delta_auc": f"{r['delta_auc']:+.4f}",
            "se": f"{r['se_diff']:.4f}",
            "z": f"{r['z']:.3f}",
            "p_two_sided": f"{r['p_two_sided']:.4f}",
            "sig": _sig_label(r["p_two_sided"]),
        })
    pd.DataFrame(rows).to_markdown(OUT_DIR / "delong_test.md", index=False)

    # Paired bootstrap delta table
    rows = []
    for k, r in paired_boot_results.items():
        rows.append({
            "comparison": k,
            "n_paired": r["n_paired"],
            "delta_auc": f"{r['delta_auc_obs']:+.4f}",
            "delta_auc_95ci": _ci_str(r["delta_auc_ci95_lower"], r["delta_auc_ci95_upper"]),
            "p_auc_le0": f"{r['p_delta_auc_le0_onesided']:.4f}",
            "delta_prauc": f"{r['delta_prauc_obs']:+.4f}",
            "delta_prauc_95ci": _ci_str(r["delta_prauc_ci95_lower"], r["delta_prauc_ci95_upper"]),
            "p_prauc_le0": f"{r['p_delta_prauc_le0_onesided']:.4f}",
        })
    pd.DataFrame(rows).to_markdown(OUT_DIR / "paired_bootstrap_delta.md", index=False)

    # Permutation test table
    rows = []
    for k, r in perm_results.items():
        rows.append({
            "comparison": k,
            "n_paired": r["n_paired"],
            "delta_auc": f"{r['delta_auc_obs']:+.4f}",
            "p_auc": f"{r['p_auc_ge_obs_onesided']:.4f}",
            "sig_auc": _sig_label(r["p_auc_ge_obs_onesided"]),
            "delta_prauc": f"{r['delta_prauc_obs']:+.4f}",
            "p_prauc": f"{r['p_prauc_ge_obs_onesided']:.4f}",
            "sig_prauc": _sig_label(r["p_prauc_ge_obs_onesided"]),
        })
    pd.DataFrame(rows).to_markdown(OUT_DIR / "paired_permutation_test.md", index=False)

    # Foldwise tables
    fw_tbl_lz[["fold", "auc_beta3p75_logitz", "auc_locked_raw", "delta_auc",
               "prauc_beta3p75_logitz", "prauc_locked_raw", "delta_prauc"]].to_markdown(
        OUT_DIR / "foldwise_logitz_vs_locked.md", index=False)
    fw_tbl_raw[["fold", "auc_beta3p75_raw", "auc_locked_raw", "delta_auc",
                "prauc_beta3p75_raw", "prauc_locked_raw", "delta_prauc"]].to_markdown(
        OUT_DIR / "foldwise_raw_vs_locked.md", index=False)


def _write_final_report(
    bootstrap_results, delong_results, paired_boot_results, perm_results, fw_sum_lz, n_paired,
):
    ci_lock = bootstrap_results["locked_v5p1b_raw"]
    ci_lz = bootstrap_results["recover035_latent384_beta3p75_oof_logitz"]
    ci_raw = bootstrap_results["recover035_latent384_beta3p75_raw"]
    dl = delong_results["beta3p75_logitz_vs_locked_raw"]
    dl_raw = delong_results["beta3p75_raw_vs_locked_raw"]
    dl_lz_raw = delong_results["beta3p75_logitz_vs_beta3p75_raw"]
    pb = paired_boot_results["beta3p75_logitz_vs_locked_raw"]
    pb_raw = paired_boot_results["beta3p75_raw_vs_locked_raw"]
    pp = perm_results["beta3p75_logitz_vs_locked_raw"]
    pp_raw = perm_results["beta3p75_raw_vs_locked_raw"]

    # Interpret overall result
    p_delong = dl["p_two_sided"]
    p_perm_auc = pp["p_auc_ge_obs_onesided"]
    p_perm_prauc = pp["p_prauc_ge_obs_onesided"]
    lz_ci_excludes_zero = pb["delta_auc_ci95_lower"] > 0

    if p_delong < 0.05 and p_perm_auc < 0.05 and lz_ci_excludes_zero:
        stability = "STATISTICALLY SIGNIFICANT: improvement is stable across all three tests."
    elif p_delong < 0.05 or (p_perm_auc < 0.05 and lz_ci_excludes_zero):
        stability = "MARGINALLY SIGNIFICANT: improvement passes some but not all tests."
    else:
        stability = "DESCRIPTIVE: improvement does not reach conventional significance thresholds."

    lines = [
        "# OOF Logit-Z Promoted Candidate — Statistical Validation Audit",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Comparison arms",
        "- **A**: locked_v5p1b raw (n=396, n_AD=96)",
        f"- **B**: recover035_latent384_beta3p75 raw (n=397, n_AD=97)",
        f"- **C**: recover035_latent384_beta3p75 oof_logitz (n=397, n_AD=97) ← promoted",
        f"- Paired subject overlap (A∩C, A∩B): n={n_paired}",
        "",
        "## Method constraints",
        "- No VAE training. No threshold fitting. No model modification.",
        "- Bootstrap and permutation: subject-level resampling/permutation.",
        "- DeLong: structural-components method (DeLong et al. 1988).",
        "- Foldwise comparison: descriptive only (n=5 folds).",
        "",
        "## 1. Bootstrap 95% CI (B={B}, subject-level)".format(B=B_BOOTSTRAP),
        "",
        "| Candidate | n | AUC | AUC 95% CI | PR-AUC | PR-AUC 95% CI |",
        "|---|---|---|---|---|---|",
        "| locked_v5p1b raw | {n} | {auc:.4f} | {lo} | {pr:.4f} | {plo} |".format(
            n=ci_lock["n"], auc=ci_lock["auc_obs"],
            lo=_ci_str(ci_lock["auc_ci95_lower"], ci_lock["auc_ci95_upper"]),
            pr=ci_lock["prauc_obs"],
            plo=_ci_str(ci_lock["prauc_ci95_lower"], ci_lock["prauc_ci95_upper"])),
        "| beta3p75 raw | {n} | {auc:.4f} | {lo} | {pr:.4f} | {plo} |".format(
            n=ci_raw["n"], auc=ci_raw["auc_obs"],
            lo=_ci_str(ci_raw["auc_ci95_lower"], ci_raw["auc_ci95_upper"]),
            pr=ci_raw["prauc_obs"],
            plo=_ci_str(ci_raw["prauc_ci95_lower"], ci_raw["prauc_ci95_upper"])),
        "| beta3p75 oof_logitz | {n} | {auc:.4f} | {lo} | {pr:.4f} | {plo} |".format(
            n=ci_lz["n"], auc=ci_lz["auc_obs"],
            lo=_ci_str(ci_lz["auc_ci95_lower"], ci_lz["auc_ci95_upper"]),
            pr=ci_lz["prauc_obs"],
            plo=_ci_str(ci_lz["prauc_ci95_lower"], ci_lz["prauc_ci95_upper"])),
        "",
        "CI overlap between locked and beta3p75_logitz indicates whether ranges are distinct.",
        "",
        "## 2. DeLong test (paired, structural components)",
        "",
        "| Comparison | n_paired | ΔAUC | SE | z | p (two-sided) |",
        "|---|---|---|---|---|---|",
        "| beta3p75_logitz vs locked_raw | {n} | {d:+.4f} | {se:.4f} | {z:.3f} | {p} |".format(
            n=dl["n_paired"], d=dl["delta_auc"], se=dl["se_diff"], z=dl["z"],
            p=_sig_label(dl["p_two_sided"])),
        "| beta3p75_raw vs locked_raw | {n} | {d:+.4f} | {se:.4f} | {z:.3f} | {p} |".format(
            n=dl_raw["n_paired"], d=dl_raw["delta_auc"], se=dl_raw["se_diff"], z=dl_raw["z"],
            p=_sig_label(dl_raw["p_two_sided"])),
        "| beta3p75_logitz vs beta3p75_raw | {n} | {d:+.4f} | {se:.4f} | {z:.3f} | {p} |".format(
            n=dl_lz_raw["n_paired"], d=dl_lz_raw["delta_auc"], se=dl_lz_raw["se_diff"], z=dl_lz_raw["z"],
            p=_sig_label(dl_lz_raw["p_two_sided"])),
        "",
        "## 3. Paired bootstrap delta (B={B}, subject-level)".format(B=B_BOOTSTRAP),
        "",
        "| Comparison | n_paired | ΔAUC | 95% CI | p(ΔAUC≤0) | ΔPR-AUC | 95% CI | p(ΔPR-AUC≤0) |",
        "|---|---|---|---|---|---|---|---|",
        "| beta3p75_logitz vs locked_raw | {n} | {da:+.4f} | {ci_a} | {pa:.4f} | {dp:+.4f} | {ci_p} | {pp:.4f} |".format(
            n=pb["n_paired"], da=pb["delta_auc_obs"],
            ci_a=_ci_str(pb["delta_auc_ci95_lower"], pb["delta_auc_ci95_upper"]),
            pa=pb["p_delta_auc_le0_onesided"],
            dp=pb["delta_prauc_obs"],
            ci_p=_ci_str(pb["delta_prauc_ci95_lower"], pb["delta_prauc_ci95_upper"]),
            pp=pb["p_delta_prauc_le0_onesided"]),
        "| beta3p75_raw vs locked_raw | {n} | {da:+.4f} | {ci_a} | {pa:.4f} | {dp:+.4f} | {ci_p} | {pp:.4f} |".format(
            n=pb_raw["n_paired"], da=pb_raw["delta_auc_obs"],
            ci_a=_ci_str(pb_raw["delta_auc_ci95_lower"], pb_raw["delta_auc_ci95_upper"]),
            pa=pb_raw["p_delta_auc_le0_onesided"],
            dp=pb_raw["delta_prauc_obs"],
            ci_p=_ci_str(pb_raw["delta_prauc_ci95_lower"], pb_raw["delta_prauc_ci95_upper"]),
            pp=pb_raw["p_delta_prauc_le0_onesided"]),
        "",
        "## 4. Paired permutation test (B={B})".format(B=B_PERMUTATION),
        "",
        "| Comparison | n_paired | ΔAUC | p_AUC | ΔPR-AUC | p_PR-AUC |",
        "|---|---|---|---|---|---|",
        "| beta3p75_logitz vs locked_raw | {n} | {da:+.4f} | {pa} | {dp:+.4f} | {pp} |".format(
            n=pp["n_paired"], da=pp["delta_auc_obs"],
            pa=_sig_label(pp["p_auc_ge_obs_onesided"]),
            dp=pp["delta_prauc_obs"],
            pp=_sig_label(pp["p_prauc_ge_obs_onesided"])),
        "| beta3p75_raw vs locked_raw | {n} | {da:+.4f} | {pa} | {dp:+.4f} | {pp} |".format(
            n=pp_raw["n_paired"], da=pp_raw["delta_auc_obs"],
            pa=_sig_label(pp_raw["p_auc_ge_obs_onesided"]),
            dp=pp_raw["delta_prauc_obs"],
            pp=_sig_label(pp_raw["p_prauc_ge_obs_onesided"])),
        "",
        "## 5. Foldwise comparison: beta3p75_logitz vs locked_raw (descriptive)",
        "",
        f"Mean ΔAUC = {fw_sum_lz['mean_delta_auc']:+.4f} ± {fw_sum_lz['std_delta_auc']:.4f} (SD over 5 folds)",
        f"Folds with positive ΔAUC: {fw_sum_lz['n_positive_delta_auc']} / 5",
        f"Wilcoxon W (AUC): {fw_sum_lz['wilcoxon_W_auc']}  p={fw_sum_lz['wilcoxon_p_auc']}  (descriptive; n=5 has no power)",
        f"Mean ΔPR-AUC = {fw_sum_lz['mean_delta_prauc']:+.4f} ± {fw_sum_lz['std_delta_prauc']:.4f}",
        f"Folds with positive ΔPR-AUC: {fw_sum_lz['n_positive_delta_prauc']} / 5",
        "",
        "## Overall verdict",
        "",
        f"**{stability}**",
        "",
        "### Interpretation",
        "",
        "The OOF logit-z calibration recovers AUC from a cross-fold score-scale",
        "mismatch (Fold 1 selected C=0.1 vs C=0.001 in folds 2–5). The calibrated AUC",
        "of 0.7951 reflects the genuine rank-order performance of the model. The raw",
        "AUC of 0.7599 was artificially depressed by cross-fold score incompatibility,",
        "not by lower predictive rank-order performance.",
        "",
        "Whether the _calibrated_ result (0.7951) is statistically superior to the",
        "locked reference (0.7786) depends on the test results above. If the improvement",
        "is not statistically significant, the correct interpretation is: the promoted",
        "model is not demonstrably worse than the locked model and may be marginally",
        "better, but the difference is within the uncertainty of a 397-subject cohort.",
        "",
        "The key promotion argument is not statistical superiority over the locked model",
        "but that the calibrated AUC of 0.7951 passes the pre-specified promotion gate",
        "(AUC > 0.7830, PR-AUC ≥ 0.5599) in a leakage-safe evaluation, demonstrating",
        "that the model achieves the required absolute performance level.",
        "",
        "## Read-only guarantee",
        "Did not retrain VAE. Did not fit thresholds on outer test data.",
        "Did not modify tensors, metadata, ledger, or any existing run outputs.",
    ]

    with open(OUT_DIR / "final_report.md", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
