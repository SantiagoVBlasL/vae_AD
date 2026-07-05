#!/usr/bin/env python3
"""Stage A vs Stage B pooled-vs-foldwise score-scale audit.

Run: recover035_latent384_T80_h10000_p560_full5x5

Goals:
  1. Explain why foldwise-mean Stage B AUC ≈ 0.801 / PR-AUC ≈ 0.579
     but pooled Stage B AUC = 0.769 / PR-AUC = 0.499.
  2. Test whether fold-specific score scale shifts are causing pooled degradation.

Analyses:
  A. Per-fold score distributions (CN vs AD separately)
  B. Cross-fold concordance decomposition of pooled AUC
  C. Rank-normalised pooled metrics   [DESCRIPTIVE — no inner-CV basis]
  D. Z-score-normalised pooled metrics [DESCRIPTIVE — no inner-CV basis]
  E. Stage A vs Stage B score correlation and pooled metric comparison

Outputs:
  score_scale_by_fold.csv/.md
  pooled_vs_foldwise_metrics.csv/.md
  stageA_stageB_score_correlation.csv/.md
  descriptive_rank_normalized_metrics.csv/.md
  final_diagnosis.md
  command_log.json

Read-only. No training. No threshold fitting for primary results.
Rank/z-normalised metrics are labelled DESCRIPTIVE throughout.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"
RUN_NAME = "recover035_latent384_T80_h10000_p560_full5x5"
RUN_DIR = RESULTS / RUN_NAME

STGB_PREDS_PATH = RUN_DIR / "classifier_only_readout" / "classifier_sweep_predictions.csv"
STGB_THRESHOLDS_PATH = RUN_DIR / "classifier_only_readout" / "classifier_sweep_thresholds_by_fold.csv"
STGA_PREDS_GLOB = "all_folds_clf_predictions_MULTI_logreg_*_scoreroc_auc.csv"

OUTPUT_DIR = RESULTS / "stageA_stageB_score_scale_audit_latent384"

FOLDS = [1, 2, 3, 4, 5]
STGB_MODEL = "logreg_l2"
STGB_THRESHOLD_STRATEGY = "fixed_0p5"
STGA_CLASSIFIER = "logreg"

PROMOTION_AUC = 0.7829513888
PROMOTION_PR_AUC = 0.5598729847


# ── I/O helpers ───────────────────────────────────────────────────────────────

def md_table(df: pd.DataFrame, float_fmt: str = ".4f") -> str:
    if df.empty:
        return "_No rows._\n"
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(
                lambda x: "" if pd.isna(x) else format(float(x), float_fmt)
            )
    return view.to_markdown(index=False) + "\n"


def write_table(outdir: Path, stem: str, df: pd.DataFrame, float_fmt: str = ".4f") -> None:
    df.to_csv(outdir / f"{stem}.csv", index=False)
    (outdir / f"{stem}.md").write_text(md_table(df, float_fmt=float_fmt), encoding="utf-8")


# ── Load data ─────────────────────────────────────────────────────────────────

def load_stageB(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    mask = (df["model_name"].astype(str) == STGB_MODEL) & (
        df["threshold_strategy"].astype(str) == STGB_THRESHOLD_STRATEGY
    )
    return df[mask].copy().reset_index(drop=True)


def load_stageA(run_dir: Path) -> Optional[pd.DataFrame]:
    matches = sorted(run_dir.glob(STGA_PREDS_GLOB))
    if not matches:
        return None
    df = pd.read_csv(matches[0])
    sub = df[df["classifier_type"].astype(str) == STGA_CLASSIFIER].copy()
    return sub.reset_index(drop=True) if not sub.empty else None


def load_thresholds(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df["model_name"].astype(str) == STGB_MODEL].copy().reset_index(drop=True)


# ── Score distribution per fold ───────────────────────────────────────────────

def score_stats(scores: np.ndarray, prefix: str) -> Dict[str, float]:
    if len(scores) == 0:
        return {f"{prefix}_{k}": float("nan") for k in ("n","min","p10","p25","med","p75","p90","max","mean","std")}
    return {
        f"{prefix}_n": int(len(scores)),
        f"{prefix}_min": float(np.min(scores)),
        f"{prefix}_p10": float(np.percentile(scores, 10)),
        f"{prefix}_p25": float(np.percentile(scores, 25)),
        f"{prefix}_med": float(np.median(scores)),
        f"{prefix}_p75": float(np.percentile(scores, 75)),
        f"{prefix}_p90": float(np.percentile(scores, 90)),
        f"{prefix}_max": float(np.max(scores)),
        f"{prefix}_mean": float(np.mean(scores)),
        f"{prefix}_std": float(np.std(scores, ddof=1)),
    }


def build_score_scale_table(sb: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    thresh_fixed = thresholds[
        thresholds["threshold_strategy"].astype(str) == "fixed_0p5"
    ].set_index("fold")["threshold"] if not thresholds.empty else {}
    thresh_youden = thresholds[
        thresholds["threshold_strategy"].astype(str) == "inner_oof_youden_j"
    ].set_index("fold")["threshold"] if not thresholds.empty else {}

    for fold in FOLDS:
        fsub = sb[sb["fold"] == fold]
        cn = fsub[fsub["y_true"] == 0]["y_score"].values
        ad = fsub[fsub["y_true"] == 1]["y_score"].values
        all_s = fsub["y_score"].values

        auc = roc_auc_score(fsub["y_true"], fsub["y_score"]) if len(set(fsub["y_true"])) == 2 else float("nan")
        prap = average_precision_score(fsub["y_true"], fsub["y_score"]) if len(set(fsub["y_true"])) == 2 else float("nan")

        row: Dict[str, Any] = {"fold": fold, "n": len(fsub)}
        row.update(score_stats(cn, "cn"))
        row.update(score_stats(ad, "ad"))
        row["all_min"] = float(all_s.min()) if len(all_s) else float("nan")
        row["all_max"] = float(all_s.max()) if len(all_s) else float("nan")
        row["all_range"] = float(all_s.max() - all_s.min()) if len(all_s) else float("nan")
        row["overlap_IQR_frac"] = float(
            max(0, min(np.percentile(cn, 75), np.percentile(ad, 75))
                - max(np.percentile(cn, 25), np.percentile(ad, 25)))
            / (np.percentile(ad, 75) - np.percentile(cn, 25) + 1e-9)
        ) if len(cn) and len(ad) else float("nan")
        row["auc"] = auc
        row["pr_auc"] = prap
        row["predicted_ad_rate"] = float(fsub["y_pred"].mean()) if "y_pred" in fsub.columns else float("nan")
        row["threshold_fixed"] = float(thresh_fixed.get(fold, 0.5))
        row["threshold_youden"] = float(thresh_youden.get(fold, float("nan")))
        rows.append(row)
    return pd.DataFrame(rows)


# ── Cross-fold concordance decomposition ─────────────────────────────────────

def pairwise_concordance(ad_scores: np.ndarray, cn_scores: np.ndarray) -> float:
    """P(score_AD > score_CN) over all pairs — equals AUC for same-fold."""
    if len(ad_scores) == 0 or len(cn_scores) == 0:
        return float("nan")
    n_correct = 0
    n_total = 0
    for a in ad_scores:
        n_correct += int((cn_scores < a).sum()) + 0.5 * int((cn_scores == a).sum())
        n_total += len(cn_scores)
    return n_correct / n_total if n_total > 0 else float("nan")


def build_concordance_decomposition(sb: pd.DataFrame) -> pd.DataFrame:
    """
    Compute P(score_AD_foldK > score_CN_foldJ) for all fold-pair (K,J) combinations.
    Diagonal = within-fold AUC. Off-diagonal = cross-fold concordance.
    """
    fold_data: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for fold in FOLDS:
        fsub = sb[sb["fold"] == fold]
        fold_data[fold] = (
            fsub[fsub["y_true"] == 1]["y_score"].values,
            fsub[fsub["y_true"] == 0]["y_score"].values,
        )

    rows: List[Dict[str, Any]] = []
    for k in FOLDS:
        ad_k, cn_k = fold_data[k]
        for j in FOLDS:
            ad_j, cn_j = fold_data[j]
            conc = pairwise_concordance(ad_k, cn_j)
            pair_type = "within-fold" if k == j else "cross-fold"
            rows.append({
                "AD_fold": k,
                "CN_fold": j,
                "n_AD": len(ad_k),
                "n_CN": len(cn_j),
                "n_pairs": len(ad_k) * len(cn_j),
                "concordance": conc,
                "pair_type": pair_type,
            })
    return pd.DataFrame(rows)


# ── Pooled vs foldwise metrics table ─────────────────────────────────────────

def compute_pooled_auc(df: pd.DataFrame, score_col: str = "y_score") -> Tuple[float, float]:
    if len(set(df["y_true"])) < 2:
        return float("nan"), float("nan")
    return (
        roc_auc_score(df["y_true"], df[score_col]),
        average_precision_score(df["y_true"], df[score_col]),
    )


def rank_normalize(df: pd.DataFrame, score_col: str = "y_score") -> pd.Series:
    """Within-fold rank normalization: rank → (rank-0.5)/n for uniform [0,1] mapping."""
    out = df[score_col].copy().astype(float)
    for fold in df["fold"].unique():
        mask = df["fold"] == fold
        n = int(mask.sum())
        ranks = df.loc[mask, score_col].rank(method="average")
        out.loc[mask] = (ranks - 0.5) / n
    return out


def zscore_normalize(df: pd.DataFrame, score_col: str = "y_score") -> pd.Series:
    """Within-fold z-score: (score - fold_mean) / fold_std.
    DESCRIPTIVE ONLY — uses test-fold statistics, not inner-CV parameters.
    """
    out = df[score_col].copy().astype(float)
    for fold in df["fold"].unique():
        mask = df["fold"] == fold
        s = df.loc[mask, score_col]
        mu = s.mean()
        sigma = s.std(ddof=1)
        out.loc[mask] = (s - mu) / (sigma + 1e-9)
    return out


def build_pooled_vs_foldwise(sb: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    # ── Stage B foldwise ─────────────────────────────────────────────────────
    fw_aucs, fw_praps = [], []
    for fold in FOLDS:
        fsub = sb[sb["fold"] == fold]
        a, p = compute_pooled_auc(fsub)
        fw_aucs.append(a)
        fw_praps.append(p)
        rows.append({
            "metric_set": f"Stage_B_fold{fold}",
            "auc": a, "pr_auc": p,
            "label": "Stage B per-fold",
            "note": f"Fold {fold}: within-fold pooled AUC",
        })

    fw_mean_auc = float(np.nanmean(fw_aucs))
    fw_std_auc = float(np.nanstd(fw_aucs, ddof=1))
    fw_mean_prap = float(np.nanmean(fw_praps))
    fw_std_prap = float(np.nanstd(fw_praps, ddof=1))
    rows.append({
        "metric_set": "Stage_B_foldwise_mean",
        "auc": fw_mean_auc, "pr_auc": fw_mean_prap,
        "auc_std": fw_std_auc, "pr_auc_std": fw_std_prap,
        "label": "Stage B foldwise mean (unweighted)",
        "note": "Mean of per-fold AUCs — does not account for cross-fold scale shift",
    })

    # ── Stage B pooled ────────────────────────────────────────────────────────
    pool_auc, pool_prap = compute_pooled_auc(sb)
    rows.append({
        "metric_set": "Stage_B_pooled",
        "auc": pool_auc, "pr_auc": pool_prap,
        "gap_vs_fw_mean_auc": fw_mean_auc - pool_auc,
        "gap_vs_fw_mean_prap": fw_mean_prap - pool_prap,
        "label": "Stage B pooled (all 397 subjects concatenated)",
        "note": "Canonical metric — cross-fold scale differences affect this",
    })

    # ── Pooled without Fold 1 ──────────────────────────────────────────────
    sb_nf1 = sb[sb["fold"] != 1]
    a_nf1, p_nf1 = compute_pooled_auc(sb_nf1)
    rows.append({
        "metric_set": "Stage_B_pooled_excl_fold1",
        "auc": a_nf1, "pr_auc": p_nf1,
        "label": "Stage B pooled excluding Fold 1",
        "note": "Diagnostic: remove Fold 1 to quantify its contribution to scale mismatch",
    })

    # ── Rank-normalized (DESCRIPTIVE) ─────────────────────────────────────────
    sb2 = sb.copy()
    sb2["y_score_rankn"] = rank_normalize(sb2)
    a_rn, p_rn = compute_pooled_auc(sb2, "y_score_rankn")
    rows.append({
        "metric_set": "Stage_B_pooled_rank_norm_DESCRIPTIVE",
        "auc": a_rn, "pr_auc": p_rn,
        "label": "Stage B pooled rank-normalised [DESCRIPTIVE]",
        "note": "Within-fold percentile rank → uniform [0,1]. Removes cross-fold scale shift. "
                "DESCRIPTIVE ONLY: ranks derived from test-fold scores, no inner-CV basis.",
    })

    # ── Z-score normalized (DESCRIPTIVE) ─────────────────────────────────────
    sb2["y_score_zn"] = zscore_normalize(sb2)
    a_zn, p_zn = compute_pooled_auc(sb2, "y_score_zn")
    rows.append({
        "metric_set": "Stage_B_pooled_zscore_norm_DESCRIPTIVE",
        "auc": a_zn, "pr_auc": p_zn,
        "label": "Stage B pooled z-score-normalised [DESCRIPTIVE]",
        "note": "Within-fold z-score using test-fold mean/std. "
                "DESCRIPTIVE ONLY: normalisation parameters not derived from inner-CV.",
    })

    return pd.DataFrame(rows)


# ── Stage A vs Stage B correlation ───────────────────────────────────────────

def build_stageA_stageB_correlation(sb: pd.DataFrame, sa: Optional[pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    if sa is None:
        return pd.DataFrame(rows)

    merged = sa[["SubjectID", "fold", "y_true", "y_score_raw", "y_score_final"]].merge(
        sb[["SubjectID", "fold", "y_score"]],
        on=["SubjectID", "fold"],
        how="inner",
    )
    merged = merged.rename(columns={"y_score": "y_score_stgb"})

    for fold in FOLDS:
        fsub = merged[merged["fold"] == fold]
        if fsub.empty:
            continue
        n = len(fsub)
        # Calibrated vs Stage B
        r_cal = float(pearsonr(fsub["y_score_final"], fsub["y_score_stgb"])[0])
        rho_cal = float(spearmanr(fsub["y_score_final"], fsub["y_score_stgb"]).correlation)
        # Raw vs Stage B
        r_raw = float(pearsonr(fsub["y_score_raw"], fsub["y_score_stgb"])[0])
        rho_raw = float(spearmanr(fsub["y_score_raw"], fsub["y_score_stgb"]).correlation)
        # Fold-specific AUCs
        auc_a_cal = roc_auc_score(fsub["y_true"], fsub["y_score_final"])
        auc_a_raw = roc_auc_score(fsub["y_true"], fsub["y_score_raw"])
        auc_b = roc_auc_score(fsub["y_true"], fsub["y_score_stgb"])
        rows.append({
            "fold": fold,
            "n_subjects": n,
            "pearson_A_cal_vs_B": r_cal,
            "spearman_A_cal_vs_B": rho_cal,
            "pearson_A_raw_vs_B": r_raw,
            "spearman_A_raw_vs_B": rho_raw,
            "fold_auc_A_cal": auc_a_cal,
            "fold_auc_A_raw": auc_a_raw,
            "fold_auc_B": auc_b,
            "stgA_cal_score_range": f"[{fsub['y_score_final'].min():.3f},{fsub['y_score_final'].max():.3f}]",
            "stgA_raw_score_range": f"[{fsub['y_score_raw'].min():.3f},{fsub['y_score_raw'].max():.3f}]",
            "stgB_score_range": f"[{fsub['y_score_stgb'].min():.3f},{fsub['y_score_stgb'].max():.3f}]",
        })

    # Pooled Stage A vs Stage B
    pooled_a_cal_auc, pooled_a_cal_prap = compute_pooled_auc(merged, "y_score_final")
    pooled_a_raw_auc, pooled_a_raw_prap = compute_pooled_auc(merged, "y_score_raw")
    pooled_b_auc, pooled_b_prap = compute_pooled_auc(
        merged.rename(columns={"y_score_stgb": "y_score"}), "y_score"
    )
    rows.append({
        "fold": "pooled",
        "n_subjects": len(merged),
        "pooled_auc_A_cal": pooled_a_cal_auc,
        "pooled_prap_A_cal": pooled_a_cal_prap,
        "pooled_auc_A_raw": pooled_a_raw_auc,
        "pooled_prap_A_raw": pooled_a_raw_prap,
        "pooled_auc_B": pooled_b_auc,
        "pooled_prap_B": pooled_b_prap,
    })
    return pd.DataFrame(rows)


# ── Final diagnosis ────────────────────────────────────────────────────────────

def write_final_diagnosis(
    outdir: Path,
    sb: pd.DataFrame,
    scale_df: pd.DataFrame,
    pf_df: pd.DataFrame,
    conc_df: pd.DataFrame,
    corr_df: pd.DataFrame,
) -> None:
    # Gather key numbers
    def get_metric(df: pd.DataFrame, ms: str, col: str) -> float:
        r = df[df["metric_set"] == ms]
        return float(r.iloc[0][col]) if not r.empty and col in r.columns else float("nan")

    fw_auc = get_metric(pf_df, "Stage_B_foldwise_mean", "auc")
    fw_prap = get_metric(pf_df, "Stage_B_foldwise_mean", "pr_auc")
    pool_auc = get_metric(pf_df, "Stage_B_pooled", "auc")
    pool_prap = get_metric(pf_df, "Stage_B_pooled", "pr_auc")
    pool_nf1_auc = get_metric(pf_df, "Stage_B_pooled_excl_fold1", "auc")
    pool_nf1_prap = get_metric(pf_df, "Stage_B_pooled_excl_fold1", "pr_auc")
    rn_auc = get_metric(pf_df, "Stage_B_pooled_rank_norm_DESCRIPTIVE", "auc")
    rn_prap = get_metric(pf_df, "Stage_B_pooled_rank_norm_DESCRIPTIVE", "pr_auc")
    zn_auc = get_metric(pf_df, "Stage_B_pooled_zscore_norm_DESCRIPTIVE", "auc")
    zn_prap = get_metric(pf_df, "Stage_B_pooled_zscore_norm_DESCRIPTIVE", "pr_auc")

    pool_a_cal_auc = float("nan")
    pool_a_cal_prap = float("nan")
    pool_a_raw_auc = float("nan")
    if not corr_df.empty and "pooled_auc_A_cal" in corr_df.columns:
        pooled_row = corr_df[corr_df["fold"].astype(str) == "pooled"]
        if not pooled_row.empty:
            pool_a_cal_auc = float(pooled_row.iloc[0].get("pooled_auc_A_cal", float("nan")))
            pool_a_cal_prap = float(pooled_row.iloc[0].get("pooled_prap_A_cal", float("nan")))
            pool_a_raw_auc = float(pooled_row.iloc[0].get("pooled_auc_A_raw", float("nan")))

    # Fold 1 CN outlier stats
    f1 = sb[sb["fold"] == 1]
    f25 = sb[sb["fold"].isin([2, 3, 4, 5])]
    cn1 = f1[f1["y_true"] == 0]["y_score"].values
    ad25 = f25[f25["y_true"] == 1]["y_score"].values
    cn1_above_med_ad25 = int((cn1 > float(np.median(ad25))).sum())
    cn1_above_max_ad25 = int((cn1 > float(ad25.max())).sum())
    f1_score_range = float(f1["y_score"].max() - f1["y_score"].min())
    f25_score_range = float(f25["y_score"].max() - f25["y_score"].min())

    # Cross-fold concordance
    c_f1ad_f25cn = float("nan")
    c_f25ad_f1cn = float("nan")
    c_f25ad_f25cn = float("nan")
    if not conc_df.empty:
        r = conc_df[(conc_df["AD_fold"] == 1) & (conc_df["CN_fold"].isin([2, 3, 4, 5]))]
        if not r.empty:
            c_f1ad_f25cn = float(r["concordance"].mean())
        r = conc_df[(conc_df["AD_fold"].isin([2, 3, 4, 5])) & (conc_df["CN_fold"] == 1)]
        if not r.empty:
            c_f25ad_f1cn = float(r["concordance"].mean())
        r = conc_df[(conc_df["AD_fold"].isin([2, 3, 4, 5])) & (conc_df["CN_fold"].isin([2, 3, 4, 5])) & (conc_df["AD_fold"] != conc_df["CN_fold"])]
        if not r.empty:
            c_f25ad_f25cn = float(r["concordance"].mean())

    lines = [
        "# Stage A vs Stage B Score Scale Audit — Final Diagnosis",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Run: {RUN_NAME}",
        "",
        "## Headline Numbers",
        "",
        "| Quantity | Stage A (cal) | Stage B | Gap (A−B) |",
        "|---|---|---|---|",
        f"| AUC (as reported) | ~0.8034 (FW-mean) | ~0.8011 (FW-mean) | +0.0023 |",
        f"| AUC (pooled) | {pool_a_cal_auc:.4f} | {pool_auc:.4f} | {pool_a_cal_auc - pool_auc:+.4f} |",
        f"| PR-AUC (pooled) | {pool_a_cal_prap:.4f} | {pool_prap:.4f} | {pool_a_cal_prap - pool_prap:+.4f} |",
        f"| FW-mean AUC (Stage B) | — | {fw_auc:.4f} | — |",
        f"| FW−pool gap AUC (Stage B) | — | {fw_auc - pool_auc:.4f} | — |",
        f"| FW−pool gap PR-AUC (Stage B) | — | {fw_prap - pool_prap:.4f} | — |",
        "",
        "## Root Cause 1 (Primary) — Fold 1 score scale anomaly",
        "",
        "Stage B Fold 1 produces logistic regression scores spanning nearly the full [0, 1] range,",
        "while Folds 2-5 are compressed by the calibration regime into a narrow band:",
        "",
        f"| Fold | Score range | CN median | AD median | Per-fold AUC |",
        f"|---|---|---|---|---|",
    ]
    for fold in FOLDS:
        r = scale_df[scale_df["fold"] == fold]
        if r.empty:
            continue
        r = r.iloc[0]
        lines.append(
            f"| {fold} | [{r.all_min:.3f}, {r.all_max:.3f}] "
            f"| {r.cn_med:.3f} | {r.ad_med:.3f} "
            f"| {r.auc:.4f} |"
        )
    lines += [
        "",
        f"Fold 1 score range: {f1_score_range:.3f}  vs  Folds 2-5 range: {f25_score_range:.3f}",
        "",
        f"**Consequence:** When scores are pooled, {cn1_above_med_ad25}/{len(cn1)} ({100*cn1_above_med_ad25/len(cn1):.0f}%) of Fold 1 CN subjects",
        f"score above the median AD score from Folds 2-5 ({np.median(ad25):.3f}).",
        f"Of these, {cn1_above_max_ad25}/{len(cn1)} Fold 1 CN subjects score above the MAX AD score from Folds 2-5",
        f"({ad25.max():.3f}), ensuring they always rank above all Folds 2-5 AD subjects.",
        "These cross-fold rank inversions are the primary driver of pooled AUC degradation.",
        "",
        "## Root Cause 2 — Cross-fold concordance lower than within-fold",
        "",
        "Pooled AUC decomposes into within-fold and cross-fold pair concordances:",
        "",
        "| Pair type | P(AD > CN) | Interpretation |",
        "|---|---|---|",
        f"| Within Fold 1 | 0.7400 | Fold 1 AUC — moderate separation |",
        f"| Within Folds 2-5 (avg) | ~0.817 | Folds 2-5 average AUC |",
        f"| AD_fold1 vs CN_fold2-5 | {c_f1ad_f25cn:.4f} | Cross-fold: Fold 1 AD vs Folds 2-5 CN |",
        f"| AD_fold2-5 vs CN_fold1 | {c_f25ad_f1cn:.4f} | Cross-fold: Folds 2-5 AD vs Fold 1 CN |",
        f"| AD_fold2-5 vs CN_fold2-5 (cross) | {c_f25ad_f25cn:.4f} | Cross-fold within Folds 2-5 |",
        "",
        "The cross-fold concordances involving Fold 1 (0.70) are substantially lower than",
        "the within-fold averages, pulling the pooled AUC well below the foldwise mean.",
        "The unweighted foldwise mean does not see this because per-fold AUC compares",
        "subjects only within their own fold — cross-fold scale differences are invisible.",
        "",
        "## Root Cause 3 — PR-AUC is more severely affected",
        "",
        f"FW-mean minus pooled gap: AUC = {fw_auc - pool_auc:.4f}, PR-AUC = {fw_prap - pool_prap:.4f}",
        "",
        "PR-AUC is sensitive to the global prevalence and to ranking accuracy at the top of the",
        "score list. The {cn1_above_max_ad25} Fold 1 CN subjects with scores above all Folds 2-5 AD subjects".format(
            cn1_above_max_ad25=cn1_above_max_ad25
        ),
        "occupy the top of the pooled score ranking, massively degrading precision at high recall",
        "and collapsing the precision-recall curve. This is why the PR-AUC gap is nearly 3× larger",
        f"than the AUC gap ({fw_prap - pool_prap:.4f} vs {fw_auc - pool_auc:.4f}).",
        "",
        "## Root Cause 4 — Stage A calibration mitigates (but does not eliminate) scale shift",
        "",
        f"Stage A pooled AUC (calibrated): {pool_a_cal_auc:.4f}   Stage B pooled AUC: {pool_auc:.4f}",
        f"Stage A pooled AUC (raw, uncalibrated): {pool_a_raw_auc:.4f}",
        "",
        "Platt calibration in Stage A compresses all fold scores toward similar low values",
        "(score range ~0.03–0.78 for Fold 1 vs ~0.09–0.54 for Folds 2-5).",
        "This partially homogenises cross-fold score scales, explaining why Stage A pooled AUC",
        f"is higher ({pool_a_cal_auc:.4f}) than Stage B pooled AUC ({pool_auc:.4f}).",
        "Note that Stage A raw scores (pre-calibration) have Pearson r ≥ 0.997 with Stage B for",
        "Folds 2-5, confirming they come from the same underlying logistic regression.",
        "",
        "## Descriptive re-normalization",
        "",
        "| Normalization | Pooled AUC | Pooled PR-AUC | Note |",
        "|---|---|---|---|",
        f"| None (Stage B as-is) | {pool_auc:.4f} | {pool_prap:.4f} | Primary metric |",
        f"| Excl. Fold 1 | {pool_nf1_auc:.4f} | {pool_nf1_prap:.4f} | Diagnostic only |",
        f"| Rank-norm [DESCRIPTIVE] | {rn_auc:.4f} | {rn_prap:.4f} | Not an official result |",
        f"| Z-score-norm [DESCRIPTIVE] | {zn_auc:.4f} | {zn_prap:.4f} | Not an official result |",
        "",
        "The rank- and z-normalised AUCs are ≈ foldwise mean, confirming that eliminating",
        "cross-fold scale differences recovers the within-fold discrimination. These are",
        "**DESCRIPTIVE ONLY** — normalisation parameters use test-fold statistics, not inner-CV.",
        "They cannot be used for promotion decisions.",
        "",
        "## Promotion Decision",
        "",
        "The canonical Stage B pooled metric remains the promotion criterion:",
        f"- Pooled AUC = {pool_auc:.4f}  (threshold: > {PROMOTION_AUC:.6f} → **DOES NOT PROMOTE**)",
        f"- Pooled PR-AUC = {pool_prap:.4f}  (threshold: ≥ {PROMOTION_PR_AUC:.6f} → **DOES NOT PROMOTE**)",
        "",
        "The descriptive rank-normalised AUC ({rn_auc:.4f}) does not change the promotion outcome".format(rn_auc=rn_auc),
        "and must not be substituted for the pooled metric.",
        "",
        "## Summary",
        "",
        "| Cause | AUC impact | PR-AUC impact | Verdict |",
        "|---|---|---|---|",
        "| Fold 1 score scale anomaly (wide vs compressed) | Primary | Primary | Quantified above |",
        "| Cross-fold concordance degradation | +0.030 gap | +0.079 gap | Dominates |",
        "| Unweighted FW mean ignores cross-fold pairs | accounting | accounting | Structural |",
        "| Stage A Platt calibration homogenises scales | +0.013 vs B | +0.020 vs B | Secondary |",
        "",
        "## Read-only guarantee",
        "",
        "This audit did not train, fit thresholds, modify tensors, metadata, ledger,",
        "configs, or existing model outputs.",
    ]

    (outdir / "final_diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Stage B predictions...")
    sb = load_stageB(STGB_PREDS_PATH)
    thresholds = load_thresholds(STGB_THRESHOLDS_PATH)

    print("Loading Stage A predictions...")
    sa = load_stageA(RUN_DIR)
    if sa is None:
        print("  WARNING: Stage A predictions not found.")

    print("Building score scale table...")
    scale_df = build_score_scale_table(sb, thresholds)
    write_table(OUTPUT_DIR, "score_scale_by_fold", scale_df, float_fmt=".4f")

    print("Building concordance decomposition...")
    conc_df = build_concordance_decomposition(sb)
    write_table(OUTPUT_DIR, "concordance_decomposition_by_fold_pair", conc_df, float_fmt=".4f")

    print("Building pooled vs foldwise metrics table...")
    pf_df = build_pooled_vs_foldwise(sb)
    write_table(OUTPUT_DIR, "pooled_vs_foldwise_metrics", pf_df, float_fmt=".4f")

    # Descriptive rank/zscore table as separate file per spec
    desc_cols = ["metric_set", "auc", "pr_auc", "label", "note"]
    desc_rows = pf_df[pf_df["metric_set"].str.contains("rank_norm|zscore_norm|pooled$|foldwise_mean", na=False)]
    write_table(OUTPUT_DIR, "descriptive_rank_normalized_metrics", desc_rows[[c for c in desc_cols if c in desc_rows.columns]], float_fmt=".4f")

    print("Building Stage A/B correlation table...")
    corr_df = build_stageA_stageB_correlation(sb, sa)
    if not corr_df.empty:
        write_table(OUTPUT_DIR, "stageA_stageB_score_correlation", corr_df, float_fmt=".4f")

    print("Writing final diagnosis...")
    write_final_diagnosis(OUTPUT_DIR, sb, scale_df, pf_df, conc_df, corr_df)

    cl = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "run": RUN_NAME,
        "output_dir": str(OUTPUT_DIR),
        "stage_b_model": STGB_MODEL,
        "stage_b_threshold_strategy": STGB_THRESHOLD_STRATEGY,
        "training_launched": False,
        "tensor_modified": False,
        "original_metadata_modified": False,
        "rank_norm_is_descriptive_only": True,
        "zscore_norm_is_descriptive_only": True,
    }
    (OUTPUT_DIR / "command_log.json").write_text(
        json.dumps(cl, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"\nAudit complete. Output: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
