#!/usr/bin/env python3
"""Read-only Stage B score-scale audit for recover035_latent384_beta3p75_T80_h10000_p560.

Key questions:
  1. Is the Fold 1 score-range anomaly reduced vs beta2.5?
     (beta2.5 Fold 1 range=0.993; Folds 2-5 range~0.43)
  2. Is the foldwise-mean vs pooled AUC gap smaller?
  3. Does stronger beta improve score-scale consistency across folds?

Per fold:
  - score distribution by diagnosis (CN/AD): min/p25/med/p75/max
  - per-fold AUC/PR-AUC
  - threshold value (youden_j and fixed_0p5)
  - score range and predicted AD rate

Across folds:
  - pooled AUC/PR-AUC vs foldwise mean
  - rank-normalized AUC/PR-AUC [DESCRIPTIVE ONLY]
  - cross-fold concordance decomposition
  - comparison of Fold 1 range vs beta2.5 baseline

Hard constraints:
  - No training.
  - No threshold fitting.
  - Rank-normalized metrics labeled DESCRIPTIVE ONLY.
  - Do not modify tensor, metadata, ledger, configs, or existing model outputs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"
BIG_DISK = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026")

DEFAULT_RUN_DIR = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
DEFAULT_OUTPUT = RESULTS / "stageB_score_scale_audit_beta3p75_latent384"

BETA2P5_FOLD1_RANGE = 0.993
BETA2P5_FOLDS25_RANGE = 0.431
BETA2P5_POOLED_AUC = 0.7694
BETA2P5_POOLED_PR_AUC = 0.4994
BETA2P5_FOLDWISE_AUC = 0.8011
BETA2P5_FOLDWISE_PR_AUC = 0.5790

PRIMARY_MODEL = "logreg_l2"
STGB_MODEL = "logreg_l2"
STGB_THRESHOLD_STRATEGY = "fixed_0p5"
FOLDS = [1, 2, 3, 4, 5]
PROMOTION_AUC = 0.782951
PROMOTION_PR_AUC = 0.559873


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths without reading artifacts.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    if path.is_absolute():
        return path
    local = RESULTS / path
    if local.exists():
        return local
    big = BIG_DISK / path.name
    if big.exists():
        return big
    return PROJECT_ROOT / path


def md_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "_No rows._\n"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
    text = view.to_markdown(index=False)
    if len(df) > max_rows:
        text += f"\n\n_Showing {max_rows} of {len(df)} rows._"
    return text + "\n"


def write_table(outdir: Path, stem: str, df: pd.DataFrame, max_rows: int = 40) -> None:
    df.to_csv(outdir / f"{stem}.csv", index=False)
    (outdir / f"{stem}.md").write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


def load_stage_b_predictions(run_dir: Path) -> pd.DataFrame:
    readout = run_dir / "classifier_only_readout"
    path = readout / "classifier_sweep_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"classifier_sweep_predictions.csv not found: {path}")
    df = pd.read_csv(path)
    mask = df["model_name"].astype(str).str.startswith(STGB_MODEL)
    if "threshold_strategy" in df.columns:
        mask &= df["threshold_strategy"].astype(str).eq(STGB_THRESHOLD_STRATEGY)
    return df[mask].copy()


def per_fold_score_distribution(sb: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for fold in FOLDS:
        fsub = sb[sb["fold"] == fold]
        for dx_label, dx_code in [("CN", 0), ("AD", 1)]:
            scores = fsub[fsub["y_true"] == dx_code]["y_score"].values
            if len(scores) == 0:
                continue
            row: Dict[str, Any] = {
                "fold": fold,
                "diagnosis": dx_label,
                "n": len(scores),
                "min": float(np.min(scores)),
                "p10": float(np.percentile(scores, 10)),
                "p25": float(np.percentile(scores, 25)),
                "median": float(np.median(scores)),
                "p75": float(np.percentile(scores, 75)),
                "p90": float(np.percentile(scores, 90)),
                "max": float(np.max(scores)),
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores, ddof=1)),
                "score_range": float(np.max(scores) - np.min(scores)),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def per_fold_metrics(sb: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for fold in FOLDS:
        fsub = sb[sb["fold"] == fold]
        y_true = fsub["y_true"].values
        y_score = fsub["y_score"].values
        if len(np.unique(y_true)) < 2:
            continue
        auc = float(roc_auc_score(y_true, y_score))
        pr_auc = float(average_precision_score(y_true, y_score))
        ad_scores = y_score[y_true == 1]
        cn_scores = y_score[y_true == 0]
        thr_vals = fsub["threshold"].values if "threshold" in fsub.columns else []
        thr = float(np.unique(thr_vals)[0]) if len(np.unique(thr_vals)) == 1 else float("nan")
        full_scores = y_score
        score_range = float(np.max(full_scores) - np.min(full_scores))
        rows.append({
            "fold": fold,
            "n_total": len(fsub),
            "n_AD": len(ad_scores),
            "n_CN": len(cn_scores),
            "auc": auc,
            "pr_auc": pr_auc,
            "score_range": score_range,
            "score_min": float(np.min(full_scores)),
            "score_max": float(np.max(full_scores)),
            "cn_median": float(np.median(cn_scores)),
            "ad_median": float(np.median(ad_scores)),
            "threshold": thr,
            "predicted_ad_rate": float((fsub["y_pred"] == 1).mean()) if "y_pred" in fsub.columns else float("nan"),
        })
    return pd.DataFrame(rows)


def pairwise_concordance(ad_scores: np.ndarray, cn_scores: np.ndarray) -> float:
    n_correct = 0.0
    n_total = 0
    for a in ad_scores:
        n_correct += float((cn_scores < a).sum()) + 0.5 * float((cn_scores == a).sum())
        n_total += len(cn_scores)
    return n_correct / n_total if n_total > 0 else float("nan")


def build_concordance_decomposition(sb: pd.DataFrame) -> pd.DataFrame:
    fold_data: Dict[int, Any] = {}
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
            rows.append({
                "AD_fold": k, "CN_fold": j,
                "n_AD": len(ad_k), "n_CN": len(cn_j),
                "n_pairs": len(ad_k) * len(cn_j),
                "concordance": conc,
                "pair_type": "within-fold" if k == j else "cross-fold",
            })
    return pd.DataFrame(rows)


def rank_normalize(df: pd.DataFrame, score_col: str = "y_score") -> pd.Series:
    out = df[score_col].copy().astype(float)
    for fold in df["fold"].unique():
        mask = df["fold"] == fold
        n = int(mask.sum())
        ranks = df.loc[mask, score_col].rank(method="average")
        out.loc[mask] = (ranks - 0.5) / n
    return out


def build_pooled_vs_foldwise(sb: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    y_true_all = sb["y_true"].values
    y_score_all = sb["y_score"].values

    # per-fold
    fw_aucs: List[float] = []
    fw_praucs: List[float] = []
    for fold in FOLDS:
        fsub = sb[sb["fold"] == fold]
        yt, ys = fsub["y_true"].values, fsub["y_score"].values
        if len(np.unique(yt)) < 2:
            continue
        fa = float(roc_auc_score(yt, ys))
        fp = float(average_precision_score(yt, ys))
        fw_aucs.append(fa)
        fw_praucs.append(fp)
        rows.append({"variant": f"Stage_B_fold{fold}", "auc": fa, "pr_auc": fp, "note": "per-fold"})

    rows.append({"variant": "Stage_B_foldwise_mean", "auc": float(np.mean(fw_aucs)), "pr_auc": float(np.mean(fw_praucs)), "note": "unweighted mean"})

    pooled_auc = float(roc_auc_score(y_true_all, y_score_all))
    pooled_pr = float(average_precision_score(y_true_all, y_score_all))
    rows.append({"variant": "Stage_B_pooled", "auc": pooled_auc, "pr_auc": pooled_pr, "note": "canonical promotion metric"})

    no_fold1 = sb[sb["fold"] != 1]
    if len(np.unique(no_fold1["y_true"].values)) >= 2:
        excl_auc = float(roc_auc_score(no_fold1["y_true"].values, no_fold1["y_score"].values))
        excl_pr = float(average_precision_score(no_fold1["y_true"].values, no_fold1["y_score"].values))
        rows.append({"variant": "Stage_B_pooled_excl_fold1", "auc": excl_auc, "pr_auc": excl_pr, "note": "diagnostic only"})

    sb2 = sb.copy()
    sb2["y_score_rank"] = rank_normalize(sb2)
    rn_auc = float(roc_auc_score(sb2["y_true"].values, sb2["y_score_rank"].values))
    rn_pr = float(average_precision_score(sb2["y_true"].values, sb2["y_score_rank"].values))
    rows.append({"variant": "Stage_B_pooled_rank_norm_DESCRIPTIVE", "auc": rn_auc, "pr_auc": rn_pr, "note": "DESCRIPTIVE ONLY — rank norm uses test statistics"})

    sb3 = sb.copy()
    zscore = sb3["y_score"].copy().astype(float)
    for fold in sb3["fold"].unique():
        mask = sb3["fold"] == fold
        s = sb3.loc[mask, "y_score"]
        mu, sigma = s.mean(), s.std(ddof=1)
        zscore.loc[mask] = (s - mu) / (sigma + 1e-9)
    sb3["y_score_z"] = zscore
    zn_auc = float(roc_auc_score(sb3["y_true"].values, sb3["y_score_z"].values))
    zn_pr = float(average_precision_score(sb3["y_true"].values, sb3["y_score_z"].values))
    rows.append({"variant": "Stage_B_pooled_zscore_norm_DESCRIPTIVE", "auc": zn_auc, "pr_auc": zn_pr, "note": "DESCRIPTIVE ONLY — z-score norm uses test statistics"})

    return pd.DataFrame(rows)


def build_beta_comparison_summary(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"metric": "Fold 1 score range", "beta2p5": BETA2P5_FOLD1_RANGE, "beta3p75": None, "note": "see fold_metrics"},
        {"metric": "Folds 2-5 score range (mean)", "beta2p5": BETA2P5_FOLDS25_RANGE, "beta3p75": None, "note": ""},
        {"metric": "Pooled AUC", "beta2p5": BETA2P5_POOLED_AUC, "beta3p75": None, "note": ""},
        {"metric": "Pooled PR-AUC", "beta2p5": BETA2P5_POOLED_PR_AUC, "beta3p75": None, "note": ""},
        {"metric": "Foldwise mean AUC", "beta2p5": BETA2P5_FOLDWISE_AUC, "beta3p75": None, "note": ""},
        {"metric": "Foldwise mean PR-AUC", "beta2p5": BETA2P5_FOLDWISE_PR_AUC, "beta3p75": None, "note": ""},
    ]
    if not fold_metrics.empty:
        f1 = fold_metrics[fold_metrics["fold"] == 1]
        f25 = fold_metrics[fold_metrics["fold"] != 1]
        if not f1.empty:
            rows[0]["beta3p75"] = float(f1["score_range"].iloc[0])
        if not f25.empty:
            rows[1]["beta3p75"] = float(f25["score_range"].mean())
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    run_dir = resolve(args.run_dir) if not args.run_dir.is_absolute() else args.run_dir
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir

    print(f"Score-scale audit: beta3p75 latent384")
    print(f"Run dir: {run_dir}")
    pred_path = run_dir / "classifier_only_readout" / "classifier_sweep_predictions.csv"
    print(f"Predictions: {pred_path}  ({'exists' if pred_path.exists() else 'NOT FOUND'})")

    if args.dry_run:
        print("\nDry-run complete. No artifact files read.")
        return 0

    outdir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    if outdir.exists() and not args.overwrite:
        print(f"Output dir exists; pass --overwrite: {outdir}")
        return 1
    outdir.mkdir(parents=True, exist_ok=True)

    sb = load_stage_b_predictions(run_dir)
    print(f"Loaded {len(sb)} prediction rows")

    dist_df = per_fold_score_distribution(sb)
    fold_metrics = per_fold_metrics(sb)
    pooled_vs_fw = build_pooled_vs_foldwise(sb)
    concordance = build_concordance_decomposition(sb)
    beta_cmp = build_beta_comparison_summary(fold_metrics)

    write_table(outdir, "score_distribution_by_fold", dist_df)
    write_table(outdir, "per_fold_metrics", fold_metrics)
    write_table(outdir, "pooled_vs_foldwise_metrics", pooled_vs_fw)
    write_table(outdir, "concordance_decomposition", concordance, max_rows=30)
    write_table(outdir, "beta_comparison_summary", beta_cmp)

    pooled_row = pooled_vs_fw[pooled_vs_fw["variant"] == "Stage_B_pooled"]
    fw_row = pooled_vs_fw[pooled_vs_fw["variant"] == "Stage_B_foldwise_mean"]
    pooled_auc = float(pooled_row["auc"].iloc[0]) if not pooled_row.empty else float("nan")
    pooled_pr = float(pooled_row["pr_auc"].iloc[0]) if not pooled_row.empty else float("nan")
    fw_auc = float(fw_row["auc"].iloc[0]) if not fw_row.empty else float("nan")
    fw_pr = float(fw_row["pr_auc"].iloc[0]) if not fw_row.empty else float("nan")

    f1_range = float(fold_metrics[fold_metrics["fold"] == 1]["score_range"].iloc[0]) if not fold_metrics.empty else float("nan")
    f25_range = float(fold_metrics[fold_metrics["fold"] != 1]["score_range"].mean()) if not fold_metrics.empty else float("nan")

    now = datetime.now(timezone.utc).isoformat()
    report_lines = [
        "# Stage B Score Scale Audit — beta3p75 latent384",
        "",
        f"Generated: {now}",
        "",
        "## Key Numbers",
        "| Quantity | beta2p5 | beta3p75 | Delta |",
        "|---|---|---|---|",
        f"| Fold 1 score range | {BETA2P5_FOLD1_RANGE:.3f} | {f1_range:.3f} | {f1_range - BETA2P5_FOLD1_RANGE:+.3f} |",
        f"| Folds 2-5 range (mean) | {BETA2P5_FOLDS25_RANGE:.3f} | {f25_range:.3f} | {f25_range - BETA2P5_FOLDS25_RANGE:+.3f} |",
        f"| Pooled AUC | {BETA2P5_POOLED_AUC:.4f} | {pooled_auc:.4f} | {pooled_auc - BETA2P5_POOLED_AUC:+.4f} |",
        f"| Pooled PR-AUC | {BETA2P5_POOLED_PR_AUC:.4f} | {pooled_pr:.4f} | {pooled_pr - BETA2P5_POOLED_PR_AUC:+.4f} |",
        f"| Foldwise mean AUC | {BETA2P5_FOLDWISE_AUC:.4f} | {fw_auc:.4f} | {fw_auc - BETA2P5_FOLDWISE_AUC:+.4f} |",
        f"| Foldwise mean PR-AUC | {BETA2P5_FOLDWISE_PR_AUC:.4f} | {fw_pr:.4f} | {fw_pr - BETA2P5_FOLDWISE_PR_AUC:+.4f} |",
        "",
        "## Fold 1 Anomaly Assessment",
        f"beta2.5 Fold 1 range: {BETA2P5_FOLD1_RANGE:.3f}  |  beta3.75 Fold 1 range: {f1_range:.3f}",
        f"Reduction: {BETA2P5_FOLD1_RANGE - f1_range:+.3f} ({'reduced' if f1_range < BETA2P5_FOLD1_RANGE else 'not reduced'})",
        "",
        "## Promotion Check",
        f"- Pooled AUC = {pooled_auc:.4f}  (threshold: > {PROMOTION_AUC} -> {'PASS' if pooled_auc > PROMOTION_AUC else 'FAIL'})",
        f"- Pooled PR-AUC = {pooled_pr:.4f}  (threshold: >= {PROMOTION_PR_AUC} -> {'PASS' if pooled_pr >= PROMOTION_PR_AUC else 'FAIL'})",
        "",
        "## Note on descriptive metrics",
        "Rank-normalized and z-score-normalized AUCs are DESCRIPTIVE ONLY.",
        "Normalization uses test-fold statistics — not inner-CV parameters.",
        "They cannot substitute for the pooled metric in promotion decisions.",
        "",
        "## Read-only guarantee",
        "Did not train, fit thresholds, modify tensors, metadata, ledger, or model outputs.",
    ]
    (outdir / "final_diagnosis.md").write_text("\n".join(report_lines), encoding="utf-8")

    cmd_log = {
        "created_utc": now,
        "run_name": "recover035_latent384_beta3p75_T80_h10000_p560_full5x5",
        "rank_norm_is_descriptive_only": True,
        "zscore_norm_is_descriptive_only": True,
        "training_launched": False,
        "threshold_fitting_performed": False,
    }
    (outdir / "command_log.json").write_text(json.dumps(cmd_log, indent=2), encoding="utf-8")

    print(f"\nPooled AUC={pooled_auc:.4f}, PR-AUC={pooled_pr:.4f}")
    print(f"Foldwise mean AUC={fw_auc:.4f}, PR-AUC={fw_pr:.4f}")
    print(f"Fold 1 range: beta2.5={BETA2P5_FOLD1_RANGE:.3f} -> beta3.75={f1_range:.3f}")
    print(f"Output: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
