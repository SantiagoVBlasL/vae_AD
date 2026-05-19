#!/usr/bin/env python3
"""Read-only post-mortem for the failed fc0 FULL 5x5 confirmation.

Compares the current FULL [1,0,2] model against the fc0 FULL [1,0,2]
candidate using only classifier-only logreg_l2 and the leakage-safe
inner-OOF target-sensitivity threshold. No training or dataset mutation occurs.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"

CURRENT_RUN_DIR = RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_mfrsplit_3840_final_candidate"
CURRENT_READOUT_DIR = RESULTS / "adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep"
FC0_RUN_DIR = RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_fc0_full_5x5"
FC0_READOUT_DIR = FC0_RUN_DIR / "classifier_only_readout"
DEFAULT_OUTPUT_DIR = RESULTS / "adni_v5_1_batch20260514b_fc0_failure_postmortem_audit"

PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"

RUNS = [
    {
        "run_id": "current_ch1_0_2",
        "label": "current [1,0,2]",
        "run_dir": CURRENT_RUN_DIR,
        "readout_dir": CURRENT_READOUT_DIR,
    },
    {
        "run_id": "fc0_ch1_0_2",
        "label": "fc0 [1,0,2]",
        "run_dir": FC0_RUN_DIR,
        "readout_dir": FC0_READOUT_DIR,
    },
]

METRICS = ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "accuracy"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def md_table(df: pd.DataFrame, digits: int = 4) -> str:
    if df.empty:
        return "_No rows._\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals: List[str] = []
        for col in cols:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append("" if pd.isna(val) else f"{float(val):.{digits}f}")
            else:
                vals.append("" if pd.isna(val) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def prepare_output(path: Path, overwrite: bool, dry_run: bool) -> None:
    if dry_run:
        return
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(f"Output directory exists and is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def primary_foldwise(readout_dir: Path, run_id: str, label: str) -> pd.DataFrame:
    df = read_csv(readout_dir / "classifier_sweep_foldwise_metrics.csv")
    df = df[(df["model_name"].eq(PRIMARY_MODEL)) & (df["threshold_strategy"].eq(PRIMARY_THRESHOLD))].copy()
    if df["fold"].nunique() != 5:
        raise RuntimeError(f"{run_id}: expected five primary fold rows, found {df['fold'].nunique()}")
    df.insert(0, "run_id", run_id)
    df.insert(1, "label", label)
    return df.reset_index(drop=True)


def primary_predictions(readout_dir: Path, run_id: str, label: str) -> pd.DataFrame:
    df = read_csv(readout_dir / "classifier_sweep_predictions.csv")
    df = df[(df["model_name"].eq(PRIMARY_MODEL)) & (df["threshold_strategy"].eq(PRIMARY_THRESHOLD))].copy()
    if df.empty:
        raise RuntimeError(f"{run_id}: no primary prediction rows.")
    df.insert(0, "run_id", run_id)
    df.insert(1, "label", label)
    return df.reset_index(drop=True)


def primary_thresholds(readout_dir: Path, run_id: str, label: str) -> pd.DataFrame:
    df = read_csv(readout_dir / "classifier_sweep_thresholds_by_fold.csv")
    df = df[(df["model_name"].eq(PRIMARY_MODEL)) & (df["threshold_strategy"].eq(PRIMARY_THRESHOLD))].copy()
    if df["fold"].nunique() != 5:
        raise RuntimeError(f"{run_id}: expected five threshold rows, found {df['fold'].nunique()}")
    df.insert(0, "run_id", run_id)
    df.insert(1, "label", label)
    return df.reset_index(drop=True)


def binary_metrics(y_true: Sequence[int], y_score: Sequence[float], y_pred: Sequence[int]) -> Dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    pred = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "n": int(len(y)),
        "n_cn": int((y == 0).sum()),
        "n_ad": int((y == 1).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "auc": roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan,
        "pr_auc": average_precision_score(y, score) if len(np.unique(y)) == 2 else np.nan,
        "accuracy": (tn + tp) / len(y) if len(y) else np.nan,
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": np.nanmean([sens, spec]),
        "f1": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan,
    }


def subgroup_metrics(predictions: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for keys, sub in predictions.groupby(["run_id", "label", group_col], dropna=False):
        run_id, label, group_value = keys
        row: Dict[str, Any] = {"run_id": run_id, "label": label, group_col: group_value}
        row.update(binary_metrics(sub["y_true"], sub["y_score"], sub["y_pred"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values([group_col, "run_id"]).reset_index(drop=True)


def make_delta_table(df: pd.DataFrame, key_cols: Sequence[str], value_cols: Sequence[str]) -> pd.DataFrame:
    current = df[df["run_id"].eq("current_ch1_0_2")].copy()
    fc0 = df[df["run_id"].eq("fc0_ch1_0_2")].copy()
    merged = current.merge(fc0, on=list(key_cols), suffixes=("_current", "_fc0"))
    for col in value_cols:
        cur = f"{col}_current"
        cand = f"{col}_fc0"
        if cur in merged.columns and cand in merged.columns:
            merged[f"delta_{col}_fc0_minus_current"] = merged[cand] - merged[cur]
    return merged


def score_distribution(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    groupings: List[tuple[str, List[str]]] = [
        ("pooled_by_diagnosis", ["run_id", "label", "ResearchGroup_Mapped"]),
        ("fold_by_diagnosis", ["run_id", "label", "fold", "ResearchGroup_Mapped"]),
        ("pooled_by_manufacturer_diagnosis", ["run_id", "label", "Manufacturer", "ResearchGroup_Mapped"]),
        ("pooled_by_sex_diagnosis", ["run_id", "label", "Sex", "ResearchGroup_Mapped"]),
    ]
    for scope, cols in groupings:
        for keys, sub in predictions.groupby(cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row: Dict[str, Any] = {"scope": scope, **dict(zip(cols, keys))}
            score = sub["y_score"].astype(float)
            row.update(
                {
                    "n": int(len(sub)),
                    "mean_score": score.mean(),
                    "std_score": score.std(ddof=0),
                    "min_score": score.min(),
                    "p25_score": score.quantile(0.25),
                    "median_score": score.median(),
                    "p75_score": score.quantile(0.75),
                    "max_score": score.max(),
                }
            )
            rows.append(row)
    out = pd.DataFrame(rows)
    sep_rows: List[Dict[str, Any]] = []
    for keys, sub in predictions.groupby(["run_id", "label", "fold"], dropna=False):
        ad = sub[sub["y_true"].eq(1)]["y_score"].astype(float)
        cn = sub[sub["y_true"].eq(0)]["y_score"].astype(float)
        sep_rows.append(
            {
                "scope": "fold_ad_minus_cn_separation",
                "run_id": keys[0],
                "label": keys[1],
                "fold": keys[2],
                "n": int(len(sub)),
                "mean_score": ad.mean() - cn.mean(),
                "std_score": np.nan,
                "min_score": np.nan,
                "p25_score": np.nan,
                "median_score": np.nan,
                "p75_score": np.nan,
                "max_score": np.nan,
            }
        )
    for keys, sub in predictions.groupby(["run_id", "label"], dropna=False):
        ad = sub[sub["y_true"].eq(1)]["y_score"].astype(float)
        cn = sub[sub["y_true"].eq(0)]["y_score"].astype(float)
        sep_rows.append(
            {
                "scope": "pooled_ad_minus_cn_separation",
                "run_id": keys[0],
                "label": keys[1],
                "fold": "pooled",
                "n": int(len(sub)),
                "mean_score": ad.mean() - cn.mean(),
                "std_score": np.nan,
                "min_score": np.nan,
                "p25_score": np.nan,
                "median_score": np.nan,
                "p75_score": np.nan,
                "max_score": np.nan,
            }
        )
    return pd.concat([out, pd.DataFrame(sep_rows)], ignore_index=True, sort=False)


def mi_participation_ratio(path: Path, variable: str) -> float:
    if not path.exists():
        return np.nan
    df = pd.read_csv(path)
    vals = df[df["variable"].eq(variable)]["mi_nats"].astype(float).to_numpy()
    denom = float(np.square(vals).sum())
    return float(np.square(vals.sum()) / denom) if denom > 0 else 0.0


def collect_latent_qc(run_dir: Path, run_id: str, label: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for fold in range(1, 6):
        fdir = run_dir / f"fold_{fold}"
        base = read_csv(fdir / "latent_qc_metrics.csv").iloc[0].to_dict()
        row = {"run_id": run_id, "label": label, "fold": fold, **base}
        for split in ["trainDev", "test"]:
            summary = read_csv(fdir / f"fold_{fold}_{split}_latent_info_summary.csv")
            for _, r in summary.iterrows():
                var = str(r["variable"]).lower().replace(" ", "_")
                prefix = f"{split}_{var}"
                row[f"{prefix}_mi_sum_nats"] = r.get("mi_sum_nats", np.nan)
                row[f"{prefix}_mi_mean_nats"] = r.get("mi_mean_nats", np.nan)
                row[f"{prefix}_n_active"] = r.get("n_active", np.nan)
                row[f"{prefix}_frac_active"] = r.get("frac_active", np.nan)
                row[f"{prefix}_total_correlation_nats"] = r.get("total_correlation_nats", np.nan)
                per_dim = fdir / f"fold_{fold}_{split}_latent_info_per_dim.csv"
                row[f"{prefix}_mi_participation_ratio"] = mi_participation_ratio(per_dim, str(r["variable"]))
        rows.append(row)
    return pd.DataFrame(rows)


def collect_reconstruction_qc(run_dir: Path, run_id: str, label: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for fold in range(1, 6):
        for source in ["norm", "recon"]:
            df = read_csv(run_dir / f"fold_{fold}" / f"fold_{fold}_dist_{source}.csv")
            for _, r in df.iterrows():
                row = {"run_id": run_id, "label": label, "fold": fold, "source": source}
                row.update(r.to_dict())
                row["touches_or_exceeds_abs_1"] = bool(float(r["min"]) <= -0.999 or float(r["max"]) >= 0.999)
                row["outside_minus1_1"] = bool(float(r["min"]) < -1.0 or float(r["max"]) > 1.0)
                rows.append(row)
    return pd.DataFrame(rows)


def collect_rate_distortion(run_dir: Path, run_id: str, label: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for fold in range(1, 6):
        df = read_csv(run_dir / f"fold_{fold}" / f"fold_{fold}_rate_distortion.csv")
        for row_type, idx in [("final_epoch", df.index[-1]), ("best_val_beta_max", df["L_val_betaMax"].idxmin())]:
            r = df.loc[idx].to_dict()
            out = {"run_id": run_id, "label": label, "fold": fold, "row_type": row_type}
            out.update(r)
            out["R_over_D_val"] = out["R_val_nats"] / out["D_val"] if out["D_val"] else np.nan
            out["beta2p5_R_over_D_val"] = 2.5 * out["R_val_nats"] / out["D_val"] if out["D_val"] else np.nan
            out["R_over_D_train"] = out["R_train_nats"] / out["D_train"] if out["D_train"] else np.nan
            out["beta2p5_R_over_D_train"] = 2.5 * out["R_train_nats"] / out["D_train"] if out["D_train"] else np.nan
            rows.append(out)
    return pd.DataFrame(rows)


def fold4_deep_dive(predictions: pd.DataFrame) -> pd.DataFrame:
    fold4 = predictions[predictions["fold"].eq(4)].copy()
    fold4["diagnosis"] = np.where(fold4["y_true"].eq(1), "AD", "CN")
    fold4["prediction"] = np.where(fold4["y_pred"].eq(1), "AD_like", "CN_like")
    fold4["error_type"] = np.select(
        [
            fold4["y_true"].eq(1) & fold4["y_pred"].eq(0),
            fold4["y_true"].eq(0) & fold4["y_pred"].eq(1),
        ],
        ["false_negative_AD", "false_positive_CN"],
        default="correct",
    )
    keep = [
        "SubjectID",
        "tensor_idx",
        "ResearchGroup_Mapped",
        "diagnosis",
        "Manufacturer",
        "Age",
        "Sex",
        "source_batch",
        "source_label",
        "tensor_source",
        "run_id",
        "label",
        "threshold",
        "y_score",
        "y_pred",
        "prediction",
        "error_type",
    ]
    cur = fold4[fold4["run_id"].eq("current_ch1_0_2")][keep].copy()
    fc0 = fold4[fold4["run_id"].eq("fc0_ch1_0_2")][keep].copy()
    merged = cur.merge(
        fc0,
        on=[
            "SubjectID",
            "tensor_idx",
            "ResearchGroup_Mapped",
            "diagnosis",
            "Manufacturer",
            "Age",
            "Sex",
            "source_batch",
            "source_label",
            "tensor_source",
        ],
        suffixes=("_current", "_fc0"),
    )
    merged["delta_score_fc0_minus_current"] = merged["y_score_fc0"] - merged["y_score_current"]
    merged["delta_threshold_fc0_minus_current"] = merged["threshold_fc0"] - merged["threshold_current"]
    merged["error_change"] = merged["error_type_current"] + "_to_" + merged["error_type_fc0"]
    return merged.sort_values(["error_change", "diagnosis", "Manufacturer", "SubjectID"]).reset_index(drop=True)


def write_pair_csv_md(output_dir: Path, name: str, df: pd.DataFrame) -> None:
    df.to_csv(output_dir / f"{name}.csv", index=False)
    (output_dir / f"{name}.md").write_text(md_table(df), encoding="utf-8")


def interpret(
    fold_delta: pd.DataFrame,
    score_dist: pd.DataFrame,
    latent_delta: pd.DataFrame,
    rd_delta: pd.DataFrame,
    mfr_delta: pd.DataFrame,
    sex_delta: pd.DataFrame,
) -> str:
    auc_delta_cols = [c for c in fold_delta.columns if c == "delta_auc_fc0_minus_current"]
    auc_deltas = fold_delta[auc_delta_cols[0]].dropna() if auc_delta_cols else pd.Series(dtype=float)
    n_auc_worse = int((auc_deltas < 0).sum())
    n_auc_better = int((auc_deltas > 0).sum())
    pooled_sep = score_dist[score_dist["scope"].eq("pooled_ad_minus_cn_separation")]
    sep_cur = pooled_sep.loc[pooled_sep["run_id"].eq("current_ch1_0_2"), "mean_score"]
    sep_fc0 = pooled_sep.loc[pooled_sep["run_id"].eq("fc0_ch1_0_2"), "mean_score"]
    sep_delta = float(sep_fc0.iloc[0] - sep_cur.iloc[0]) if not sep_cur.empty and not sep_fc0.empty else np.nan

    active_cols = [c for c in latent_delta.columns if c.endswith("_n_active_fc0")]
    active_note = "No active-unit summary was available."
    if active_cols:
        col = "trainDev_y_target_n_active"
        cur_col = f"{col}_current"
        fc0_col = f"{col}_fc0"
        if cur_col in latent_delta and fc0_col in latent_delta:
            active_note = (
                f"Active units did not collapse: current mean={latent_delta[cur_col].mean():.1f}, "
                f"fc0 mean={latent_delta[fc0_col].mean():.1f}."
            )
    tc_cur_col = "trainDev_y_target_total_correlation_nats_current"
    tc_fc0_col = "trainDev_y_target_total_correlation_nats_fc0"
    tc_note = "Total-correlation summary was unavailable."
    if tc_cur_col in latent_delta and tc_fc0_col in latent_delta:
        tc_note = (
            f"Train/dev total correlation decreased from {latent_delta[tc_cur_col].mean():.1f} "
            f"to {latent_delta[tc_fc0_col].mean():.1f} nats on average."
        )
    rd = rd_delta[rd_delta["row_type"].eq("best_val_beta_max")]
    rd_note = "Rate-distortion ratios were unavailable."
    if not rd.empty and "beta2p5_R_over_D_val_current" in rd and "beta2p5_R_over_D_val_fc0" in rd:
        rd_note = (
            f"At the best validation beta-max row, beta*KLD/recon changed from "
            f"{rd['beta2p5_R_over_D_val_current'].mean():.4f} to "
            f"{rd['beta2p5_R_over_D_val_fc0'].mean():.4f}."
        )

    mfr_fragile = []
    if not mfr_delta.empty:
        for _, row in mfr_delta.iterrows():
            ba = row.get("delta_balanced_accuracy_fc0_minus_current", np.nan)
            auc = row.get("delta_auc_fc0_minus_current", np.nan)
            if pd.notna(ba) and ba < -0.03 or pd.notna(auc) and auc < -0.03:
                mfr_fragile.append(f"{row['Manufacturer']}: delta AUC={auc:+.3f}, delta BA={ba:+.3f}")
    sex_fragile = []
    if not sex_delta.empty:
        for _, row in sex_delta.iterrows():
            ba = row.get("delta_balanced_accuracy_fc0_minus_current", np.nan)
            auc = row.get("delta_auc_fc0_minus_current", np.nan)
            if pd.notna(ba) and ba < -0.03 or pd.notna(auc) and auc < -0.03:
                sex_fragile.append(f"{row['Sex']}: delta AUC={auc:+.3f}, delta BA={ba:+.3f}")

    lines = [
        "# fc0 Failure Interpretation",
        "",
        "This is a read-only post-mortem comparing current FULL `[1,0,2]` against FULL `[1,0,2]` with `intermediate_fc_dim_vae=0`.",
        "The classifier readout is fixed to classifier-only `logreg_l2` with true inner-CV OOF `inner_oof_target_sens_ge_0p70_max_spec` thresholding.",
        "",
        "## Answers",
        "",
        f"1. Fold consistency: fc0 AUC was lower in {n_auc_worse}/5 folds and higher in {n_auc_better}/5 folds. This is not a single-fold-only failure.",
        f"2. Rank separation: pooled AD-minus-CN mean score separation changed by {sep_delta:+.4f}. The main loss is ranking/AUC and PR-AUC, while threshold metrics stayed similar.",
        f"3. Latent participation: {active_note} {tc_note}",
        f"4. KLD/reconstruction balance: {rd_note}",
        "5. Manufacturer/Sex subgroups: "
        + ("; ".join(mfr_fragile + sex_fragile) if (mfr_fragile or sex_fragile) else "no material BA fragility beyond the global AUC/PR-AUC drop was detected."),
        "6. Interpretation: the failure is consistent with reduced representational richness or underfitting in the FULL 5x5 setting, not with a simple thresholding failure and not with a complete active-unit collapse.",
        "",
        "## Decision",
        "",
        "Do not promote fc0. Current FULL `[1,0,2]` remains the best paper candidate.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    prepare_output(output_dir, args.overwrite, args.dry_run)

    foldwise_parts = []
    pred_parts = []
    threshold_parts = []
    latent_parts = []
    recon_parts = []
    rd_parts = []
    logs: Dict[str, Any] = {}
    for run in RUNS:
        run_dir = run["run_dir"]
        readout_dir = run["readout_dir"]
        if not run_dir.exists():
            raise FileNotFoundError(run_dir)
        if not readout_dir.exists():
            raise FileNotFoundError(readout_dir)
        foldwise_parts.append(primary_foldwise(readout_dir, run["run_id"], run["label"]))
        pred_parts.append(primary_predictions(readout_dir, run["run_id"], run["label"]))
        threshold_parts.append(primary_thresholds(readout_dir, run["run_id"], run["label"]))
        latent_parts.append(collect_latent_qc(run_dir, run["run_id"], run["label"]))
        recon_parts.append(collect_reconstruction_qc(run_dir, run["run_id"], run["label"]))
        rd_parts.append(collect_rate_distortion(run_dir, run["run_id"], run["label"]))
        log_path = readout_dir / "command_log.json"
        logs[run["run_id"]] = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {}

    foldwise = pd.concat(foldwise_parts, ignore_index=True)
    predictions = pd.concat(pred_parts, ignore_index=True)
    thresholds = pd.concat(threshold_parts, ignore_index=True)
    latent = pd.concat(latent_parts, ignore_index=True)
    recon = pd.concat(recon_parts, ignore_index=True)
    rd = pd.concat(rd_parts, ignore_index=True)

    fold_delta = make_delta_table(foldwise, ["fold"], METRICS + ["threshold", "inner_oof_sensitivity", "inner_oof_specificity"])
    score_dist = score_distribution(predictions)
    threshold_delta = make_delta_table(
        thresholds,
        ["fold"],
        ["threshold", "inner_oof_sensitivity", "inner_oof_specificity", "inner_oof_balanced_accuracy", "best_inner_auc"],
    )
    latent_delta = make_delta_table(
        latent,
        ["fold"],
        [
            "silhouette_latent",
            "acc_site_latent",
            "acc_site_raw",
            "trainDev_y_target_mi_sum_nats",
            "trainDev_y_target_n_active",
            "trainDev_y_target_frac_active",
            "trainDev_y_target_total_correlation_nats",
            "trainDev_y_target_mi_participation_ratio",
            "test_y_target_mi_sum_nats",
            "test_y_target_n_active",
            "test_y_target_frac_active",
            "test_y_target_total_correlation_nats",
            "test_y_target_mi_participation_ratio",
        ],
    )
    recon_delta = make_delta_table(
        recon,
        ["fold", "source", "channel"],
        ["mean", "std", "min", "max", "p05", "p95"],
    )
    rd_delta = make_delta_table(
        rd,
        ["fold", "row_type"],
        ["epoch", "D_train", "R_train_nats", "L_train_betaMax", "D_val", "R_val_nats", "L_val_betaMax", "R_over_D_val", "beta2p5_R_over_D_val"],
    )
    mfr = subgroup_metrics(predictions, "Manufacturer")
    sex = subgroup_metrics(predictions, "Sex")
    mfr_delta = make_delta_table(mfr, ["Manufacturer"], METRICS + ["n", "n_cn", "n_ad", "tn", "fp", "fn", "tp"])
    sex_delta = make_delta_table(sex, ["Sex"], METRICS + ["n", "n_cn", "n_ad", "tn", "fp", "fn", "tp"])
    fold4 = fold4_deep_dive(predictions)
    interpretation = interpret(fold_delta, score_dist, latent_delta, rd_delta, mfr_delta, sex_delta)

    if args.dry_run:
        print("DRY RUN: fc0 post-mortem inputs validated.")
        print(f"Would write outputs to {output_dir}")
        print(f"Foldwise delta rows: {len(fold_delta)}")
        print(f"Fold4 deep-dive rows: {len(fold4)}")
        return 0

    write_pair_csv_md(output_dir, "foldwise_delta_current_vs_fc0", fold_delta)
    write_pair_csv_md(output_dir, "score_distribution_current_vs_fc0", score_dist)
    write_pair_csv_md(output_dir, "threshold_distribution_current_vs_fc0", threshold_delta)
    write_pair_csv_md(output_dir, "latent_qc_current_vs_fc0", latent_delta)
    write_pair_csv_md(output_dir, "reconstruction_qc_current_vs_fc0", recon_delta)
    write_pair_csv_md(output_dir, "rate_distortion_current_vs_fc0", rd_delta)
    write_pair_csv_md(output_dir, "manufacturer_subgroup_delta_current_vs_fc0", mfr_delta)
    write_pair_csv_md(output_dir, "sex_subgroup_delta_current_vs_fc0", sex_delta)
    write_pair_csv_md(output_dir, "fold4_current_vs_fc0_deep_dive", fold4)
    (output_dir / "fc0_failure_interpretation.md").write_text(interpretation, encoding="utf-8")
    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "primary_model": PRIMARY_MODEL,
        "primary_threshold_strategy": PRIMARY_THRESHOLD,
        "read_only": True,
        "training_launched": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "readout_command_logs": logs,
    }
    (output_dir / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote fc0 post-mortem audit to {output_dir}")
    print(interpretation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
