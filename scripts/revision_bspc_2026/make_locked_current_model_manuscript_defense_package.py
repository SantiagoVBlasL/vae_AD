#!/usr/bin/env python3
"""Create a read-only manuscript defense package for locked FULL [1,0,2].

The package consolidates existing locked-model predictions and prior negative
optimization audits. It does not train, modify tensors, metadata, ledgers,
configs, or any existing run outputs.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results" / "revision_bspc_2026"
OUTDIR = RESULTS_ROOT / "adni_v5_1_batch20260514b_manuscript_defense_locked_current_model"

READOUT_DIR = RESULTS_ROOT / "adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep"
SITE_AUDIT_DIR = RESULTS_ROOT / "adni_v5_1_batch20260514b_locked_subject_error_qc_site_audit"
REVISION_PACKAGE_DIR = RESULTS_ROOT / "adni_v5_1_batch20260514b_revision_package_locked_current_model"
SCANNER_QC = RESULTS_ROOT / "adni_v5_1_batch20260514b_canonical_reconciliation_capacity_audit" / "current_full_scanner_leakage_qc.csv"

FC0_MAIN = RESULTS_ROOT / "adni_v5_1_batch20260514b_ch1_0_2_fc0_full_5x5_comparison" / "fc0_full_main_comparison.csv"
ACT_NONE_MAIN = (
    RESULTS_ROOT
    / "adni_v5_1_batch20260514b_ch1_0_2_final_activation_none_full_5x5_comparison"
    / "final_activation_none_full_main_comparison.csv"
)
MFR_SAMPLER_MAIN = (
    RESULTS_ROOT
    / "adni_v5_1_batch20260514b_ch1_0_2_manufacturer_balanced_sampler_full_5x5_comparison"
    / "main_model_comparison.csv"
)
AUC_SPRINT_FAST = RESULTS_ROOT / "adni_v5_1_batch20260514b_auc_sprint_v2_representation_fast3x3" / "final_fast_decision_table.csv"
CLASSIFIER_SELECTION = RESULTS_ROOT / "adni_v5_1_batch20260514b_frozen_latent_classifier_selection_audit" / "primary_model_comparison.csv"

PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
FIXED_THRESHOLD = "fixed_0p5"
RNG_SEED = 20260519
N_BOOT = 2000


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fmt(x: Any, digits: int = 4) -> str:
    try:
        if pd.isna(x):
            return ""
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def md_table(df: pd.DataFrame, max_rows: int = 120) -> str:
    if df.empty:
        return "_No rows._\n"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(fmt)
    text = view.to_markdown(index=False)
    if len(df) > max_rows:
        text += f"\n\n_Showing {max_rows} of {len(df)} rows._"
    return text + "\n"


def write_pair(stem: str, df: pd.DataFrame, max_rows: int = 120) -> None:
    df.to_csv(OUTDIR / f"{stem}.csv", index=False)
    (OUTDIR / f"{stem}.md").write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def binary_metrics(y_true: Iterable[int], y_score: Iterable[float], y_pred: Iterable[int]) -> Dict[str, Any]:
    y = np.asarray(list(y_true), dtype=int)
    s = np.asarray(list(y_score), dtype=float)
    p = np.asarray(list(y_pred), dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "n_cn": int((y == 0).sum()),
        "n_ad": int((y == 1).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": safe_div(tp + tn, len(y)),
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "balanced_accuracy": float(np.nanmean([safe_div(tp, tp + fn), safe_div(tn, tn + fp)])),
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
        "predicted_ad_rate": float(p.mean()) if len(p) else float("nan"),
        "auc": float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan"),
        "pr_auc": float(average_precision_score(y, s)) if len(np.unique(y)) == 2 else float("nan"),
        "brier": float(brier_score_loss(y, np.clip(s, 0.0, 1.0))),
    }


def load_predictions() -> pd.DataFrame:
    path = READOUT_DIR / "classifier_sweep_predictions.csv"
    preds = read_csv(path)
    if preds.empty:
        raise FileNotFoundError(f"Missing predictions: {path}")
    preds = preds[preds["model_name"].eq(PRIMARY_MODEL)].copy()
    if preds.empty:
        raise RuntimeError(f"No predictions for {PRIMARY_MODEL}")
    return preds


def primary_predictions(preds: pd.DataFrame) -> pd.DataFrame:
    primary = preds[preds["threshold_strategy"].eq(PRIMARY_THRESHOLD)].copy()
    if primary.empty:
        raise RuntimeError(f"No primary predictions for threshold {PRIMARY_THRESHOLD}")
    primary["SiteCode"] = primary["SubjectID"].astype(str).str.extract(r"^(\d{3})", expand=False).fillna("UNKNOWN")
    return primary


def final_locked_model_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for strategy, sub in preds.groupby("threshold_strategy", dropna=False):
        row = {"model": PRIMARY_MODEL, "threshold_strategy": strategy, "threshold": "fold_specific" if strategy != FIXED_THRESHOLD else 0.5}
        row.update(binary_metrics(sub["y_true"], sub["y_score"], sub["y_pred"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("threshold_strategy")


def confusion_table(primary: pd.DataFrame) -> pd.DataFrame:
    tn, fp, fn, tp = confusion_matrix(primary["y_true"], primary["y_pred"], labels=[0, 1]).ravel()
    return pd.DataFrame(
        [
            {"true_label": "CN", "predicted_CN": int(tn), "predicted_AD": int(fp), "n": int(tn + fp)},
            {"true_label": "AD", "predicted_CN": int(fn), "predicted_AD": int(tp), "n": int(fn + tp)},
        ]
    )


def stratified_bootstrap_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    neg = np.where(y == 0)[0]
    pos = np.where(y == 1)[0]
    return np.concatenate([rng.choice(neg, size=len(neg), replace=True), rng.choice(pos, size=len(pos), replace=True)])


def roc_pr_bootstrap(primary: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = primary["y_true"].astype(int).to_numpy()
    s = primary["y_score"].astype(float).to_numpy()
    rng = np.random.default_rng(RNG_SEED)
    fpr_grid = np.linspace(0.0, 1.0, 201)
    recall_grid = np.linspace(0.0, 1.0, 201)
    aucs: List[float] = []
    aps: List[float] = []
    tpr_boot: List[np.ndarray] = []
    prec_boot: List[np.ndarray] = []
    for _ in range(N_BOOT):
        idx = stratified_bootstrap_indices(y, rng)
        yb = y[idx]
        sb = s[idx]
        aucs.append(float(roc_auc_score(yb, sb)))
        aps.append(float(average_precision_score(yb, sb)))
        fpr, tpr, _ = roc_curve(yb, sb)
        tpr_boot.append(np.interp(fpr_grid, fpr, tpr))
        precision, recall, _ = precision_recall_curve(yb, sb)
        order = np.argsort(recall)
        prec_boot.append(np.interp(recall_grid, recall[order], precision[order]))

    fpr, tpr, _ = roc_curve(y, s)
    precision, recall, _ = precision_recall_curve(y, s)
    roc_df = pd.DataFrame(
        {
            "fpr": fpr_grid,
            "tpr": np.interp(fpr_grid, fpr, tpr),
            "tpr_ci_low": np.percentile(np.vstack(tpr_boot), 2.5, axis=0),
            "tpr_ci_high": np.percentile(np.vstack(tpr_boot), 97.5, axis=0),
        }
    )
    order = np.argsort(recall)
    pr_df = pd.DataFrame(
        {
            "recall": recall_grid,
            "precision": np.interp(recall_grid, recall[order], precision[order]),
            "precision_ci_low": np.percentile(np.vstack(prec_boot), 2.5, axis=0),
            "precision_ci_high": np.percentile(np.vstack(prec_boot), 97.5, axis=0),
        }
    )
    summary = pd.DataFrame(
        [
            {
                "metric": "ROC-AUC",
                "estimate": float(roc_auc_score(y, s)),
                "bootstrap_ci_low": float(np.percentile(aucs, 2.5)),
                "bootstrap_ci_high": float(np.percentile(aucs, 97.5)),
                "n_bootstrap": N_BOOT,
            },
            {
                "metric": "PR-AUC",
                "estimate": float(average_precision_score(y, s)),
                "bootstrap_ci_low": float(np.percentile(aps, 2.5)),
                "bootstrap_ci_high": float(np.percentile(aps, 97.5)),
                "n_bootstrap": N_BOOT,
            },
        ]
    )
    return roc_df, pr_df, summary


def plot_roc_pr(roc_df: pd.DataFrame, pr_df: pd.DataFrame, summary: pd.DataFrame) -> None:
    auc = summary.loc[summary["metric"].eq("ROC-AUC"), "estimate"].iloc[0]
    ap = summary.loc[summary["metric"].eq("PR-AUC"), "estimate"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.plot(roc_df["fpr"], roc_df["tpr"], color="#1f77b4", lw=2, label=f"ROC-AUC={auc:.3f}")
    ax.fill_between(roc_df["fpr"], roc_df["tpr_ci_low"], roc_df["tpr_ci_high"], color="#1f77b4", alpha=0.18, label="95% bootstrap CI")
    ax.plot([0, 1], [0, 1], color="0.5", lw=1, linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Pooled OOF ROC")
    ax.legend(loc="lower right")
    ax = axes[1]
    ax.plot(pr_df["recall"], pr_df["precision"], color="#d62728", lw=2, label=f"PR-AUC={ap:.3f}")
    ax.fill_between(pr_df["recall"], pr_df["precision_ci_low"], pr_df["precision_ci_high"], color="#d62728", alpha=0.18, label="95% bootstrap CI")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Pooled OOF PR")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUTDIR / "pooled_oof_roc_pr_curves_bootstrap_ci.png", dpi=180)
    plt.close(fig)


def calibration_outputs(primary: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y = primary["y_true"].astype(int).to_numpy()
    s = np.clip(primary["y_score"].astype(float).to_numpy(), 0.0, 1.0)
    prob_true, prob_pred = calibration_curve(y, s, n_bins=10, strategy="uniform")
    rows: List[Dict[str, Any]] = []
    bins = np.linspace(0, 1, 11)
    bin_ids = np.digitize(s, bins[1:-1], right=True)
    for i in range(10):
        mask = bin_ids == i
        rows.append(
            {
                "bin": i + 1,
                "bin_low": bins[i],
                "bin_high": bins[i + 1],
                "n": int(mask.sum()),
                "mean_predicted_probability": float(s[mask].mean()) if mask.any() else np.nan,
                "observed_ad_fraction": float(y[mask].mean()) if mask.any() else np.nan,
            }
        )
    cal = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "metric": "Brier",
                "estimate": float(brier_score_loss(y, s)),
                "n": int(len(y)),
            }
        ]
    )
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", lw=1)
    ax.plot(prob_pred, prob_true, marker="o", color="#2ca02c", lw=2)
    ax.set_xlabel("Mean predicted AD probability")
    ax.set_ylabel("Observed AD fraction")
    ax.set_title(f"Calibration curve (Brier={summary['estimate'].iloc[0]:.3f})")
    fig.tight_layout()
    fig.savefig(OUTDIR / "calibration_curve_brier.png", dpi=180)
    plt.close(fig)
    return cal, summary


def manufacturer_table(primary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for manufacturer, sub in primary.groupby("Manufacturer", dropna=False):
        row = {"Manufacturer": manufacturer}
        row.update(binary_metrics(sub["y_true"], sub["y_score"], sub["y_pred"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Manufacturer")


def scanner_leakage_summary() -> pd.DataFrame:
    scanner = read_csv(SCANNER_QC)
    if scanner.empty:
        scanner = read_csv(REVISION_PACKAGE_DIR / "scanner_leakage_summary.csv")
    if scanner.empty:
        return scanner
    rows: List[Dict[str, Any]] = []
    for split, sub in scanner.groupby("split", dropna=False):
        rows.append(
            {
                "split": split,
                "n_rows": int(len(sub)),
                "mean_acc_site_raw": float(sub["acc_site_raw"].mean()),
                "mean_acc_site_latent": float(sub["acc_site_latent"].mean()),
                "mean_latent_minus_raw_site_acc": float(sub["latent_minus_raw_site_acc"].mean()),
                "max_acc_site_latent": float(sub["acc_site_latent"].max()),
                "chance_level": float(sub["chance_level"].dropna().iloc[0]) if sub["chance_level"].notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def sitecode_summary() -> pd.DataFrame:
    site = read_csv(SITE_AUDIT_DIR / "sitecode_auc_report.csv")
    if site.empty:
        return site
    summary = pd.DataFrame(
        [
            {
                "n_sites": int(len(site)),
                "sites_with_both_classes": int(site["site_class_status"].eq("both_classes").sum()),
                "sites_one_class_only": int((~site["site_class_status"].eq("both_classes")).sum()),
                "sites_auc_eligible": int(site["site_auc_eligible"].sum()),
                "eligible_site_auc_median": float(site.loc[site["site_auc_eligible"], "auc"].median()),
                "eligible_site_auc_min": float(site.loc[site["site_auc_eligible"], "auc"].min()),
                "eligible_site_auc_max": float(site.loc[site["site_auc_eligible"], "auc"].max()),
            }
        ]
    )
    return summary


def failed_optimization_table() -> pd.DataFrame:
    pooled = read_csv(READOUT_DIR / "classifier_sweep_pooled_metrics.csv")
    current = pooled[(pooled["model_name"].eq(PRIMARY_MODEL)) & (pooled["threshold_strategy"].eq(PRIMARY_THRESHOLD))].iloc[0]

    def base_row(label: str, category: str, stage: str, metrics: Dict[str, Any], decision: str) -> Dict[str, Any]:
        row = {
            "candidate": label,
            "category": category,
            "evaluation_stage": stage,
            "decision": decision,
            "auc": float(metrics.get("auc", np.nan)),
            "pr_auc": float(metrics.get("pr_auc", np.nan)),
            "balanced_accuracy": float(metrics.get("balanced_accuracy", np.nan)),
            "sensitivity": float(metrics.get("sensitivity", np.nan)),
            "specificity": float(metrics.get("specificity", np.nan)),
            "f1": float(metrics.get("f1", np.nan)),
        }
        row["delta_auc_vs_locked"] = row["auc"] - float(current["auc"]) if np.isfinite(row["auc"]) else np.nan
        row["delta_pr_auc_vs_locked"] = row["pr_auc"] - float(current["pr_auc"]) if np.isfinite(row["pr_auc"]) else np.nan
        return row

    rows: List[Dict[str, Any]] = []
    fc0 = read_csv(FC0_MAIN)
    if not fc0.empty:
        r = fc0[fc0["run_id"].eq("fc0_ch1_0_2")].iloc[0]
        rows.append(base_row("fc0", "VAE capacity", "FULL 5x5", r, "not promoted: lower AUC/PR-AUC"))
    act = read_csv(ACT_NONE_MAIN)
    if not act.empty:
        r = act[act["run_id"].eq("final_activation_none_ch1_0_2")].iloc[0]
        rows.append(base_row("final_activation_none", "decoder output", "FULL 5x5", r, "not promoted: lower AUC/PR-AUC"))
    mfr = read_csv(MFR_SAMPLER_MAIN)
    if not mfr.empty:
        r = mfr[mfr["run_id"].eq("manufacturer_balanced_sampler_full_5x5")].iloc[0]
        rows.append(base_row("manufacturer_balanced_sampler", "VAE sampling", "FULL 5x5", r, "not promoted: threshold metrics improved but AUC/PR-AUC decreased"))
    fast = read_csv(AUC_SPRINT_FAST)
    if not fast.empty:
        for cid in ["channel_dropout_fast3x3", "batch32_fast3x3"]:
            sub = fast[fast["candidate_id"].eq(cid)]
            if not sub.empty:
                r = sub.iloc[0]
                rows.append(base_row(cid.replace("_fast3x3", ""), "FAST representation screen", "FAST 3x3", r, "not promoted from FAST screen"))
    clf = read_csv(CLASSIFIER_SELECTION)
    if not clf.empty:
        alt = clf[~clf["strategy"].eq("baseline_logreg_l2_current_selection")].sort_values("auc", ascending=False).iloc[0]
        rows.append(base_row("classifier_selection_best_alternative", "frozen-latent readout", "read-only classifier audit", alt, "not promoted: no AUC/PR-AUC improvement without subgroup tradeoff"))
    return pd.DataFrame(rows)


def reviewer_text(metrics: pd.Series, ci: pd.DataFrame, failed: pd.DataFrame) -> str:
    auc_ci = ci[ci["metric"].eq("ROC-AUC")].iloc[0]
    pr_ci = ci[ci["metric"].eq("PR-AUC")].iloc[0]
    return f"""# Reviewer-Response-Ready Text

## Final Locked Model

We locked the final ADNI v5.1 model before manuscript revision as the FULL 5x5 `[1,0,2]` beta-VAE with the original reconstruction objective, tanh decoder output, manufacturer-aware nested splits, and classifier-only `logreg_l2` readout on latent `mu + Age + Sex`. The selected operating point was chosen inside training data only using inner-CV out-of-fold predictions to target sensitivity >= 0.70 while maximizing specificity.

The locked pooled OOF performance is ROC-AUC {float(metrics['auc']):.4f} (bootstrap 95% CI {float(auc_ci['bootstrap_ci_low']):.4f}-{float(auc_ci['bootstrap_ci_high']):.4f}) and PR-AUC {float(metrics['pr_auc']):.4f} (bootstrap 95% CI {float(pr_ci['bootstrap_ci_low']):.4f}-{float(pr_ci['bootstrap_ci_high']):.4f}). At the pre-specified sensitivity-constrained threshold, the confusion matrix is TN={int(metrics['tn'])}, FP={int(metrics['fp'])}, FN={int(metrics['fn'])}, TP={int(metrics['tp'])}, with sensitivity {float(metrics['sensitivity']):.4f}, specificity {float(metrics['specificity']):.4f}, balanced accuracy {float(metrics['balanced_accuracy']):.4f}, and F1 {float(metrics['f1']):.4f}.

## Why AUC Optimization Was Stopped

After locking the model, we ran only controlled negative-control and sensitivity analyses to test whether small, scientifically plausible changes could improve ranking without increasing bias. These included reducing the intermediate fully connected bottleneck (`fc0`), removing the decoder tanh, changing VAE sampling to manufacturer-balanced batches, channel dropout, batch size 32, and frozen-latent classifier-selection rules. None improved both ROC-AUC and PR-AUC versus the locked model. Several improved threshold-dependent metrics while reducing ROC-AUC/PR-AUC, indicating a changed operating point rather than better AD-vs-CN ranking.

We therefore stopped micro-optimization to avoid post-hoc model chasing. The locked model is retained because it had the best threshold-independent ranking metrics, used leakage-safe threshold selection, and survived scanner leakage, calibration, subgroup, and SiteCode audits without revealing a predefinable QC or metadata exclusion rule.

## Conservative Interpretation

We do not claim near-diagnostic performance. The observed AUC reflects a realistic ADNI AD-vs-CN task with heterogeneous manufacturers, sites, and age distributions. Remaining errors are concentrated partly in known subgroups, especially Philips CN false positives and GE AD false negatives, but subject-level audit did not identify a defensible post-hoc exclusion rule. We report these limitations directly and treat external validation/data expansion as the appropriate next step rather than further tuning on the same cohort.

## Failed Optimization Summary

{failed.to_markdown(index=False)}
"""


def final_recommendation_text() -> str:
    return """# Final Recommendation

Keep the locked current FULL tanh `[1,0,2]` model as the manuscript model.

Do not promote fc0, final_activation=none, manufacturer-balanced VAE sampling, channel dropout, batch_size=32, or frozen-latent classifier-selection variants. None satisfy the core promotion rule of improving ROC-AUC and PR-AUC without subgroup degradation.

The manuscript should present the locked model with conservative claims, bootstrap uncertainty, threshold-independent metrics, the leakage-safe threshold rule, manufacturer and SiteCode audits, calibration/Brier score, and the negative optimization table as evidence against cherry-picking.
"""


def main() -> int:
    if OUTDIR.exists():
        shutil.rmtree(OUTDIR)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    all_preds = load_predictions()
    primary = primary_predictions(all_preds)
    metrics_all = final_locked_model_metrics(all_preds)
    primary_metrics = metrics_all[
        (metrics_all["model"].eq(PRIMARY_MODEL)) & (metrics_all["threshold_strategy"].eq(PRIMARY_THRESHOLD))
    ].iloc[0]

    write_pair("final_locked_model_metrics", metrics_all)
    write_pair("confusion_matrix", confusion_table(primary))

    roc_df, pr_df, ci_df = roc_pr_bootstrap(primary)
    roc_df.to_csv(OUTDIR / "pooled_oof_roc_curve_bootstrap_ci.csv", index=False)
    pr_df.to_csv(OUTDIR / "pooled_oof_pr_curve_bootstrap_ci.csv", index=False)
    write_pair("pooled_oof_auc_pr_bootstrap_ci", ci_df)
    plot_roc_pr(roc_df, pr_df, ci_df)

    threshold = metrics_all[metrics_all["threshold_strategy"].isin([FIXED_THRESHOLD, PRIMARY_THRESHOLD])].copy()
    write_pair("threshold_comparison_fixed_0p5_vs_inner_oof_sens_ge_0p70", threshold)
    write_pair("manufacturer_subgroup_table", manufacturer_table(primary))
    write_pair("sitecode_audit_summary", sitecode_summary())
    site_detail = read_csv(SITE_AUDIT_DIR / "sitecode_auc_report.csv")
    if not site_detail.empty:
        write_pair("sitecode_auc_report", site_detail, max_rows=200)

    scanner = scanner_leakage_summary()
    write_pair("scanner_leakage_summary", scanner)
    cal, brier = calibration_outputs(primary)
    write_pair("calibration_curve_brier", cal)
    write_pair("calibration_brier_summary", brier)

    failed = failed_optimization_table()
    write_pair("failed_optimization_table", failed)

    (OUTDIR / "reviewer_response_ready_text.md").write_text(reviewer_text(primary_metrics, ci_df, failed), encoding="utf-8")
    (OUTDIR / "final_recommendation.md").write_text(final_recommendation_text(), encoding="utf-8")

    readme = f"""# Locked Current FULL [1,0,2] Manuscript Defense Package

This package is read-only with respect to model/data artifacts. It consolidates locked predictions, bootstrap curves, calibration, subgroup/site audits, scanner leakage summaries, and failed optimization evidence.

## Final Locked Metrics

- ROC-AUC: `{float(primary_metrics['auc']):.4f}`
- PR-AUC: `{float(primary_metrics['pr_auc']):.4f}`
- Balanced accuracy: `{float(primary_metrics['balanced_accuracy']):.4f}`
- Sensitivity: `{float(primary_metrics['sensitivity']):.4f}`
- Specificity: `{float(primary_metrics['specificity']):.4f}`
- F1: `{float(primary_metrics['f1']):.4f}`
- Brier: `{float(primary_metrics['brier']):.4f}`
- Confusion: TN=`{int(primary_metrics['tn'])}`, FP=`{int(primary_metrics['fp'])}`, FN=`{int(primary_metrics['fn'])}`, TP=`{int(primary_metrics['tp'])}`

## Decision

Keep the locked current FULL tanh `[1,0,2]` model. Stop AUC micro-optimization on this cohort.
"""
    (OUTDIR / "README.md").write_text(readme, encoding="utf-8")

    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "output_dir": str(OUTDIR),
        "readout_dir": str(READOUT_DIR),
        "site_audit_dir": str(SITE_AUDIT_DIR),
        "n_bootstrap": N_BOOT,
        "primary_model": PRIMARY_MODEL,
        "primary_threshold": PRIMARY_THRESHOLD,
        "training_launched": False,
        "vae_retrained": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "configs_modified": False,
        "existing_results_modified": False,
    }
    write_json(OUTDIR / "command_log.json", command_log)

    print(f"output_dir={OUTDIR}")
    print("training_launched=False")
    print("tensor_modified=False")
    print("metadata_modified=False")
    print("ledger_modified=False")
    print(
        f"AUC={float(primary_metrics['auc']):.4f} PR_AUC={float(primary_metrics['pr_auc']):.4f} "
        f"BA={float(primary_metrics['balanced_accuracy']):.4f} F1={float(primary_metrics['f1']):.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
