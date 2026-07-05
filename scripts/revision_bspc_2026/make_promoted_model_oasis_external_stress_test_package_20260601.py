#!/usr/bin/env python3
"""Build a manuscript-facing OASIS external stress-test package.

This script is read-only with respect to model, tensor, metadata, and scoring
artifacts. It only reads completed OASIS/ADNI outputs and writes a synthesis
package with tables, figures, and reviewer-safe text.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import auc as skl_auc
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "revision_bspc_2026"

OUT = RESULTS / "promoted_model_oasis_external_stress_test_package_20260601"
FIG = OUT / "figures"
TAB = OUT / "tables"

PILOT_POST = RESULTS / "oasis_tanda_2026_05_25_external_scoring_postmortem"
NEW_SCORING = RESULTS / "oasis_next_60cn_60ad_external_scoring_20260530"
NEW_POST = RESULTS / "oasis_next_60cn_60ad_external_scoring_postmortem_20260530"
MEGA = RESULTS / "oasis_mega_90cn_90ad_pooled_external_validation_20260531"
CLIN_MOTION = RESULTS / "oasis_pilot_vs_new_clinical_motion_difficulty_audit_20260531"
ADNI_RANKING = RESULTS / "final_best_model_deep_audit_and_ranking_20260601"

PROMOTED_EXTERNAL_LABEL = "recover035_oof_logitz"
PROMOTED_ADNI_ROW = "promoted_p015_latent384_beta3p75"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def md_table(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        path.write_text("_No rows available._\n", encoding="utf-8")
        return
    try:
        text = df.to_markdown(index=index)
    except Exception:
        text = df.to_string(index=index)
    path.write_text(text + "\n", encoding="utf-8")


def write_table(df: pd.DataFrame, stem: str) -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{stem}.csv", index=False)
    md_table(df, TAB / f"{stem}.md")


def safe_float(x) -> float:
    try:
        if pd.isna(x):
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def metric_ci_from_scores(y_true: np.ndarray, y_score: np.ndarray, metric: str, n_boot: int = 1000, seed: int = 20260601):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan, np.nan
    if metric == "auc":
        estimate = roc_auc_score(y_true, y_score)
    elif metric == "pr_auc":
        estimate = average_precision_score(y_true, y_score)
    else:
        raise ValueError(metric)
    vals = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(roc_auc_score(y_true[idx], y_score[idx]) if metric == "auc" else average_precision_score(y_true[idx], y_score[idx]))
    if not vals:
        return estimate, np.nan, np.nan
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return estimate, lo, hi


def add_score_summary(rows: list[dict], label: str, dataset: str, subset: str, build: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    d = df.copy()
    if "model_label" in d.columns:
        d = d[d["model_label"].eq(PROMOTED_EXTERNAL_LABEL)]
    if "adni_model" in d.columns:
        d = d[d["adni_model"].eq(PROMOTED_EXTERNAL_LABEL)]
    if "build_candidate" in d.columns:
        d = d[d["build_candidate"].eq(build)]
    if "prediction_level" in d.columns:
        d = d[d["prediction_level"].eq("ensemble_mean_score_majority_vote")]
    if "fold" in d.columns:
        d = d[d["fold"].astype(str).eq("ensemble")]
    if d.empty:
        return
    y_true_col = "y_true" if "y_true" in d.columns else "y"
    score_col = "y_score" if "y_score" in d.columns else "prob_ensemble"
    if y_true_col not in d.columns or score_col not in d.columns:
        return
    y_true = pd.to_numeric(d[y_true_col], errors="coerce").to_numpy()
    y_score = pd.to_numeric(d[score_col], errors="coerce").to_numpy()
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[mask].astype(int)
    y_score = y_score[mask].astype(float)
    if len(y_true) == 0:
        return
    auc_est, auc_lo, auc_hi = metric_ci_from_scores(y_true, y_score, "auc")
    pr_est, pr_lo, pr_hi = metric_ci_from_scores(y_true, y_score, "pr_auc")
    rows.append(
        {
            "analysis_label": label,
            "dataset": dataset,
            "subset": subset,
            "build_candidate": build,
            "model_label": PROMOTED_EXTERNAL_LABEL,
            "n": len(y_true),
            "n_cn": int((y_true == 0).sum()),
            "n_ad": int((y_true == 1).sum()),
            "auc": auc_est,
            "auc_ci_low": auc_lo,
            "auc_ci_high": auc_hi,
            "pr_auc": pr_est,
            "pr_auc_ci_low": pr_lo,
            "pr_auc_ci_high": pr_hi,
        }
    )


def make_auc_pr_summary() -> pd.DataFrame:
    rows: list[dict] = []
    builds = ["concatenated_timeseries", "runwise_140TR_pilot_parity", "runwise164_pilot_parity"]
    mega_pred = read_csv(MEGA / "predictions.csv")
    for build in builds:
        d_build = mega_pred[mega_pred.get("build_candidate", pd.Series(dtype=str)).eq(build)] if not mega_pred.empty else pd.DataFrame()
        add_score_summary(rows, "pilot_30CN_30AD", "OASIS pilot", "pilot", build, d_build[d_build.get("source_batch", "").eq("pilot")] if not d_build.empty else d_build)
        add_score_summary(rows, "new_60CN_60AD_combined", "OASIS new", "calibration_plus_locked_test", build, d_build[d_build.get("source_batch", "").eq("new")] if not d_build.empty else d_build)
        add_score_summary(rows, "mega_90CN_90AD", "OASIS mega", "pooled_secondary_exploratory", build, d_build)

    # Also include the pre-specified new calibration/test rows from the original new scoring output.
    for split, fn in [("calibration", "predictions_calibration.csv"), ("locked_test", "predictions_locked_test.csv")]:
        d = read_csv(NEW_SCORING / fn)
        for build in ["concatenated_timeseries", "runwise_140TR_connectome_average", "runwise164_connectome_average"]:
            add_score_summary(rows, f"new_60CN_60AD_{split}", "OASIS new", split, build, d)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["analysis_label", "build_candidate"]).reset_index(drop=True)


def make_primary_metric_context(summary: pd.DataFrame) -> pd.DataFrame:
    adni = read_csv(ADNI_RANKING / "model_ranking_table.csv")
    rows = []
    if not adni.empty:
        sel = adni[
            adni["model_id"].eq(PROMOTED_ADNI_ROW)
            & adni["calib_method"].astype(str).eq("oof_ecdf")
            & adni["threshold_strategy"].astype(str).eq("inner_oof_target_sens_ge_0p70_max_spec")
        ]
        for _, r in sel.iterrows():
            rows.append(
                {
                    "analysis": "ADNI internal CV",
                    "build_candidate": "ADNI promoted model",
                    "model_label": "recover035_latent384_beta3p75_oof_ecdf",
                    "n": r.get("n"),
                    "n_cn": r.get("n_cn"),
                    "n_ad": r.get("n_ad"),
                    "auc": r.get("auc"),
                    "pr_auc": r.get("pr_auc"),
                    "balanced_accuracy": r.get("balanced_accuracy"),
                    "sensitivity": r.get("sensitivity"),
                    "specificity": r.get("specificity"),
                    "f1": r.get("f1"),
                    "role": "promoted_internal_reference",
                }
            )
    if not summary.empty:
        for _, r in summary.iterrows():
            rows.append(
                {
                    "analysis": r["analysis_label"],
                    "build_candidate": r["build_candidate"],
                    "model_label": r["model_label"],
                    "n": r["n"],
                    "n_cn": r["n_cn"],
                    "n_ad": r["n_ad"],
                    "auc": r["auc"],
                    "pr_auc": r["pr_auc"],
                    "balanced_accuracy": np.nan,
                    "sensitivity": np.nan,
                    "specificity": np.nan,
                    "f1": np.nan,
                    "role": "external_ranking_stress_test",
                }
            )
    return pd.DataFrame(rows)


def make_threshold_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    all_splits = read_csv(NEW_POST / "metrics_all_splits.csv")
    new_thr = read_csv(NEW_POST / "threshold_transfer_summary.csv")
    mega_thr = read_csv(MEGA / "threshold_metrics.csv")

    rows = []
    if not all_splits.empty:
        d = all_splits[all_splits["model_label"].eq(PROMOTED_EXTERNAL_LABEL)].copy()
        d["source_table"] = "new_60cn_60ad_metrics_all_splits"
        rows.append(d)
    if not mega_thr.empty:
        d = mega_thr[mega_thr["adni_model"].eq(PROMOTED_EXTERNAL_LABEL)].copy()
        d = d.rename(columns={"adni_model": "model_label", "evaluation_subset": "split_subset"})
        d["source_table"] = "mega_90cn_90ad_threshold_metrics"
        rows.append(d)
    threshold_metrics = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()

    threshold_summary = pd.DataFrame()
    if not new_thr.empty:
        threshold_summary = new_thr[new_thr["model_label"].eq(PROMOTED_EXTERNAL_LABEL)].copy()
    return threshold_metrics, threshold_summary


def make_threshold_value_report(threshold_metrics: pd.DataFrame) -> pd.DataFrame:
    if threshold_metrics.empty:
        return pd.DataFrame()
    rows = []
    cols = [
        "source_table",
        "build_candidate",
        "split_subset",
        "threshold_strategy",
        "threshold_fit_subset",
        "threshold",
        "n",
        "n_cn",
        "n_ad",
        "sensitivity",
        "specificity",
        "balanced_accuracy",
        "f1",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    keep = threshold_metrics[threshold_metrics["model_label"].eq(PROMOTED_EXTERNAL_LABEL)].copy()
    keep = keep[
        keep["threshold_strategy"].astype(str).isin(
            [
                "adni_fixed",
                "oasis_calibration",
                "new_calibration_sens_ge_0p70_max_spec",
                "pilot_calibrated_sens_ge_0p70_max_spec",
            ]
        )
    ]
    for _, r in keep.iterrows():
        rows.append({c: r.get(c, np.nan) for c in cols})
    return pd.DataFrame(rows)


def make_source_batch_instability() -> pd.DataFrame:
    strat = read_csv(MEGA / "stratified_metrics.csv")
    if strat.empty:
        return strat
    d = strat[
        strat["adni_model"].eq(PROMOTED_EXTERNAL_LABEL)
        & strat["stratum"].eq("source_batch")
        & strat["threshold_strategy"].eq("adni_fixed")
    ].copy()
    deltas = []
    for build, g in d.groupby("build_candidate"):
        p = g[g["stratum_value"].eq("pilot")]
        n = g[g["stratum_value"].eq("new")]
        if not p.empty and not n.empty:
            pr = p.iloc[0]
            nr = n.iloc[0]
            deltas.append(
                {
                    "build_candidate": build,
                    "pilot_auc": pr["auc"],
                    "new_auc": nr["auc"],
                    "new_minus_pilot_auc": nr["auc"] - pr["auc"],
                    "pilot_pr_auc": pr["pr_auc"],
                    "new_pr_auc": nr["pr_auc"],
                    "new_minus_pilot_pr_auc": nr["pr_auc"] - pr["pr_auc"],
                    "pilot_ba": pr["balanced_accuracy"],
                    "new_ba": nr["balanced_accuracy"],
                    "new_minus_pilot_ba": nr["balanced_accuracy"] - pr["balanced_accuracy"],
                }
            )
    return pd.DataFrame(deltas)


def make_motion_severity_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    motion = read_csv(CLIN_MOTION / "cohort_motion.csv")
    cdr = read_csv(CLIN_MOTION / "cdr_distribution.csv")
    err = read_csv(NEW_POST / "motion_error_sensitivity.csv")
    if not err.empty:
        err = err[err["model_label"].eq(PROMOTED_EXTERNAL_LABEL)].copy()
    return motion, cdr, err


def plot_auc_pr(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    plot = summary[
        summary["analysis_label"].isin(
            ["pilot_30CN_30AD", "new_60CN_60AD_locked_test", "mega_90CN_90AD"]
        )
    ].copy()
    if plot.empty:
        return
    label_map = {
        "pilot_30CN_30AD": "Pilot 30/30",
        "new_60CN_60AD_locked_test": "New locked test 30/30",
        "mega_90CN_90AD": "Mega 90/90",
    }
    plot["x_label"] = plot["analysis_label"].map(label_map) + "\n" + plot["build_candidate"].str.replace("_", "\n")
    x = np.arange(len(plot))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(10, len(plot) * 0.75), 5.5))
    auc_y = plot["auc"].to_numpy(float)
    pr_y = plot["pr_auc"].to_numpy(float)
    auc_err = np.vstack([auc_y - plot["auc_ci_low"].to_numpy(float), plot["auc_ci_high"].to_numpy(float) - auc_y])
    pr_err = np.vstack([pr_y - plot["pr_auc_ci_low"].to_numpy(float), plot["pr_auc_ci_high"].to_numpy(float) - pr_y])
    ax.bar(x - width / 2, auc_y, width, yerr=auc_err, label="ROC-AUC", color="#3b6ea8", capsize=3)
    ax.bar(x + width / 2, pr_y, width, yerr=pr_err, label="PR-AUC", color="#b65c32", capsize=3)
    ax.axhline(0.5, color="0.5", lw=1, ls="--")
    ax.set_ylim(0.2, 0.9)
    ax.set_ylabel("Metric")
    ax.set_title("OASIS external ranking stress test with bootstrap 95% CI")
    ax.set_xticks(x)
    ax.set_xticklabels(plot["x_label"], rotation=45, ha="right", fontsize=8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig01_auc_pr_bar_with_ci.png", dpi=200)
    plt.close(fig)


def plot_score_distributions() -> None:
    pred = read_csv(MEGA / "predictions.csv")
    if pred.empty:
        return
    d = pred[
        pred["model_label"].eq(PROMOTED_EXTERNAL_LABEL)
        & pred["prediction_level"].eq("ensemble_mean_score_majority_vote")
        & pred["build_candidate"].isin(["concatenated_timeseries", "runwise_140TR_pilot_parity", "runwise164_pilot_parity"])
    ].copy()
    if d.empty:
        return
    builds = list(d["build_candidate"].drop_duplicates())
    fig, axes = plt.subplots(1, len(builds), figsize=(5.2 * len(builds), 4.5), sharey=True)
    if len(builds) == 1:
        axes = [axes]
    colors = {("pilot", "CN"): "#6aaed6", ("pilot", "AD_DEMENTIA"): "#e07a5f", ("new", "CN"): "#22577a", ("new", "AD_DEMENTIA"): "#9d0208"}
    for ax, build in zip(axes, builds):
        sub = d[d["build_candidate"].eq(build)]
        positions = []
        data = []
        labels = []
        for i, (batch, diag) in enumerate([("pilot", "CN"), ("pilot", "AD_DEMENTIA"), ("new", "CN"), ("new", "AD_DEMENTIA")]):
            vals = sub[sub["source_batch"].eq(batch) & sub["diagnosis"].eq(diag)]["y_score"].dropna().to_numpy(float)
            if len(vals):
                data.append(vals)
                positions.append(i)
                labels.append(f"{batch}\n{diag.replace('_DEMENTIA', '')}")
        bp = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True, showfliers=False)
        for patch, lab in zip(bp["boxes"], labels):
            batch, diag = lab.split("\n")
            full_diag = "AD_DEMENTIA" if diag == "AD" else diag
            patch.set_facecolor(colors[(batch, full_diag)])
            patch.set_alpha(0.75)
        for i, vals in zip(positions, data):
            jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else np.array([0.0])
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=8, color="black", alpha=0.35)
        ax.set_title(build.replace("_", " "))
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("AD score" if ax is axes[0] else "")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Promoted model OASIS score distributions by source batch and diagnosis", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "fig02_score_distributions_by_batch_diagnosis.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_calibration() -> None:
    frames = []
    for fn in ["predictions_calibration.csv", "predictions_locked_test.csv"]:
        d = read_csv(NEW_SCORING / fn)
        if not d.empty:
            frames.append(d)
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    d = df[df["model_label"].eq(PROMOTED_EXTERNAL_LABEL) & df["fold"].astype(str).eq("ensemble")].copy()
    if d.empty:
        return
    builds = list(d["build_candidate"].drop_duplicates())
    fig, axes = plt.subplots(1, len(builds), figsize=(5.2 * len(builds), 4.5), sharex=True, sharey=True)
    if len(builds) == 1:
        axes = [axes]
    for ax, build in zip(axes, builds):
        for subset, color in [("calibration", "#3b6ea8"), ("locked_test", "#b65c32")]:
            sub = d[d["build_candidate"].eq(build) & d["split_subset"].eq(subset)]
            if sub.empty or sub["y_true"].nunique() < 2:
                continue
            frac_pos, mean_pred = calibration_curve(sub["y_true"].astype(int), sub["y_score"].astype(float), n_bins=5, strategy="quantile")
            ax.plot(mean_pred, frac_pos, marker="o", lw=1.8, label=subset, color=color)
        ax.plot([0, 1], [0, 1], ls="--", color="0.45", lw=1)
        ax.set_title(build.replace("_", " "))
        ax.set_xlabel("Mean predicted score")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Observed AD fraction")
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle("OASIS descriptive calibration curves; no locked-test threshold fitting", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "fig03_calibration_curve_new_oasis.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(threshold_metrics: pd.DataFrame) -> None:
    if threshold_metrics.empty:
        return
    d = threshold_metrics[
        threshold_metrics["source_table"].eq("new_60cn_60ad_metrics_all_splits")
        & threshold_metrics["split_subset"].eq("locked_test")
        & threshold_metrics["threshold_strategy"].isin(["adni_fixed", "oasis_calibration"])
    ].copy()
    if d.empty:
        return
    d = d[d["build_candidate"].isin(["concatenated_timeseries", "runwise_140TR_connectome_average", "runwise164_connectome_average"])]
    d = d.sort_values(["build_candidate", "threshold_strategy"])
    n = len(d)
    fig, axes = plt.subplots(2, int(np.ceil(n / 2)), figsize=(4.2 * int(np.ceil(n / 2)), 7.5))
    axes = np.ravel(axes)
    vmax = float(d[["tn", "fp", "fn", "tp"]].max().max()) if not d.empty else 30
    for ax, (_, r) in zip(axes, d.iterrows()):
        mat = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]], dtype=float)
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=vmax)
        for (i, j), val in np.ndenumerate(mat):
            ax.text(j, i, str(int(val)), ha="center", va="center", fontsize=12)
        ax.set_xticks([0, 1], ["Pred CN", "Pred AD"], fontsize=8)
        ax.set_yticks([0, 1], ["True CN", "True AD"], fontsize=8)
        ax.set_title(
            f"{str(r['build_candidate']).replace('_', ' ')}\n{r['threshold_strategy']}\nBA={r['balanced_accuracy']:.3f}, F1={r['f1']:.3f}",
            fontsize=9,
        )
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Locked-test threshold transfer confusion matrices", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG / "fig04_threshold_transfer_confusion_matrices.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_markdown(summary: pd.DataFrame, source_instability: pd.DataFrame, motion: pd.DataFrame, cdr: pd.DataFrame) -> None:
    best_mega = summary[
        summary["analysis_label"].eq("mega_90CN_90AD") & summary["build_candidate"].str.contains("runwise164", regex=False)
    ]
    best_mega_text = "not available"
    if not best_mega.empty:
        r = best_mega.iloc[0]
        best_mega_text = f"AUC={r['auc']:.3f}, PR-AUC={r['pr_auc']:.3f}"

    readme = f"""# Promoted Model OASIS External Stress-Test Package

Generated: {datetime.now().isoformat(timespec='seconds')}

Promoted ADNI model: `recover035_latent384_beta3p75_T80_h10000_p560_full5x5`.

Available OASIS scoring artifacts use `recover035_oof_logitz` for the score-harmonized promoted external scorer. The internal best row is OOF-ECDF (`AUC=0.795155`, `PR-AUC=0.573934`); no separate OASIS OOF-ECDF scoring artifact was found, so this package uses the completed score-harmonized promoted OASIS outputs rather than refitting or rescoring.

## Main Finding

External transfer is moderate and source-batch dependent. The strongest secondary pooled mega-OASIS result for the promoted score-harmonized model is {best_mega_text} using the pilot-parity runwise164 build, but the pre-specified new 60CN/60AD calibration/test analysis is weaker and unstable across build variants. OASIS results are therefore treated as an external stress test, not as model selection or model promotion evidence.

## Guardrails

- No training.
- No OASIS-based model selection.
- No OASIS-based promotion.
- Thresholds for the new OASIS protocol are fitted only on the calibration subset.
- Locked-test labels are not used for threshold fitting.
- Mega-OASIS 90CN/90AD is secondary/exploratory.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    protocol = """# OASIS Calibration/Test Threshold Protocol

The new OASIS 60CN/60AD batch was split before scoring into a 30CN/30AD calibration subset and a disjoint 30CN/30AD locked-test subset.

Primary threshold-transfer analyses:

1. **ADNI fixed threshold baseline**: apply the ADNI-derived threshold directly to OASIS.
2. **OASIS calibration-only threshold**: choose a threshold on the calibration subset targeting sensitivity >= 0.70 and maximizing specificity, then apply that locked threshold to the held-out OASIS test subset.

The locked-test subset is never used to fit thresholds. OASIS labels are used only for evaluation and descriptive stress-test reporting.
"""
    (OUT / "threshold_protocol.md").write_text(protocol, encoding="utf-8")

    manuscript = """# Manuscript-Ready External Stress-Test Paragraph

As an external stress test, we applied the promoted ADNI-trained model to independent OASIS resting-state fMRI cohorts without retraining the VAE or classifier. In the pre-specified OASIS 60CN/60AD protocol, the cohort was split before scoring into a calibration subset and an untouched locked-test subset; threshold recalibration, when reported, was performed only in the calibration subset and then applied to the locked test. Ranking performance transferred only moderately and varied across OASIS source batches and run-handling choices. In the secondary pooled 90CN/90AD analysis, the promoted score-harmonized model showed a positive ranking signal for the pilot-parity runwise builds, whereas the new locked-test split remained weaker. These findings support the presence of some external signal but also demonstrate threshold-transfer and domain-shift instability. They do not invalidate the internally validated ADNI model; rather, they motivate scanner/protocol harmonization and prospective external calibration before clinical use, and they prevent overclaiming from internal cross-validation alone.
"""
    (OUT / "manuscript_ready_oasis_external_stress_test_paragraph.md").write_text(manuscript, encoding="utf-8")

    recommendation = """# Final Recommendation

Decision: `external_stress_test_moderate_unstable_no_oasis_based_promotion`.

The promoted ADNI model remains supported by internal nested CV, but OASIS transfer is not strong or stable enough to claim scanner/protocol-general clinical deployment. Report OASIS as an external stress test. Use the pre-specified new 60CN/60AD calibration/test protocol as the primary external protocol and label the pooled 90CN/90AD analysis as secondary/exploratory.

Next external step: process a larger independent OASIS or non-ADNI cohort with locked preprocessing, perform threshold calibration only on a designated calibration subset, and evaluate on a prospectively locked test subset.
"""
    (OUT / "final_recommendation.md").write_text(recommendation, encoding="utf-8")


def write_artifact_manifest() -> None:
    paths = [
        PILOT_POST / "primary_metrics_with_ci.csv",
        NEW_SCORING / "metrics_locked_test.csv",
        NEW_SCORING / "predictions_calibration.csv",
        NEW_SCORING / "predictions_locked_test.csv",
        NEW_POST / "metrics_all_splits.csv",
        NEW_POST / "threshold_transfer_summary.csv",
        NEW_POST / "motion_error_sensitivity.csv",
        MEGA / "pooled_ranking_metrics.csv",
        MEGA / "threshold_metrics.csv",
        MEGA / "stratified_metrics.csv",
        MEGA / "predictions.csv",
        CLIN_MOTION / "summary.md",
        CLIN_MOTION / "cohort_motion.csv",
        CLIN_MOTION / "cdr_distribution.csv",
        ADNI_RANKING / "model_ranking_table.csv",
    ]
    rows = [{"path": str(p.relative_to(ROOT)), "exists": p.exists(), "bytes": p.stat().st_size if p.exists() else np.nan} for p in paths]
    write_table(pd.DataFrame(rows), "source_artifact_manifest")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    required = [NEW_SCORING, NEW_POST, MEGA, ADNI_RANKING]
    missing = [str(p) for p in required if not p.exists()]
    if args.dry_run:
        print("Output:", OUT)
        print("Missing required directories:", missing)
        return 1 if missing else 0
    if missing:
        raise FileNotFoundError("Missing required directories: " + ", ".join(missing))

    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)

    summary = make_auc_pr_summary()
    write_table(summary, "oasis_auc_pr_summary_with_bootstrap_ci")

    context = make_primary_metric_context(summary)
    write_table(context, "adni_and_oasis_metric_context")

    threshold_metrics, threshold_summary = make_threshold_tables()
    write_table(threshold_metrics, "threshold_transfer_metrics_promoted")
    write_table(threshold_summary, "new_oasis_threshold_transfer_summary_promoted")
    write_table(make_threshold_value_report(threshold_metrics), "threshold_values_and_confusion_promoted")

    source_instability = make_source_batch_instability()
    write_table(source_instability, "source_batch_instability_promoted")

    motion, cdr, motion_errors = make_motion_severity_tables()
    write_table(motion, "motion_summary_pilot_vs_new")
    write_table(cdr, "severity_cdr_summary_pilot_vs_new")
    write_table(motion_errors, "motion_error_sensitivity_promoted")

    mega_rank = read_csv(MEGA / "pooled_ranking_metrics.csv")
    write_table(mega_rank[mega_rank["adni_model"].eq(PROMOTED_EXTERNAL_LABEL)].copy() if not mega_rank.empty else mega_rank, "mega_oasis_pooled_promoted_metrics")

    plot_auc_pr(summary)
    plot_score_distributions()
    plot_calibration()
    plot_confusion(threshold_metrics)

    make_markdown(summary, source_instability, motion, cdr)
    write_artifact_manifest()

    log = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "output": str(OUT.relative_to(ROOT)),
        "promoted_adni_internal_best": {
            "model": "recover035_latent384_beta3p75_T80_h10000_p560_full5x5",
            "best_row": "OOF-ECDF score-harmonized logreg_l2 z_plus_age_sex",
            "auc": 0.795155,
            "pr_auc": 0.573934,
        },
        "oasis_scored_model_label_used": PROMOTED_EXTERNAL_LABEL,
        "safety": {
            "training": False,
            "model_selection": False,
            "oasis_based_promotion": False,
            "tensor_modified": False,
            "metadata_modified": False,
            "model_artifacts_modified": False,
        },
    }
    (OUT / "command_log.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote OASIS external stress-test package to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
