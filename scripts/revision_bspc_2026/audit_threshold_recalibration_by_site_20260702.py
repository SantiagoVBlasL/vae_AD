#!/usr/bin/env python3
"""Oracle per-site threshold recalibration audit for promoted OOF predictions.

Read-only with respect to model artifacts. Writes small derived tables/plot.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    average_precision_score,
    roc_auc_score,
)


ROOT = Path("/home/diego/proyectos/vae_AD")
PRED_PATH = ROOT / "results/revision_bspc_2026/recover035_latent384_beta3p75_T80_h10000_p560_full5x5/all_folds_clf_predictions_MULTI_logreg_vaeconvtranspose4l_ld384_beta3.75_normzscore_offdiag_ch3sel_intFCquarter_drop0.15_ln0_outer5x1_scoreroc_auc.csv"
META_PATH = ROOT / "results/revision_bspc_2026/adni_035_metadata_rescue_preflight/patched_metadata_candidate.csv"
OUT = ROOT / "results/revision_bspc_2026/post_revision_exploratory_20260630/threshold_recalibration_by_site_20260702"


def site_label(x: object) -> str:
    if pd.isna(x):
        return "MISSING"
    try:
        return str(int(float(x)))
    except Exception:
        return str(x)


def metrics_at_threshold(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "threshold": float(threshold),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "balanced_accuracy": float((sens + spec) / 2) if np.isfinite(sens) and np.isfinite(spec) else np.nan,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def oracle_threshold_max_ba(y: np.ndarray, score: np.ndarray) -> dict:
    # y_pred = score >= threshold. Include thresholds above max and below min.
    vals = np.unique(score)
    candidates = [float(np.nextafter(vals.min(), -np.inf))]
    candidates += [float(v) for v in vals]
    candidates += [float(np.nextafter(vals.max(), np.inf))]
    rows = []
    for t in candidates:
        m = metrics_at_threshold(y, score, t)
        rows.append(m)
    df = pd.DataFrame(rows)
    # Max BA; tie-break toward sensitivity recovery, then specificity, then higher threshold.
    df = df.sort_values(
        ["balanced_accuracy", "sensitivity", "specificity", "threshold"],
        ascending=[False, False, False, False],
    )
    return df.iloc[0].to_dict()


def implied_global_threshold(df: pd.DataFrame) -> dict:
    neg_scores = df.loc[df["y_pred"].eq(0), "y_score_final"]
    pos_scores = df.loc[df["y_pred"].eq(1), "y_score_final"]
    max_pred0 = float(neg_scores.max())
    min_pred1 = float(pos_scores.min())
    if max_pred0 < min_pred1:
        mid = (max_pred0 + min_pred1) / 2.0
        note = "Any threshold in (max_pred0_score, min_pred1_score] reproduces y_pred."
    else:
        mid = float("nan")
        note = "No clean threshold interval exactly separates saved y_pred from y_score_final."
    return {
        "max_pred0_score": max_pred0,
        "min_pred1_score": min_pred1,
        "implied_threshold_midpoint": float(mid),
        "threshold_note": note,
    }


def main() -> None:
    started = datetime.now(timezone.utc)
    OUT.mkdir(parents=True, exist_ok=True)

    pred_all = pd.read_csv(PRED_PATH)
    pred = pred_all[pred_all["classifier_type"].eq("logreg")].copy()
    if pred["SubjectID"].duplicated().any():
        raise RuntimeError("Logreg-filtered OOF predictions still contain duplicated SubjectID rows.")
    meta = pd.read_csv(META_PATH)
    join_cols = ["SubjectID", "Site3", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex"]
    merged = pred.merge(meta[join_cols], on="SubjectID", how="left", validate="one_to_one")
    merged["Site3_label"] = merged["Site3"].map(site_label)

    global_info = implied_global_threshold(merged)
    global_t = global_info["implied_threshold_midpoint"]
    if not np.isfinite(global_t):
        # Fallback to min positive to reproduce score>=threshold behavior as closely as possible.
        global_t = global_info["min_pred1_score"]

    rows = []
    pooled_parts = []
    excluded = []
    for site, sub in sorted(merged.groupby("Site3_label"), key=lambda kv: kv[0]):
        y = sub["y_true"].to_numpy(dtype=int)
        score = sub["y_score_final"].to_numpy(dtype=float)
        n = len(sub)
        n_cn = int((y == 0).sum())
        n_ad = int((y == 1).sum())
        eligible = n >= 15 and n_cn >= 3 and n_ad >= 3
        reason = ""
        if not eligible:
            parts = []
            if n < 15:
                parts.append("N<15")
            if n_cn < 3:
                parts.append("CN<3")
            if n_ad < 3:
                parts.append("AD<3")
            reason = ";".join(parts)
            excluded.append({"Site3": site, "N": n, "CN": n_cn, "AD": n_ad, "reason": reason})
            rows.append({
                "Site3": site, "eligible": False, "exclusion_reason": reason,
                "N": n, "CN": n_cn, "AD": n_ad,
                "global_threshold": global_t,
            })
            continue
        current = metrics_at_threshold(y, score, global_t)
        oracle = oracle_threshold_max_ba(y, score)
        auc = roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan
        pr_auc = average_precision_score(y, score) if len(np.unique(y)) == 2 else np.nan
        row = {
            "Site3": site,
            "eligible": True,
            "exclusion_reason": "",
            "N": n,
            "CN": n_cn,
            "AD": n_ad,
            "auc": float(auc),
            "pr_auc": float(pr_auc),
            "global_threshold": global_t,
            "oracle_threshold": float(oracle["threshold"]),
            "threshold_delta_oracle_minus_global": float(oracle["threshold"] - global_t),
            "current_sensitivity_global_threshold": current["sensitivity"],
            "current_specificity_global_threshold": current["specificity"],
            "current_balanced_accuracy_global_threshold": current["balanced_accuracy"],
            "current_f1_global_threshold": current["f1"],
            "current_tn": current["tn"], "current_fp": current["fp"],
            "current_fn": current["fn"], "current_tp": current["tp"],
            "oracle_sensitivity": oracle["sensitivity"],
            "oracle_specificity": oracle["specificity"],
            "oracle_balanced_accuracy": oracle["balanced_accuracy"],
            "oracle_f1": oracle["f1"],
            "oracle_tn": int(oracle["tn"]), "oracle_fp": int(oracle["fp"]),
            "oracle_fn": int(oracle["fn"]), "oracle_tp": int(oracle["tp"]),
            "delta_sensitivity": oracle["sensitivity"] - current["sensitivity"],
            "delta_specificity": oracle["specificity"] - current["specificity"],
            "delta_balanced_accuracy": oracle["balanced_accuracy"] - current["balanced_accuracy"],
        }
        rows.append(row)
        pooled = sub[["SubjectID", "y_true", "y_score_final"]].copy()
        pooled["global_pred"] = (pooled["y_score_final"] >= global_t).astype(int)
        pooled["oracle_pred"] = (pooled["y_score_final"] >= oracle["threshold"]).astype(int)
        pooled["Site3"] = site
        pooled_parts.append(pooled)

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "threshold_recalibration_by_site.csv", index=False)

    if pooled_parts:
        pooled = pd.concat(pooled_parts, ignore_index=True)
        y = pooled["y_true"].to_numpy(dtype=int)
        global_pred = pooled["global_pred"].to_numpy(dtype=int)
        oracle_pred = pooled["oracle_pred"].to_numpy(dtype=int)
        global_pool = metrics_at_threshold(y, global_pred.astype(float), 0.5)
        oracle_pool = metrics_at_threshold(y, oracle_pred.astype(float), 0.5)
        pooled_report = {
            "eligible_sites_n": int(result["eligible"].sum()),
            "pooled_eligible_subjects_n": int(len(pooled)),
            "pooled_eligible_cn": int((y == 0).sum()),
            "pooled_eligible_ad": int((y == 1).sum()),
            "pooled_global_sensitivity": global_pool["sensitivity"],
            "pooled_global_specificity": global_pool["specificity"],
            "pooled_global_balanced_accuracy": global_pool["balanced_accuracy"],
            "pooled_global_f1": global_pool["f1"],
            "pooled_oracle_sensitivity": oracle_pool["sensitivity"],
            "pooled_oracle_specificity": oracle_pool["specificity"],
            "pooled_oracle_balanced_accuracy": oracle_pool["balanced_accuracy"],
            "pooled_oracle_f1": oracle_pool["f1"],
        }
    else:
        pooled_report = {}

    # Plot sensitivity before/after.
    elig = result[result["eligible"].eq(True)].copy()
    elig = elig.sort_values("current_sensitivity_global_threshold")
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.35 * len(elig))))
    y_pos = np.arange(len(elig))
    ax.hlines(
        y_pos,
        elig["current_sensitivity_global_threshold"],
        elig["oracle_sensitivity"],
        color="#bdbdbd",
        lw=2,
    )
    ax.scatter(elig["current_sensitivity_global_threshold"], y_pos, color="#377eb8", label="Global threshold", zorder=3)
    ax.scatter(elig["oracle_sensitivity"], y_pos, color="#e41a1c", label="Oracle site threshold", zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Site {s} (N={n})" for s, n in zip(elig["Site3"], elig["N"])])
    ax.set_xlabel("Sensitivity")
    ax.set_xlim(-0.03, 1.03)
    ax.set_title("Oracle per-site threshold recalibration: sensitivity before/after")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "threshold_recalibration_plot.pdf")
    plt.close(fig)

    excluded_df = pd.DataFrame(excluded)
    eligible_cols = [
        "Site3", "N", "CN", "AD", "auc", "pr_auc", "global_threshold", "oracle_threshold",
        "threshold_delta_oracle_minus_global",
        "current_sensitivity_global_threshold", "current_specificity_global_threshold", "current_balanced_accuracy_global_threshold",
        "oracle_sensitivity", "oracle_specificity", "oracle_balanced_accuracy",
        "delta_sensitivity", "delta_specificity", "delta_balanced_accuracy",
    ]
    report = f"""# Oracle Per-Site Threshold Recalibration Audit

## ORACLE / Upper-Bound Caveat

This is **not** a deployable calibration method. Each site's oracle threshold is fit using that site's own OOF labels. Those labels would not be available in a true prospective external-transfer setting. The analysis is an upper-bound diagnostic for the manuscript claim that ranking may be partly preserved while threshold transfer is poor.

## Inputs

- OOF prediction file: `{PRED_PATH}`
- Metadata file: `{META_PATH}`
- Rows in prediction file before classifier filter: {len(pred_all)}
- Rows after `classifier_type == logreg`: {len(pred)}
- Unique logreg OOF subjects: {pred['SubjectID'].nunique()}

## Global Threshold Recovery

- max score among saved `y_pred=0`: {global_info['max_pred0_score']:.12f}
- min score among saved `y_pred=1`: {global_info['min_pred1_score']:.12f}
- implied midpoint threshold used for this audit: {global_t:.12f}
- note: {global_info['threshold_note']}

## Pooled Eligible-Site Comparison

{pd.DataFrame([pooled_report]).to_markdown(index=False) if pooled_report else 'No eligible sites.'}

## Eligible Site Results

{elig[eligible_cols].to_markdown(index=False)}

## Excluded Sites

Sites were excluded from oracle threshold fitting if `N<15` or either class had fewer than 3 subjects. They are listed here rather than silently dropped.

{excluded_df.to_markdown(index=False) if not excluded_df.empty else 'No excluded sites.'}

## Interpretation

If oracle thresholds raise sensitivity at several sites, that supports a threshold-transfer problem rather than a complete loss of rank information. Because thresholds are fit on each site's own labels, these results should be described only as an upper-bound diagnostic. They should not be described as validation of a prospective site-specific recalibration procedure.
"""
    (OUT / "threshold_recalibration_by_site_report.md").write_text(report, encoding="utf-8")

    log = {
        "timestamp_utc": started.isoformat(),
        "did_train": False,
        "did_infer": False,
        "did_modify_inputs": False,
        "prediction_file": str(PRED_PATH),
        "metadata_file": str(META_PATH),
        "output_dir": str(OUT),
        "logreg_rows": int(len(pred)),
        "unique_subjects": int(pred["SubjectID"].nunique()),
        "global_threshold": global_t,
        "eligible_sites": int(result["eligible"].sum()),
        "excluded_sites": int((~result["eligible"]).sum()),
        "commands": [
            "/home/diego/anaconda3/envs/vae_ad/bin/python -m py_compile scripts/revision_bspc_2026/audit_threshold_recalibration_by_site_20260702.py",
            "/home/diego/anaconda3/envs/vae_ad/bin/python scripts/revision_bspc_2026/audit_threshold_recalibration_by_site_20260702.py",
        ],
        "outputs": [
            "threshold_recalibration_by_site.csv",
            "threshold_recalibration_by_site_report.md",
            "threshold_recalibration_plot.pdf",
            "command_log.json",
        ],
    }
    (OUT / "command_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")
    print(result[result["eligible"].eq(True)][["Site3", "N", "CN", "AD", "current_sensitivity_global_threshold", "oracle_sensitivity", "current_balanced_accuracy_global_threshold", "oracle_balanced_accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
