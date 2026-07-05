#!/usr/bin/env python3
"""Manufacturer-stratified OOF audit for the final BSPC ADNI model.

Read-only with respect to model/data artifacts. Writes only derived audit
tables into a new results directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


ROOT = Path("/home/diego/proyectos/vae_AD")
PRED_PATH = ROOT / "results/revision_bspc_2026/final_ad_classification_figure_table_20260622/primary_oof_predictions_validated.csv"
META_PATH = ROOT / "results/revision_bspc_2026/adni_035_metadata_rescue_preflight/patched_metadata_candidate.csv"
OUT_DIR = ROOT / "results/revision_bspc_2026/final_model_manufacturer_stratified_oof_audit_20260626"

PRIMARY_MODEL = "logreg_l2_original"
PRIMARY_FEATURES = "z_plus_age_sex"
PRIMARY_CALIB = "oof_ecdf"
PRIMARY_THRESHOLD_STRATEGY = "inner_oof_target_sens_ge_0p70_max_spec"


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else np.nan


def to_markdown_file(df: pd.DataFrame, path: Path, title: str | None = None) -> None:
    lines: list[str] = []
    if title:
        lines.extend([f"# {title}", ""])
    lines.append(df.to_markdown(index=False, floatfmt=".6g"))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def bootstrap_auc_pr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 20260626,
) -> dict[str, float | int | str]:
    n = len(y_true)
    min_class = int(min((y_true == 0).sum(), (y_true == 1).sum()))
    if len(np.unique(y_true)) < 2:
        return {
            "auc_ci_low": np.nan,
            "auc_ci_high": np.nan,
            "prauc_ci_low": np.nan,
            "prauc_ci_high": np.nan,
            "bootstrap_valid_resamples": 0,
            "ci_status": "insufficient_classes",
        }

    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    praucs: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        if len(np.unique(yb)) < 2:
            continue
        sb = y_score[idx]
        aucs.append(float(roc_auc_score(yb, sb)))
        praucs.append(float(average_precision_score(yb, sb)))

    if len(aucs) < 200:
        status = "unstable_too_few_valid_bootstraps"
    elif n < 30 or min_class < 8:
        status = "unstable_low_sample_or_minority_class"
    else:
        status = "ok"

    return {
        "auc_ci_low": float(np.quantile(aucs, 0.025)) if aucs else np.nan,
        "auc_ci_high": float(np.quantile(aucs, 0.975)) if aucs else np.nan,
        "prauc_ci_low": float(np.quantile(praucs, 0.025)) if praucs else np.nan,
        "prauc_ci_high": float(np.quantile(praucs, 0.975)) if praucs else np.nan,
        "bootstrap_valid_resamples": len(aucs),
        "ci_status": status,
    }


def compute_group_metrics(df: pd.DataFrame, manufacturer: str) -> dict[str, object]:
    y_true = df["y_true"].astype(int).to_numpy()
    y_score = df["y_score"].astype(float).to_numpy()
    y_pred = df["y_pred"].astype(int).to_numpy()

    n_cn = int((y_true == 0).sum())
    n_ad = int((y_true == 1).sum())
    n = int(len(df))

    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

    has_both = n_cn > 0 and n_ad > 0
    metrics: dict[str, object] = {
        "Manufacturer": manufacturer,
        "n_total": n,
        "n_cn": n_cn,
        "n_ad": n_ad,
        "roc_auc": float(roc_auc_score(y_true, y_score)) if has_both else np.nan,
        "pr_auc": float(average_precision_score(y_true, y_score)) if has_both else np.nan,
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "balanced_accuracy": np.nan,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)) if has_both else np.nan,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "cn_false_positive_rate": safe_div(fp, fp + tn),
        "ad_false_negative_rate": safe_div(fn, fn + tp),
        "mean_score_cn": float(df.loc[df["y_true"] == 0, "y_score"].mean()) if n_cn else np.nan,
        "mean_score_ad": float(df.loc[df["y_true"] == 1, "y_score"].mean()) if n_ad else np.nan,
        "threshold_min": float(df["threshold"].min()) if "threshold" in df else np.nan,
        "threshold_max": float(df["threshold"].max()) if "threshold" in df else np.nan,
        "metric_status": "ok_both_classes" if has_both else "insufficient_classes_for_auc",
    }
    if has_both:
        metrics["balanced_accuracy"] = float(np.nanmean([metrics["sensitivity"], metrics["specificity"]]))
        metrics.update(bootstrap_auc_pr(y_true, y_score))
    else:
        metrics.update(
            {
                "auc_ci_low": np.nan,
                "auc_ci_high": np.nan,
                "prauc_ci_low": np.nan,
                "prauc_ci_high": np.nan,
                "bootstrap_valid_resamples": 0,
                "ci_status": "insufficient_classes",
            }
        )
    return metrics


def validate_primary_predictions(pred: pd.DataFrame) -> list[str]:
    notes: list[str] = []
    if pred["SubjectID"].duplicated().any():
        dupes = pred.loc[pred["SubjectID"].duplicated(), "SubjectID"].tolist()
        raise ValueError(f"Duplicated SubjectID in primary OOF predictions: {dupes[:10]}")
    if len(pred) != 397:
        raise ValueError(f"Expected N=397 primary OOF rows, found {len(pred)}")
    counts = pred["y_true"].value_counts().to_dict()
    if counts.get(0, 0) != 300 or counts.get(1, 0) != 97:
        raise ValueError(f"Expected y_true CN=300/AD=97, found {counts}")
    expected = {
        "model_name": PRIMARY_MODEL,
        "feature_set": PRIMARY_FEATURES,
        "calib_method": PRIMARY_CALIB,
        "threshold_strategy": PRIMARY_THRESHOLD_STRATEGY,
    }
    for col, val in expected.items():
        vals = sorted(pred[col].dropna().unique().tolist())
        if vals != [val]:
            raise ValueError(f"Expected {col}={val}, found {vals}")
    notes.append("Primary OOF predictions validated: N=397, CN=300, AD=97, one row per subject.")
    notes.append(f"Readout: {PRIMARY_MODEL} / {PRIMARY_FEATURES} / {PRIMARY_CALIB} / {PRIMARY_THRESHOLD_STRATEGY}.")
    return notes


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(PRED_PATH)
    meta = pd.read_csv(META_PATH)
    validation_notes = validate_primary_predictions(pred)

    needed_meta = ["SubjectID", "ResearchGroup_Mapped", "Diagnosis", "Manufacturer", "Site3", "Age", "Sex"]
    missing_meta_cols = [c for c in needed_meta if c not in meta.columns]
    if missing_meta_cols:
        raise ValueError(f"Metadata missing expected columns: {missing_meta_cols}")

    merged = pred.merge(
        meta[needed_meta],
        on="SubjectID",
        how="left",
        suffixes=("_pred", "_meta"),
        validate="one_to_one",
    )

    merged["Manufacturer_final"] = merged["Manufacturer_meta"].fillna(merged["Manufacturer_pred"])
    manufacturer_conflicts = merged[
        merged["Manufacturer_meta"].notna()
        & merged["Manufacturer_pred"].notna()
        & (merged["Manufacturer_meta"].astype(str) != merged["Manufacturer_pred"].astype(str))
    ]
    if not manufacturer_conflicts.empty:
        raise ValueError(
            "Manufacturer mismatch between OOF predictions and harmonized metadata: "
            + ",".join(manufacturer_conflicts["SubjectID"].head(10).astype(str).tolist())
        )
    if merged["Manufacturer_final"].isna().any():
        missing = merged.loc[merged["Manufacturer_final"].isna(), "SubjectID"].tolist()
        raise ValueError(f"Missing manufacturer after merge for subjects: {missing[:10]}")

    rows = []
    for manufacturer, g in merged.groupby("Manufacturer_final", dropna=False):
        rows.append(compute_group_metrics(g.copy(), str(manufacturer)))
    metrics = pd.DataFrame(rows).sort_values(["n_total", "Manufacturer"], ascending=[False, True])

    metric_cols = [
        "Manufacturer",
        "n_total",
        "n_cn",
        "n_ad",
        "roc_auc",
        "auc_ci_low",
        "auc_ci_high",
        "pr_auc",
        "prauc_ci_low",
        "prauc_ci_high",
        "sensitivity",
        "specificity",
        "balanced_accuracy",
        "f1",
        "cn_false_positive_rate",
        "ad_false_negative_rate",
        "mean_score_cn",
        "mean_score_ad",
        "bootstrap_valid_resamples",
        "ci_status",
        "metric_status",
    ]
    confusion_cols = ["Manufacturer", "n_total", "n_cn", "n_ad", "tn", "fp", "fn", "tp"]
    fpr_cols = [
        "Manufacturer",
        "n_cn",
        "n_ad",
        "tn",
        "fp",
        "fn",
        "tp",
        "cn_false_positive_rate",
        "ad_false_negative_rate",
    ]

    metrics[metric_cols].to_csv(OUT_DIR / "manufacturer_stratified_metrics.csv", index=False)
    metrics[confusion_cols].to_csv(OUT_DIR / "manufacturer_confusion_matrices.csv", index=False)
    metrics[fpr_cols].to_csv(OUT_DIR / "manufacturer_cn_fpr_ad_fnr.csv", index=False)

    to_markdown_file(metrics[metric_cols], OUT_DIR / "manufacturer_stratified_metrics.md", "Manufacturer-Stratified Metrics")

    notes = []
    notes.extend(validation_notes)
    notes.append("Manufacturer values were cross-checked against the harmonized metadata; no conflicts were detected.")
    notes.append("Manufacturer and site were not used as classifier features in this audit; metrics are post hoc strata.")
    notes.append("")
    notes.append("## Interpretation")
    notes.append("")
    notes.append(
        "ROC-AUC and PR-AUC quantify within-manufacturer ranking performance when both CN and AD are present. "
        "Sensitivity, specificity, balanced accuracy, F1, CN FPR, and AD FNR quantify transfer of the locked primary "
        "operating threshold into each manufacturer stratum."
    )
    notes.append("")
    for _, r in metrics.iterrows():
        manu = r["Manufacturer"]
        if r["metric_status"] != "ok_both_classes":
            notes.append(
                f"- {manu}: insufficient class counts for stable AUC/PR-AUC estimation "
                f"(CN={int(r['n_cn'])}, AD={int(r['n_ad'])}); threshold error rates are descriptive only."
            )
            continue
        ci_note = (
            "bootstrap CI accepted"
            if r["ci_status"] == "ok"
            else f"bootstrap CI flagged as {r['ci_status']}"
        )
        notes.append(
            f"- {manu}: ROC-AUC={r['roc_auc']:.3f}, PR-AUC={r['pr_auc']:.3f}, "
            f"BA={r['balanced_accuracy']:.3f}, CN FPR={r['cn_false_positive_rate']:.3f}, "
            f"AD FNR={r['ad_false_negative_rate']:.3f}; {ci_note}."
        )
    notes.append("")
    notes.append(
        "A high within-stratum AUC with a high CN false-positive rate should be interpreted as a threshold-transfer "
        "or score-distribution problem rather than necessarily poor rank separation. Conversely, unstable or wide "
        "bootstrap intervals reflect limited manufacturer-specific sample size and should not be over-interpreted."
    )
    (OUT_DIR / "audit_interpretation.md").write_text("\n".join(notes) + "\n", encoding="utf-8")

    command_log = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "prediction_input": str(PRED_PATH),
        "metadata_input": str(META_PATH),
        "output_dir": str(OUT_DIR),
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 20260626,
        "guardrails": {
            "did_train_models": False,
            "did_modify_tensors": False,
            "did_modify_metadata": False,
            "did_modify_model_outputs": False,
            "did_modify_manuscript": False,
            "manufacturer_or_site_used_as_features": False,
        },
        "primary_readout": {
            "model_name": PRIMARY_MODEL,
            "feature_set": PRIMARY_FEATURES,
            "calib_method": PRIMARY_CALIB,
            "threshold_strategy": PRIMARY_THRESHOLD_STRATEGY,
        },
        "validation_notes": validation_notes,
        "output_files": [
            "manufacturer_stratified_metrics.csv",
            "manufacturer_stratified_metrics.md",
            "manufacturer_confusion_matrices.csv",
            "manufacturer_cn_fpr_ad_fnr.csv",
            "audit_interpretation.md",
            "command_log.txt",
        ],
    }
    (OUT_DIR / "command_log.txt").write_text(json.dumps(command_log, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote audit outputs to {OUT_DIR}")
    print(metrics[metric_cols].to_string(index=False))


if __name__ == "__main__":
    main()
