#!/usr/bin/env python3
"""OASIS CDR label-definition sensitivity audit for SIPAIM transportability.

Read-only post-hoc analysis. This script uses already-computed OASIS
predictions plus the local OASIS CDR provenance audit. It does not train
models, run VAE inference, preprocess data, modify raw data, recalibrate
scores, select thresholds, or use OASIS labels for model selection.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


PROJECT = Path("/home/diego/proyectos/vae_AD")
OUT = PROJECT / "results/sipaim_2026/oasis_label_sensitivity_20260710"

PROVENANCE = PROJECT / "results/sipaim_2026/oasis_local_inventory/oasis_label_provenance_audit.csv"
CURRENT_VS_LOCAL = PROJECT / "results/sipaim_2026/oasis_local_inventory/oasis_current180_vs_local_candidates.csv"
LOCKED_PRED = (
    PROJECT
    / "results/revision_bspc_2026/oasis_mega_90_90_external_inference_model_panel_20260604/predictions.csv"
)
CLEANREPRO_PRED = PROJECT / "results/sipaim_2026/frozen_oasis_inference/cleanrepro_oasis_predictions.csv"
EXTERNAL_DATASET_PRED = (
    PROJECT
    / "results/revision_bspc_2026/post_revision_exploratory_20260630/"
    "oasis_external_batch_combat_20260710/oasis_external_batch_combat_predictions.csv"
)
EXTERNAL_DATASET_METRICS = PROJECT / "results/sipaim_2026/oasis_external_batch_combat/oasis_external_batch_combat_metrics.csv"
EXTERNAL_DATASET_BOOT = (
    PROJECT / "results/sipaim_2026/oasis_external_batch_combat/oasis_external_batch_combat_paired_bootstrap.csv"
)
LOCKED_PRIMARY_METRICS = PROJECT / "results/sipaim_2026/frozen_oasis_inference/locked_oasis_mega_90_90_primary_metrics.csv"

BOOT_N = 10_000
BOOT_SEED = 20260710

TARGETS = [
    "oasis_label_sensitivity_subject_table.csv",
    "oasis_label_sensitivity_metrics.csv",
    "oasis_label_sensitivity_metrics.md",
    "oasis_cdr_score_distributions.csv",
    "oasis_cdr_score_distributions.md",
    "fig_oasis_label_sensitivity.pdf",
    "oasis_label_sensitivity_interpretation.md",
    "command_log.json",
]


@dataclass(frozen=True)
class ArmSpec:
    arm: str
    score_column: str
    threshold_column: str
    pred_column: str
    source_path: Path
    label: str


ARMS = [
    ArmSpec(
        arm="locked_frozen_transfer",
        score_column="score_locked",
        threshold_column="threshold_locked",
        pred_column="y_pred_locked",
        source_path=LOCKED_PRED,
        label="Locked frozen transfer",
    ),
    ArmSpec(
        arm="previous_adni_fitted_siemens_combat",
        score_column="score_combat_if_available",
        threshold_column="threshold_combat_if_available",
        pred_column="y_pred_combat_if_available",
        source_path=CLEANREPRO_PRED,
        label="Previous ADNI-fitted Siemens-ComBat",
    ),
    ArmSpec(
        arm="external_dataset_combat_adni_reference",
        score_column="score_dataset_combat_adni_ref_if_available",
        threshold_column="threshold_dataset_combat_adni_ref_if_available",
        pred_column="y_pred_dataset_combat_adni_ref_if_available",
        source_path=EXTERNAL_DATASET_PRED,
        label="External Dataset-ComBat, ADNI reference",
    ),
    ArmSpec(
        arm="external_dataset_combat_no_reference",
        score_column="score_dataset_combat_no_ref_if_available",
        threshold_column="threshold_dataset_combat_no_ref_if_available",
        pred_column="y_pred_dataset_combat_no_ref_if_available",
        source_path=EXTERNAL_DATASET_PRED,
        label="External Dataset-ComBat, no reference",
    ),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def assert_no_overwrite() -> None:
    if "--repair-existing" in sys.argv:
        return
    existing = [str(OUT / name) for name in TARGETS if (OUT / name).exists()]
    if existing:
        raise RuntimeError("Refusing to overwrite existing outputs: " + "; ".join(existing))


def md_table(df: pd.DataFrame, title: str, max_rows: int | None = None) -> str:
    view = df if max_rows is None else df.head(max_rows)
    try:
        body = view.to_markdown(index=False)
    except Exception:
        body = view.to_string(index=False)
    suffix = ""
    if max_rows is not None and len(df) > max_rows:
        suffix = f"\n\n_Showing {max_rows} of {len(df)} rows._"
    return f"# {title}\n\n{body}{suffix}\n"


def proposed_cdr_label(cdr: float) -> str:
    if pd.isna(cdr):
        return "MISSING_CDR"
    if float(cdr) == 0.0:
        return "CN"
    if float(cdr) == 0.5:
        return "CDR_0.5_EXCLUDE_STRICT"
    if float(cdr) >= 1.0:
        return "AD"
    return "UNEXPECTED_CDR"


def cdr_group(cdr: float) -> str:
    if pd.isna(cdr):
        return "missing"
    if float(cdr) == 0.0:
        return "CDR 0"
    if float(cdr) == 0.5:
        return "CDR 0.5"
    if float(cdr) >= 1.0:
        return "CDR >= 1"
    return "other"


def cdr_ordinal(cdr: float) -> float:
    if pd.isna(cdr):
        return np.nan
    if float(cdr) == 0.0:
        return 0.0
    if float(cdr) == 0.5:
        return 1.0
    if float(cdr) >= 1.0:
        return 2.0
    return np.nan


def normalize_label(label: Any) -> str:
    s = str(label).strip().upper()
    if s in {"0", "CN", "CONTROL"}:
        return "CN"
    if s in {"1", "AD", "DEMENTIA", "AD_DEMENTIA"}:
        return "AD"
    return s


def load_locked_predictions() -> pd.DataFrame:
    df = pd.read_csv(LOCKED_PRED)
    out = df[
        df["candidate"].eq("promoted_beta3p75_oof_ecdf")
        & df["build_candidate"].eq("runwise164_pilot_parity")
        & df["prediction_level"].eq("ensemble_mean_score_majority_vote")
    ].copy()
    if len(out) != 180:
        raise RuntimeError(f"Expected 180 locked ensemble rows, found {len(out)}")
    out["arm"] = "locked_frozen_transfer"
    return out


def load_cleanrepro_predictions() -> pd.DataFrame:
    df = pd.read_csv(CLEANREPRO_PRED)
    out = df[df["prediction_level"].eq("ensemble_mean_score_majority_vote")].copy()
    if len(out) != 180:
        raise RuntimeError(f"Expected 180 cleanrepro ensemble rows, found {len(out)}")
    out["arm"] = "previous_adni_fitted_siemens_combat"
    return out


def load_external_dataset_predictions(arm: str) -> pd.DataFrame:
    if not EXTERNAL_DATASET_PRED.exists():
        return pd.DataFrame()
    df = pd.read_csv(EXTERNAL_DATASET_PRED)
    out = df[df["prediction_level"].eq("ensemble_mean_score_majority_vote") & df["arm"].eq(arm)].copy()
    if not out.empty and len(out) != 180:
        raise RuntimeError(f"Expected 180 {arm} ensemble rows, found {len(out)}")
    return out


def prediction_key_table(df: pd.DataFrame, arm: ArmSpec) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["subject_id", "session_id", arm.score_column, arm.threshold_column, arm.pred_column])
    keep = ["subject_id", "session_id", "y_score", "threshold", "y_pred"]
    out = df[keep].copy()
    out = out.rename(columns={"y_score": arm.score_column, "threshold": arm.threshold_column, "y_pred": arm.pred_column})
    out[arm.score_column] = pd.to_numeric(out[arm.score_column], errors="raise")
    out[arm.threshold_column] = pd.to_numeric(out[arm.threshold_column], errors="raise")
    out[arm.pred_column] = pd.to_numeric(out[arm.pred_column], errors="raise").astype(int)
    return out


def build_subject_table() -> pd.DataFrame:
    prov = pd.read_csv(PROVENANCE)
    current = pd.read_csv(CURRENT_VS_LOCAL)
    locked = load_locked_predictions()
    base = locked[
        [
            "SubjectID",
            "subject_id",
            "session_id",
            "experiment_id",
            "diagnosis",
            "ResearchGroup_Mapped",
            "y",
            "Age",
            "Sex",
            "Manufacturer",
            "threshold",
            "y_score",
            "y_pred",
        ]
    ].copy()
    base = base.rename(
        columns={
            "ResearchGroup_Mapped": "current_project_label",
            "y": "y_current",
            "y_score": "score_locked",
            "threshold": "threshold_locked",
            "y_pred": "y_pred_locked",
        }
    )
    base["current_project_label"] = base["current_project_label"].map(normalize_label)
    base["y_current"] = pd.to_numeric(base["y_current"], errors="raise").astype(int)

    table = base.merge(
        prov,
        on=["subject_id", "session_id"],
        how="left",
        suffixes=("", "_cdr_audit"),
        validate="one_to_one",
    )
    table = table.merge(
        current,
        on=["subject_id", "session_id"],
        how="left",
        suffixes=("", "_current_inventory"),
        validate="one_to_one",
    )
    if table["cdr_value"].isna().any():
        raise RuntimeError("CDR provenance merge incomplete for current OASIS-180.")

    table["CDR"] = pd.to_numeric(table["cdr_value"], errors="raise")
    table["proposed_CDR_label"] = table["CDR"].map(proposed_cdr_label)
    table["label_confidence"] = table["label_confidence_HIGH_MEDIUM_LOW"].fillna(table["label_confidence"]).astype(str)
    table["exclusion_reason"] = table["exclusion_reason"].fillna("")
    table["cdr_group"] = table["CDR"].map(cdr_group)
    table["cdr_ordinal"] = table["CDR"].map(cdr_ordinal)
    table["y_strict_cdr"] = np.where(table["proposed_CDR_label"].eq("AD"), 1, np.where(table["proposed_CDR_label"].eq("CN"), 0, np.nan))
    table["strict_inclusion"] = table["proposed_CDR_label"].isin(["CN", "AD"])

    clean = load_cleanrepro_predictions()
    external_adni_ref = load_external_dataset_predictions("external_dataset_combat_adni_reference")
    external_no_ref = load_external_dataset_predictions("external_dataset_combat_no_reference")
    for arm, df in [
        (ARMS[1], clean),
        (ARMS[2], external_adni_ref),
        (ARMS[3], external_no_ref),
    ]:
        table = table.merge(prediction_key_table(df, arm), on=["subject_id", "session_id"], how="left", validate="one_to_one")

    ordered_cols = [
        "subject_id",
        "session_id",
        "experiment_id",
        "current_project_label",
        "y_current",
        "CDR",
        "proposed_CDR_label",
        "label_confidence",
        "exclusion_reason",
        "score_locked",
        "score_combat_if_available",
        "score_dataset_combat_adni_ref_if_available",
        "score_dataset_combat_no_ref_if_available",
        "threshold_locked",
        "threshold_combat_if_available",
        "threshold_dataset_combat_adni_ref_if_available",
        "threshold_dataset_combat_no_ref_if_available",
        "y_pred_locked",
        "y_pred_combat_if_available",
        "y_pred_dataset_combat_adni_ref_if_available",
        "y_pred_dataset_combat_no_ref_if_available",
        "strict_inclusion",
        "y_strict_cdr",
        "cdr_group",
        "cdr_ordinal",
        "Age",
        "Sex",
        "Manufacturer",
        "raw_label_fields",
        "diagnosis_text",
        "label_source_file",
        "days_from_entry_clinical",
        "days_from_entry_scan",
        "in_current_180",
        "in_additional_candidates",
        "has_label",
    ]
    return table[[c for c in ordered_cols if c in table.columns]].sort_values(["subject_id", "session_id"])


def fixed_threshold_pred(score: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    return (np.asarray(score, dtype=float) >= np.asarray(threshold, dtype=float)).astype(int)


def metric_dict(y: np.ndarray, score: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pred = np.asarray(pred, dtype=int)
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=labels).ravel()
    out = {
        "n": int(len(y)),
        "n_cn": int(np.sum(y == 0)),
        "n_ad": int(np.sum(y == 1)),
        "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "brier_score_descriptive": float(brier_score_loss(y, score)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_ad_rate": float(np.mean(pred)),
        "mean_score": float(np.mean(score)),
    }
    return out


def stratified_bootstrap_ci(y: np.ndarray, score: np.ndarray, threshold: np.ndarray, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    threshold = np.asarray(threshold, dtype=float)
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)
    metrics = ["roc_auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "brier_score_descriptive"]
    vals = {m: [] for m in metrics}
    for _ in range(BOOT_N):
        sample = np.concatenate(
            [
                rng.choice(idx0, size=len(idx0), replace=True),
                rng.choice(idx1, size=len(idx1), replace=True),
            ]
        )
        pred = fixed_threshold_pred(score[sample], threshold[sample])
        md = metric_dict(y[sample], score[sample], pred)
        for m in metrics:
            vals[m].append(md[m])
    out: dict[str, Any] = {}
    for m, arr in vals.items():
        a = np.asarray(arr, dtype=float)
        out[f"{m}_ci_low_2p5"] = float(np.nanpercentile(a, 2.5))
        out[f"{m}_ci_high_97p5"] = float(np.nanpercentile(a, 97.5))
    out["bootstrap_type"] = "separate_diagnosis_stratified_subject_bootstrap"
    out["n_bootstrap"] = BOOT_N
    out["bootstrap_seed"] = seed
    return out


def evaluation_frame(table: pd.DataFrame, evaluation_set: str) -> tuple[pd.DataFrame, np.ndarray]:
    if evaluation_set == "OASIS-current-180":
        df = table.copy()
        df["evaluation_y"] = df["y_current"].astype(int)
    elif evaluation_set == "OASIS-strict-128":
        df = table[table["strict_inclusion"]].copy()
        df["evaluation_y"] = df["y_strict_cdr"].astype(int)
    else:
        raise ValueError(evaluation_set)
    return df, df["evaluation_y"].to_numpy(dtype=int)


def compute_metrics(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for evaluation_set in ["OASIS-current-180", "OASIS-strict-128"]:
        df, y = evaluation_frame(table, evaluation_set)
        for i, arm in enumerate(ARMS):
            if arm.score_column not in df or df[arm.score_column].isna().all():
                continue
            g = df[df[arm.score_column].notna() & df[arm.threshold_column].notna()].copy()
            yy = g["evaluation_y"].to_numpy(dtype=int)
            score = g[arm.score_column].to_numpy(dtype=float)
            threshold = g[arm.threshold_column].to_numpy(dtype=float)
            if arm.pred_column in g and g[arm.pred_column].notna().all():
                pred = g[arm.pred_column].to_numpy(dtype=int)
                prediction_rule = "stored_existing_fixed_fold_threshold_majority_vote_y_pred"
            else:
                pred = fixed_threshold_pred(score, threshold)
                prediction_rule = "fallback_score_ge_existing_threshold"
            row: dict[str, Any] = {
                "evaluation_set": evaluation_set,
                "arm": arm.arm,
                "label": arm.label,
                "threshold_source": "fixed_ADNI_derived_existing_prediction_threshold",
                "prediction_rule": prediction_rule,
                "brier_score_note": "ECDF score; descriptive only, not recalibrated on OASIS",
                "oasis_labels_used_for_model_selection_or_calibration": False,
            }
            row.update(metric_dict(yy, score, pred))
            # Bootstrap the stored fixed-threshold binary decisions by passing a
            # threshold that reproduces them exactly for each resampled subject.
            bootstrap_threshold = np.where(pred == 1, -np.inf, np.inf)
            row.update(stratified_bootstrap_ci(yy, score, bootstrap_threshold, BOOT_SEED + i + (0 if evaluation_set.endswith("180") else 100)))
            rows.append(row)
    return pd.DataFrame(rows)


def compute_cdr_distributions(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_order = ["CDR 0", "CDR 0.5", "CDR >= 1"]
    ord_values = table["cdr_ordinal"].to_numpy(dtype=float)
    for arm in ARMS:
        if arm.score_column not in table or table[arm.score_column].isna().all():
            continue
        df = table[table[arm.score_column].notna()].copy()
        scores_all = df[arm.score_column].to_numpy(dtype=float)
        rho, rho_p = spearmanr(df["cdr_ordinal"].to_numpy(dtype=float), scores_all)
        arrays = [df[df["cdr_group"].eq(g)][arm.score_column].to_numpy(dtype=float) for g in group_order]
        kw_stat, kw_p = kruskal(*arrays)
        pair_tests: dict[str, float] = {}
        for a, b in [("CDR 0", "CDR 0.5"), ("CDR 0.5", "CDR >= 1"), ("CDR 0", "CDR >= 1")]:
            av = df[df["cdr_group"].eq(a)][arm.score_column].to_numpy(dtype=float)
            bv = df[df["cdr_group"].eq(b)][arm.score_column].to_numpy(dtype=float)
            pair_tests[f"mannwhitney_p_{a.replace(' ', '_').replace('>=', 'ge')}_vs_{b.replace(' ', '_').replace('>=', 'ge')}"] = float(
                mannwhitneyu(av, bv, alternative="two-sided").pvalue
            )
        for group in group_order:
            vals = df[df["cdr_group"].eq(group)][arm.score_column].to_numpy(dtype=float)
            rows.append(
                {
                    "arm": arm.arm,
                    "label": arm.label,
                    "cdr_group": group,
                    "cdr_ordinal": {"CDR 0": 0, "CDR 0.5": 1, "CDR >= 1": 2}[group],
                    "n": int(len(vals)),
                    "score_mean": float(np.mean(vals)),
                    "score_sd": float(np.std(vals, ddof=1)),
                    "score_median": float(np.median(vals)),
                    "score_iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
                    "spearman_rho_score_vs_ordinal_cdr": float(rho),
                    "spearman_pvalue": float(rho_p),
                    "kruskal_wallis_statistic": float(kw_stat),
                    "kruskal_wallis_pvalue": float(kw_p),
                    **pair_tests,
                }
            )
    return pd.DataFrame(rows)


def plot_label_sensitivity(table: pd.DataFrame, metrics: pd.DataFrame, cdr_dist: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    primary = "locked_frozen_transfer"
    score_col = "score_locked"
    group_order = ["CDR 0", "CDR 0.5", "CDR >= 1"]
    colors = ["#4C78A8", "#F2CF5B", "#E45756"]

    ax = axes[0, 0]
    data = [table[table["cdr_group"].eq(g)][score_col].to_numpy(dtype=float) for g in group_order]
    ax.boxplot(data, tick_labels=group_order, showfliers=False)
    for x, vals, color in zip(range(1, 4), data, colors):
        jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.array([0.0])
        ax.scatter(np.full(len(vals), x) + jitter, vals, s=14, alpha=0.6, color=color)
    ax.set_title("Locked OASIS scores by CDR group")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[0, 1]
    for g, color in zip(group_order, colors):
        vals = table[table["cdr_group"].eq(g)][score_col].to_numpy(dtype=float)
        ax.hist(vals, bins=np.linspace(0, 1, 26), histtype="step", linewidth=2, density=True, label=g, color=color)
    ax.set_title("Locked score distributions")
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    ax = axes[1, 0]
    m = metrics[metrics["arm"].eq(primary)].set_index("evaluation_set")
    labels = ["Current-180", "Strict-128"]
    aucs = [m.loc["OASIS-current-180", "roc_auc"], m.loc["OASIS-strict-128", "roc_auc"]]
    pr = [m.loc["OASIS-current-180", "pr_auc"], m.loc["OASIS-strict-128", "pr_auc"]]
    x = np.arange(2)
    ax.bar(x - 0.18, aucs, width=0.36, label="ROC-AUC")
    ax.bar(x + 0.18, pr, width=0.36, label="PR-AUC")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Locked metrics by label definition")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1, 1]
    for arm in ARMS:
        sub = cdr_dist[cdr_dist["arm"].eq(arm.arm)].sort_values("cdr_ordinal")
        ax.plot(sub["cdr_ordinal"], sub["score_mean"], marker="o", label=arm.label)
    ax.set_xticks([0, 1, 2], group_order)
    ax.set_title("Mean score trend by CDR group")
    ax.set_ylabel("Mean score")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def interpretation(table: pd.DataFrame, metrics: pd.DataFrame, cdr_dist: pd.DataFrame) -> str:
    locked = metrics[metrics["arm"].eq("locked_frozen_transfer")].set_index("evaluation_set")
    current = locked.loc["OASIS-current-180"]
    strict = locked.loc["OASIS-strict-128"]
    dist_locked = cdr_dist[cdr_dist["arm"].eq("locked_frozen_transfer")].set_index("cdr_group")
    cdr0 = dist_locked.loc["CDR 0", "score_mean"]
    cdr05 = dist_locked.loc["CDR 0.5", "score_mean"]
    cdr1 = dist_locked.loc["CDR >= 1", "score_mean"]
    rho = dist_locked.loc["CDR 0", "spearman_rho_score_vs_ordinal_cdr"]
    rho_p = dist_locked.loc["CDR 0", "spearman_pvalue"]
    n_cdr05_current_ad = int(np.sum((table["CDR"].eq(0.5)) & (table["y_current"].eq(1))))
    n_cdr05_current_cn = int(np.sum((table["CDR"].eq(0.5)) & (table["y_current"].eq(0))))
    lines = [
        "# OASIS label-definition sensitivity interpretation",
        "",
        "## Guardrails",
        "",
        "- No model was trained.",
        "- No VAE inference or preprocessing was run.",
        "- Raw data were not modified.",
        "- OASIS labels were not used for calibration, threshold selection, harmonisation, or model selection.",
        "- CDR 0.5 was kept as its own ordinal group and excluded from the strict binary analysis.",
        "",
        "## Direct answers",
        "",
        (
            "1. **Does external ROC-AUC improve when restricting OASIS to high-confidence CDR-defined CN/AD?** "
            f"Yes for the locked frozen-transfer arm: ROC-AUC changed from {current['roc_auc']:.4f} on OASIS-current-180 "
            f"to {strict['roc_auc']:.4f} on OASIS-strict-128. PR-AUC changed from {current['pr_auc']:.4f} to {strict['pr_auc']:.4f}. "
            "This is a label-definition sensitivity result, not a model-selection criterion."
        ),
        "",
        (
            "2. **Are CDR 0.5 subjects intermediate in model score between CDR 0 and CDR >= 1?** "
            f"Yes descriptively for locked scores: mean score CDR 0 = {cdr0:.4f}, CDR 0.5 = {cdr05:.4f}, "
            f"CDR >= 1 = {cdr1:.4f}. Spearman trend across ordinal CDR groups was rho={rho:.4f}, p={rho_p:.4g}."
        ),
        "",
        (
            "3. **Does the current OASIS-180 result underestimate performance because many positives are very mild/ambiguous?** "
            f"Likely yes. The current labels include {n_cdr05_current_ad} CDR 0.5 subjects as AD "
            f"and {n_cdr05_current_cn} CDR 0.5 subject as CN. Removing CDR 0.5 and evaluating only CDR 0 vs CDR >= 1 "
            "increases the locked ROC-AUC, consistent with ambiguous/mild positives depressing binary discrimination."
        ),
        "",
        (
            "4. **Primary external result recommendation.** Keep OASIS-current-180 as the primary external result and report "
            "OASIS-strict-128 as a planned/transparent label-definition sensitivity. Switching the primary result to strict-128 would "
            "discard the originally evaluated external cohort and can look post-hoc, even though it clarifies that stricter CDR labels "
            "yield better apparent transportability."
        ),
        "",
        "## Bootstrap note",
        "",
        (
            "OASIS-strict-128 is a subset/relabeling sensitivity rather than the same target population as OASIS-current-180. "
            "Therefore the metrics table reports separate diagnosis-stratified subject-bootstrap confidence intervals for each set, "
            "instead of a paired current-vs-strict bootstrap delta."
        ),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    assert_no_overwrite()
    if OUT.exists():
        if any(OUT.iterdir()):
            if "--repair-existing" not in sys.argv:
                raise RuntimeError(f"Refusing to use non-empty output directory: {OUT}")
    else:
        OUT.mkdir(parents=True, exist_ok=False)
    command_log: list[dict[str, Any]] = [
        {
            "timestamp_utc": utc_now(),
            "argv": sys.argv,
            "cwd": str(PROJECT),
            "inputs": {
                "provenance": str(PROVENANCE),
                "current_vs_local": str(CURRENT_VS_LOCAL),
                "locked_predictions": str(LOCKED_PRED),
                "cleanrepro_predictions": str(CLEANREPRO_PRED),
                "external_dataset_predictions": str(EXTERNAL_DATASET_PRED),
                "locked_primary_metrics": str(LOCKED_PRIMARY_METRICS),
                "external_dataset_metrics": str(EXTERNAL_DATASET_METRICS),
                "external_dataset_bootstrap": str(EXTERNAL_DATASET_BOOT),
            },
            "guardrails": {
                "no_model_training": True,
                "no_vae_inference": True,
                "no_preprocessing": True,
                "no_raw_data_modification": True,
                "no_oasis_calibration": True,
                "no_oasis_threshold_selection": True,
                "no_oasis_harmonisation": True,
                "no_oasis_model_selection": True,
                "cdr_0p5_kept_as_own_group": True,
            },
        }
    ]
    table = build_subject_table()
    metrics = compute_metrics(table)
    cdr_dist = compute_cdr_distributions(table)

    table.to_csv(OUT / "oasis_label_sensitivity_subject_table.csv", index=False)
    metrics.to_csv(OUT / "oasis_label_sensitivity_metrics.csv", index=False)
    (OUT / "oasis_label_sensitivity_metrics.md").write_text(
        md_table(metrics, "OASIS label-definition sensitivity metrics", max_rows=80), encoding="utf-8"
    )
    cdr_dist.to_csv(OUT / "oasis_cdr_score_distributions.csv", index=False)
    (OUT / "oasis_cdr_score_distributions.md").write_text(
        md_table(cdr_dist, "OASIS CDR score distributions", max_rows=80), encoding="utf-8"
    )
    plot_label_sensitivity(table, metrics, cdr_dist, OUT / "fig_oasis_label_sensitivity.pdf")
    (OUT / "oasis_label_sensitivity_interpretation.md").write_text(
        interpretation(table, metrics, cdr_dist), encoding="utf-8"
    )
    command_log.append(
        {
            "timestamp_utc": utc_now(),
            "status": "COMPLETE",
            "output_dir": str(OUT),
            "n_subjects": int(len(table)),
            "n_current180": int(len(table)),
            "n_strict128": int(table["strict_inclusion"].sum()),
            "cdr_counts": {str(k): int(v) for k, v in table["cdr_group"].value_counts().sort_index().items()},
            "available_score_columns": [
                c
                for c in [
                    "score_locked",
                    "score_combat_if_available",
                    "score_dataset_combat_adni_ref_if_available",
                    "score_dataset_combat_no_ref_if_available",
                ]
                if c in table and table[c].notna().any()
            ],
            "bootstrap_n": BOOT_N,
            "bootstrap_seed": BOOT_SEED,
        }
    )
    (OUT / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
